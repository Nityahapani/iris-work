"""
experiments/temporal_vs_static.py

Directly tests the core thesis: does retiring edges DURING training beat
selecting them ONCE before training, at the same final sparsity?

Static path:
  1. Score all edges using TGS's influence estimator (deg-product based)
  2. Remove the bottom-k% upfront (before any training)
  3. Train fresh GCN on that sparse graph for full 300 epochs

Temporal TGS path:
  1. Start with dense graph
  2. Train while gradually retiring edges using the same scoring function
  3. Final graph has the same sparsity as the static path

Metrics compared at matched sparsity:
  - Test accuracy (TGS run)
  - Fresh GCN accuracy on final edge set
  - FLOPs (training cost)
  - Wall-clock time
  - Peak memory
  - Representation distortion bound
"""

import sys, os, json, time, tracemalloc
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import degree

from tgs.core.temporal_graph import TemporalGraph
from tgs.core.influence import GradientNormEstimator
from tgs.models.gcn import TemporalGCN
from tgs.schedulers.adaptive_scheduler import AdaptiveRetirementScheduler
from tgs.evaluation.flops import FLOPsCounter
from tgs.utils.reproducibility import set_seed

DEVICE = torch.device("cpu")
SEED   = 42
EPOCHS = 300


# ── Static baseline: score-then-prune ──────────────────────────────────────

def static_prune(edge_index, num_nodes, target_sparsity, method="degree"):
    """
    Remove edges upfront using the same structural scoring as TGS.
    method: 'degree' uses deg_product (our estimator's primary score)
            'random' for ablation
    """
    m = edge_index.shape[1]
    n_remove = int(m * target_sparsity)
    if n_remove == 0:
        return edge_index

    src, dst = edge_index[0], edge_index[1]
    deg = degree(dst, num_nodes, dtype=torch.float)

    if method == "degree":
        # Same formula as TGS estimator: retire high deg_product first
        score = deg[src] * deg[dst]          # high = safe to remove
        # Additionally protect top-10% er_proxy bridges (same gate as TGS)
        er = 1.0 / deg[src].clamp(min=1) + 1.0 / deg[dst].clamp(min=1)
        er_thresh = torch.quantile(er, 0.90)
        protected = er >= er_thresh
        score[protected] = -1.0             # never remove protected edges
    elif method == "random":
        rng = torch.Generator(); rng.manual_seed(SEED)
        score = torch.rand(m, generator=rng)
    else:
        raise ValueError(method)

    # Remove the n_remove highest-scored edges
    _, sorted_idx = score.sort(descending=True)
    remove_set = set(sorted_idx[:n_remove].tolist())
    keep = torch.tensor([i not in remove_set for i in range(m)], dtype=torch.bool)
    return edge_index[:, keep]


def train_static(data, num_features, num_classes, edge_index, label):
    """Train fresh GCN on a fixed sparse graph. Returns metrics + timing."""
    set_seed(SEED)
    m0_full = data.edge_index.shape[1]
    m_sparse = edge_index.shape[1]

    tracemalloc.start()
    t0 = time.perf_counter()

    model = TemporalGCN(num_features, 64, num_classes, 2, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    flops = FLOPsCounter(m_sparse, 2, 64)  # constant throughout

    best_val = best_test = 0.0
    val_history = []

    for epoch in range(EPOCHS):
        model.train()
        loss = F.cross_entropy(
            model(data.x, edge_index)[data.train_mask],
            data.y[data.train_mask])
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad(): el = model(data.x, edge_index)
        preds    = el.argmax(-1)
        val_acc  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        val_history.append(val_acc)
        flops.record_step(m_sparse)
        if val_acc > best_val: best_val, best_test = val_acc, test_acc

    elapsed = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "label":         label,
        "test_acc":      best_test,
        "sparsity":      1.0 - m_sparse / m0_full,
        "flops_red":     flops.summary()["flops_reduction"],
        "total_flops":   flops.total_flops,
        "runtime_s":     elapsed,
        "peak_mem_mb":   peak_mem / 1e6,
        "val_history":   val_history,
        "distortion":    None,   # static has no theoretical bound
        "method":        "static",
    }


# ── Temporal TGS run ────────────────────────────────────────────────────────

def train_temporal(data, num_features, num_classes):
    set_seed(SEED)
    m0 = data.edge_index.shape[1]

    tracemalloc.start()
    t0 = time.perf_counter()

    tg  = TemporalGraph(data.edge_index, data.num_nodes, device=DEVICE)
    est = GradientNormEstimator(m0, DEVICE,
            edge_index=data.edge_index, num_nodes=data.num_nodes,
            alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
    model = TemporalGCN(num_features, 64, num_classes, 2, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(
                list(model.parameters()) + [est.edge_weights],
                lr=0.01, weight_decay=5e-4)
    sched = AdaptiveRetirementScheduler(tg,
                epsilon_max=5e-3, epsilon_min=1e-5, anneal_steps=100,
                warmup_steps=40, max_retire_frac=0.10,
                max_sparsity=0.65, retire_every=2)
    flops     = FLOPsCounter(m0, 2, 64)
    best_val  = best_test = 0.0
    val_history = []

    for epoch in range(EPOCHS):
        model.train(); am = tg.active_mask
        logits = model(data.x, tg.edge_index, est.edge_weights[am])
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        opt.zero_grad(); loss.backward(); est.update_influence(am); opt.step()

        model.eval()
        with torch.no_grad(): el = model(data.x, tg.edge_index)
        preds    = el.argmax(-1)
        val_acc  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        val_history.append(val_acc)
        sched.update_val_acc(val_acc); sched.step(est.influence_scores(am))
        flops.record_step(tg.mt); tg.step()
        if val_acc > best_val: best_val, best_test = val_acc, test_acc

    elapsed = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    sp = tg.sparsity
    f  = flops.summary()
    rs = sched.summary()

    # Fresh GCN on TGS-selected edges (key comparison point)
    tgs_ei = tg.edge_index.clone()
    set_seed(SEED)
    m2 = TemporalGCN(num_features, 64, num_classes, 2, 0.5).to(DEVICE)
    o2 = torch.optim.Adam(m2.parameters(), lr=0.01, weight_decay=5e-4)
    bv2 = bt2 = 0.0
    for epoch in range(EPOCHS):
        m2.train()
        F.cross_entropy(m2(data.x, tgs_ei)[data.train_mask],
                        data.y[data.train_mask]).backward()
        o2.step(); o2.zero_grad()
        m2.eval()
        with torch.no_grad(): el2 = m2(data.x, tgs_ei)
        p2 = el2.argmax(-1)
        v2 = (p2[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        t2 = (p2[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        if v2 > bv2: bv2, bt2 = v2, t2

    return {
        "label":          "Temporal TGS",
        "test_acc":       best_test,
        "fresh_acc":      bt2,
        "sparsity":       sp,
        "flops_red":      f["flops_reduction"],
        "total_flops":    flops.total_flops,
        "runtime_s":      elapsed,
        "peak_mem_mb":    peak_mem / 1e6,
        "val_history":    val_history,
        "distortion":     rs["cumulative_distortion_bound"],
        "method":         "temporal",
    }, tgs_ei, sp


def main():
    for ds_name in ["Cora", "CiteSeer"]:
        dataset = Planetoid(root="./data", name=ds_name, transform=NormalizeFeatures())
        data    = dataset[0].to(DEVICE)
        nf, nc  = dataset.num_features, dataset.num_classes
        m0      = data.edge_index.shape[1]

        print(f"\n{'='*80}")
        print(f"Temporal vs Static — {ds_name}  (n={data.num_nodes}, m0={m0})")
        print(f"{'='*80}")

        # 1. Run temporal TGS first to learn the final sparsity
        print("  Running Temporal TGS...")
        tgs_result, tgs_ei, tgs_sp = train_temporal(data, nf, nc)
        print(f"  → test={tgs_result['test_acc']:.4f}  fresh={tgs_result['fresh_acc']:.4f}  "
              f"sp={tgs_sp:.3f}  time={tgs_result['runtime_s']:.1f}s")

        # 2. Static paths at the SAME sparsity
        results = [tgs_result]
        for method, label in [
            ("degree", "Static (degree score)"),
            ("random", "Static (random)"),
        ]:
            print(f"  Running {label} @ sp={tgs_sp:.3f}...")
            ei_pruned = static_prune(data.edge_index, data.num_nodes, tgs_sp, method)
            r = train_static(data, nf, nc, ei_pruned, label)
            results.append(r)
            print(f"  → test={r['test_acc']:.4f}  sp={r['sparsity']:.3f}  time={r['runtime_s']:.1f}s")

        # 3. Dense baseline
        print("  Running Dense GCN...")
        ei_dense = data.edge_index
        r_dense  = train_static(data, nf, nc, ei_dense, "Dense GCN")
        results.append(r_dense)
        print(f"  → test={r_dense['test_acc']:.4f}  sp=0.000  time={r_dense['runtime_s']:.1f}s")

        # 4. Print table
        dense_time = r_dense["runtime_s"]
        dense_mem  = r_dense["peak_mem_mb"]
        dense_flop = r_dense["total_flops"]

        print(f"\n  {'Method':<25} {'Test':>7} {'Fresh':>7} {'Sparsity':>9} "
              f"{'FLOPs↓':>7} {'Runtime':>9} {'Memory':>8} {'Distortion':>11}")
        print(f"  {'-'*95}")
        for r in results:
            fresh = f"{r.get('fresh_acc', r['test_acc']):.4f}"
            dist  = f"{r['distortion']:.4f}" if r.get('distortion') else "N/A (static)"
            rt    = f"{r['runtime_s']/dense_time*100:.0f}%"
            mem   = f"{r['peak_mem_mb']/dense_mem*100:.0f}%"
            print(f"  {r['label']:<25} {r['test_acc']:>7.4f} {fresh:>7} "
                  f"{r['sparsity']:>9.3f} {r['flops_red']:>7.3f} "
                  f"{rt:>9} {mem:>8} {dist:>11}")

        # 5. Save
        os.makedirs("results", exist_ok=True)
        out = {"dataset": ds_name, "m0": m0, "results": results}
        with open(f"results/temporal_vs_static_{ds_name.lower()}.json", "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\n  Saved results/temporal_vs_static_{ds_name.lower()}.json")


if __name__ == "__main__":
    main()
