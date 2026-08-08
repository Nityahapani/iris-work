"""
experiments/runtime_analysis.py

Real runtime measurements for the ISEF Software Design category.

Measures per method:
  - Total training time (wall clock)
  - Time per epoch (mean ± std)
  - Peak memory (tracemalloc)
  - Estimator overhead (time inside update_influence)
  - Scheduler overhead (time inside step)
  - Inference time (one forward pass on full graph)
  - FLOPs (theoretical)

Produces a normalised table with Dense = 100% baseline.
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

from tgs.core.temporal_graph import TemporalGraph
from tgs.core.influence import GradientNormEstimator
from tgs.models.gcn import TemporalGCN
from tgs.schedulers.adaptive_scheduler import AdaptiveRetirementScheduler
from tgs.evaluation.flops import FLOPsCounter
from tgs.utils.reproducibility import set_seed

DEVICE = torch.device("cpu")
SEED   = 42
EPOCHS = 300


def measure_dense(data, num_features, num_classes):
    set_seed(SEED)
    model = TemporalGCN(num_features, 64, num_classes, 2, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    m0    = data.edge_index.shape[1]
    flops = FLOPsCounter(m0, 2, 64)

    tracemalloc.start()
    epoch_times = []
    best_val = best_test = 0.0

    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        model.train()
        loss = F.cross_entropy(
            model(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): el = model(data.x, data.edge_index)
        preds    = el.argmax(-1)
        val_acc  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        flops.record_step(m0)
        epoch_times.append(time.perf_counter() - t0)
        if val_acc > best_val: best_val, best_test = val_acc, test_acc

    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

    # Inference time (10 runs, take median)
    model.eval()
    inf_times = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter()
            model(data.x, data.edge_index)
            inf_times.append(time.perf_counter() - t0)

    return {
        "label":          "Dense GCN",
        "test_acc":       best_test,
        "sparsity":       0.0,
        "total_time_s":   sum(epoch_times),
        "epoch_time_mean":np.mean(epoch_times),
        "epoch_time_std": np.std(epoch_times),
        "peak_mem_mb":    peak / 1e6,
        "flops_red":      0.0,
        "total_flops":    flops.total_flops,
        "inference_ms":   np.median(inf_times) * 1000,
        "estimator_overhead_ms": 0.0,
        "scheduler_overhead_ms": 0.0,
    }


def measure_static(data, num_features, num_classes, target_sparsity):
    """Static pruning: degree-score then train fresh."""
    from torch_geometric.utils import degree as pyg_deg
    set_seed(SEED)
    m0  = data.edge_index.shape[1]
    src = data.edge_index[0]; dst = data.edge_index[1]
    deg = pyg_deg(dst, data.num_nodes, dtype=torch.float)
    er  = 1.0/deg[src].clamp(min=1) + 1.0/deg[dst].clamp(min=1)
    er_thresh = torch.quantile(er, 0.90)
    score = deg[src] * deg[dst]; score[er >= er_thresh] = -1.0
    n_remove = int(m0 * target_sparsity)
    _, sidx = score.sort(descending=True)
    remove  = set(sidx[:n_remove].tolist())
    keep    = torch.tensor([i not in remove for i in range(m0)], dtype=torch.bool)
    ei      = data.edge_index[:, keep]; m_sp = ei.shape[1]

    model = TemporalGCN(num_features, 64, num_classes, 2, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    flops = FLOPsCounter(m_sp, 2, 64)

    tracemalloc.start()
    epoch_times = []; best_val = best_test = 0.0

    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        model.train()
        loss = F.cross_entropy(model(data.x, ei)[data.train_mask], data.y[data.train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): el = model(data.x, ei)
        preds    = el.argmax(-1)
        val_acc  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        flops.record_step(m_sp)
        epoch_times.append(time.perf_counter() - t0)
        if val_acc > best_val: best_val, best_test = val_acc, test_acc

    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

    model.eval()
    inf_times = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter(); model(data.x, ei)
            inf_times.append(time.perf_counter() - t0)

    return {
        "label":          "Static (degree-prune)",
        "test_acc":       best_test,
        "sparsity":       1 - m_sp / m0,
        "total_time_s":   sum(epoch_times),
        "epoch_time_mean":np.mean(epoch_times),
        "epoch_time_std": np.std(epoch_times),
        "peak_mem_mb":    peak / 1e6,
        "flops_red":      flops.summary()["flops_reduction"],
        "total_flops":    flops.total_flops,
        "inference_ms":   np.median(inf_times) * 1000,
        "estimator_overhead_ms": 0.0,
        "scheduler_overhead_ms": 0.0,
    }


def measure_tgs(data, num_features, num_classes):
    set_seed(SEED)
    m0  = data.edge_index.shape[1]
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
    flops = FLOPsCounter(m0, 2, 64)

    tracemalloc.start()
    epoch_times = []; est_times = []; sched_times = []
    best_val = best_test = 0.0

    for epoch in range(EPOCHS):
        t0 = time.perf_counter()

        model.train(); am = tg.active_mask
        logits = model(data.x, tg.edge_index, est.edge_weights[am])
        loss   = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        opt.zero_grad(); loss.backward()

        # Time estimator update
        te0 = time.perf_counter()
        est.update_influence(am)
        est_times.append(time.perf_counter() - te0)

        opt.step()

        model.eval()
        with torch.no_grad(): el = model(data.x, tg.edge_index)
        preds    = el.argmax(-1)
        val_acc  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        # Time scheduler
        ts0 = time.perf_counter()
        sched.update_val_acc(val_acc)
        scores = est.influence_scores(am)
        sched.step(scores)
        sched_times.append(time.perf_counter() - ts0)

        flops.record_step(tg.mt); tg.step()
        epoch_times.append(time.perf_counter() - t0)
        if val_acc > best_val: best_val, best_test = val_acc, test_acc

    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

    model.eval(); final_ei = tg.edge_index
    inf_times = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter(); model(data.x, final_ei)
            inf_times.append(time.perf_counter() - t0)

    f  = flops.summary()
    rs = sched.summary()
    return {
        "label":           "TGS (ours)",
        "test_acc":        best_test,
        "sparsity":        tg.sparsity,
        "total_time_s":    sum(epoch_times),
        "epoch_time_mean": np.mean(epoch_times),
        "epoch_time_std":  np.std(epoch_times),
        "peak_mem_mb":     peak / 1e6,
        "flops_red":       f["flops_reduction"],
        "total_flops":     flops.total_flops,
        "inference_ms":    np.median(inf_times) * 1000,
        "estimator_overhead_ms": np.mean(est_times) * 1000,
        "scheduler_overhead_ms": np.mean(sched_times) * 1000,
        "distortion":      rs["cumulative_distortion_bound"],
    }


def print_table(results, ref):
    ref_time = ref["total_time_s"]
    ref_mem  = ref["peak_mem_mb"]
    ref_flop = ref["total_flops"]
    ref_inf  = ref["inference_ms"]

    print(f"\n  {'Method':<25} {'Acc':>6} {'Sp':>6} {'FLOPs':>7} "
          f"{'Runtime':>8} {'Memory':>8} {'Inference':>10} {'Est OH':>8} {'Sched OH':>9}")
    print(f"  {'-'*100}")
    for r in results:
        rt  = f"{r['total_time_s']/ref_time*100:.0f}%"
        mem = f"{r['peak_mem_mb']/ref_mem*100:.0f}%"
        fl  = f"{(1-r['flops_red'])*100:.0f}%"
        inf = f"{r['inference_ms']:.2f}ms"
        eoh = f"{r.get('estimator_overhead_ms',0):.2f}ms"
        soh = f"{r.get('scheduler_overhead_ms',0):.2f}ms"
        print(f"  {r['label']:<25} {r['test_acc']:>6.4f} {r['sparsity']:>6.3f} "
              f"{fl:>7} {rt:>8} {mem:>8} {inf:>10} {eoh:>8} {soh:>9}")


def main():
    all_output = {}
    for ds_name in ["Cora", "CiteSeer"]:
        dataset = Planetoid(root="./data", name=ds_name, transform=NormalizeFeatures())
        data    = dataset[0].to(DEVICE)
        nf, nc  = dataset.num_features, dataset.num_classes

        print(f"\n{'='*75}")
        print(f"Runtime Analysis — {ds_name}  (n={data.num_nodes}, m={data.edge_index.shape[1]})")
        print(f"{'='*75}")

        results = []

        print("  Measuring Dense GCN...")
        r_dense = measure_dense(data, nf, nc)
        results.append(r_dense)
        print(f"  → {r_dense['total_time_s']:.1f}s  mem={r_dense['peak_mem_mb']:.1f}MB")

        print("  Measuring TGS...")
        r_tgs = measure_tgs(data, nf, nc)
        results.append(r_tgs)
        print(f"  → {r_tgs['total_time_s']:.1f}s  sp={r_tgs['sparsity']:.3f}")

        print(f"  Measuring Static @ sp={r_tgs['sparsity']:.3f}...")
        r_static = measure_static(data, nf, nc, r_tgs["sparsity"])
        results.append(r_static)
        print(f"  → {r_static['total_time_s']:.1f}s")

        print_table(results, r_dense)

        all_output[ds_name] = {
            "results": results,
            "tgs_overhead_pct": (
                (r_tgs["estimator_overhead_ms"] + r_tgs["scheduler_overhead_ms"])
                / (r_tgs["epoch_time_mean"] * 1000) * 100
            ),
        }

    os.makedirs("results", exist_ok=True)
    with open("results/runtime_analysis.json", "w") as f:
        json.dump(all_output, f, indent=2, default=float)
    print("\nSaved results/runtime_analysis.json")


if __name__ == "__main__":
    main()
