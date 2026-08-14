"""
experiments/temporal_order_ablation.py

Experiment 5: Same final edges, different retirement order.
Experiment 6: Retirement timing sweep.

Experiment 5 — Temporal ordering causal test:
  All variants use the EXACT same final edge set (from TGS run).
  Only the ORDER and TIMING of edge removal changes.

  A) TGS order       — retire edges in the order TGS chose (low-influence first)
  B) Random order    — same edges, random retirement order
  C) Reverse order   — retire highest-influence edges first
  D) Static          — remove all edges before training

  If A > B > C > D → temporal ordering is causal, not just edge selection.
  If A ≈ B > C > D → timing matters but not the specific order.
  If A ≈ B ≈ C > D → just having the right final graph matters.

Experiment 6 — Retirement timing sweep:
  Keep final sparsity fixed at TGS's natural level.
  Vary when retirement starts: epoch 0, 20, 40, 80, 120.
  Measure final accuracy. Should show an optimum around epoch 40
  (after representations have begun to stabilize).
"""

import sys, os, json
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


def load_cora():
    dataset = Planetoid(root="./data", name="Cora", transform=NormalizeFeatures())
    return dataset[0].to(DEVICE), dataset.num_features, dataset.num_classes


# ── Run TGS, capture full retirement schedule ────────────────────────────────

def run_tgs_capture_schedule(data, nf, nc):
    """
    Run TGS and return:
      - final test accuracy
      - final edge set (which edges survived)
      - full retirement log: list of (epoch, edge_idx) in order retired
    """
    set_seed(SEED)
    m0 = data.edge_index.shape[1]
    tg  = TemporalGraph(data.edge_index, data.num_nodes, device=DEVICE)
    est = GradientNormEstimator(m0, DEVICE,
            edge_index=data.edge_index, num_nodes=data.num_nodes,
            alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
    model = TemporalGCN(nf, 64, nc, 2, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(list(model.parameters()) + [est.edge_weights],
                             lr=0.01, weight_decay=5e-4)
    sched = AdaptiveRetirementScheduler(tg,
                epsilon_max=5e-3, epsilon_min=1e-5, anneal_steps=100,
                warmup_steps=40, max_retire_frac=0.10,
                max_sparsity=0.65, retire_every=2)

    retirement_log = []   # (epoch, list of edge indices retired)
    bv = bt = 0.0

    for epoch in range(EPOCHS):
        model.train(); am = tg.active_mask
        F.cross_entropy(model(data.x, tg.edge_index, est.edge_weights[am])[data.train_mask],
                        data.y[data.train_mask]).backward()
        est.update_influence(am); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad(): out = model(data.x, tg.edge_index)
        p = out.argmax(-1)
        va = (p[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        ta = (p[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        # Record which edges get retired this step
        active_before = tg.active_mask.clone()
        sched.update_val_acc(va); sched.step(est.influence_scores(am))
        newly_retired = (active_before & ~tg.active_mask).nonzero(as_tuple=False).squeeze(1)
        if len(newly_retired) > 0:
            retirement_log.append({
                "epoch": epoch,
                "retired_indices": newly_retired.tolist(),
            })
        tg.step()
        if va > bv: bv, bt = va, ta

    return bt, tg.edge_index.clone(), retirement_log, tg.sparsity


# ── Replay retirement in a different order ───────────────────────────────────

def replay_retirement(
    data, nf, nc,
    final_edge_index: torch.Tensor,
    retirement_log: list,
    mode: str,   # 'tgs', 'random', 'reverse', 'static'
    seed: int = SEED,
) -> float:
    """
    Train GCN while retiring edges according to different orderings.

    All modes retire the SAME SET of edges — only the timing/order changes.
    - 'static':  all edges removed before training (epoch 0)
    - 'tgs':     original TGS order and timing
    - 'random':  same edges retired, but in a random epoch-order
    - 'reverse': retire highest-influence edges first (reverse of TGS)
    """
    set_seed(seed)
    m0 = data.edge_index.shape[1]

    # Build the full retirement schedule
    # Flatten: list of edge indices in retirement order
    tgs_order = []
    for entry in retirement_log:
        tgs_order.extend(entry["retired_indices"])

    if mode == "static":
        # All retired edges removed upfront
        keep = torch.ones(m0, dtype=torch.bool)
        keep[torch.tensor(tgs_order, dtype=torch.long)] = False
        ei_static = data.edge_index[:, keep]
        model = TemporalGCN(nf, 64, nc, 2, 0.5).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        bv = bt = 0.0
        for epoch in range(EPOCHS):
            model.train()
            F.cross_entropy(model(data.x, ei_static)[data.train_mask],
                            data.y[data.train_mask]).backward()
            opt.step(); opt.zero_grad()
            model.eval()
            with torch.no_grad(): out = model(data.x, ei_static)
            p = out.argmax(-1)
            va = (p[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
            ta = (p[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            if va > bv: bv, bt = va, ta
        return bt

    # For TGS / random / reverse: replay with temporal graph
    # Determine the epoch schedule for each retired edge
    if mode == "tgs":
        # Original schedule
        retire_schedule = {}  # epoch -> list of edge indices
        for entry in retirement_log:
            retire_schedule[entry["epoch"]] = entry["retired_indices"]

    elif mode == "random":
        # Same edges, random epoch assignment within the same time window
        rng = np.random.default_rng(seed)
        all_epochs = [entry["epoch"] for entry in retirement_log
                      for _ in entry["retired_indices"]]
        rng.shuffle(all_epochs)
        retire_schedule = {}
        for idx, epoch in zip(tgs_order, all_epochs):
            retire_schedule.setdefault(epoch, []).append(idx)

    elif mode == "reverse":
        # Reverse the order of retirement (highest-influence first)
        reversed_order = list(reversed(tgs_order))
        all_epochs = [entry["epoch"] for entry in retirement_log
                      for _ in entry["retired_indices"]]
        retire_schedule = {}
        for idx, epoch in zip(reversed_order, all_epochs):
            retire_schedule.setdefault(epoch, []).append(idx)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Replay training with schedule
    tg = TemporalGraph(data.edge_index, data.num_nodes, device=DEVICE)
    model = TemporalGCN(nf, 64, nc, 2, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    bv = bt = 0.0

    for epoch in range(EPOCHS):
        # Retire edges scheduled for this epoch
        if epoch in retire_schedule:
            idx_tensor = torch.tensor(retire_schedule[epoch], dtype=torch.long)
            tg.retire_edges(idx_tensor)

        ei_t = tg.edge_index
        model.train()
        F.cross_entropy(model(data.x, ei_t)[data.train_mask],
                        data.y[data.train_mask]).backward()
        opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad(): out = model(data.x, ei_t)
        p = out.argmax(-1)
        va = (p[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        ta = (p[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        tg.step()
        if va > bv: bv, bt = va, ta

    return bt


# ── Experiment 6: Retirement timing sweep ────────────────────────────────────

def run_timing_sweep(data, nf, nc, target_sparsity: float) -> list:
    """
    Fix final sparsity = target_sparsity.
    Vary warmup_steps: retirement starts at different epochs.
    Compare final accuracy.
    """
    results = []
    print(f"\n=== Exp 6: Retirement timing (target_sp={target_sparsity:.2f}) ===")
    print(f"  {'Warmup':>8} {'TGS acc':>9} {'Static acc':>11} {'Delta':>8} {'FinalSp':>8}")

    # Static baseline at target sparsity
    m0 = data.edge_index.shape[1]
    src_np, dst_np = data.edge_index[0].numpy(), data.edge_index[1].numpy()
    deg = degree(data.edge_index[1], data.num_nodes, dtype=torch.float).numpy()
    er  = 1.0 / deg[src_np].clip(1) + 1.0 / deg[dst_np].clip(1)
    score = torch.from_numpy(deg[src_np] * deg[dst_np]).float()
    score[torch.from_numpy(er) >= torch.quantile(torch.from_numpy(er), 0.90)] = -1.0
    n_rem = int(m0 * target_sparsity)
    _, sidx = score.sort(descending=True)
    rm  = set(sidx[:n_rem].tolist())
    ei_s = data.edge_index[:, torch.tensor([i not in rm for i in range(m0)], dtype=torch.bool)]
    set_seed(SEED)
    ms  = TemporalGCN(nf, 64, nc, 2, 0.5).to(DEVICE)
    os_ = torch.optim.Adam(ms.parameters(), lr=0.01, weight_decay=5e-4)
    bvs = bts = 0.0
    for e in range(EPOCHS):
        ms.train()
        F.cross_entropy(ms(data.x, ei_s)[data.train_mask], data.y[data.train_mask]).backward()
        os_.step(); os_.zero_grad()
        ms.eval()
        with torch.no_grad(): out = ms(data.x, ei_s)
        p = out.argmax(-1)
        va = (p[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        ta = (p[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        if va > bvs: bvs, bts = va, ta
    static_acc = bts

    for warmup in [0, 20, 40, 60, 80, 120]:
        set_seed(SEED)
        tg  = TemporalGraph(data.edge_index, data.num_nodes, device=DEVICE)
        est = GradientNormEstimator(m0, DEVICE,
                edge_index=data.edge_index, num_nodes=data.num_nodes,
                alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
        model = TemporalGCN(nf, 64, nc, 2, 0.5).to(DEVICE)
        opt   = torch.optim.Adam(list(model.parameters()) + [est.edge_weights],
                                 lr=0.01, weight_decay=5e-4)
        sched = AdaptiveRetirementScheduler(tg,
                    epsilon_max=5e-3, epsilon_min=1e-5, anneal_steps=100,
                    warmup_steps=warmup, max_retire_frac=0.10,
                    max_sparsity=target_sparsity, retire_every=2)
        flops = FLOPsCounter(m0, 2, 64)
        bv = bt = 0.0

        for epoch in range(EPOCHS):
            model.train(); am = tg.active_mask
            F.cross_entropy(model(data.x, tg.edge_index, est.edge_weights[am])[data.train_mask],
                            data.y[data.train_mask]).backward()
            est.update_influence(am); opt.step(); opt.zero_grad()
            model.eval()
            with torch.no_grad(): out = model(data.x, tg.edge_index)
            p = out.argmax(-1)
            va = (p[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
            ta = (p[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            sched.update_val_acc(va); sched.step(est.influence_scores(am))
            flops.record_step(tg.mt); tg.step()
            if va > bv: bv, bt = va, ta

        delta = bt - static_acc
        print(f"  {warmup:>8} {bt:>9.4f} {static_acc:>11.4f} {delta:>+8.4f} {tg.sparsity:>8.3f}")
        results.append({
            "warmup": warmup, "tgs_acc": float(bt),
            "static_acc": float(static_acc),
            "delta": float(delta), "sparsity": float(tg.sparsity),
        })

    return results


def main():
    data, nf, nc = load_cora()
    all_results = {}

    # ── Experiment 5: Temporal order ablation ───────────────────────────────
    print("\n=== Exp 5: Temporal Order Ablation (Cora) ===")
    print("Running TGS to capture retirement schedule...")
    tgs_acc, final_ei, ret_log, sp = run_tgs_capture_schedule(data, nf, nc)
    n_retired = sum(len(e["retired_indices"]) for e in ret_log)
    print(f"TGS: test={tgs_acc:.4f}  sparsity={sp:.3f}  edges_retired={n_retired}")

    order_results = {"TGS order": tgs_acc}
    for mode, label in [("random", "Random order"), ("reverse", "Reverse order"), ("static", "Static (all upfront)")]:
        acc = replay_retirement(data, nf, nc, final_ei, ret_log, mode=mode)
        order_results[label] = acc
        print(f"  {label:<22}: {acc:.4f}  (vs TGS: {acc-tgs_acc:+.4f})")

    all_results["exp5_temporal_order"] = order_results
    print(f"\nOrdering matters: TGS({tgs_acc:.4f}) vs Random({order_results['Random order']:.4f}) "
          f"vs Reverse({order_results['Reverse order']:.4f}) vs Static({order_results['Static (all upfront)']:.4f})")

    # ── Experiment 6: Retirement timing ─────────────────────────────────────
    timing_results = run_timing_sweep(data, nf, nc, target_sparsity=sp)
    all_results["exp6_timing"] = timing_results

    os.makedirs("results", exist_ok=True)
    with open("results/temporal_order_ablation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved results/temporal_order_ablation.json")


if __name__ == "__main__":
    main()
