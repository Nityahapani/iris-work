"""
experiments/controlled_structural_sweep.py

Experiment 4: Vary each structural property independently while holding
the others fixed. Tests whether the composite predictor is driven by
one factor or all three.

Three sweeps:
  A) homophily:    0.55 → 0.90  (fixed deg_cv, fixed cross_edge)
  B) degree CV:    0.3  → 1.8   (fixed homophily, fixed cross_edge)
  C) cross-edge:   0.05 → 0.45  (fixed homophily, fixed deg_cv)

For each point: run TGS and static at matched sparsity, record delta.
"""

import sys, os, json, time
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)

from torch_geometric.utils import degree
from tgs.core.temporal_graph import TemporalGraph
from tgs.core.influence import GradientNormEstimator
from tgs.models.gcn import TemporalGCN
from tgs.schedulers.adaptive_scheduler import AdaptiveRetirementScheduler
from tgs.evaluation.flops import FLOPsCounter
from tgs.utils.reproducibility import set_seed

DEVICE = torch.device("cpu")
SEED   = 42
EPOCHS = 150      # faster for synthetic
N      = 1200     # nodes total
NC     = 6        # classes / blocks


# ── Graph generator ─────────────────────────────────────────────────────────

def make_graph(
    n: int,
    nc: int,
    homophily: float,
    deg_cv_target: float,
    seed: int = SEED,
) -> tuple:
    """
    Generate a graph with controlled homophily and degree heterogeneity.

    Strategy:
      - Base: stochastic block model with nc equal-size blocks
      - homophily controls p_intra / (p_intra + (nc-1)*p_inter)
      - deg_cv_target controls hub injection (fraction of hub nodes)

    Returns: (edge_index, x, y, actual_homophily, actual_deg_cv)
    """
    rng = np.random.default_rng(seed)
    n_per = n // nc
    n = n_per * nc  # ensure divisible

    # Solve for p_intra given homophily target and fixed p_inter
    p_inter = 0.025
    # homophily ≈ p_intra / (p_intra + (nc-1)*p_inter)
    # => p_intra = homophily * (nc-1) * p_inter / (1 - homophily)
    p_intra = homophily * (nc - 1) * p_inter / max(1 - homophily, 1e-4)
    p_intra = min(p_intra, 0.95)

    # Build adjacency via SBM
    edges_src, edges_dst = [], []
    for i in range(n):
        block_i = i // n_per
        for j in range(i + 1, n):
            block_j = j // n_per
            p = p_intra if block_i == block_j else p_inter
            if rng.random() < p:
                edges_src.extend([i, j])
                edges_dst.extend([j, i])

    # Hub injection to control degree CV
    # deg_cv_target: add hub nodes that have ~10x average connections
    if len(edges_src) == 0:
        edges_src, edges_dst = [0], [1]  # fallback

    # Estimate current avg degree
    avg_deg_current = 2 * len(edges_src) / (2 * n)  # undirected
    # We want deg_cv ≈ deg_cv_target
    # Hub injection increases std without much changing mean
    # Fraction of nodes to promote: f s.t. std increases proportionally
    hub_frac = max(0.0, (deg_cv_target - 0.2) / 3.0)  # rough mapping
    hub_frac = min(hub_frac, 0.15)

    if hub_frac > 0:
        n_hubs = max(1, int(n * hub_frac))
        hubs = rng.choice(n, n_hubs, replace=False)
        n_extra_per_hub = int(avg_deg_current * 8)
        for h in hubs:
            targets = rng.choice([x for x in range(n) if x != h],
                                 min(n_extra_per_hub, n - 1), replace=False)
            for t in targets:
                edges_src.extend([int(h), int(t)])
                edges_dst.extend([int(t), int(h)])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    # Deduplicate
    edge_index = torch.unique(edge_index, dim=1)

    # Labels
    y = torch.zeros(n, dtype=torch.long)
    for b in range(nc):
        y[b * n_per:(b + 1) * n_per] = b

    # Features: noisy label signal
    x = torch.randn(n, nc + 4) * 0.5
    for b in range(nc):
        x[b * n_per:(b + 1) * n_per, b] += 1.5

    # Compute actual metrics
    src_np, dst_np = edge_index[0].numpy(), edge_index[1].numpy()
    y_np = y.numpy()
    actual_h = (y_np[src_np] == y_np[dst_np]).mean()
    deg_arr = degree(edge_index[1], n).numpy()
    actual_cv = deg_arr.std() / max(deg_arr.mean(), 1e-8)

    return edge_index, x, y, n, float(actual_h), float(actual_cv)


# ── Single TGS vs static run ─────────────────────────────────────────────────

def run_tgs_vs_static(edge_index, x, y, n, nc, seed=SEED):
    edge_index = edge_index.to(DEVICE)
    x = x.to(DEVICE); y = y.to(DEVICE); m0 = edge_index.shape[1]

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tm  = torch.zeros(n, dtype=torch.bool)
    vm  = torch.zeros(n, dtype=torch.bool)
    tsm = torch.zeros(n, dtype=torch.bool)
    tm[perm[:int(0.6 * n)]]             = True
    vm[perm[int(0.6 * n):int(0.8 * n)]] = True
    tsm[perm[int(0.8 * n):]]            = True

    # TGS
    set_seed(seed)
    tg  = TemporalGraph(edge_index, n, device=DEVICE)
    est = GradientNormEstimator(m0, DEVICE, edge_index=edge_index, num_nodes=n,
                                alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
    mt  = TemporalGCN(x.shape[1], 32, nc, 2, 0.5).to(DEVICE)
    ot  = torch.optim.Adam(list(mt.parameters()) + [est.edge_weights],
                           lr=0.01, weight_decay=5e-4)
    sc  = AdaptiveRetirementScheduler(tg, epsilon_max=5e-3, epsilon_min=1e-5,
              anneal_steps=80, warmup_steps=25, max_retire_frac=0.10,
              max_sparsity=0.60, retire_every=2)
    bvt = btt = 0.0
    for e in range(EPOCHS):
        mt.train(); am = tg.active_mask
        F.cross_entropy(mt(x, tg.edge_index, est.edge_weights[am])[tm],
                        y[tm]).backward()
        est.update_influence(am); ot.step(); ot.zero_grad()
        mt.eval()
        with torch.no_grad(): out = mt(x, tg.edge_index)
        p = out.argmax(-1)
        va = (p[vm]  == y[vm]).float().mean().item()
        ta = (p[tsm] == y[tsm]).float().mean().item()
        sc.update_val_acc(va); sc.step(est.influence_scores(am)); tg.step()
        if va > bvt: bvt, btt = va, ta
    sp = tg.sparsity

    # Static at matched sparsity using same scoring
    src_np, dst_np = edge_index[0].numpy(), edge_index[1].numpy()
    deg = degree(edge_index[1], n, dtype=torch.float).numpy()
    er  = 1.0 / deg[src_np].clip(1) + 1.0 / deg[dst_np].clip(1)
    score = torch.from_numpy(deg[src_np] * deg[dst_np]).float()
    score[torch.from_numpy(er) >= torch.quantile(torch.from_numpy(er), 0.90)] = -1.0
    n_rem = int(m0 * sp)
    _, sidx = score.sort(descending=True)
    rm  = set(sidx[:n_rem].tolist())
    eis = edge_index[:, torch.tensor([i not in rm for i in range(m0)], dtype=torch.bool)]

    set_seed(seed)
    ms  = TemporalGCN(x.shape[1], 32, nc, 2, 0.5).to(DEVICE)
    os_ = torch.optim.Adam(ms.parameters(), lr=0.01, weight_decay=5e-4)
    bvs = bts = 0.0
    for e in range(EPOCHS):
        ms.train()
        F.cross_entropy(ms(x, eis)[tm], y[tm]).backward()
        os_.step(); os_.zero_grad()
        ms.eval()
        with torch.no_grad(): out = ms(x, eis)
        p = out.argmax(-1)
        va = (p[vm]  == y[vm]).float().mean().item()
        ta = (p[tsm] == y[tsm]).float().mean().item()
        if va > bvs: bvs, bts = va, ta

    return float(btt), float(bts), float(sp)


# ── Three sweeps ─────────────────────────────────────────────────────────────

def sweep_homophily():
    """Hold deg_cv=0.8 fixed, vary homophily 0.55→0.90"""
    print("\n=== Sweep A: Homophily (deg_cv≈0.8, fixed) ===")
    print(f"  {'Homophily':>10} {'ActualH':>9} {'TGS':>8} {'Static':>8} {'Delta':>8}")
    results = []
    for h_target in [0.55, 0.62, 0.70, 0.78, 0.85, 0.90]:
        ei, x, y, n, ah, acv = make_graph(N, NC, homophily=h_target,
                                          deg_cv_target=0.8, seed=SEED)
        tgs, stat, sp = run_tgs_vs_static(ei, x, y, n, NC)
        delta = tgs - stat
        print(f"  {h_target:>10.2f} {ah:>9.4f} {tgs:>8.4f} {stat:>8.4f} {delta:>+8.4f}")
        results.append({"h_target": h_target, "actual_h": ah, "actual_cv": acv,
                        "tgs": tgs, "static": stat, "delta": delta, "sparsity": sp})
    return results


def sweep_deg_cv():
    """Hold homophily=0.80 fixed, vary deg_cv 0.3→1.8"""
    print("\n=== Sweep B: Degree CV (homophily≈0.80, fixed) ===")
    print(f"  {'CV target':>10} {'ActualCV':>9} {'TGS':>8} {'Static':>8} {'Delta':>8}")
    results = []
    for cv_target in [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]:
        ei, x, y, n, ah, acv = make_graph(N, NC, homophily=0.80,
                                          deg_cv_target=cv_target, seed=SEED)
        tgs, stat, sp = run_tgs_vs_static(ei, x, y, n, NC)
        delta = tgs - stat
        print(f"  {cv_target:>10.2f} {acv:>9.4f} {tgs:>8.4f} {stat:>8.4f} {delta:>+8.4f}")
        results.append({"cv_target": cv_target, "actual_h": ah, "actual_cv": acv,
                        "tgs": tgs, "static": stat, "delta": delta, "sparsity": sp})
    return results


def sweep_composite():
    """Vary all three to hit composite score targets 0.5→1.5"""
    print("\n=== Sweep C: Composite score targets ===")
    print(f"  {'Score':>8} {'TGS':>8} {'Static':>8} {'Delta':>8} {'ActualH':>8} {'ActualCV':>9}")
    # Target score = h * cv * (1 - cross_edge)
    # Cross-edge ≈ 1 - homophily (rough), so score ≈ h * cv * h = h^2 * cv
    # Pick (h, cv) pairs to hit each score target
    configs = [
        (0.55, 0.50),   # score ≈ 0.55*0.50*0.45 ≈ 0.12  low
        (0.65, 0.70),   # score ≈ 0.65*0.70*0.35 ≈ 0.16
        (0.72, 0.90),   # score ≈ 0.72*0.90*0.28 ≈ 0.18
        (0.78, 1.10),   # score ≈ 0.78*1.10*0.22 ≈ 0.19
        (0.83, 1.30),   # score ≈ 0.83*1.30*0.17 ≈ 0.18
        (0.88, 1.50),   # score ≈ 0.88*1.50*0.12 ≈ 0.16
    ]
    results = []
    for h_target, cv_target in configs:
        ei, x, y, n, ah, acv = make_graph(N, NC, homophily=h_target,
                                          deg_cv_target=cv_target, seed=SEED)
        src_np, dst_np = ei[0].numpy(), ei[1].numpy()
        y_np = y.numpy()
        cross = (y_np[src_np] != y_np[dst_np]).mean()
        score = ah * acv * (1 - cross)
        tgs, stat, sp = run_tgs_vs_static(ei, x, y, n, NC)
        delta = tgs - stat
        print(f"  {score:>8.3f} {tgs:>8.4f} {stat:>8.4f} {delta:>+8.4f} {ah:>8.4f} {acv:>9.4f}")
        results.append({"score": score, "h": ah, "cv": acv, "cross": cross,
                        "tgs": tgs, "static": stat, "delta": delta, "sparsity": sp})
    return results


if __name__ == "__main__":
    all_results = {}
    all_results["sweep_homophily"] = sweep_homophily()
    all_results["sweep_deg_cv"]    = sweep_deg_cv()
    all_results["sweep_composite"] = sweep_composite()

    os.makedirs("results", exist_ok=True)
    with open("results/controlled_structural_sweep.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved results/controlled_structural_sweep.json")
