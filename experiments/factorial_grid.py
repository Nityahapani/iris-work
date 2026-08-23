"""
experiments/factorial_grid.py

2D Controlled Factorial Experiment
====================================

Construct graphs on an 8×7 grid:
    H   ∈ {0.10, 0.18, 0.26, 0.34, 0.42, 0.50, 0.58, 0.66}
    CV  ∈ {0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 1.10}

while holding everything else approximately constant:
    n_nodes ≈ 720, n_classes = 4, feature signal = constant.

Measure: Δ = TGS_acc − static_er_acc  (matched sparsity)

Then fit the interaction regression:
    Δ = β₀ + β₁·H + β₂·CV + β₃·(H×CV) + ε

Test whether:
    - β₃ (interaction) is significant (t-test / permutation)
    - H×CV alone explains most variance (confirms product structure)
    - The advantage is localised to high-H, high-CV region

This directly tests the proposed mechanism, not just a correlation.

Checkpointed in chunks of 8 cells (one H row).
"""

import sys, os, json, time
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)

from torch_geometric.utils import degree, stochastic_blockmodel_graph
from tgs.core.temporal_graph import TemporalGraph
from tgs.core.influence import GradientNormEstimator
from tgs.models.gcn import TemporalGCN
from tgs.schedulers.adaptive_scheduler import AdaptiveRetirementScheduler
from tgs.utils.reproducibility import set_seed

DEVICE = torch.device("cpu")
EPOCHS = 200
N_PER  = 180; N_BLOCKS = 4
SEEDS  = [42, 123]   # 2 seeds per cell — 56 cells × 2 = 112 runs each method

# Target grid values (actual values will be close but not exact)
H_TARGETS  = [0.10, 0.18, 0.26, 0.34, 0.42, 0.50, 0.58, 0.66]
CV_TARGETS = [0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 1.10]

CHECKPOINT = "results/factorial_checkpoint.json"


# ─── Graph generator with independent H and CV control ───────────────────────

def make_factorial_graph(h_target, cv_target, seed):
    """
    Generate graph targeting specific (homophily, deg_cv) values independently.

    H control:  p_intra / p_inter ratio in SBM
    CV control: within-block hub injection (doesn't change homophily)
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n  = N_PER * N_BLOCKS; nc = N_BLOCKS

    y_np = np.zeros(n, dtype=int)
    for b in range(nc): y_np[b*N_PER:(b+1)*N_PER] = b

    # H control: solve for p_intra/p_inter from h_target
    # Expected homophily ≈ p_intra / (p_intra + (nc-1)*p_inter)
    p_inter = 0.012   # fixed base density
    # h ≈ p_in / (p_in + 3*p_out)  → p_in = 3*p_out*h / (1-h)
    p_intra = max(3 * p_inter * h_target / max(1 - h_target, 0.01), 0.005)
    p_intra = min(p_intra, 0.40)

    ep = np.full((nc, nc), p_inter)
    np.fill_diagonal(ep, p_intra)
    ei_base = stochastic_blockmodel_graph([N_PER]*nc, torch.tensor(ep))
    es = list(ei_base[0].numpy()); ed = list(ei_base[1].numpy())

    # CV control: within-block hub injection
    # Calibrate hub_pct and extra from cv_target empirically
    # cv_target 0.20 → no hubs; cv_target 1.10 → ~15% hubs with 80 extra edges
    hub_pct   = max(0, (cv_target - 0.22) / 6.0)
    extra_hub = int(hub_pct * 500)   # roughly calibrated
    hub_pct   = min(hub_pct, 0.20)
    extra_hub = min(extra_hub, 80)

    if hub_pct > 0:
        hubs = rng.choice(n, int(n * hub_pct), replace=False)
        for h in hubs:
            bn = [i for i in range(n) if y_np[i] == y_np[h] and i != h]
            for t in rng.choice(bn, min(extra_hub, len(bn)), replace=False):
                es.extend([int(h), int(t)]); ed.extend([int(t), int(h)])

    ei = torch.unique(torch.tensor([es, ed], dtype=torch.long), dim=1)
    src = ei[0].numpy(); dst = ei[1].numpy()

    # Measure actual values
    h_actual  = float((y_np[src] == y_np[dst]).mean())
    dega      = degree(ei[1], n).numpy()
    cv_actual = float(dega.std() / max(dega.mean(), 1e-8))

    y = torch.tensor(y_np)
    x = torch.randn(n, nc + 4) * 0.9
    for b in range(nc): x[b*N_PER:(b+1)*N_PER, b] += 1.0

    return ei, x, y, n, nc, h_actual, cv_actual


def run_tgs(ei, x, y, n, nc, tm, vm, tsm, seed):
    m0 = ei.shape[1]; set_seed(seed)
    tg  = TemporalGraph(ei, n, device=DEVICE)
    est = GradientNormEstimator(m0, DEVICE, edge_index=ei, num_nodes=n,
                                alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
    mt  = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
    ot  = torch.optim.Adam(list(mt.parameters()) + [est.edge_weights],
                           lr=0.01, weight_decay=5e-4)
    sc  = AdaptiveRetirementScheduler(tg, epsilon_max=5e-3, epsilon_min=1e-5,
              anneal_steps=100, warmup_steps=40, max_retire_frac=0.10,
              max_sparsity=0.65, retire_every=2)
    bvt = btt = 0.0
    for e in range(EPOCHS):
        mt.train(); am = tg.active_mask
        F.cross_entropy(mt(x, tg.edge_index, est.edge_weights[am])[tm], y[tm]).backward()
        est.update_influence(am); ot.step(); ot.zero_grad()
        mt.eval()
        with torch.no_grad(): out = mt(x, tg.edge_index)
        p  = out.argmax(-1)
        va = (p[vm]==y[vm]).float().mean().item()
        ta = (p[tsm]==y[tsm]).float().mean().item()
        sc.update_val_acc(va); sc.step(est.influence_scores(am)); tg.step()
        if va > bvt: bvt, btt = va, ta
    return float(btt), float(tg.sparsity)


def run_static_er(ei, x, y, n, nc, tm, vm, tsm, seed, target_sp):
    m0 = ei.shape[1]
    src = ei[0].numpy(); dega = degree(ei[1], n, dtype=torch.float).numpy()
    er  = 1.0/dega[src].clip(1) + 1.0/dega[ei[1].numpy()].clip(1)
    n_rem = int(m0 * target_sp)
    _, sidx = torch.from_numpy(er).float().sort()
    rm = set(sidx[:n_rem].tolist())
    ei_s = ei[:, torch.tensor([i not in rm for i in range(m0)], dtype=torch.bool)]
    set_seed(seed)
    ms  = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
    os_ = torch.optim.Adam(ms.parameters(), lr=0.01, weight_decay=5e-4)
    bvs = bts = 0.0
    for e in range(EPOCHS):
        ms.train(); F.cross_entropy(ms(x, ei_s)[tm], y[tm]).backward()
        os_.step(); os_.zero_grad()
        ms.eval()
        with torch.no_grad(): out = ms(x, ei_s)
        p  = out.argmax(-1)
        va = (p[vm]==y[vm]).float().mean().item()
        ta = (p[tsm]==y[tsm]).float().mean().item()
        if va > bvs: bvs, bts = va, ta
    return float(bts)


def run_cell(h_target, cv_target, seed):
    ei, x, y, n, nc, h_act, cv_act = make_factorial_graph(h_target, cv_target, seed)
    ei = ei.to(DEVICE); x = x.to(DEVICE); y = y.to(DEVICE)
    g  = torch.Generator().manual_seed(seed); perm = torch.randperm(n, generator=g)
    tm = torch.zeros(n, dtype=torch.bool); vm = torch.zeros(n, dtype=torch.bool)
    tsm = torch.zeros(n, dtype=torch.bool)
    tm[perm[:int(0.6*n)]] = True; vm[perm[int(0.6*n):int(0.8*n)]] = True
    tsm[perm[int(0.8*n):]] = True
    tgs_acc, sp = run_tgs(ei, x, y, n, nc, tm, vm, tsm, seed)
    stat_acc    = run_static_er(ei, x, y, n, nc, tm, vm, tsm, seed, sp)
    return {
        "h_target": h_target, "cv_target": cv_target, "seed": seed,
        "h_actual": h_act, "cv_actual": cv_act,
        "score_actual": h_act * cv_act,
        "tgs_acc": tgs_acc, "static_acc": stat_acc,
        "delta": tgs_acc - stat_acc, "sparsity": sp,
    }


def run_next_chunk():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f: state = json.load(f)
    else:
        state = {"cells": []}

    done_keys = set(f"{r['h_target']}_{r['cv_target']}_{r['seed']}" for r in state["cells"])
    all_cells = [(h, cv, s) for h in H_TARGETS for cv in CV_TARGETS for s in SEEDS]
    total = len(all_cells)
    todo = [(h, cv, s) for h, cv, s in all_cells
            if f"{h}_{cv}_{s}" not in done_keys]

    if not todo: print("All done."); return True

    CHUNK = 8   # 8 (h, cv, seed) triples per chunk ≈ 4 min
    t0 = time.time()
    for h, cv, s in todo[:CHUNK]:
        r = run_cell(h, cv, s)
        state["cells"].append(r)
        done_now = len(done_keys) + 1
        print(f"  [{done_now:3d}/{total}] "
              f"H={r['h_actual']:.3f}(tgt={h:.2f}) "
              f"CV={r['cv_actual']:.3f}(tgt={cv:.2f}) "
              f"Δ={r['delta']:+.4f}  {time.time()-t0:.0f}s")
        done_keys.add(f"{h}_{cv}_{s}")
        with open(CHECKPOINT, "w") as f: json.dump(state, f, indent=2)

    remaining = len(todo) - CHUNK
    if remaining > 0:
        elapsed = time.time() - t0
        print(f"\nChunk done. {remaining} cells remain "
              f"(~{elapsed/CHUNK*remaining/60:.0f} min).")
        return False
    return True


def finalise():
    with open(CHECKPOINT) as f: state = json.load(f)
    cells = state["cells"]

    # Aggregate: mean delta per (h_target, cv_target)
    from collections import defaultdict
    grid = defaultdict(list)
    for r in cells:
        grid[(r["h_target"], r["cv_target"])].append(r)

    agg = {}
    for (h, cv), group in grid.items():
        h_act  = np.mean([r["h_actual"]  for r in group])
        cv_act = np.mean([r["cv_actual"] for r in group])
        delta  = np.mean([r["delta"]     for r in group])
        agg[(h, cv)] = {"h": h_act, "cv": cv_act, "delta": float(delta),
                        "score": h_act * cv_act, "n": len(group)}

    # Fit interaction regression: Δ = β₀ + β₁H + β₂CV + β₃(H×CV)
    rows = list(agg.values())
    H_v  = np.array([r["h"]     for r in rows])
    CV_v = np.array([r["cv"]    for r in rows])
    D_v  = np.array([r["delta"] for r in rows])
    HCV  = H_v * CV_v

    # Design matrix
    X = np.column_stack([np.ones(len(rows)), H_v, CV_v, HCV])
    # OLS
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, D_v, rcond=None)
        pred    = X @ beta
        ss_tot  = np.sum((D_v - D_v.mean())**2)
        ss_res  = np.sum((D_v - pred)**2)
        r2_full = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Permutation test for β₃ (interaction term)
        n_perm = 2000
        rng    = np.random.default_rng(42)
        beta3_null = []
        for _ in range(n_perm):
            perm_D = rng.permutation(D_v)
            b_null, _, _, _ = np.linalg.lstsq(X, perm_D, rcond=None)
            beta3_null.append(b_null[3])
        p_value = float(np.mean(np.abs(beta3_null) >= abs(beta[3])))

        # Also fit model without interaction
        X_no_int = np.column_stack([np.ones(len(rows)), H_v, CV_v])
        b_no_int, _, _, _ = np.linalg.lstsq(X_no_int, D_v, rcond=None)
        pred_no_int = X_no_int @ b_no_int
        r2_no_int = 1 - np.sum((D_v - pred_no_int)**2) / ss_tot if ss_tot > 0 else 0

        print("\n" + "="*68)
        print("FACTORIAL GRID — INTERACTION REGRESSION")
        print("="*68)
        print(f"\n  Model: Δ = β₀ + β₁·H + β₂·CV + β₃·(H×CV)")
        print(f"  {'Coefficient':12s}  {'Value':>10}  {'Interpretation'}")
        print(f"  {'─'*55}")
        names = ["β₀ (intercept)", "β₁ (H)", "β₂ (CV)", "β₃ (H×CV)"]
        for name, b in zip(names, beta):
            print(f"  {name:14s}  {b:>+10.4f}")
        print(f"\n  R² (full model):          {r2_full:.4f}")
        print(f"  R² (without interaction): {r2_no_int:.4f}")
        print(f"  ΔR² from interaction:     {r2_full - r2_no_int:+.4f}")
        print(f"  β₃ p-value (permutation): {p_value:.4f}  "
              f"{'(significant)' if p_value < 0.05 else '(not significant)'}")
        print(f"\n  Mean Δ by region:")
        low_H_low_CV  = np.mean([r["delta"] for r in rows if r["h"] < 0.35 and r["cv"] < 0.60])
        low_H_high_CV = np.mean([r["delta"] for r in rows if r["h"] < 0.35 and r["cv"] >= 0.60])
        high_H_low_CV = np.mean([r["delta"] for r in rows if r["h"] >= 0.35 and r["cv"] < 0.60])
        high_H_high_CV= np.mean([r["delta"] for r in rows if r["h"] >= 0.35 and r["cv"] >= 0.60])
        print(f"    Low H,  Low CV:   {low_H_low_CV*100:+.2f} pp")
        print(f"    Low H,  High CV:  {low_H_high_CV*100:+.2f} pp")
        print(f"    High H, Low CV:   {high_H_low_CV*100:+.2f} pp")
        print(f"    High H, High CV:  {high_H_high_CV*100:+.2f} pp  ← predicted region")

        regression = {
            "beta": beta.tolist(), "r2_full": float(r2_full),
            "r2_no_interaction": float(r2_no_int),
            "delta_r2": float(r2_full - r2_no_int),
            "beta3_pvalue": p_value,
            "quadrant_means": {
                "low_H_low_CV":   float(low_H_low_CV),
                "low_H_high_CV":  float(low_H_high_CV),
                "high_H_low_CV":  float(high_H_low_CV),
                "high_H_high_CV": float(high_H_high_CV),
            }
        }
    except Exception as ex:
        print(f"Regression failed: {ex}")
        regression = {}

    # Grid table
    print(f"\n  Mean Δ (pp) grid — rows=H, cols=CV:")
    print(f"  {'H\\CV':>8}", end="")
    for cv in CV_TARGETS:
        print(f"  {cv:.2f}", end="")
    print()
    for h in H_TARGETS:
        print(f"  {h:.2f}     ", end="")
        for cv in CV_TARGETS:
            key = (h, cv)
            if key in agg:
                print(f"  {agg[key]['delta']*100:>+4.1f}", end="")
            else:
                print(f"  {'N/A':>5}", end="")
        print()

    os.makedirs("results", exist_ok=True)
    out = {
        "h_targets": H_TARGETS, "cv_targets": CV_TARGETS,
        "regression": regression,
        "grid_agg": [{"h_target": h, "cv_target": cv, **v}
                     for (h, cv), v in agg.items()],
        "raw_cells": cells,
    }
    with open("results/factorial_grid.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → results/factorial_grid.json")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--finalise", action="store_true")
    args = p.parse_args()
    if args.finalise:
        finalise()
    else:
        done = run_next_chunk()
        if done:
            finalise()
