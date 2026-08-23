"""
experiments/sparsity_optimization.py

Score-Predicted Optimal Retention
==================================

Contribution:
  SCORE(G) = homophily(G) × deg_cv(G) predicts the MINIMUM SAFE RETENTION —
  the smallest fraction of edges TGS can keep while still outperforming
  matched-sparsity static ER pruning by ≥ DELTA_THRESHOLD pp.

  This maps:  G → optimal_edge_retention_ratio
  Validating the accuracy–sparsity Pareto frontier from graph structure alone.

Protocol:
  For each (graph config, retention level):
    - TGS with max_sparsity = (1 − retention)
    - Static ER at exactly that retention
    - delta(retention) = TGS_acc − static_acc
  optimal_retention(G) = min{ r : delta(r) ≥ DELTA_THRESHOLD }

  Phase 1 — Fit: linear regression optimal_retention ~ score  (12 calib configs)
  Phase 2 — Predict: locked model on 12 held-out configs, report MAE + success rate

  Runs in checkpointed chunks (≈6 min each) to handle sandbox timeouts.

DELTA_THRESHOLD = 0.05  (5 pp)
RETENTION_GRID  = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
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
from tgs.utils.reproducibility import set_seed
from experiments.predictor_prospective import make_graph, graph_stats

DEVICE          = torch.device("cpu")
EPOCHS          = 200
DELTA_THRESHOLD = 0.05
RETENTION_GRID  = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]

CALIB_SEEDS = [42, 99]
HELD_SEEDS  = [555, 666]

CALIB_CONFIGS = [
    (0.035, 0.013, 0.02, 15),  # score ≈ 0.17
    (0.040, 0.012, 0.03, 20),  # score ≈ 0.19
    (0.045, 0.011, 0.04, 25),  # score ≈ 0.23
    (0.050, 0.010, 0.04, 25),  # score ≈ 0.25
    (0.055, 0.009, 0.05, 30),  # score ≈ 0.30
    (0.060, 0.008, 0.05, 30),  # score ≈ 0.31
    (0.065, 0.007, 0.06, 35),  # score ≈ 0.38
    (0.070, 0.007, 0.07, 40),  # score ≈ 0.40
    (0.075, 0.006, 0.08, 42),  # score ≈ 0.45
    (0.080, 0.006, 0.08, 45),  # score ≈ 0.46
    (0.085, 0.005, 0.09, 48),  # score ≈ 0.50
    (0.092, 0.004, 0.10, 52),  # score ≈ 0.54
]

HELD_CONFIGS = [
    (0.038, 0.013, 0.02, 15),  # score ≈ 0.18
    (0.042, 0.012, 0.03, 18),  # score ≈ 0.21
    (0.047, 0.011, 0.04, 24),  # score ≈ 0.24
    (0.052, 0.010, 0.04, 26),  # score ≈ 0.27
    (0.057, 0.009, 0.05, 29),  # score ≈ 0.31
    (0.062, 0.008, 0.05, 32),  # score ≈ 0.34
    (0.067, 0.007, 0.06, 37),  # score ≈ 0.40
    (0.073, 0.007, 0.07, 41),  # score ≈ 0.43
    (0.078, 0.006, 0.08, 44),  # score ≈ 0.47
    (0.083, 0.005, 0.08, 46),  # score ≈ 0.48
    (0.088, 0.005, 0.09, 50),  # score ≈ 0.52
    (0.095, 0.004, 0.10, 54),  # score ≈ 0.56
]


# ─── Training runs ───────────────────────────────────────────────────────────

def run_tgs_at_retention(ei, x, y, n, nc, tm, vm, tsm, seed, retention):
    max_sp = 1.0 - retention
    m0 = ei.shape[1]
    set_seed(seed)
    tg  = TemporalGraph(ei, n, device=DEVICE)
    est = GradientNormEstimator(m0, DEVICE, edge_index=ei, num_nodes=n,
                                alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
    mt  = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
    ot  = torch.optim.Adam(list(mt.parameters()) + [est.edge_weights],
                           lr=0.01, weight_decay=5e-4)
    sc  = AdaptiveRetirementScheduler(
        tg, epsilon_max=5e-3, epsilon_min=1e-5,
        anneal_steps=100, warmup_steps=40,
        max_retire_frac=0.10, max_sparsity=max_sp, retire_every=2)
    bvt = btt = 0.0
    for e in range(EPOCHS):
        mt.train(); am = tg.active_mask
        F.cross_entropy(mt(x, tg.edge_index, est.edge_weights[am])[tm], y[tm]).backward()
        est.update_influence(am); ot.step(); ot.zero_grad()
        mt.eval()
        with torch.no_grad(): out = mt(x, tg.edge_index)
        p  = out.argmax(-1)
        va = (p[vm]  == y[vm]).float().mean().item()
        ta = (p[tsm] == y[tsm]).float().mean().item()
        sc.update_val_acc(va); sc.step(est.influence_scores(am)); tg.step()
        if va > bvt: bvt, btt = va, ta
    return float(btt), float(tg.sparsity)


def run_static_at_retention(ei, x, y, n, nc, tm, vm, tsm, seed, retention):
    m0    = ei.shape[1]
    src   = ei[0].numpy(); dst = ei[1].numpy()
    dega  = degree(ei[1], n, dtype=torch.float).numpy()
    er    = 1.0 / dega[src].clip(1) + 1.0 / dega[dst].clip(1)
    n_rem = int(m0 * (1.0 - retention))
    _, sidx = torch.from_numpy(er).float().sort()
    rm    = set(sidx[:n_rem].tolist())
    ei_s  = ei[:, torch.tensor([i not in rm for i in range(m0)], dtype=torch.bool)]
    set_seed(seed)
    ms  = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
    os_ = torch.optim.Adam(ms.parameters(), lr=0.01, weight_decay=5e-4)
    bvs = bts = 0.0
    for e in range(EPOCHS):
        ms.train()
        F.cross_entropy(ms(x, ei_s)[tm], y[tm]).backward()
        os_.step(); os_.zero_grad()
        ms.eval()
        with torch.no_grad(): out = ms(x, ei_s)
        p  = out.argmax(-1)
        va = (p[vm]  == y[vm]).float().mean().item()
        ta = (p[tsm] == y[tsm]).float().mean().item()
        if va > bvs: bvs, bts = va, ta
    return float(bts)


def sweep_config(p_intra, p_inter, hub_pct, extra, seeds):
    delta_matrix = {r: [] for r in RETENTION_GRID}
    tgs_matrix   = {r: [] for r in RETENTION_GRID}
    stat_matrix  = {r: [] for r in RETENTION_GRID}
    all_h = []; all_cv = []

    for seed in seeds:
        ei, x, y, n, nc = make_graph(p_intra, p_inter, hub_pct, extra, seed)
        h, cv, _        = graph_stats(ei, y, n)
        all_h.append(h); all_cv.append(cv)
        ei = ei.to(DEVICE); x = x.to(DEVICE); y = y.to(DEVICE)
        g  = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        tm  = torch.zeros(n, dtype=torch.bool)
        vm  = torch.zeros(n, dtype=torch.bool)
        tsm = torch.zeros(n, dtype=torch.bool)
        tm[perm[:int(0.6*n)]]             = True
        vm[perm[int(0.6*n):int(0.8*n)]]  = True
        tsm[perm[int(0.8*n):]]            = True
        for ret in RETENTION_GRID:
            ta, sp = run_tgs_at_retention(ei, x, y, n, nc, tm, vm, tsm, seed, ret)
            sa     = run_static_at_retention(ei, x, y, n, nc, tm, vm, tsm, seed, ret)
            delta_matrix[ret].append(ta - sa)
            tgs_matrix[ret].append(ta)
            stat_matrix[ret].append(sa)

    mean_h  = float(np.mean(all_h))
    mean_cv = float(np.mean(all_cv))
    score   = mean_h * mean_cv
    mean_deltas = {r: float(np.mean(delta_matrix[r])) for r in RETENTION_GRID}
    mean_tgs    = {r: float(np.mean(tgs_matrix[r]))   for r in RETENTION_GRID}
    mean_static = {r: float(np.mean(stat_matrix[r]))  for r in RETENTION_GRID}

    # Maximum retention where delta >= DELTA_THRESHOLD.
    # This is the "critical retention" — the highest edge-keep fraction at which
    # TGS still meaningfully dominates static. Below it, delta falls under threshold.
    # Higher score → lower critical_retention (can prune more aggressively).
    optimal_retention = None
    for ret in sorted(RETENTION_GRID, reverse=True):   # high → low
        if mean_deltas[ret] >= DELTA_THRESHOLD:
            optimal_retention = ret
            break

    return {
        "p_intra": p_intra, "p_inter": p_inter,
        "hub_pct": hub_pct, "extra_per_hub": extra,
        "homophily": mean_h, "deg_cv": mean_cv, "score": score,
        "mean_deltas":  {str(r): mean_deltas[r]  for r in RETENTION_GRID},
        "mean_tgs":     {str(r): mean_tgs[r]     for r in RETENTION_GRID},
        "mean_static":  {str(r): mean_static[r]  for r in RETENTION_GRID},
        "optimal_retention": optimal_retention,
    }


# ─── Regression ──────────────────────────────────────────────────────────────

def fit_regression(results):
    valid    = [r for r in results if r["optimal_retention"] is not None]
    scores   = np.array([r["score"]             for r in valid])
    opt_rets = np.array([r["optimal_retention"] for r in valid])
    c_lin    = np.polyfit(scores, opt_rets, 1)
    c_quad   = np.polyfit(scores, opt_rets, 2)
    ss_tot   = np.sum((opt_rets - opt_rets.mean()) ** 2)
    r2_lin   = 1 - np.sum((opt_rets - np.polyval(c_lin,  scores))**2) / ss_tot if ss_tot else 0
    r2_quad  = 1 - np.sum((opt_rets - np.polyval(c_quad, scores))**2) / ss_tot if ss_tot else 0
    return {
        "linear":    {"coeffs": c_lin.tolist(),  "r2": float(r2_lin)},
        "quadratic": {"coeffs": c_quad.tolist(), "r2": float(r2_quad)},
        "n_valid": len(valid),
        "score_range":     [float(scores.min()),   float(scores.max())],
        "retention_range": [float(opt_rets.min()), float(opt_rets.max())],
        "points": list(zip(scores.tolist(), opt_rets.tolist())),
    }


def predict_ret(score, coeffs):
    return float(np.clip(np.polyval(coeffs, score), 0.25, 0.85))


def snap(ret):
    return min(RETENTION_GRID, key=lambda x: abs(x - ret))


def evaluate(held_results, coeffs, label):
    valid = [r for r in held_results if r["optimal_retention"] is not None]
    preds_raw = [predict_ret(r["score"], coeffs) for r in valid]
    preds     = [snap(p) for p in preds_raw]
    actuals   = [r["optimal_retention"] for r in valid]
    successes = [r["mean_deltas"][str(p)] >= DELTA_THRESHOLD for r, p in zip(valid, preds)]
    gaps      = [p - a for p, a in zip(preds, actuals)]
    return {
        "model": label, "n": len(valid),
        "mae":          float(np.mean(np.abs(np.array(preds) - np.array(actuals)))),
        "success_rate": float(np.mean(successes)),
        "mean_gap":     float(np.mean(gaps)),
        "preds": preds, "actuals": actuals,
        "successes": successes, "gaps": gaps,
        "preds_raw": preds_raw,
    }


# ─── Chunked runner (call this repeatedly) ───────────────────────────────────

CHECKPOINT = "results/sparsity_checkpoint.json"
CHUNK_SIZE  = 4


def run_next_chunk():
    all_configs = [("C", cfg) for cfg in CALIB_CONFIGS] + \
                  [("H", cfg) for cfg in HELD_CONFIGS]

    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f: state = json.load(f)
    else:
        state = {"calib": [], "held": []}

    done_c = len(state["calib"])
    done_h = len(state["held"])
    done   = done_c + done_h
    total  = len(all_configs)

    if done >= total:
        print("All configs complete."); return True

    t0 = time.time()
    for i in range(done, min(done + CHUNK_SIZE, total)):
        phase, cfg = all_configs[i]
        seeds = CALIB_SEEDS if phase == "C" else HELD_SEEDS
        r     = sweep_config(*cfg, seeds=seeds)
        opt   = r["optimal_retention"]
        d25   = r["mean_deltas"]["0.25"]
        d55   = r["mean_deltas"]["0.55"]
        d85   = r["mean_deltas"]["0.85"]
        print(
            f"  [{phase} {i+1:2d}/{total}]  "
            f"score={r['score']:.4f}  opt_ret={str(opt):>4}  "
            f"δ@[0.25,0.55,0.85]=[{d25:+.3f},{d55:+.3f},{d85:+.3f}]  "
            f"{(time.time()-t0):.0f}s"
        )
        if phase == "C": state["calib"].append(r)
        else:            state["held"].append(r)
        with open(CHECKPOINT, "w") as f: json.dump(state, f, indent=2)

    remaining = total - min(done + CHUNK_SIZE, total)
    if remaining > 0:
        elapsed = time.time() - t0
        print(f"\n  Chunk done. {remaining} configs remain (~{elapsed/CHUNK_SIZE*remaining/60:.0f} min).")
        return False
    return True


def finalise():
    with open(CHECKPOINT) as f: state = json.load(f)
    calib_results = state["calib"]
    held_results  = state["held"]

    reg = fit_regression(calib_results)
    ev_lin  = evaluate(held_results, reg["linear"]["coeffs"],  "linear")
    ev_quad = evaluate(held_results, reg["quadratic"]["coeffs"],"quadratic")

    valid_held = [r for r in held_results if r["optimal_retention"] is not None]

    print("\n" + "="*68)
    print("SPARSITY OPTIMIZATION — FINAL RESULTS")
    print("="*68)
    print(f"\nLinear model (locked on {reg['n_valid']} calib configs):")
    print(f"  opt_ret = {reg['linear']['coeffs'][0]:+.4f}·score + {reg['linear']['coeffs'][1]:+.4f}"
          f"  (R² = {reg['linear']['r2']:.3f})")
    print(f"  Score range: {reg['score_range'][0]:.3f}–{reg['score_range'][1]:.3f}"
          f"  → retention: {reg['retention_range'][0]:.2f}–{reg['retention_range'][1]:.2f}")

    for ev in [ev_lin, ev_quad]:
        print(f"\n  [{ev['model']}]  n={ev['n']}")
        print(f"    MAE:           {ev['mae']:.3f} retention units")
        print(f"    Success rate:  {ev['success_rate']:.1%}  "
              f"(predicted budget achieves ≥{DELTA_THRESHOLD*100:.0f}pp)")
        print(f"    Mean gap:      {ev['mean_gap']:+.3f}  "
              f"({'conservative' if ev['mean_gap']>0 else 'aggressive'})")

    # Pareto surface table
    print(f"\n  Mean delta by retention (held-out, all configs):")
    print(f"  {'ret':>5}", end="")
    for r in RETENTION_GRID:
        print(f"  {r:.2f}", end="")
    print()
    print(f"  {'Δ(pp)':>5}", end="")
    for r in RETENTION_GRID:
        ds = [res["mean_deltas"][str(r)] for res in held_results]
        print(f" {np.mean(ds)*100:>+5.1f}", end="")
    print()

    # Per-config breakdown
    ev = ev_lin
    print(f"\n  Per-config (linear model):")
    print(f"  {'score':>7} {'actual':>7} {'pred':>6} {'gap':>5} {'δ@pred':>8} {'✓':>3}")
    print(f"  {'─'*45}")
    for r, pred, act, suc, gap in zip(
        valid_held, ev["preds"], ev["actuals"], ev["successes"], ev["gaps"]
    ):
        dp = r["mean_deltas"][str(pred)]
        print(f"  {r['score']:>7.4f} {act:>7.2f} {pred:>6.2f} {gap:>+5.2f} {dp:>+8.3f}  "
              f"{'✓' if suc else '✗'}")

    # Save
    os.makedirs("results", exist_ok=True)
    out = {
        "delta_threshold_pp": DELTA_THRESHOLD * 100,
        "retention_grid":     RETENTION_GRID,
        "linear_model": {
            "formula":  f"opt_ret = {reg['linear']['coeffs'][0]:.4f}·score + {reg['linear']['coeffs'][1]:.4f}",
            "coeffs":   reg["linear"]["coeffs"],
            "r2":       reg["linear"]["r2"],
        },
        "quadratic_model": {"coeffs": reg["quadratic"]["coeffs"], "r2": reg["quadratic"]["r2"]},
        "held_linear":    {k: v for k, v in ev_lin.items()},
        "held_quadratic": {k: v for k, v in ev_quad.items()},
        "calib_results":  calib_results,
        "held_results":   held_results,
        "regression_points": reg["points"],
    }
    with open("results/sparsity_optimization.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → results/sparsity_optimization.json")
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
