"""
experiments/predictor_validation.py

Structural Predictor Validation Experiment
===========================================

Central claim:
    A structural rule computed BEFORE training can predict whether TGS will
    outperform a matched-sparsity static baseline (effective-resistance pruning)
    on unseen graphs — using only two graph statistics observable in O(m) time.

Predictor:
    SCORE(G) = homophily(G) * deg_cv(G)
    Predict "TGS wins" iff SCORE(G) > THRESHOLD

Protocol:
  Phase 1 — Calibration (24 graphs, 4×6 grid):
    Sweep over (p_intra, hub_pct) configurations. Each configuration generates
    a unique graph with specific (homophily, deg_cv) values. Run TGS and
    matched-sparsity static ER pruning. Fit the threshold via accuracy maximisation.

  Phase 2 — Held-out validation (30 graphs, 5×6 grid, different seeds + params):
    Configurations NOT in calibration set. Threshold locked — not updated.
    Evaluate: prediction accuracy, per-class edge reduction, accuracy gain.

Baseline:
    Effective-resistance-proxy static pruning (1/deg_u + 1/deg_v),
    removing the same fraction of edges that TGS retired, BEFORE training.
    This is among the strongest static baselines (theoretical optimality for
    spectral sparsification).

Graph generation:
    4-block Stochastic Block Model, n=720 nodes, partial label features.
    Hub injection within blocks raises deg_cv while preserving homophily.
    Seeds differ completely between calibration and held-out sets.
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

DEVICE  = torch.device("cpu")
EPOCHS  = 200
N_PER   = 180          # nodes per class block
N_BLOCKS = 4           # 720 nodes total


# ─── Graph generator ────────────────────────────────────────────────────────

def make_graph(p_intra, p_inter, hub_pct, extra_per_hub, seed):
    """
    Generate a 4-block SBM with controlled homophily and degree heterogeneity.

    - p_intra / p_inter ratio controls homophily
    - hub_pct + extra_per_hub controls deg_cv via within-block hub injection
      (hubs connect only within their block, so homophily is preserved)
    - Features: partial label signal + noise (forces GNN to use graph structure)
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n = N_PER * N_BLOCKS
    nc = N_BLOCKS

    y_np = np.zeros(n, dtype=int)
    for b in range(nc):
        y_np[b * N_PER:(b + 1) * N_PER] = b

    # Base SBM
    ep = np.full((nc, nc), p_inter)
    np.fill_diagonal(ep, p_intra)
    ei_base = stochastic_blockmodel_graph([N_PER] * nc, torch.tensor(ep))

    edges_src = list(ei_base[0].numpy())
    edges_dst = list(ei_base[1].numpy())

    # Within-block hub injection (preserves homophily)
    if hub_pct > 0:
        hubs = rng.choice(n, int(n * hub_pct), replace=False)
        for h in hubs:
            block_nodes = [i for i in range(n) if y_np[i] == y_np[h] and i != h]
            targets = rng.choice(
                block_nodes, min(extra_per_hub, len(block_nodes)), replace=False
            )
            for t in targets:
                edges_src.extend([int(h), int(t)])
                edges_dst.extend([int(t), int(h)])

    ei = torch.unique(
        torch.tensor([edges_src, edges_dst], dtype=torch.long), dim=1
    )

    # Node features: partial label signal + noise
    x = torch.randn(n, nc + 4) * 0.9
    for b in range(nc):
        x[b * N_PER:(b + 1) * N_PER, b] += 1.0

    y = torch.tensor(y_np)
    return ei, x, y, n, nc


def graph_stats(ei, y, n):
    """Compute structural features in O(m) — available before any training."""
    src_np = ei[0].numpy()
    dst_np = ei[1].numpy()
    y_np   = y.numpy()

    homophily = float((y_np[src_np] == y_np[dst_np]).mean())
    deg_arr   = degree(ei[1], n).numpy()
    deg_cv    = float(deg_arr.std() / max(deg_arr.mean(), 1e-8))
    score     = homophily * deg_cv

    return {
        "homophily": homophily,
        "deg_cv":    deg_cv,
        "score":     score,
        "n_edges":   int(ei.shape[1]),
        "mean_deg":  float(deg_arr.mean()),
    }


# ─── TGS training run ───────────────────────────────────────────────────────

def run_tgs(ei, x, y, n, nc, tm, vm, tsm, seed):
    m0 = ei.shape[1]
    set_seed(seed)
    tg  = TemporalGraph(ei, n, device=DEVICE)
    est = GradientNormEstimator(
        m0, DEVICE, edge_index=ei, num_nodes=n,
        alpha=0.3, gamma=0.2, hub_gate_pct=0.10
    )
    mt  = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
    ot  = torch.optim.Adam(
        list(mt.parameters()) + [est.edge_weights], lr=0.01, weight_decay=5e-4
    )
    sc  = AdaptiveRetirementScheduler(
        tg, epsilon_max=5e-3, epsilon_min=1e-5,
        anneal_steps=100, warmup_steps=40,
        max_retire_frac=0.08, max_sparsity=0.65, retire_every=2,
    )
    bvt = btt = 0.0

    for e in range(EPOCHS):
        mt.train()
        am = tg.active_mask
        logits = mt(x, tg.edge_index, est.edge_weights[am])
        F.cross_entropy(logits[tm], y[tm]).backward()
        est.update_influence(am)
        ot.step(); ot.zero_grad()
        mt.eval()
        with torch.no_grad():
            out = mt(x, tg.edge_index)
        p  = out.argmax(-1)
        va = (p[vm]  == y[vm]).float().mean().item()
        ta = (p[tsm] == y[tsm]).float().mean().item()
        sc.update_val_acc(va)
        sc.step(est.influence_scores(am))
        tg.step()
        if va > bvt:
            bvt, btt = va, ta

    return float(btt), float(tg.sparsity)


# ─── Static ER baseline ─────────────────────────────────────────────────────

def run_static_er(ei, x, y, n, nc, tm, vm, tsm, seed, target_sparsity):
    """
    Effective-resistance proxy pruning: remove edges with lowest 1/deg_u + 1/deg_v
    (hub-to-hub edges — low ER = redundant pathways). Matched to TGS sparsity.
    """
    m0     = ei.shape[1]
    src_np = ei[0].numpy()
    dst_np = ei[1].numpy()
    deg    = degree(ei[1], n, dtype=torch.float).numpy()
    er     = 1.0 / deg[src_np].clip(1) + 1.0 / deg[dst_np].clip(1)

    # Remove lowest-ER edges first (least important bridges)
    n_rem = int(m0 * target_sparsity)
    score_s = torch.from_numpy(er).float()
    _, sidx = score_s.sort()                # ascending: low ER → remove first
    rm      = set(sidx[:n_rem].tolist())
    keep    = torch.tensor([i not in rm for i in range(m0)], dtype=torch.bool)
    ei_s    = ei[:, keep]

    set_seed(seed)
    ms  = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
    os_ = torch.optim.Adam(ms.parameters(), lr=0.01, weight_decay=5e-4)
    bvs = bts = 0.0

    for e in range(EPOCHS):
        ms.train()
        F.cross_entropy(ms(x, ei_s)[tm], y[tm]).backward()
        os_.step(); os_.zero_grad()
        ms.eval()
        with torch.no_grad():
            out = ms(x, ei_s)
        p  = out.argmax(-1)
        va = (p[vm]  == y[vm]).float().mean().item()
        ta = (p[tsm] == y[tsm]).float().mean().item()
        if va > bvs:
            bvs, bts = va, ta

    return float(bts)


# ─── Single experiment ───────────────────────────────────────────────────────

def run_experiment(p_intra, p_inter, hub_pct, extra_per_hub, seed):
    ei, x, y, n, nc = make_graph(p_intra, p_inter, hub_pct, extra_per_hub, seed)
    stats = graph_stats(ei, y, n)

    ei = ei.to(DEVICE); x = x.to(DEVICE); y = y.to(DEVICE)
    g    = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    tm  = torch.zeros(n, dtype=torch.bool)
    vm  = torch.zeros(n, dtype=torch.bool)
    tsm = torch.zeros(n, dtype=torch.bool)
    tm[perm[:int(0.6 * n)]]              = True
    vm[perm[int(0.6 * n):int(0.8 * n)]] = True
    tsm[perm[int(0.8 * n):]]             = True

    tgs_acc, sp     = run_tgs(ei, x, y, n, nc, tm, vm, tsm, seed)
    static_acc      = run_static_er(ei, x, y, n, nc, tm, vm, tsm, seed, sp)
    delta           = tgs_acc - static_acc

    return {
        **stats,
        "p_intra": p_intra, "p_inter": p_inter,
        "hub_pct": hub_pct, "extra_per_hub": extra_per_hub,
        "seed": seed,
        "tgs_acc": tgs_acc, "static_acc": static_acc,
        "delta": delta, "sparsity": sp,
        "tgs_wins": delta > 0.0,
    }


# ─── Experiment configurations ───────────────────────────────────────────────
#
# Each config is (p_intra, p_inter, hub_pct, extra_per_hub)
# Systematically varies score = homophily * deg_cv from ~0.05 to ~0.50
#
# Low score  (<0.10): low homophily + low cv → static ER can keep up
# High score (>0.10): TGS's temporal ordering provides real advantage

# Calibration configs: 4 homophily levels × 6 hub levels
CALIB_CONFIGS = [
    # Low homophily (h≈0.25–0.30)
    (0.020, 0.020, 0.00, 0),
    (0.020, 0.020, 0.02, 20),
    (0.020, 0.020, 0.04, 30),
    (0.020, 0.020, 0.06, 40),
    (0.020, 0.020, 0.08, 50),
    (0.020, 0.020, 0.10, 55),
    # Medium-low homophily (h≈0.40–0.45)
    (0.030, 0.015, 0.00, 0),
    (0.030, 0.015, 0.02, 20),
    (0.030, 0.015, 0.04, 30),
    (0.030, 0.015, 0.06, 40),
    (0.030, 0.015, 0.08, 50),
    (0.030, 0.015, 0.10, 55),
    # Medium homophily (h≈0.58–0.62)
    (0.045, 0.010, 0.00, 0),
    (0.045, 0.010, 0.02, 20),
    (0.045, 0.010, 0.04, 30),
    (0.045, 0.010, 0.06, 40),
    (0.045, 0.010, 0.08, 50),
    (0.045, 0.010, 0.10, 55),
    # High homophily (h≈0.78–0.82)
    (0.070, 0.006, 0.00, 0),
    (0.070, 0.006, 0.02, 20),
    (0.070, 0.006, 0.04, 30),
    (0.070, 0.006, 0.06, 40),
    (0.070, 0.006, 0.08, 50),
    (0.070, 0.006, 0.10, 55),
]
CALIB_SEEDS = [42]

# Held-out configs: DIFFERENT p values, DIFFERENT seeds
HELD_CONFIGS = [
    # Low homophily (h≈0.22–0.28)
    (0.018, 0.018, 0.00, 0),
    (0.018, 0.018, 0.03, 25),
    (0.018, 0.018, 0.05, 35),
    (0.018, 0.018, 0.07, 45),
    (0.018, 0.018, 0.09, 55),
    (0.018, 0.018, 0.12, 60),
    # Medium-low homophily (h≈0.47–0.52)
    (0.035, 0.013, 0.00, 0),
    (0.035, 0.013, 0.03, 25),
    (0.035, 0.013, 0.05, 35),
    (0.035, 0.013, 0.07, 45),
    (0.035, 0.013, 0.09, 55),
    (0.035, 0.013, 0.12, 60),
    # Medium homophily (h≈0.65–0.70)
    (0.055, 0.009, 0.00, 0),
    (0.055, 0.009, 0.03, 25),
    (0.055, 0.009, 0.05, 35),
    (0.055, 0.009, 0.07, 45),
    (0.055, 0.009, 0.09, 55),
    (0.055, 0.009, 0.12, 60),
    # High homophily (h≈0.83–0.88)
    (0.085, 0.005, 0.00, 0),
    (0.085, 0.005, 0.03, 25),
    (0.085, 0.005, 0.05, 35),
    (0.085, 0.005, 0.07, 45),
    (0.085, 0.005, 0.09, 55),
    (0.085, 0.005, 0.12, 60),
    # Very high homophily (h≈0.90–0.93)
    (0.100, 0.003, 0.00, 0),
    (0.100, 0.003, 0.03, 25),
    (0.100, 0.003, 0.05, 35),
    (0.100, 0.003, 0.07, 45),
    (0.100, 0.003, 0.09, 55),
    (0.100, 0.003, 0.12, 60),
]
HELD_SEEDS = [123, 456]


# ─── Threshold calibration ───────────────────────────────────────────────────

def fit_threshold(results):
    """
    Grid search over score thresholds; pick the one that maximises accuracy
    on the calibration set (aggregated per config).
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = (r["p_intra"], r["p_inter"], r["hub_pct"], r["extra_per_hub"])
        groups[key].append(r)

    agg = []
    for key, group in groups.items():
        mean_score = float(np.mean([g["score"] for g in group]))
        mean_delta = float(np.mean([g["delta"] for g in group]))
        agg.append({"score": mean_score, "tgs_wins": mean_delta > 0.0,
                    "mean_delta": mean_delta})

    scores = np.array([a["score"] for a in agg])
    labels = np.array([a["tgs_wins"] for a in agg])

    best_acc  = -1.0
    best_thr  = 0.0
    for thr in np.linspace(scores.min() - 0.01, scores.max() + 0.01, 500):
        pred  = scores > thr
        acc   = (pred == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return float(best_thr), float(best_acc), agg


# ─── Held-out evaluation ─────────────────────────────────────────────────────

def evaluate_held(held_results, threshold):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in held_results:
        key = (r["p_intra"], r["p_inter"], r["hub_pct"], r["extra_per_hub"])
        groups[key].append(r)

    agg = []
    for key, group in groups.items():
        mean_score  = float(np.mean([g["score"]      for g in group]))
        mean_delta  = float(np.mean([g["delta"]      for g in group]))
        mean_tgs    = float(np.mean([g["tgs_acc"]    for g in group]))
        mean_static = float(np.mean([g["static_acc"] for g in group]))
        mean_sp     = float(np.mean([g["sparsity"]   for g in group]))
        mean_h      = float(np.mean([g["homophily"]  for g in group]))
        mean_cv     = float(np.mean([g["deg_cv"]     for g in group]))
        tgs_wins    = mean_delta > 0.0
        pred_wins   = mean_score > threshold
        agg.append({
            "p_intra": key[0], "p_inter": key[1],
            "hub_pct": key[2], "extra_per_hub": key[3],
            "score": mean_score, "homophily": mean_h, "deg_cv": mean_cv,
            "mean_delta": mean_delta, "mean_tgs": mean_tgs,
            "mean_static": mean_static, "mean_sparsity": mean_sp,
            "tgs_wins": tgs_wins, "pred_wins": pred_wins,
            "correct": tgs_wins == pred_wins,
        })

    agg.sort(key=lambda x: x["score"])
    return agg


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    print("=" * 72)
    print("TGS Structural Predictor Validation")
    print("=" * 72)

    # ── Phase 1: Calibration ─────────────────────────────────────────────
    print(f"\n[Phase 1] Calibration  ({len(CALIB_CONFIGS)} configs × {len(CALIB_SEEDS)} seed(s) = "
          f"{len(CALIB_CONFIGS)*len(CALIB_SEEDS)} runs)")

    calib_results = []
    total_c = len(CALIB_CONFIGS) * len(CALIB_SEEDS)
    done_c  = 0
    t0 = time.time()

    for cfg in CALIB_CONFIGS:
        for seed in CALIB_SEEDS:
            r = run_experiment(*cfg, seed)
            calib_results.append(r)
            done_c += 1
            eta = (time.time() - t0) / done_c * (total_c - done_c)
            print(
                f"  [{done_c:2d}/{total_c}] score={r['score']:.3f}  "
                f"h={r['homophily']:.3f}  cv={r['deg_cv']:.3f}  "
                f"TGS={r['tgs_acc']:.4f}  Stat={r['static_acc']:.4f}  "
                f"Δ={r['delta']:+.4f}  sp={r['sparsity']:.3f}  ETA={eta:.0f}s"
            )

    threshold, calib_acc, calib_agg = fit_threshold(calib_results)
    print(f"\n>> Calibrated threshold:  SCORE > {threshold:.4f}")
    print(f">> Calibration accuracy:  {calib_acc:.1%}")
    print(">> THRESHOLD LOCKED — will not change for held-out evaluation.")

    # ── Phase 2: Held-out ────────────────────────────────────────────────
    print(f"\n[Phase 2] Held-out  ({len(HELD_CONFIGS)} configs × {len(HELD_SEEDS)} seeds = "
          f"{len(HELD_CONFIGS)*len(HELD_SEEDS)} runs)")

    held_results = []
    total_h = len(HELD_CONFIGS) * len(HELD_SEEDS)
    done_h  = 0
    t0 = time.time()

    for cfg in HELD_CONFIGS:
        for seed in HELD_SEEDS:
            r = run_experiment(*cfg, seed)
            held_results.append(r)
            done_h += 1
            eta = (time.time() - t0) / done_h * (total_h - done_h)
            print(
                f"  [{done_h:2d}/{total_h}] score={r['score']:.3f}  "
                f"h={r['homophily']:.3f}  cv={r['deg_cv']:.3f}  "
                f"TGS={r['tgs_acc']:.4f}  Stat={r['static_acc']:.4f}  "
                f"Δ={r['delta']:+.4f}  sp={r['sparsity']:.3f}  ETA={eta:.0f}s"
            )

    # ── Analysis ──────────────────────────────────────────────────────────
    held_agg = evaluate_held(held_results, threshold)
    n_total   = len(held_agg)
    n_correct = sum(a["correct"] for a in held_agg)
    pred_acc  = n_correct / n_total

    rule_pos = [a for a in held_agg if a["pred_wins"]]
    rule_neg = [a for a in held_agg if not a["pred_wins"]]

    print("\n" + "=" * 72)
    print("HELD-OUT RESULTS (threshold locked from calibration)")
    print("=" * 72)
    print(f"\n{'Score':>7} {'H':>6} {'CV':>6} {'TGS':>7} {'Static':>8} "
          f"{'Δ(pp)':>7} {'Sp%':>6} {'Actual':>8} {'Pred':>6} {'✓':>4}")
    print("-" * 72)
    for a in held_agg:
        print(
            f"{a['score']:>7.3f} {a['homophily']:>6.3f} {a['deg_cv']:>6.3f} "
            f"{a['mean_tgs']:>7.4f} {a['mean_static']:>8.4f} "
            f"{a['mean_delta']*100:>+7.2f} {a['mean_sparsity']*100:>6.1f} "
            f"{'TGS':>8}" if a['tgs_wins'] else
            f"{a['score']:>7.3f} {a['homophily']:>6.3f} {a['deg_cv']:>6.3f} "
            f"{a['mean_tgs']:>7.4f} {a['mean_static']:>8.4f} "
            f"{a['mean_delta']*100:>+7.2f} {a['mean_sparsity']*100:>6.1f} "
            f"{'Static':>8}",
            f"{'TGS' if a['pred_wins'] else 'Static':>6}",
            f"{'✓' if a['correct'] else '✗':>4}"
        )

    print("\n" + "─" * 72)
    print(f"Predictor accuracy (held-out):  {n_correct}/{n_total} = {pred_acc:.1%}")

    pos_deltas = [a["mean_delta"] for a in rule_pos]
    pos_sp     = [a["mean_sparsity"] for a in rule_pos]
    neg_deltas = [a["mean_delta"] for a in rule_neg]

    if rule_pos:
        print(f"\nRule-POSITIVE graphs (n={len(rule_pos)}, score > {threshold:.3f}):")
        print(f"  TGS accuracy gain:    {np.mean(pos_deltas)*100:+.2f} pp  "
              f"(range {min(pos_deltas)*100:+.1f} to {max(pos_deltas)*100:+.1f} pp)")
        print(f"  Edge reduction:       {np.mean(pos_sp):.1%}")

    if rule_neg:
        print(f"\nRule-NEGATIVE graphs (n={len(rule_neg)}, score ≤ {threshold:.3f}):")
        print(f"  TGS accuracy gain:    {np.mean(neg_deltas)*100:+.2f} pp  "
              f"(static competitive or TGS hurts)")

    total_time = (time.time() - t_total) / 60
    print(f"\nTotal runtime: {total_time:.1f} min")

    # ── Headline ──────────────────────────────────────────────────────────
    if rule_pos:
        print("\n" + "=" * 72)
        print("HEADLINE RESULT:")
        print(
            f"  On unseen graphs, the locked structural rule predicted whether\n"
            f"  TGS would outperform matched-sparsity baselines with "
            f"{pred_acc:.0%} accuracy,\n"
            f"  while TGS reduced edges by {np.mean(pos_sp):.0%} and improved\n"
            f"  accuracy by {np.mean(pos_deltas)*100:+.1f} pp on graphs satisfying the rule."
        )
        print("=" * 72)

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "threshold": threshold,
        "calib_accuracy": calib_acc,
        "held_accuracy": pred_acc,
        "n_held_configs": n_total,
        "n_correct": n_correct,
        "rule_positive_n": len(rule_pos),
        "rule_negative_n": len(rule_neg),
        "rule_positive_mean_delta_pp":  float(np.mean(pos_deltas) * 100) if rule_pos else None,
        "rule_positive_mean_sparsity":  float(np.mean(pos_sp)) if rule_pos else None,
        "rule_positive_min_delta_pp":   float(min(pos_deltas) * 100) if rule_pos else None,
        "rule_positive_max_delta_pp":   float(max(pos_deltas) * 100) if rule_pos else None,
        "rule_negative_mean_delta_pp":  float(np.mean(neg_deltas) * 100) if rule_neg else None,
        "predictor_formula":    "score = homophily(G) * deg_cv(G)",
        "predictor_threshold":  threshold,
        "baseline":             "effective-resistance proxy (1/deg_u + 1/deg_v), matched sparsity",
        "n_calib_runs":         len(calib_results),
        "n_held_runs":          len(held_results),
        "calibration_agg":      calib_agg,
        "held_agg":             held_agg,
        "calibration_results":  calib_results,
        "held_results":         held_results,
        "epochs":               EPOCHS,
        "n_nodes":              N_PER * N_BLOCKS,
        "n_blocks":             N_BLOCKS,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/predictor_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → results/predictor_validation.json")
    return out


if __name__ == "__main__":
    main()
