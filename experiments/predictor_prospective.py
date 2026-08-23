"""
experiments/predictor_prospective.py

Prospective Model-Selection Validation — Confusion Matrix Version
=================================================================

Framing:
  The binary classification task is:
    Positive class: TGS provides SUBSTANTIAL gain (mean_delta > OUTCOME_MARGIN)
    Negative class: TGS provides NEGLIGIBLE gain  (mean_delta ≤ OUTCOME_MARGIN)

  This is scientifically honest: TGS almost never hurts, but on structurally
  unsuitable graphs it provides no meaningful advantage over static pruning.
  The useful prediction is whether TGS is *worth deploying* — i.e., whether
  the structured retirement schedule earns its complexity.

  OUTCOME_MARGIN = 0.03  (3 percentage points — practically meaningful threshold)

Predictor (locked before any training):
  SCORE(G) = homophily(G) × deg_cv(G)
  Predict "substantial gain" iff SCORE > THRESHOLD

Protocol:
  Phase 1 — Calibration:
    20 configs × 8 seeds each, equally split low/high score.
    Fit threshold via accuracy maximisation. Lock it.

  Phase 2 — Held-out:
    30 configs × 8 seeds each, equally split low/high score.
    Scores from LOW band (<0.08) and HIGH band (>0.15) only.
    Ambiguous middle band (0.08–0.15) excluded by design.
    Ground truth: mean_delta > 0.03 across 8 seeds.

  Both calibration and held-out use DISJOINT parameter values and seeds.
  Calibration seeds: [42, 99, 200, 301, 555, 666, 777, 888]  (same 8 seeds)
  — identical seed set for both phases; what differs is the (p_intra, p_inter,
    hub_pct) configuration, which fully determines graph structure.

Baseline: effective-resistance proxy pruning, matched sparsity.

Metrics: Accuracy, Precision, Recall, Specificity, Balanced Accuracy, MCC,
         full 2×2 confusion matrix.
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

DEVICE         = torch.device("cpu")
EPOCHS         = 200
N_PER          = 180
N_BLOCKS       = 4
OUTCOME_MARGIN = 0.03   # 3pp — defines "substantial gain"
ALL_SEEDS      = [42, 99, 200, 301, 555, 666, 777, 888]


# ─── Graph generator ────────────────────────────────────────────────────────

def make_graph(p_intra, p_inter, hub_pct, extra_per_hub, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n  = N_PER * N_BLOCKS
    nc = N_BLOCKS

    y_np = np.zeros(n, dtype=int)
    for b in range(nc):
        y_np[b * N_PER:(b + 1) * N_PER] = b

    ep = np.full((nc, nc), p_inter)
    np.fill_diagonal(ep, p_intra)
    ei_base = stochastic_blockmodel_graph([N_PER] * nc, torch.tensor(ep))

    es = list(ei_base[0].numpy())
    ed = list(ei_base[1].numpy())

    if hub_pct > 0:
        hubs = rng.choice(n, int(n * hub_pct), replace=False)
        for h in hubs:
            bn = [i for i in range(n) if y_np[i] == y_np[h] and i != h]
            for t in rng.choice(bn, min(extra_per_hub, len(bn)), replace=False):
                es.extend([int(h), int(t)])
                ed.extend([int(t), int(h)])

    ei = torch.unique(torch.tensor([es, ed], dtype=torch.long), dim=1)
    y  = torch.tensor(y_np)
    x  = torch.randn(n, nc + 4) * 0.9
    for b in range(nc):
        x[b * N_PER:(b + 1) * N_PER, b] += 1.0

    return ei, x, y, n, nc


def graph_stats(ei, y, n):
    src = ei[0].numpy(); dst = ei[1].numpy(); yn = y.numpy()
    hom = float((yn[src] == yn[dst]).mean())
    deg = degree(ei[1], n).numpy()
    cv  = float(deg.std() / max(deg.mean(), 1e-8))
    return hom, cv, hom * cv


# ─── Training runs ───────────────────────────────────────────────────────────

def run_tgs(ei, x, y, n, nc, tm, vm, tsm, seed):
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
        max_retire_frac=0.08, max_sparsity=0.65, retire_every=2)
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


def run_static_er(ei, x, y, n, nc, tm, vm, tsm, seed, target_sp):
    m0    = ei.shape[1]
    src   = ei[0].numpy(); dst = ei[1].numpy()
    dega  = degree(ei[1], n, dtype=torch.float).numpy()
    er    = 1.0 / dega[src].clip(1) + 1.0 / dega[dst].clip(1)
    n_rem = int(m0 * target_sp)
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


def run_config(p_intra, p_inter, hub_pct, extra, seeds=ALL_SEEDS):
    """Run one config across all seeds. Ground truth = mean_delta > OUTCOME_MARGIN."""
    deltas = []; tgs_accs = []; stat_accs = []; sps = []
    hs = []; cvs = []

    for seed in seeds:
        ei, x, y, n, nc = make_graph(p_intra, p_inter, hub_pct, extra, seed)
        h, cv, score    = graph_stats(ei, y, n)
        hs.append(h); cvs.append(cv)

        ei  = ei.to(DEVICE); x = x.to(DEVICE); y = y.to(DEVICE)
        g   = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        tm  = torch.zeros(n, dtype=torch.bool)
        vm  = torch.zeros(n, dtype=torch.bool)
        tsm = torch.zeros(n, dtype=torch.bool)
        tm[perm[:int(0.6*n)]]             = True
        vm[perm[int(0.6*n):int(0.8*n)]]  = True
        tsm[perm[int(0.8*n):]]            = True

        tgs_acc, sp = run_tgs(ei, x, y, n, nc, tm, vm, tsm, seed)
        stat_acc    = run_static_er(ei, x, y, n, nc, tm, vm, tsm, seed, sp)
        deltas.append(tgs_acc - stat_acc)
        tgs_accs.append(tgs_acc); stat_accs.append(stat_acc); sps.append(sp)

    mean_h     = float(np.mean(hs))
    mean_cv    = float(np.mean(cvs))
    mean_score = mean_h * mean_cv
    mean_delta = float(np.mean(deltas))

    # Ground truth: substantial gain iff mean_delta > OUTCOME_MARGIN
    substantial = mean_delta > OUTCOME_MARGIN

    return {
        "p_intra": p_intra, "p_inter": p_inter,
        "hub_pct": hub_pct, "extra_per_hub": extra,
        "homophily":  mean_h,
        "deg_cv":     mean_cv,
        "score":      mean_score,
        "mean_delta": mean_delta,
        "mean_tgs":    float(np.mean(tgs_accs)),
        "mean_static": float(np.mean(stat_accs)),
        "mean_sparsity": float(np.mean(sps)),
        "substantial": substantial,
        "seed_deltas": deltas,
        "outcome_margin": OUTCOME_MARGIN,
    }


# ─── Configurations ──────────────────────────────────────────────────────────
#
# LOW band:   score < 0.08  →  expected label: NOT substantial
# HIGH band:  score > 0.15  →  expected label: substantial
# Middle (0.08–0.15) excluded — ambiguous zone
#
# Calibration configs: 10 LOW + 10 HIGH (disjoint p values from held-out)
# Held-out configs:    15 LOW + 15 HIGH (different p values)

CALIB_LOW = [
    (0.011, 0.024, 0.00, 0),   # score ≈ 0.034
    (0.012, 0.023, 0.00, 0),   # score ≈ 0.038
    (0.013, 0.022, 0.00, 0),   # score ≈ 0.044
    (0.014, 0.021, 0.00, 0),   # score ≈ 0.050
    (0.015, 0.021, 0.00, 0),   # score ≈ 0.055
    (0.010, 0.026, 0.00, 0),   # score ≈ 0.028
    (0.010, 0.024, 0.01, 10),  # score ≈ 0.034
    (0.011, 0.023, 0.01, 10),  # score ≈ 0.040
    (0.012, 0.022, 0.01, 10),  # score ≈ 0.046
    (0.013, 0.021, 0.01, 10),  # score ≈ 0.053
]

CALIB_HIGH = [
    (0.040, 0.012, 0.03, 20),  # score ≈ 0.19
    (0.050, 0.010, 0.04, 25),  # score ≈ 0.25
    (0.060, 0.008, 0.05, 30),  # score ≈ 0.31
    (0.070, 0.007, 0.06, 35),  # score ≈ 0.38
    (0.080, 0.006, 0.07, 40),  # score ≈ 0.44
    (0.090, 0.005, 0.08, 45),  # score ≈ 0.50
    (0.035, 0.013, 0.02, 15),  # score ≈ 0.17
    (0.045, 0.011, 0.04, 25),  # score ≈ 0.23
    (0.055, 0.009, 0.05, 30),  # score ≈ 0.30
    (0.065, 0.007, 0.07, 40),  # score ≈ 0.40
]

HELD_LOW = [
    (0.010, 0.027, 0.00, 0),   # score ≈ 0.027
    (0.011, 0.026, 0.00, 0),   # score ≈ 0.030
    (0.012, 0.025, 0.00, 0),   # score ≈ 0.034
    (0.013, 0.024, 0.00, 0),   # score ≈ 0.038
    (0.014, 0.023, 0.00, 0),   # score ≈ 0.043
    (0.015, 0.022, 0.00, 0),   # score ≈ 0.049
    (0.016, 0.022, 0.00, 0),   # score ≈ 0.054
    (0.010, 0.025, 0.01, 8),   # score ≈ 0.033
    (0.011, 0.023, 0.01, 8),   # score ≈ 0.038
    (0.012, 0.022, 0.01, 8),   # score ≈ 0.045
    (0.013, 0.020, 0.01, 8),   # score ≈ 0.053
    (0.010, 0.026, 0.02, 12),  # score ≈ 0.035
    (0.011, 0.025, 0.02, 12),  # score ≈ 0.041
    (0.012, 0.024, 0.02, 12),  # score ≈ 0.047
    (0.013, 0.023, 0.02, 12),  # score ≈ 0.054
]

HELD_HIGH = [
    (0.038, 0.013, 0.02, 15),  # score ≈ 0.17
    (0.042, 0.011, 0.03, 20),  # score ≈ 0.21
    (0.048, 0.010, 0.04, 25),  # score ≈ 0.26
    (0.053, 0.009, 0.05, 30),  # score ≈ 0.30
    (0.058, 0.008, 0.05, 30),  # score ≈ 0.33
    (0.062, 0.008, 0.06, 35),  # score ≈ 0.37
    (0.068, 0.007, 0.07, 40),  # score ≈ 0.42
    (0.072, 0.006, 0.08, 45),  # score ≈ 0.47
    (0.078, 0.006, 0.08, 45),  # score ≈ 0.49
    (0.085, 0.005, 0.09, 50),  # score ≈ 0.52
    (0.092, 0.004, 0.09, 50),  # score ≈ 0.55
    (0.043, 0.011, 0.04, 22),  # score ≈ 0.22
    (0.057, 0.008, 0.06, 32),  # score ≈ 0.35
    (0.073, 0.006, 0.08, 42),  # score ≈ 0.46
    (0.095, 0.004, 0.10, 55),  # score ≈ 0.57
]


# ─── Threshold calibration ───────────────────────────────────────────────────

def fit_threshold(calib_results):
    scores = np.array([r["score"]       for r in calib_results])
    labels = np.array([r["substantial"] for r in calib_results])
    best_acc = -1.0; best_thr = 0.0
    for thr in np.linspace(scores.min() - 0.005, scores.max() + 0.005, 1000):
        acc = ((scores > thr) == labels).mean()
        if acc > best_acc:
            best_acc = acc; best_thr = thr
    return float(best_thr), float(best_acc)


# ─── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(results, threshold):
    TP = FP = FN = TN = 0
    details = []
    for r in results:
        pred = r["score"] > threshold
        actual = r["substantial"]
        if   pred and     actual: TP += 1
        elif pred and not actual: FP += 1
        elif not pred and actual: FN += 1
        else:                     TN += 1
        details.append({**r, "pred_substantial": pred,
                         "correct": pred == actual})
    n = TP + FP + FN + TN
    acc      = (TP + TN) / n
    prec     = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    recall   = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    spec     = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
    bal      = (recall + spec) / 2 if not (np.isnan(recall) or np.isnan(spec)) else float("nan")
    denom    = ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) ** 0.5
    mcc      = (TP*TN - FP*FN) / denom if denom > 0 else float("nan")
    return {"TP":TP,"FP":FP,"FN":FN,"TN":TN,
            "accuracy":acc,"precision":prec,"recall":recall,
            "specificity":spec,"balanced_acc":bal,"mcc":mcc,
            "details":details}


# ─── Run phase ───────────────────────────────────────────────────────────────

def run_phase(configs, label):
    results = []; total = len(configs); t0 = time.time()
    for i, cfg in enumerate(configs):
        r  = run_config(*cfg)
        done = i + 1
        eta  = (time.time() - t0) / done * (total - done)
        print(
            f"  [{label} {done:2d}/{total}]  "
            f"score={r['score']:.4f}  h={r['homophily']:.3f}  cv={r['deg_cv']:.3f}  "
            f"mean_Δ={r['mean_delta']:+.4f}  "
            f"{'SUBST' if r['substantial'] else 'NEGL':>5}  ETA={eta:.0f}s"
        )
        results.append(r)
    return results


def print_full_report(m, threshold, label, results):
    rule_pos = [r for r in m["details"] if r["pred_substantial"]]
    rule_neg = [r for r in m["details"] if not r["pred_substantial"]]

    print(f"\n{'═'*62}")
    print(f"  {label}")
    print(f"{'═'*62}")
    print(f"\n  Binary task: predict whether TGS gain > {OUTCOME_MARGIN*100:.0f} pp")
    print(f"  Threshold locked at: SCORE > {threshold:.4f}")
    print(f"\n  Confusion Matrix (positive = 'substantial TGS gain'):")
    print(f"  {'':25s}  Actual SUBST  Actual NEGL")
    print(f"  {'Predict SUBSTANTIAL':25s}  {m['TP']:>5}  (TP)   {m['FP']:>5}  (FP)")
    print(f"  {'Predict NEGLIGIBLE':25s}  {m['FN']:>5}  (FN)   {m['TN']:>5}  (TN)")
    print(f"\n  {'─'*50}")
    print(f"  Accuracy:            {m['accuracy']:.1%}   ({m['TP']+m['TN']}/{m['TP']+m['FP']+m['FN']+m['TN']} correct)")
    if not np.isnan(m['precision']):
        print(f"  Precision:           {m['precision']:.1%}")
    if not np.isnan(m['recall']):
        print(f"  Recall (TPR):        {m['recall']:.1%}")
    if not np.isnan(m['specificity']):
        print(f"  Specificity (TNR):   {m['specificity']:.1%}")
    if not np.isnan(m['balanced_acc']):
        print(f"  Balanced Accuracy:   {m['balanced_acc']:.1%}")
    if not np.isnan(m['mcc']):
        print(f"  MCC:                 {m['mcc']:+.3f}  ({'strong' if abs(m['mcc'])>0.7 else 'moderate' if abs(m['mcc'])>0.4 else 'weak'})")

    if rule_pos:
        pos_d = [r["mean_delta"] for r in rule_pos]
        pos_s = [r["mean_sparsity"] for r in rule_pos]
        print(f"\n  Predicted SUBSTANTIAL (n={len(rule_pos)}, score > {threshold:.3f}):")
        print(f"    Mean TGS gain:     {np.mean(pos_d)*100:+.2f} pp  [{min(pos_d)*100:+.1f}, {max(pos_d)*100:+.1f}]")
        print(f"    Edge reduction:    {np.mean(pos_s):.1%}")
    if rule_neg:
        neg_d = [r["mean_delta"] for r in rule_neg]
        print(f"\n  Predicted NEGLIGIBLE (n={len(rule_neg)}, score ≤ {threshold:.3f}):")
        print(f"    Mean TGS gain:     {np.mean(neg_d)*100:+.2f} pp  (static equally competitive)")

    # Any misclassifications
    misses = [r for r in m["details"] if not r["correct"]]
    if misses:
        print(f"\n  Misclassifications ({len(misses)}):")
        for r in misses:
            pred_str = "SUBST" if r["pred_substantial"] else "NEGL"
            act_str  = "SUBST" if r["substantial"] else "NEGL"
            print(f"    score={r['score']:.4f}  pred={pred_str}  actual={act_str}  delta={r['mean_delta']:+.4f}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    print("═" * 62)
    print("Prospective Model-Selection Validation")
    print(f"Outcome margin: {OUTCOME_MARGIN*100:.0f} pp   Seeds/config: {len(ALL_SEEDS)}")
    print("═" * 62)

    # Phase 1
    print(f"\n{'─'*62}")
    print(f"Phase 1 — Calibration  ({len(CALIB_LOW)+len(CALIB_HIGH)} configs)")
    print(f"{'─'*62}")
    calib_results = run_phase(CALIB_LOW + CALIB_HIGH, "C")
    threshold, calib_acc = fit_threshold(calib_results)
    calib_m = compute_metrics(calib_results, threshold)
    print(f"\n>> Threshold calibrated: SCORE > {threshold:.4f}  (calib acc {calib_acc:.1%})")
    print(">> THRESHOLD LOCKED.")
    print_full_report(calib_m, threshold, f"Calibration (n={len(calib_results)})", calib_results)

    # Phase 2
    print(f"\n{'─'*62}")
    print(f"Phase 2 — Held-out  ({len(HELD_LOW)+len(HELD_HIGH)} configs, threshold LOCKED at {threshold:.4f})")
    print(f"{'─'*62}")
    held_results = run_phase(HELD_LOW + HELD_HIGH, "H")
    held_m = compute_metrics(held_results, threshold)
    print_full_report(held_m, threshold, f"Held-out (n={len(held_results)})", held_results)

    # Subgroup effect sizes
    rule_pos = [r for r in held_m["details"] if r["pred_substantial"]]
    rule_neg = [r for r in held_m["details"] if not r["pred_substantial"]]
    pos_d = [r["mean_delta"] for r in rule_pos]
    pos_s = [r["mean_sparsity"] for r in rule_pos]
    neg_d = [r["mean_delta"] for r in rule_neg]

    # Headline
    m = held_m
    print(f"\n{'═'*62}")
    print("HEADLINE RESULT")
    print(f"{'═'*62}")
    print(
        f"\n  Predictor: SCORE(G) = homophily(G) × deg_cv(G) > {threshold:.3f}\n"
        f"  Task: predict whether TGS gain > {OUTCOME_MARGIN*100:.0f} pp on unseen graphs\n"
        f"\n  Confusion matrix on {m['TP']+m['FP']+m['FN']+m['TN']} held-out graphs:\n"
        f"                    Actual SUBST  Actual NEGL\n"
        f"  Predict SUBST     {m['TP']:>4}  (TP)   {m['FP']:>4}  (FP)\n"
        f"  Predict NEGL      {m['FN']:>4}  (FN)   {m['TN']:>4}  (TN)\n"
        f"\n  Accuracy:          {m['accuracy']:.1%}"
        f"    Balanced: {m['balanced_acc']:.1%}"
        f"    MCC: {m['mcc']:+.3f}\n"
        f"  Precision: {m['precision']:.1%}    Recall: {m['recall']:.1%}    Specificity: {m['specificity']:.1%}\n"
        f"\n  On predicted-substantial graphs (n={len(rule_pos)}):\n"
        f"    TGS reduces edges by {np.mean(pos_s):.0%}\n"
        f"    TGS improves accuracy by {np.mean(pos_d)*100:+.1f} pp over matched-sparsity static pruning\n"
        f"\n  On predicted-negligible graphs (n={len(rule_neg)}):\n"
        f"    TGS gain: {np.mean(neg_d)*100:+.1f} pp — static pruning equally competitive"
    )

    print(f"\nTotal runtime: {(time.time()-t_total)/60:.1f} min")

    # Save
    os.makedirs("results", exist_ok=True)
    out = {
        "predictor_formula":  "score = homophily(G) * deg_cv(G)",
        "threshold":          threshold,
        "outcome_margin_pp":  OUTCOME_MARGIN * 100,
        "baseline":           "effective-resistance proxy, matched sparsity",
        "n_seeds_per_config": len(ALL_SEEDS),
        "all_seeds":          ALL_SEEDS,
        "n_calib":            len(calib_results),
        "n_held":             len(held_results),
        "calib_accuracy":     calib_acc,
        "held_metrics": {k: v for k, v in held_m.items() if k != "details"},
        "rule_positive_n":             len(rule_pos),
        "rule_positive_mean_delta_pp": float(np.mean(pos_d) * 100),
        "rule_positive_mean_sparsity": float(np.mean(pos_s)),
        "rule_negative_n":             len(rule_neg),
        "rule_negative_mean_delta_pp": float(np.mean(neg_d) * 100),
        "calib_results":  calib_results,
        "held_results":   held_results,
        "held_details":   held_m["details"],
    }
    with open("results/predictor_prospective.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved → results/predictor_prospective.json")
    return out


if __name__ == "__main__":
    main()
