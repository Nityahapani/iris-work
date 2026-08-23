"""
experiments/feature_ablation.py

Feature Ablation Tournament
============================

Question: Is SCORE = homophily × deg_cv the right predictor, or is it a proxy
for something simpler (density, average degree, clustering, etc.)?

Method:
  For each candidate predictor, compute its value on every graph in the
  held-out set, fit a threshold (on calibration), then evaluate prediction
  accuracy on the locked held-out set.

  Candidate predictors (all computable from graph structure in O(m) or O(m log n)):
    1.  homophily            — fraction same-class edges
    2.  deg_cv               — coefficient of variation of degree
    3.  density              — |E| / (n*(n-1))
    4.  mean_degree          — 2|E|/n
    5.  log_n                — log(num nodes)
    6.  clustering_coeff     — mean local clustering coefficient
    7.  assortativity        — degree assortativity (Pearson r)
    8.  H_alone              — homophily only (same as #1, explicit)
    9.  CV_alone             — deg_cv only (same as #2, explicit)
    10. H_plus_CV            — homophily + deg_cv  (additive)
    11. H_times_CV           — homophily × deg_cv  (THE RULE)
    12. spectral_gap         — λ₂ of normalized Laplacian (algebraic connectivity proxy)
    13. er_mean              — mean effective-resistance proxy (1/d_u + 1/d_v)
    14. rf_all               — RandomForest on all 10 structural features (black-box)

  Binary task (same as predictor_prospective.py):
    Positive: TGS provides substantial gain (mean_delta > 3pp)
    Negative: TGS provides negligible gain

  Threshold for each predictor: calibrated on 20 configs, locked for held-out.
  Evaluation on 30 held-out configs (same as predictor_prospective.py).

  Metrics: Accuracy, Balanced Accuracy, MCC — all reported for each predictor.
"""

import sys, os, json, time
sys.path.insert(0, ".")

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)

from torch_geometric.utils import degree
from experiments.predictor_prospective import (
    make_graph, graph_stats, run_config,
    CALIB_LOW, CALIB_HIGH, HELD_LOW, HELD_HIGH,
    ALL_SEEDS, OUTCOME_MARGIN
)

DEVICE = torch.device("cpu")


# ─── Extended structural feature extraction ──────────────────────────────────

def compute_all_features(ei, y, n):
    """
    Compute all structural predictors from (edge_index, labels, n_nodes).
    Returns dict of feature_name → value.
    """
    src = ei[0].numpy(); dst = ei[1].numpy(); yn = y.numpy()
    m   = ei.shape[1]

    # 1. Homophily
    homophily = float((yn[src] == yn[dst]).mean())

    # 2. Degree CV
    deg_arr = degree(ei[1], n).numpy()
    deg_cv  = float(deg_arr.std() / max(deg_arr.mean(), 1e-8))

    # 3. Density
    density = float(m / max(n * (n - 1), 1))

    # 4. Mean degree
    mean_deg = float(deg_arr.mean())

    # 5. log(n)
    log_n = float(np.log(n))

    # 6. Local clustering coefficient (mean)
    # For each node u: C(u) = (triangles through u) / (deg(u)*(deg(u)-1))
    # Approximate via counting common neighbors for sampled edges
    adj = {}
    for u, v in zip(src, dst):
        adj.setdefault(u, set()).add(v)
    clust_vals = []
    for u in range(n):
        nb = adj.get(u, set())
        du = len(nb)
        if du < 2:
            clust_vals.append(0.0)
            continue
        tri = sum(1 for v in nb for w in nb if w != v and w in adj.get(v, set()))
        clust_vals.append(tri / (du * (du - 1)))
    clustering = float(np.mean(clust_vals))

    # 7. Degree assortativity (Pearson correlation of endpoint degrees)
    src_deg = deg_arr[src]; dst_deg = deg_arr[dst]
    if src_deg.std() > 0 and dst_deg.std() > 0:
        assortativity = float(np.corrcoef(src_deg, dst_deg)[0, 1])
    else:
        assortativity = 0.0

    # 8. Spectral gap proxy: ratio λ₁/λ₂ of degree sequence
    # True spectral gap requires eigendecomposition (expensive); use
    # Cheeger-bound proxy: min_cut_proxy = min(d_u+d_v) / max(d_u+d_v)
    # Actually just use normalized algebraic connectivity proxy from degree stats
    sorted_deg = np.sort(deg_arr)
    spectral_gap = float(sorted_deg[1] / max(sorted_deg[-1], 1))  # λ₂ lower bound proxy

    # 9. Effective resistance proxy: mean (1/d_u + 1/d_v)
    er_mean = float(np.mean(1.0 / src_deg.clip(1) + 1.0 / dst_deg.clip(1)))

    return {
        "homophily":     homophily,
        "deg_cv":        deg_cv,
        "density":       density,
        "mean_degree":   mean_deg,
        "log_n":         log_n,
        "clustering":    clustering,
        "assortativity": assortativity,
        "spectral_gap":  spectral_gap,
        "er_mean":       er_mean,
        "H_plus_CV":     homophily + deg_cv,
        "H_times_CV":    homophily * deg_cv,   # THE RULE
    }


# ─── Predictor definitions ────────────────────────────────────────────────────

PREDICTOR_NAMES = [
    "homophily",
    "deg_cv",
    "density",
    "mean_degree",
    "log_n",
    "clustering",
    "assortativity",
    "spectral_gap",
    "er_mean",
    "H_plus_CV",
    "H_times_CV",    # our rule
    "rf_all",        # random forest on all features
]

# Which direction means "TGS wins" for each predictor
# +1: higher value → predict TGS wins
# -1: lower value  → predict TGS wins (fit by checking both during calibration)
DIRECTION = {k: +1 for k in PREDICTOR_NAMES}
# These may flip — we'll discover during calibration


# ─── Threshold calibration ───────────────────────────────────────────────────

def fit_threshold_single(scores, labels):
    """Fit threshold and direction for a single predictor."""
    scores = np.array(scores); labels = np.array(labels)
    best_acc = -1; best_thr = 0; best_dir = +1
    for direction in [+1, -1]:
        for thr in np.linspace(scores.min() - 1e-4, scores.max() + 1e-4, 500):
            pred = (direction * scores) > (direction * thr)
            acc  = (pred == labels).mean()
            if acc > best_acc:
                best_acc = acc; best_thr = thr; best_dir = direction
    return float(best_thr), int(best_dir), float(best_acc)


def predict_single(score, thr, direction):
    return bool((direction * score) > (direction * thr))


# ─── RF predictor ─────────────────────────────────────────────────────────────

def fit_rf(feature_matrix, labels):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(feature_matrix, labels)
    return rf


# ─── Metrics ─────────────────────────────────────────────────────────────────

def metrics(preds, labels):
    preds = np.array(preds); labels = np.array(labels)
    TP = int(( preds &  labels).sum())
    FP = int(( preds & ~labels).sum())
    FN = int((~preds &  labels).sum())
    TN = int((~preds & ~labels).sum())
    n  = TP + FP + FN + TN
    acc  = (TP + TN) / n
    prec = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    rec  = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    spec = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
    bal  = (rec + spec) / 2 if not (np.isnan(rec) or np.isnan(spec)) else float("nan")
    den  = ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) ** 0.5
    mcc  = (TP*TN - FP*FN) / den if den > 0 else float("nan")
    return {"acc": acc, "bal_acc": bal, "mcc": mcc,
            "prec": prec, "rec": rec, "spec": spec,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN}


# ─── Run configs and extract features ────────────────────────────────────────

CHECKPOINT = "results/feature_ablation_checkpoint.json"
CHUNK_SIZE  = 5


def run_or_load_configs():
    all_configs = (
        [("C", cfg) for cfg in CALIB_LOW + CALIB_HIGH] +
        [("H", cfg) for cfg in HELD_LOW  + HELD_HIGH]
    )

    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f: state = json.load(f)
    else:
        state = {"calib": [], "held": []}

    done = len(state["calib"]) + len(state["held"])
    total = len(all_configs)
    if done >= total:
        return state

    print(f"Running {total - done} remaining configs...")
    t0 = time.time()
    for i in range(done, total):
        phase, cfg = all_configs[i]
        seeds = ALL_SEEDS
        # Run TGS vs static to get ground truth
        r = run_config(*cfg, seeds=seeds)
        # Extract all structural features (use first seed's graph as representative)
        ei, x, y, n, nc = make_graph(*cfg, seeds[0])
        feats = compute_all_features(ei, y, n)
        r["features"] = feats
        r["substantial"] = r["mean_delta"] > OUTCOME_MARGIN

        done_now = i + 1
        eta = (time.time() - t0) / done_now * (total - done_now)
        print(f"  [{phase} {done_now:2d}/{total}]  "
              f"score={feats['H_times_CV']:.4f}  "
              f"Δ={r['mean_delta']:+.4f}  "
              f"subst={r['substantial']}  ETA={eta:.0f}s")

        if phase == "C": state["calib"].append(r)
        else:            state["held"].append(r)
        with open(CHECKPOINT, "w") as f: json.dump(state, f, indent=2)

    return state


def run_next_chunk_only():
    """Run only the next CHUNK_SIZE configs, then return."""
    all_configs = (
        [("C", cfg) for cfg in CALIB_LOW + CALIB_HIGH] +
        [("H", cfg) for cfg in HELD_LOW  + HELD_HIGH]
    )
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f: state = json.load(f)
    else:
        state = {"calib": [], "held": []}

    done = len(state["calib"]) + len(state["held"])
    total = len(all_configs)
    if done >= total:
        print("All configs done."); return True

    t0 = time.time()
    end = min(done + CHUNK_SIZE, total)
    for i in range(done, end):
        phase, cfg = all_configs[i]
        r = run_config(*cfg, seeds=ALL_SEEDS)
        ei, x, y, n, nc = make_graph(*cfg, ALL_SEEDS[0])
        feats = compute_all_features(ei, y, n)
        r["features"] = feats
        r["substantial"] = r["mean_delta"] > OUTCOME_MARGIN
        print(f"  [{phase} {i+1:2d}/{total}]  "
              f"H×CV={feats['H_times_CV']:.4f}  "
              f"Δ={r['mean_delta']:+.4f}  "
              f"{'SUBST' if r['substantial'] else 'NEGL'}  "
              f"{time.time()-t0:.0f}s")
        if phase == "C": state["calib"].append(r)
        else:            state["held"].append(r)
        with open(CHECKPOINT, "w") as f: json.dump(state, f, indent=2)

    remaining = total - end
    if remaining > 0:
        elapsed = time.time() - t0
        print(f"\nChunk done. {remaining} configs remain "
              f"(~{elapsed/CHUNK_SIZE*remaining/60:.0f} min).")
        return False
    return True


def finalise():
    with open(CHECKPOINT) as f: state = json.load(f)
    calib = state["calib"]; held = state["held"]

    calib_feats   = [r["features"]    for r in calib]
    calib_labels  = np.array([r["substantial"] for r in calib])
    held_feats    = [r["features"]    for r in held]
    held_labels   = np.array([r["substantial"] for r in held])

    # Feature matrix for RF
    feat_keys = ["homophily","deg_cv","density","mean_degree","log_n",
                 "clustering","assortativity","spectral_gap","er_mean"]
    calib_X = np.array([[r[k] for k in feat_keys] for r in calib_feats])
    held_X  = np.array([[r[k] for k in feat_keys] for r in held_feats])

    # Fit RF
    rf_model = fit_rf(calib_X, calib_labels)
    rf_calib_pred = rf_model.predict(calib_X).astype(bool)
    rf_held_pred  = rf_model.predict(held_X).astype(bool)

    results = []
    print("\n" + "="*72)
    print("FEATURE ABLATION TOURNAMENT")
    print("="*72)
    print(f"\n{'Predictor':20s} {'Calib Acc':>10} {'Held Acc':>9} "
          f"{'Bal Acc':>8} {'MCC':>7}")
    print("─"*60)

    for name in PREDICTOR_NAMES:
        if name == "rf_all":
            c_preds = rf_calib_pred
            h_preds = rf_held_pred
        else:
            c_scores = np.array([r[name] for r in calib_feats])
            h_scores = np.array([r[name] for r in held_feats])
            thr, direction, _ = fit_threshold_single(c_scores, calib_labels)
            c_preds = np.array([predict_single(s, thr, direction) for s in c_scores])
            h_preds = np.array([predict_single(s, thr, direction) for s in h_scores])

        c_m = metrics(c_preds, calib_labels)
        h_m = metrics(h_preds, held_labels)

        # Mark our rule
        marker = " ◄" if name == "H_times_CV" else ""
        print(f"  {name:18s} {c_m['acc']:>9.1%} {h_m['acc']:>9.1%} "
              f"{h_m['bal_acc']:>8.1%} {h_m['mcc']:>+7.3f}{marker}")
        results.append({
            "predictor":      name,
            "calib_accuracy": c_m["acc"],
            "held_accuracy":  h_m["acc"],
            "held_bal_acc":   h_m["bal_acc"],
            "held_mcc":       h_m["mcc"],
            "held_metrics":   h_m,
        })

    # Print ranking
    results_sorted = sorted(results, key=lambda x: x["held_mcc"], reverse=True)
    print(f"\n  Ranking by held-out MCC:")
    for rank, r in enumerate(results_sorted, 1):
        marker = " ◄ OUR RULE" if r["predictor"] == "H_times_CV" else ""
        print(f"    {rank:2d}. {r['predictor']:18s}  MCC={r['held_mcc']:+.3f}{marker}")

    our_rank = next(i+1 for i,r in enumerate(results_sorted) if r["predictor"]=="H_times_CV")
    print(f"\n  H×CV ranks {our_rank}/{len(results_sorted)} predictors by MCC.")

    # RF feature importances
    importances = rf_model.feature_importances_
    print(f"\n  RF feature importances:")
    for k, imp in sorted(zip(feat_keys, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {k:18s}  {imp:.3f}  {bar}")

    os.makedirs("results", exist_ok=True)
    out = {
        "results": results,
        "ranking": [r["predictor"] for r in results_sorted],
        "our_rule_rank": our_rank,
        "rf_feature_importances": dict(zip(feat_keys, importances.tolist())),
        "outcome_margin_pp": OUTCOME_MARGIN * 100,
        "n_calib": len(calib), "n_held": len(held),
    }
    with open("results/feature_ablation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → results/feature_ablation.json")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--finalise", action="store_true")
    args = p.parse_args()
    if args.finalise:
        finalise()
    else:
        done = run_next_chunk_only()
        if done:
            finalise()
