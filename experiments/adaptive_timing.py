"""
experiments/adaptive_timing.py

Adaptive Timing Experiment
============================

Central hypothesis:
  Can SCORE(G) predict the optimal retirement start epoch (τ_G)
  without access to test labels, enabling a fully pre-training
  policy for TGS that matches oracle timing?

Policy (locked from score_to_tau.py calibration):
  if SCORE <= 0.051:   tau_hat = "never"  (TGS not useful)
  if SCORE > 0.051:    tau_hat = max(10, round(-36 * SCORE + 31))

This is computable in O(m) before any training begins.

Comparison table for each graph:
  Static/oracle topology @ t=0    (train from scratch on TGS final graph)
  TGS @ fixed t=20                (warmup=20, always)
  TGS @ fixed t=40                (warmup=40, always)
  TGS @ fixed t=80                (warmup=80, always)
  TGS @ adaptive τ̂               (warmup = score-predicted)
  Oracle τ*                       (best fixed t from TAU_GRID)
  Dense                           (upper bound)

Run on 30 new high-score graphs (score > 0.13, seeds not used before).
All policies locked before evaluation.
"""

import sys, os, json, time
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)

from tgs.core.temporal_graph import TemporalGraph
from tgs.core.influence import GradientNormEstimator
from tgs.models.gcn import TemporalGCN
from tgs.schedulers.adaptive_scheduler import AdaptiveRetirementScheduler
from tgs.utils.reproducibility import set_seed
from torch_geometric.utils import degree
from experiments.predictor_prospective import make_graph, graph_stats

DEVICE = torch.device("cpu")
EPOCHS = 200
SEED   = 42

# Adaptive policy coefficients (locked from score_to_tau.py calibration)
TAU_COEFF_A = -36.1
TAU_COEFF_B = 30.9
TAU_MIN     = 10
SCORE_THRESHOLD = 0.051   # below this: TGS not useful

# Fixed comparison timings
FIXED_TIMINGS = [0, 20, 40, 80]
TAU_GRID      = [0, 10, 20, 30, 40, 50, 60, 80]  # for oracle

CHECKPOINT = "results/adaptive_timing_checkpoint.json"
CHUNK_SIZE  = 3

# 30 new high-score configs (score > 0.13, different from all prior experiments)
CONFIGS = [
    (0.037, 0.014, 0.02, 14),
    (0.039, 0.013, 0.03, 18),
    (0.041, 0.013, 0.03, 19),
    (0.043, 0.012, 0.03, 20),
    (0.044, 0.012, 0.04, 22),
    (0.046, 0.012, 0.04, 23),
    (0.049, 0.011, 0.04, 24),
    (0.051, 0.011, 0.04, 26),
    (0.053, 0.010, 0.05, 27),
    (0.054, 0.010, 0.05, 28),
    (0.056, 0.010, 0.05, 29),
    (0.059, 0.009, 0.05, 30),
    (0.061, 0.009, 0.06, 32),
    (0.063, 0.009, 0.06, 33),
    (0.066, 0.008, 0.06, 34),
    (0.068, 0.008, 0.07, 36),
    (0.071, 0.008, 0.07, 38),
    (0.073, 0.007, 0.07, 39),
    (0.076, 0.007, 0.08, 41),
    (0.078, 0.007, 0.08, 42),
    (0.081, 0.006, 0.08, 43),
    (0.083, 0.006, 0.08, 44),
    (0.086, 0.006, 0.09, 46),
    (0.088, 0.005, 0.09, 47),
    (0.091, 0.005, 0.09, 49),
    (0.093, 0.005, 0.09, 50),
    (0.096, 0.004, 0.10, 52),
    (0.098, 0.004, 0.10, 53),
    (0.044, 0.011, 0.04, 24),
    (0.074, 0.007, 0.07, 40),
]


def predict_tau(score):
    """Adaptive policy: score -> tau_hat. Locked from calibration."""
    if score <= SCORE_THRESHOLD:
        return None   # TGS not useful
    return max(TAU_MIN, round(TAU_COEFF_A * score + TAU_COEFF_B))


def get_splits(n):
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g)
    tm = torch.zeros(n, dtype=torch.bool)
    vm = torch.zeros(n, dtype=torch.bool)
    tsm = torch.zeros(n, dtype=torch.bool)
    tm[perm[:int(0.6*n)]]             = True
    vm[perm[int(0.6*n):int(0.8*n)]]  = True
    tsm[perm[int(0.8*n):]]            = True
    return tm, vm, tsm


def acc_fn(model, ei, x, y, mask):
    model.eval()
    with torch.no_grad():
        return float((model(x, ei).argmax(-1)[mask] == y[mask]).float().mean())


def run_tgs_get_sparse(ei, x, y, n, nc, tm, vm, tsm):
    """Run TGS, return final sparse graph and accuracy."""
    m0 = ei.shape[1]
    set_seed(SEED)
    tg  = TemporalGraph(ei, n, device=DEVICE)
    est = GradientNormEstimator(m0, DEVICE, edge_index=ei, num_nodes=n,
                                alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
    mt  = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
    ot  = torch.optim.Adam(list(mt.parameters()) + [est.edge_weights],
                           lr=0.01, weight_decay=5e-4)
    sc  = AdaptiveRetirementScheduler(tg, epsilon_max=5e-3, epsilon_min=1e-5,
              anneal_steps=100, warmup_steps=40, max_retire_frac=0.06,
              max_sparsity=0.40, retire_every=2)
    bvv = bvt = 0.0
    for e in range(EPOCHS):
        mt.train(); am = tg.active_mask
        F.cross_entropy(mt(x, tg.edge_index, est.edge_weights[am])[tm], y[tm]).backward()
        est.update_influence(am); ot.step(); ot.zero_grad()
        sc.update_val_acc(acc_fn(mt, tg.edge_index, x, y, vm))
        sc.step(est.influence_scores(am)); tg.step()
        va = acc_fn(mt, tg.edge_index, x, y, vm)
        ta = acc_fn(mt, tg.edge_index, x, y, tsm)
        if va > bvv: bvv, bvt = va, ta
    return tg.active_mask.clone(), float(tg.sparsity)


def train_from_t(ei_dense, ei_sparse, t_retire, x, y, tm, vm, tsm, nc):
    """Dense for t_retire epochs, then switch to sparse. Return best val-gated acc."""
    set_seed(SEED)
    mt  = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
    ot  = torch.optim.Adam(mt.parameters(), lr=0.01, weight_decay=5e-4)
    bvv = bvt = 0.0
    for e in range(EPOCHS):
        ei_cur = ei_dense if e < t_retire else ei_sparse
        mt.train(); F.cross_entropy(mt(x, ei_cur)[tm], y[tm]).backward()
        ot.step(); ot.zero_grad()
        va = acc_fn(mt, ei_cur, x, y, vm)
        ta = acc_fn(mt, ei_cur, x, y, tsm)
        if va > bvv: bvv, bvt = va, ta
    return bvt


def process_config(p_in, p_out, hub, extra):
    """Run all policies on one graph config."""
    ei, x, y, n, nc = make_graph(p_in, p_out, hub, extra, seed=SEED)
    h, cv, score     = graph_stats(ei, y, n)
    ei = ei.to(DEVICE); x = x.to(DEVICE); y = y.to(DEVICE)
    tm, vm, tsm = get_splits(n)

    # Adaptive tau prediction (BEFORE any training)
    tau_hat = predict_tau(score)

    # TGS final sparse graph
    active_mask, sparsity = run_tgs_get_sparse(ei, x, y, n, nc, tm, vm, tsm)
    ei_sparse = ei[:, active_mask]

    # Dense upper bound
    set_seed(SEED)
    mt_d = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
    ot_d = torch.optim.Adam(mt_d.parameters(), lr=0.01, weight_decay=5e-4)
    bvv = bvt_dense = 0.0
    for e in range(EPOCHS):
        mt_d.train(); F.cross_entropy(mt_d(x, ei)[tm], y[tm]).backward()
        ot_d.step(); ot_d.zero_grad()
        va = acc_fn(mt_d, ei, x, y, vm); ta = acc_fn(mt_d, ei, x, y, tsm)
        if va > bvv: bvv, bvt_dense = va, ta

    # All fixed timings + adaptive + oracle sweep
    timing_results = {}
    for t in FIXED_TIMINGS:
        timing_results[f"fixed_t{t}"] = train_from_t(ei, ei_sparse, t, x, y, tm, vm, tsm, nc)

    # Adaptive policy
    if tau_hat is not None:
        timing_results["adaptive"] = train_from_t(ei, ei_sparse, tau_hat, x, y, tm, vm, tsm, nc)
    else:
        timing_results["adaptive"] = bvt_dense  # don't retire = dense

    # Oracle: sweep TAU_GRID, pick best
    oracle_accs = {}
    for t in TAU_GRID:
        oracle_accs[t] = train_from_t(ei, ei_sparse, t, x, y, tm, vm, tsm, nc)
    tau_star = max(oracle_accs, key=oracle_accs.get)
    timing_results["oracle"] = oracle_accs[tau_star]

    return {
        "p_intra": p_in, "p_inter": p_out, "hub_pct": hub, "extra": extra,
        "score": float(score), "homophily": float(h), "deg_cv": float(cv),
        "tau_hat": tau_hat, "tau_star": int(tau_star),
        "sparsity": float(sparsity), "dense_acc": float(bvt_dense),
        "timings": timing_results,
        "oracle_curve": {str(t): v for t, v in oracle_accs.items()},
    }


def run_next_chunk():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f: state = json.load(f)
    else:
        state = {"results": []}

    done = set((r["p_intra"], r["p_inter"], r["hub_pct"]) for r in state["results"])
    todo = [(p,q,h,e) for p,q,h,e in CONFIGS if (p,q,h) not in done]
    total = len(CONFIGS)

    if not todo:
        print("All done."); return True

    t0 = time.time()
    for i, (p_in, p_out, hub, extra) in enumerate(todo[:CHUNK_SIZE]):
        r = process_config(p_in, p_out, hub, extra)
        state["results"].append(r)
        done_now = len(state["results"]); remaining = total - done_now
        eta = (time.time() - t0) / (i + 1) * remaining
        t = r["timings"]
        print(
            f"  [{done_now:2d}/{total}] score={r['score']:.4f} τ̂={r['tau_hat']:>3} τ*={r['tau_star']:>3}  "
            f"t=0:{t['fixed_t0']:.3f} t=20:{t['fixed_t20']:.3f} "
            f"adaptive:{t['adaptive']:.3f} oracle:{t['oracle']:.3f} dense:{r['dense_acc']:.3f}  "
            f"ETA={eta:.0f}s"
        )
        with open(CHECKPOINT, "w") as f: json.dump(state, f, indent=2)

    remaining = total - len(state["results"])
    if remaining > 0:
        print(f"\nChunk done. {remaining} remain (~{(time.time()-t0)/CHUNK_SIZE*remaining/60:.0f} min).")
        return False
    return True


def finalise():
    with open(CHECKPOINT) as f: state = json.load(f)
    results = state["results"]

    print("\n" + "=" * 72)
    print("ADAPTIVE TIMING EXPERIMENT — FINAL RESULTS")
    print(f"N={len(results)} graphs | Policy locked from score_to_tau.py calibration")
    print("=" * 72)

    # Aggregate policy performances (relative to dense)
    policies = ["fixed_t0", "fixed_t20", "fixed_t40", "fixed_t80", "adaptive", "oracle"]
    labels   = ["Static/Oracle @ t=0", "Fixed t=20", "Fixed t=40", "Fixed t=80",
                 "Adaptive τ̂ (score-pred)", "Oracle τ*"]

    print(f"\n  {'Policy':26s}  {'Mean acc':>9}  {'vs Dense':>9}  {'vs Oracle':>9}")
    print("  " + "─" * 58)

    dense_accs = np.array([r["dense_acc"] for r in results])
    mean_dense = float(dense_accs.mean())

    for policy, label in zip(policies, labels):
        accs = np.array([r["timings"][policy] for r in results])
        mean_acc  = accs.mean()
        vs_dense  = mean_acc - mean_dense
        oracle_accs = np.array([r["timings"]["oracle"] for r in results])
        vs_oracle = mean_acc - oracle_accs.mean()
        marker = " ◄" if policy == "adaptive" else ""
        print(f"  {label:26s}  {mean_acc:>9.4f}  {vs_dense:>+9.4f}  {vs_oracle:>+9.4f}{marker}")

    print(f"\n  Dense upper bound:          {mean_dense:.4f}")

    # Adaptive vs oracle gap
    adapt_accs  = np.array([r["timings"]["adaptive"] for r in results])
    oracle_accs = np.array([r["timings"]["oracle"]   for r in results])
    adapt_vs_oracle = (adapt_accs - oracle_accs).mean()
    print(f"\n  Adaptive vs Oracle gap:    {adapt_vs_oracle:+.4f} ({adapt_vs_oracle*100:+.2f}pp)")
    print(f"  Adaptive matches Oracle:   {abs(adapt_vs_oracle) < 0.01}")

    # τ̂ accuracy
    tau_errors = [abs((r["tau_hat"] or 200) - r["tau_star"]) for r in results]
    print(f"\n  τ̂ prediction MAE:          {np.mean(tau_errors):.1f} epochs")

    os.makedirs("results", exist_ok=True)
    policy_means = {}
    for policy, label in zip(policies, labels):
        accs = np.array([r["timings"][policy] for r in results])
        policy_means[policy] = {"label": label, "mean": float(accs.mean()),
                                "std": float(accs.std()), "vs_dense": float(accs.mean()-mean_dense)}
    out = {
        "n": len(results), "policy_means": policy_means,
        "mean_dense": mean_dense,
        "adaptive_vs_oracle_pp": float(adapt_vs_oracle * 100),
        "tau_mae": float(np.mean(tau_errors)),
        "results": results,
    }
    with open("results/adaptive_timing.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → results/adaptive_timing.json")
    return out


if __name__ == "__main__":
    import argparse; p = argparse.ArgumentParser()
    p.add_argument("--finalise", action="store_true")
    args = p.parse_args()
    if args.finalise: finalise()
    else:
        done = run_next_chunk()
        if done: finalise()
