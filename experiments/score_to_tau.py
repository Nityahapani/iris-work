"""
experiments/score_to_tau.py

SCORE → τ_G: Structural Rule Predicts Representation Maturation Speed
======================================================================

For each graph G, define:
    τ_G = min t such that oracle_acc(t) >= dense_acc - 0.01
          i.e. the earliest retirement epoch where the sparse graph
          fully recovers the dense baseline (within 1pp).

Test: does SCORE(G) = homophily × deg_cv predict τ_G?
If SCORE negatively predicts τ_G, the mechanistic chain is complete:

    High H×CV → representations mature faster (small τ_G)
             → edges become redundant sooner
             → TGS can retire them earlier
             → larger accuracy gain

This closes the loop from graph structure → representation dynamics → outcome.

Run on 8 graphs spanning score 0.03 to 0.48.
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
DENSE_TOL = 0.01  # τ_G defined as: oracle_acc(t) >= dense_acc - DENSE_TOL

# Retirement epochs to sweep for τ_G
TAU_GRID = [0, 10, 20, 30, 40, 50, 60, 80, 100, 120, 160, 200]

# 8 configs spanning score 0.03–0.48 (from verified configs in feature_ablation_checkpoint)
CONFIGS = [
    dict(p_intra=0.011, p_inter=0.026, hub_pct=0.00, extra_per_hub=0,  score_approx=0.030),
    dict(p_intra=0.013, p_inter=0.020, hub_pct=0.01, extra_per_hub=8,  score_approx=0.053),
    dict(p_intra=0.035, p_inter=0.013, hub_pct=0.02, extra_per_hub=15, score_approx=0.146),
    dict(p_intra=0.040, p_inter=0.012, hub_pct=0.03, extra_per_hub=20, score_approx=0.193),
    dict(p_intra=0.048, p_inter=0.010, hub_pct=0.04, extra_per_hub=25, score_approx=0.245),
    dict(p_intra=0.058, p_inter=0.008, hub_pct=0.05, extra_per_hub=30, score_approx=0.311),
    dict(p_intra=0.065, p_inter=0.007, hub_pct=0.07, extra_per_hub=40, score_approx=0.400),
    dict(p_intra=0.095, p_inter=0.004, hub_pct=0.10, extra_per_hub=55, score_approx=0.482),
]


def acc(model, ei, x, y, mask):
    model.eval()
    with torch.no_grad():
        return float((model(x, ei).argmax(-1)[mask] == y[mask]).float().mean())


def run_dense(ei, x, y, n, nc, tm, vm, tsm):
    """Run full dense training, return final acc."""
    set_seed(SEED)
    mt = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
    ot = torch.optim.Adam(mt.parameters(), lr=0.01, weight_decay=5e-4)
    bvv = bvt = 0.0
    for e in range(EPOCHS):
        mt.train(); F.cross_entropy(mt(x, ei)[tm], y[tm]).backward()
        ot.step(); ot.zero_grad()
        va = acc(mt, ei, x, y, vm); ta = acc(mt, ei, x, y, tsm)
        if va > bvv: bvv, bvt = va, ta
    return bvt


def run_tgs_get_sparse(ei, x, y, n, nc, tm, vm, tsm):
    """Run TGS, return the final sparse graph (active mask)."""
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
        sc.update_val_acc(acc(mt, tg.edge_index, x, y, vm))
        sc.step(est.influence_scores(am)); tg.step()
        va = acc(mt, tg.edge_index, x, y, vm)
        ta = acc(mt, tg.edge_index, x, y, tsm)
        if va > bvv: bvv, bvt = va, ta
    return tg.active_mask.clone(), bvt, float(tg.sparsity)


def train_from_t(ei_dense, ei_sparse, t_retire):
    """Dense until t_retire, then switch to sparse. Return best val-gated test acc."""
    set_seed(SEED)
    mt  = TemporalGCN(ei_dense.shape[0] if False else x_g.shape[1], 40, nc_g, 2, 0.5).to(DEVICE)
    ot  = torch.optim.Adam(mt.parameters(), lr=0.01, weight_decay=5e-4)
    bvv = bvt = 0.0
    for e in range(EPOCHS):
        ei_cur = ei_dense if e < t_retire else ei_sparse
        mt.train(); F.cross_entropy(mt(x_g, ei_cur)[tm_g], y_g[tm_g]).backward()
        ot.step(); ot.zero_grad()
        va = acc(mt, ei_cur, x_g, y_g, vm_g)
        ta = acc(mt, ei_cur, x_g, y_g, tsm_g)
        if va > bvv: bvv, bvt = va, ta
    return bvt


# Globals set per-graph for train_from_t (avoids passing everywhere)
x_g = y_g = tm_g = vm_g = tsm_g = nc_g = None


def process_graph(cfg):
    global x_g, y_g, tm_g, vm_g, tsm_g, nc_g

    ei, x, y, n, nc = make_graph(
        cfg["p_intra"], cfg["p_inter"], cfg["hub_pct"], cfg["extra_per_hub"], seed=SEED)
    h, cv, score = graph_stats(ei, y, n)

    ei  = ei.to(DEVICE); x = x.to(DEVICE); y = y.to(DEVICE)
    g   = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g)
    tm  = torch.zeros(n, dtype=torch.bool)
    vm  = torch.zeros(n, dtype=torch.bool)
    tsm = torch.zeros(n, dtype=torch.bool)
    tm[perm[:int(0.6*n)]]             = True
    vm[perm[int(0.6*n):int(0.8*n)]]  = True
    tsm[perm[int(0.8*n):]]            = True

    # Set globals for train_from_t
    x_g = x; y_g = y; tm_g = tm; vm_g = vm; tsm_g = tsm; nc_g = nc

    # 1. Dense upper bound
    dense_acc = run_dense(ei, x, y, n, nc, tm, vm, tsm)

    # 2. TGS sparse graph
    active_mask, tgs_acc, sparsity = run_tgs_get_sparse(ei, x, y, n, nc, tm, vm, tsm)
    ei_sparse = ei[:, active_mask]
    n_retired  = int((~active_mask).sum().item())

    # 3. Sweep t_retire using oracle topology (TGS final graph)
    tau_curve = {}
    for t in TAU_GRID:
        a = train_from_t(ei, ei_sparse, t_retire=t)
        tau_curve[t] = float(a)

    # 4. Find τ_G: first t where oracle_acc >= dense_acc - DENSE_TOL
    tau_G = None
    for t in TAU_GRID:
        if tau_curve[t] >= dense_acc - DENSE_TOL:
            tau_G = t
            break
    if tau_G is None:
        tau_G = TAU_GRID[-1]  # never fully recovers

    # 5. Linear probe maturation: at which epoch does probe acc plateau?
    # Proxy: run partial training and measure linear probe accuracy
    from sklearn.linear_model import LogisticRegression
    probe_epochs = [0, 10, 20, 30, 40, 60, 80, 120]
    probe_accs = {}
    set_seed(SEED)
    mt_probe = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
    ot_probe  = torch.optim.Adam(mt_probe.parameters(), lr=0.01, weight_decay=5e-4)
    for e in range(EPOCHS + 1):
        if e in probe_epochs:
            mt_probe.eval()
            with torch.no_grad():
                h_rep = x
                for conv in mt_probe.convs[:-1]:
                    h_rep = torch.relu(conv(h_rep, ei))
            h_np = h_rep.detach().cpu().numpy()
            X_tr = h_np[tm.cpu()]; y_tr = y[tm].cpu().numpy()
            X_te = h_np[tsm.cpu()]; y_te = y[tsm].cpu().numpy()
            if len(np.unique(y_tr)) >= 2:
                clf = LogisticRegression(max_iter=300, random_state=42)
                clf.fit(X_tr, y_tr)
                probe_accs[e] = float(clf.score(X_te, y_te))
            else:
                probe_accs[e] = 0.0
        if e < EPOCHS:
            mt_probe.train()
            F.cross_entropy(mt_probe(x, ei)[tm], y[tm]).backward()
            ot_probe.step(); ot_probe.zero_grad()

    return {
        "score":      float(score),
        "homophily":  float(h),
        "deg_cv":     float(cv),
        "dense_acc":  float(dense_acc),
        "tgs_acc":    float(tgs_acc),
        "sparsity":   float(sparsity),
        "n_retired":  n_retired,
        "tau_curve":  {str(t): v for t, v in tau_curve.items()},
        "tau_G":      tau_G,
        "probe_accs": {str(k): v for k, v in probe_accs.items()},
        "delta":      float(tgs_acc - run_dense(ei, x, y, n, nc, tm, vm, tsm)
                           if False else tgs_acc - dense_acc),
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("SCORE → τ_G: Structure Predicts Maturation Speed")
    print("=" * 60)

    results = []
    for i, cfg in enumerate(CONFIGS):
        t1 = time.time()
        print(f"\n[{i+1}/{len(CONFIGS)}] score≈{cfg['score_approx']:.3f}  "
              f"p_in={cfg['p_intra']:.3f} p_out={cfg['p_inter']:.3f}")
        r = process_graph(cfg)
        elapsed = time.time() - t1
        print(f"  score={r['score']:.4f}  dense={r['dense_acc']:.4f}  "
              f"tgs={r['tgs_acc']:.4f}  τ_G={r['tau_G']}  ({elapsed:.0f}s)")
        print(f"  tau_curve: " + "  ".join(
            f"t{t}={v:.3f}" for t, v in sorted(r["tau_curve"].items(), key=lambda x: int(x[0]))
        ))
        results.append(r)

    # Correlation: score vs tau_G
    scores = np.array([r["score"] for r in results])
    taus   = np.array([r["tau_G"] for r in results])
    from scipy import stats
    r_val, p_val = stats.pearsonr(scores, taus)
    r_sp,  p_sp  = stats.spearmanr(scores, taus)

    print(f"\n{'='*60}")
    print(f"RESULTS: SCORE → τ_G")
    print(f"{'='*60}")
    print(f"\n{'score':>7} {'tau_G':>6} {'dense':>7} {'tgs':>7} {'probe@20':>9} {'probe@40':>9}")
    print("─"*55)
    for r in results:
        print(f"  {r['score']:>5.3f} {r['tau_G']:>6}  {r['dense_acc']:>7.4f} {r['tgs_acc']:>7.4f}  "
              f"{r['probe_accs'].get('20', 0):>9.4f}  {r['probe_accs'].get('40', 0):>9.4f}")

    print(f"\nCorrelation (score vs τ_G):")
    print(f"  Pearson r = {r_val:+.3f}  p = {p_val:.4f}")
    print(f"  Spearman ρ = {r_sp:+.3f}  p = {p_sp:.4f}")
    if r_val < -0.5:
        print(f"  → Higher score predicts faster maturation (smaller τ_G)")
        print(f"  → Mechanistic chain CONFIRMED: H×CV → fast maturation → safe early retirement")

    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")

    os.makedirs("results", exist_ok=True)
    out = {
        "results": results,
        "pearson_r": float(r_val), "pearson_p": float(p_val),
        "spearman_r": float(r_sp), "spearman_p": float(p_sp),
        "tau_grid": TAU_GRID, "dense_tol": DENSE_TOL,
    }
    with open("results/score_to_tau.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved → results/score_to_tau.json")
    return out


if __name__ == "__main__":
    main()
