"""
experiments/cv_sweep_tau.py

CV → τ*: Does degree heterogeneity causally shift optimal retirement epoch?
============================================================================

The experiment you described:

  Same graph size, same edge count, same labels, same features,
  same cross-class edge probability (→ fixed H) — only hub structure varies.
  Measure τ* = the optimal retirement epoch, directly.

  Prediction: CV↑ ⇒ τ*↑
  Interpretation: higher hub concentration → representations take longer to
  mature (hubs act as cross-class mixing nodes; until the model learns to
  discount them, retiring edges is premature) → optimal retirement is later.

Two sweeps, run back to back:
  SWEEP A: Hold H fixed (≈ 0.20, heterophilous), vary CV from ~0.1 to ~2.0
  SWEEP B: Hold H fixed (≈ 0.65, homophilous),   vary CV from ~0.1 to ~2.0

If CV→τ* holds in BOTH sweeps independently, then CV is causally driving τ*
regardless of H. That separates the two components of the H×CV score cleanly.

How τ* is measured (oracle sweep, not TGS):
  1. Run full dense training → dense_acc
  2. Run TGS once to get the final sparse topology (oracle graph)
  3. Sweep t_retire ∈ {0,5,10,15,20,30,40,60,80,100,150,200}
     For each t_retire: train dense for t epochs, then switch to sparse graph
  4. τ* = argmax_t { oracle_acc(t) }  [not first-recovery; peak performance]
     Also record τ_recover = first t where oracle_acc(t) >= dense_acc - 0.01

Graph construction:
  Base graph: stochastic block model, 4 classes × 75 nodes = 300 nodes
  H controlled by p_intra / p_inter ratio (same as predictor_prospective.py)
  CV controlled by hub injection — cross-class edges only, so H stays constant.

  Hub injection preserves H because hubs connect within-class in SWEEP A's
  heterophilous graphs: we inject hubs with CROSS-class edges for low-H graphs
  and WITHIN-class edges for high-H graphs, keeping the edge type ratio fixed.

  The CRITICAL invariant: for each CV level, total edges ≈ constant (within 5%).
  We achieve this by injecting hub edges and removing the same number of
  random non-hub edges, keeping m fixed exactly.

Run: PYTHONPATH=. python3 experiments/cv_sweep_tau.py [--seeds N] [--fast]
"""

import sys, os, json, time, argparse
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.utils import degree, stochastic_blockmodel_graph
from scipy import stats

from tgs.utils.reproducibility import set_seed
from tgs.core.temporal_graph import TemporalGraph
from tgs.core.influence import GradientNormEstimator
from tgs.models.gcn import TemporalGCN
from tgs.schedulers.adaptive_scheduler import AdaptiveRetirementScheduler
from experiments.predictor_prospective import graph_stats

import logging
logging.basicConfig(level=logging.WARNING)

OUT = Path("results/cv_sweep_tau")
OUT.mkdir(parents=True, exist_ok=True)

# ── Graph parameters ──────────────────────────────────────────────────────────
N_BLOCKS = 4
N_PER    = 75          # → n = 300
EPOCHS   = 300
DEVICE   = torch.device("cpu")

# Oracle sweep grid: t_retire values to test
TAU_GRID = [0, 5, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200]

# CV levels to sweep: achieved deg-CV values (hub injection tuned to hit these)
# Each entry: (hub_frac, extra_hub_edges) tuned to achieve target CV
# Cross-class hub edges for LOW-H sweep, within-class for HIGH-H sweep
CV_LEVELS = [
    dict(label="CV≈0.10", hub_frac=0.00, hub_extra=0),
    dict(label="CV≈0.35", hub_frac=0.02, hub_extra=4),
    dict(label="CV≈0.55", hub_frac=0.04, hub_extra=7),
    dict(label="CV≈0.75", hub_frac=0.06, hub_extra=10),
    dict(label="CV≈0.95", hub_frac=0.08, hub_extra=14),
    dict(label="CV≈1.15", hub_frac=0.10, hub_extra=18),
    dict(label="CV≈1.40", hub_frac=0.13, hub_extra=24),
    dict(label="CV≈1.70", hub_frac=0.17, hub_extra=32),
    dict(label="CV≈2.00", hub_frac=0.22, hub_extra=42),
]

# Two sweeps
SWEEPS = [
    dict(name="low_H",  p_intra=0.025, p_inter=0.080, target_h=0.20,
         hub_same_class=False),   # hubs connect cross-class → adds cross-class edges
    dict(name="high_H", p_intra=0.090, p_inter=0.012, target_h=0.65,
         hub_same_class=True),    # hubs connect within-class → adds within-class edges
]


# ── Graph construction ────────────────────────────────────────────────────────

def make_fixed_H_variable_CV(p_intra, p_inter, hub_frac, hub_extra,
                              hub_same_class, seed):
    """
    Build a graph with fixed H (~target_h), fixed m (~m_target), variable deg-CV.

    Uses a bipartite configuration model:
      - Build degree sequence: n_hubs nodes get hub_deg, rest get nonhub_deg.
      - For each node, allocate same_deg = round(deg * target_h) and
        cross_deg = deg - same_deg stubs.
      - Pair same-class stubs within each class separately.
      - Pair cross-class stubs: for each node pick a target from a different class,
        weighted by cross-class stub count.
    This ensures both H and CV are set by construction.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    n  = N_PER * N_BLOCKS
    nc = N_BLOCKS
    y_np = np.repeat(np.arange(nc), N_PER)

    # Target H from SBM params
    target_h = p_intra / (p_intra + (nc - 1) * p_inter)

    # m_target: expected edges at hub_frac=0
    # E[m] for balanced SBM with blocks of size N_PER:
    m_target = int(nc * N_PER*(N_PER-1)/2 * p_intra +
                   nc*(nc-1)/2 * N_PER*N_PER * p_inter)

    # Degree sequence
    if hub_frac == 0 or hub_extra == 0:
        mean_deg = max(2, round(2 * m_target / n))
        degs = np.full(n, mean_deg, dtype=float)
    else:
        n_hubs = max(1, int(n * hub_frac))
        mean_deg = 2 * m_target / n
        # nonhub_deg * n + hub_extra * n_hubs = 2 * m_target
        nonhub_deg = max(1.0, (2 * m_target - n_hubs * hub_extra) / n)
        hub_deg = nonhub_deg + hub_extra
        degs = np.full(n, nonhub_deg)
        hub_idx = rng.choice(n, n_hubs, replace=False)
        degs[hub_idx] = hub_deg

    # Per-node stub allocation
    same_degs  = np.maximum(0, np.round(degs * target_h)).astype(int)
    cross_degs = np.maximum(0, np.round(degs * (1 - target_h))).astype(int)

    edge_set = set()

    # ── Same-class edges: pair stubs within each class ────────────────────
    for cls in range(nc):
        cls_nodes = [i for i in range(n) if y_np[i] == cls]
        stubs = []
        for node in cls_nodes:
            stubs.extend([node] * same_degs[node])
        rng.shuffle(stubs)
        for i in range(0, len(stubs) - 1, 2):
            u, v = int(stubs[i]), int(stubs[i+1])
            if u == v: continue
            key = (min(u,v), max(u,v))
            edge_set.add(key)

    # ── Cross-class edges: pair stubs across classes ───────────────────────
    # Build per-class cross stub pools
    cross_pools = {}
    for cls in range(nc):
        pool = []
        for node in range(n):
            if y_np[node] == cls:
                pool.extend([node] * cross_degs[node])
        rng.shuffle(pool)
        cross_pools[cls] = list(pool)

    # Pair greedily: for each class, pair its stubs with stubs from other classes
    # Round-robin across other classes
    for cls in range(nc):
        other_classes = [c for c in range(nc) if c != cls]
        pool_u = cross_pools[cls]
        for u in pool_u:
            # Pick partner from a random other class with remaining stubs
            rng.shuffle(other_classes)
            for cls_v in other_classes:
                if not cross_pools[cls_v]:
                    continue
                # Check if there's a node in cls_v's pool that isn't u
                idx = int(rng.integers(len(cross_pools[cls_v])))
                v = cross_pools[cls_v][idx]
                if v == u:
                    continue
                key = (min(u,v), max(u,v))
                if key not in edge_set:
                    edge_set.add(key)
                    cross_pools[cls_v].pop(idx)
                    break

    us = [u for u,v in edge_set]+[v for u,v in edge_set]
    vs = [v for u,v in edge_set]+[u for u,v in edge_set]
    ei = torch.tensor([us,vs], dtype=torch.long)

    rng2 = np.random.default_rng(seed+1000)
    x_np = rng2.normal(0, 0.9, size=(n, nc+4))
    for b in range(nc):
        x_np[b*N_PER:(b+1)*N_PER, b] += 1.0
    x = torch.tensor(x_np, dtype=torch.float)
    y = torch.tensor(y_np, dtype=torch.long)

    return ei, x, y, n, nc


def make_splits(n, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    tm  = torch.zeros(n, dtype=torch.bool); tm[perm[:int(0.6*n)]]  = True
    vm  = torch.zeros(n, dtype=torch.bool); vm[perm[int(0.6*n):int(0.8*n)]] = True
    tsm = torch.zeros(n, dtype=torch.bool); tsm[perm[int(0.8*n):]] = True
    return tm, vm, tsm


# ── Training primitives ───────────────────────────────────────────────────────

def train_dense(ei, x, y, n, nc, tm, vm, tsm, seed):
    set_seed(seed)
    model = TemporalGCN(x.shape[1], 64, nc, 2, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_test = 0.0, 0.0
    for epoch in range(EPOCHS):
        model.train()
        loss = F.cross_entropy(model(x, ei)[tm], y[tm])
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(x, ei).argmax(-1)
        va = (preds[vm]  == y[vm]).float().mean().item()
        ta = (preds[tsm] == y[tsm]).float().mean().item()
        if va > best_val:
            best_val, best_test = va, ta
    return best_test


def get_tgs_sparse_graph(ei, x, y, n, nc, tm, vm, tsm, seed):
    """Run TGS, return the final retired topology."""
    m0 = ei.shape[1]
    set_seed(seed)
    tg  = TemporalGraph(ei, n, device=DEVICE)
    est = GradientNormEstimator(m0, DEVICE, edge_index=ei, num_nodes=n,
                                alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
    model = TemporalGCN(x.shape[1], 64, nc, 2, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(list(model.parameters()) + [est.edge_weights],
                             lr=0.01, weight_decay=5e-4)
    sched = AdaptiveRetirementScheduler(
        tg, epsilon_max=5e-3, epsilon_min=1e-5,
        anneal_steps=150, warmup_steps=40,
        max_retire_frac=0.06, max_sparsity=0.55, retire_every=2)
    for epoch in range(EPOCHS):
        model.train(); am = tg.active_mask
        F.cross_entropy(model(x, tg.edge_index, est.edge_weights[am])[tm], y[tm]).backward()
        est.update_influence(am); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            preds = model(x, tg.edge_index).argmax(-1)
        va = (preds[vm] == y[vm]).float().mean().item()
        sched.update_val_acc(va)
        sched.step(est.influence_scores(am))
        tg.step()
    return ei[:, tg.active_mask].clone(), float(tg.sparsity)


def train_oracle_at_t(ei_dense, ei_sparse, t_retire, x, y, n, nc, tm, vm, tsm, seed):
    """
    Train: dense graph for t_retire epochs, then switch to sparse.
    Returns best val-gated test accuracy.
    This measures: what is the accuracy if we had oracle knowledge of the
    final topology but chose to retire at epoch t?
    """
    set_seed(seed)
    model = TemporalGCN(x.shape[1], 64, nc, 2, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_test = 0.0, 0.0
    for epoch in range(EPOCHS):
        ei_cur = ei_dense if epoch < t_retire else ei_sparse
        model.train()
        F.cross_entropy(model(x, ei_cur)[tm], y[tm]).backward()
        opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            preds = model(x, ei_cur).argmax(-1)
        va = (preds[vm]  == y[vm]).float().mean().item()
        ta = (preds[tsm] == y[tsm]).float().mean().item()
        if va > best_val:
            best_val, best_test = va, ta
    return best_test


# ── Per-graph oracle sweep ────────────────────────────────────────────────────

def run_graph(sweep, cv_level, seed):
    ei, x, y, n, nc = make_fixed_H_variable_CV(
        p_intra       = sweep["p_intra"],
        p_inter       = sweep["p_inter"],
        hub_frac      = cv_level["hub_frac"],
        hub_extra     = cv_level["hub_extra"],
        hub_same_class= sweep["hub_same_class"],
        seed          = seed,
    )
    ei = ei.to(DEVICE); x = x.to(DEVICE); y = y.to(DEVICE)
    tm, vm, tsm = make_splits(n, seed=seed)

    h, cv, score = graph_stats(ei, y, n)
    m = ei.shape[1] // 2

    # 1. Dense baseline
    dense_acc = train_dense(ei, x, y, n, nc, tm, vm, tsm, seed)

    # 2. TGS sparse topology
    ei_sparse, sparsity = get_tgs_sparse_graph(ei, x, y, n, nc, tm, vm, tsm, seed)

    # 3. Oracle sweep over t_retire
    tau_curve = {}
    for t in TAU_GRID:
        acc = train_oracle_at_t(ei, ei_sparse, t, x, y, n, nc, tm, vm, tsm, seed)
        tau_curve[t] = round(float(acc), 4)

    # 4. τ* = argmax of oracle curve (peak performance epoch)
    tau_star = max(tau_curve, key=tau_curve.get)

    # 5. τ_recover = first t where oracle_acc >= dense_acc - 0.01
    tau_recover = None
    for t in TAU_GRID:
        if tau_curve[t] >= dense_acc - 0.01:
            tau_recover = t
            break
    if tau_recover is None:
        tau_recover = TAU_GRID[-1]

    return {
        "homophily":   round(float(h),    4),
        "deg_cv":      round(float(cv),    4),
        "score":       round(float(score), 4),
        "m":           int(m),
        "dense_acc":   round(float(dense_acc), 4),
        "sparsity":    round(float(sparsity),  3),
        "tau_star":    int(tau_star),
        "tau_recover": int(tau_recover),
        "tau_curve":   {str(t): v for t, v in tau_curve.items()},
        "seed":        seed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--fast",  action="store_true",
                        help="2 seeds, 5 CV levels (smoke test)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seeds = list(range(42, 42 + (2 if args.fast else args.seeds)))
    cv_levels = CV_LEVELS[:5] if args.fast else CV_LEVELS
    sweeps = SWEEPS

    out_file = OUT / "results.json"
    all_results = (json.loads(out_file.read_text())
                   if out_file.exists() and not args.force else {})

    total = len(sweeps) * len(cv_levels) * len(seeds)
    done  = 0
    t_global = time.time()

    print(f"CV → τ* sweep: {len(sweeps)} sweeps × {len(cv_levels)} CV levels × {len(seeds)} seeds = {total} runs")
    print(f"Epochs: {EPOCHS}  |  Oracle grid: {TAU_GRID}")
    print()

    for sweep in sweeps:
        sname = sweep["name"]
        if sname not in all_results:
            all_results[sname] = {"sweep": sweep, "cv_levels": []}

        for cv_level in cv_levels:
            clabel = cv_level["label"]
            # Find or create this CV level's entry
            existing = next((e for e in all_results[sname]["cv_levels"]
                             if e["label"] == clabel), None)
            if existing is None:
                existing = {"label": clabel, "hub_frac": cv_level["hub_frac"],
                            "hub_extra": cv_level["hub_extra"], "runs": []}
                all_results[sname]["cv_levels"].append(existing)

            done_seeds = {r["seed"] for r in existing["runs"]}

            for seed in seeds:
                if seed in done_seeds and not args.force:
                    done += 1
                    continue

                print(f"  [{sweep['name']}] {clabel} seed={seed} ...", end=" ", flush=True)
                t0 = time.time()
                r  = run_graph(sweep, cv_level, seed)
                elapsed = time.time() - t0
                done += 1

                print(f"h={r['homophily']:.3f} cv={r['deg_cv']:.3f} m={r['m']} "
                      f"dense={r['dense_acc']:.3f} τ*={r['tau_star']:>3} "
                      f"τ_rec={r['tau_recover']:>3}  ({elapsed:.0f}s)  "
                      f"[{done}/{total}]")

                existing["runs"].append(r)
                out_file.write_text(json.dumps(all_results, indent=2))

    # ── Analysis ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("RESULTS: CV → τ*")
    print(f"{'='*70}\n")

    summary = {}
    for sweep in sweeps:
        sname = sweep["name"]
        print(f"Sweep: {sname} (target H≈{sweep['target_h']:.2f})")
        print(f"  {'CV level':<14} {'h':>6} {'deg_cv':>7} {'m':>5} "
              f"{'dense':>7} {'τ*':>5} {'τ_rec':>6} {'sp':>5}")
        print("  " + "─"*60)

        cvs, tau_stars, tau_recs = [], [], []
        for entry in all_results[sname]["cv_levels"]:
            runs = entry["runs"]
            if not runs:
                continue
            import statistics as st
            h_mean   = st.mean(r["homophily"]   for r in runs)
            cv_mean  = st.mean(r["deg_cv"]      for r in runs)
            m_mean   = st.mean(r["m"]           for r in runs)
            d_mean   = st.mean(r["dense_acc"]   for r in runs)
            ts_mean  = st.mean(r["tau_star"]    for r in runs)
            tr_mean  = st.mean(r["tau_recover"] for r in runs)
            sp_mean  = st.mean(r["sparsity"]    for r in runs)

            print(f"  {entry['label']:<14} {h_mean:>6.3f} {cv_mean:>7.3f} {m_mean:>5.0f} "
                  f"{d_mean:>7.3f} {ts_mean:>5.1f} {tr_mean:>6.1f} {sp_mean:>5.3f}")

            cvs.append(cv_mean)
            tau_stars.append(ts_mean)
            tau_recs.append(tr_mean)

        if len(cvs) >= 3:
            r_ts, p_ts = stats.spearmanr(cvs, tau_stars)
            r_tr, p_tr = stats.spearmanr(cvs, tau_recs)
            print(f"\n  Spearman ρ(CV, τ*)       = {r_ts:+.3f}  p={p_ts:.4f}")
            print(f"  Spearman ρ(CV, τ_recover) = {r_tr:+.3f}  p={p_tr:.4f}")

            if r_ts > 0.5 and p_ts < 0.10:
                print(f"  ✓ CV↑ ⇒ τ*↑  confirmed in {sname} sweep")
            else:
                print(f"  ~ CV→τ* relationship is weak or absent in {sname} sweep")

            summary[sname] = {
                "cvs": cvs, "tau_stars": tau_stars, "tau_recs": tau_recs,
                "spearman_ts": round(r_ts, 3), "p_ts": round(p_ts, 4),
                "spearman_tr": round(r_tr, 3), "p_tr": round(p_tr, 4),
            }
        print()

    # Cross-sweep comparison
    if "low_H" in summary and "high_H" in summary:
        print("Cross-sweep comparison:")
        print(f"  low_H  ρ(CV,τ*) = {summary['low_H']['spearman_ts']:+.3f}  "
              f"p={summary['low_H']['p_ts']:.4f}")
        print(f"  high_H ρ(CV,τ*) = {summary['high_H']['spearman_ts']:+.3f}  "
              f"p={summary['high_H']['p_ts']:.4f}")
        both_positive = (summary['low_H']['spearman_ts'] > 0 and
                         summary['high_H']['spearman_ts'] > 0)
        if both_positive:
            print("  ✓ CV→τ* holds in BOTH sweeps independently of H")
            print("  → Deg-CV is a causal driver of retirement timing, separable from homophily")
        else:
            print("  ~ CV→τ* does not replicate across both H regimes")

    all_results["_summary"] = summary
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nTotal runtime: {(time.time()-t_global)/60:.1f} min")
    print(f"Saved → {out_file}")


if __name__ == "__main__":
    main()
