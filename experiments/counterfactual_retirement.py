"""
experiments/counterfactual_retirement.py

Counterfactual Edge-Retirement Experiment
==========================================

Tests the proposed causal mechanism:
    representation maturation → edge redundancy → safe retirement

Five components:

A. Representation Maturity
   Linear probe accuracy at each checkpoint t ∈ {0,10,20,...,200}
   Shows when representations stabilise.

B. Retirement Damage vs Maturity
   At checkpoint t, retire a fixed set of "redundant" edges (identified
   by TGS influence scores at convergence), continue training, measure
   Δacc = acc_keep − acc_retire.
   Expected: Δacc falls as t increases → same edges become safe later.

C. Edge Redundancy Distribution
   R_e(t) = acc_with_e − acc_without_e, measured at checkpoints.
   Expected: distribution shifts toward 0 as representations mature.

D. Temporal Swap (the killer condition)
   Take the EXACT edges TGS retires and retire them all at different epochs:
   t_retire ∈ {0, 20, 40, 60, 80, convergence}.
   Same final edge set. Only retirement time changes.
   Expected: accuracy rises monotonically with t_retire.
   This proves: "edge redundancy is not a property of the graph,
   it is a property of the graph–representation pair at time t."

E. Static-Final-Oracle Baseline
   1. Train dense model to convergence → identify final TGS sparse graph.
   2. Rewind to epoch 0. Train from scratch with that exact sparse graph.
   Compare: Dense, Static-ER, Static-Final-Oracle, TGS.
   If TGS > Oracle → the MODEL NEEDS THE DENSE GRAPH EARLY, even knowing
   the perfect final topology. The trajectory is the explanation.

Run on one high-score graph (score≈0.31) — deep single-graph analysis.
All components checkpointed.
"""

import sys, os, json, copy, time
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

DEVICE = torch.device("cpu")
EPOCHS = 200
SEED   = 42

# Graph: high-score, proven TGS advantage
CFG = dict(p_intra=0.045, p_inter=0.011, hub_pct=0.04, extra_per_hub=25)  # score≈0.24, delta≈+14pp

CHECKPOINT_EPOCHS = list(range(0, EPOCHS + 1, 10))   # 0,10,...,200
SWAP_RETIRE_EPOCHS = [0, 20, 40, 60, 80, 120, 160, 200]
REDUNDANCY_SAMPLE  = 30    # edges sampled for Panel C
COUNTERFACTUAL_N   = 50    # edges retired in Panel B


# ─── Graph setup ────────────────────────────────────────────────────────────

def setup_graph():
    from experiments.predictor_prospective import make_graph, graph_stats
    ei, x, y, n, nc = make_graph(**CFG, seed=SEED)
    h, cv, score = graph_stats(ei, y, n)
    print(f"Graph: n={n} m0={ei.shape[1]} nc={nc} H={h:.3f} CV={cv:.3f} score={score:.4f}")

    ei = ei.to(DEVICE); x = x.to(DEVICE); y = y.to(DEVICE)
    g    = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g)
    tm  = torch.zeros(n, dtype=torch.bool)
    vm  = torch.zeros(n, dtype=torch.bool)
    tsm = torch.zeros(n, dtype=torch.bool)
    tm[perm[:int(0.6*n)]]             = True
    vm[perm[int(0.6*n):int(0.8*n)]]  = True
    tsm[perm[int(0.8*n):]]            = True

    return ei, x, y, n, nc, tm, vm, tsm, score


def acc(model, ei_, x_, y_, mask):
    model.eval()
    with torch.no_grad():
        return float((model(x_, ei_).argmax(-1)[mask] == y_[mask]).float().mean())


# ─── Linear probe ─────────────────────────────────────────────────────────

def linear_probe_acc(model, x, y, tm, tsm, ei):
    """Train a linear classifier on frozen representations."""
    model.eval()
    with torch.no_grad():
        # Get penultimate layer representations (before last linear)
        # We use the model's forward but stop before final layer
        h = x
        for i, conv in enumerate(model.convs[:-1]):
            h = conv(h, ei)
            h = F.relu(h)
            h = F.dropout(h, p=model.dropout, training=False)
        # h is now [n, hidden_channels] — the representation
    h = h.detach()

    # Fit logistic regression on train set, eval on test
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500, random_state=42)
    X_train = h[tm].cpu().numpy()
    y_train = y[tm].cpu().numpy()
    X_test  = h[tsm].cpu().numpy()
    y_test  = y[tsm].cpu().numpy()
    if len(np.unique(y_train)) < 2:
        return 0.0
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


# ─── Component A+B+C: Dense TGS run with checkpointing ───────────────────

def run_dense_with_checkpoints(ei, x, y, n, nc, tm, vm, tsm):
    """
    Run full TGS training. At each checkpoint:
    - Record model state_dict
    - Record test accuracy
    - Record linear probe accuracy (representation maturity)
    - Record influence scores (edge importance at this epoch)
    """
    m0 = ei.shape[1]
    set_seed(SEED)
    tg  = TemporalGraph(ei, n, device=DEVICE)
    est = GradientNormEstimator(m0, DEVICE, edge_index=ei, num_nodes=n,
                                alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
    mt  = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
    ot  = torch.optim.Adam(list(mt.parameters()) + [est.edge_weights],
                           lr=0.01, weight_decay=5e-4)
    sc  = AdaptiveRetirementScheduler(tg, epsilon_max=5e-3, epsilon_min=1e-5,
              anneal_steps=100, warmup_steps=40, max_retire_frac=0.08,
              max_sparsity=0.35, retire_every=2)

    checkpoints = {}   # epoch → {state_dict, test_acc, probe_acc, influence, active_mask, retired_set}

    print("  Running dense TGS with checkpoints...")
    for e in range(EPOCHS + 1):
        # Checkpoint BEFORE training step
        if e in CHECKPOINT_EPOCHS:
            probe = linear_probe_acc(mt, x, y, tm, tsm, tg.edge_index)
            ta    = acc(mt, tg.edge_index, x, y, tsm)
            infl  = est.influence_scores(tg.active_mask).detach().cpu().numpy().tolist()
            checkpoints[e] = {
                "epoch":      e,
                "test_acc":   ta,
                "probe_acc":  probe,
                "sparsity":   tg.sparsity,
                "active_mask": tg.active_mask.cpu().numpy().tolist(),
                "influence_scores": infl,
                "state_dict": copy.deepcopy(mt.state_dict()),
            }
            print(f"    epoch={e:3d}  acc={ta:.4f}  probe={probe:.4f}  sp={tg.sparsity:.3f}")

        if e == EPOCHS:
            break

        # Training step
        mt.train(); am = tg.active_mask
        F.cross_entropy(mt(x, tg.edge_index, est.edge_weights[am])[tm], y[tm]).backward()
        est.update_influence(am); ot.step(); ot.zero_grad()
        sc.update_val_acc(acc(mt, tg.edge_index, x, y, vm))
        sc.step(est.influence_scores(am)); tg.step()

    # Final retired set (the exact edges TGS chose to remove)
    retired_mask = ~tg.active_mask
    retired_indices = retired_mask.nonzero(as_tuple=True)[0].tolist()

    return checkpoints, retired_indices, tg.sparsity


# ─── Component B: Retirement damage as function of maturity ──────────────

def measure_retirement_damage(checkpoints, retired_indices, ei, x, y, n, nc, tm, vm, tsm):
    """
    At each checkpoint t, take the model state, retire `retired_indices`,
    continue training for remaining epochs, record final accuracy.
    Δacc = acc_keep(t) − acc_retire(t)
    """
    m0 = ei.shape[1]
    # The "keep" baseline: full TGS final acc (from checkpoint at epoch 200)
    keep_acc = checkpoints[EPOCHS]["test_acc"]

    # Sparse graph: the TGS final edge set
    keep_mask = torch.ones(m0, dtype=torch.bool, device=DEVICE)
    if retired_indices:
        keep_mask[torch.tensor(retired_indices, dtype=torch.long)] = False
    ei_sparse = ei[:, keep_mask]

    damage_results = {}
    print("  Measuring retirement damage at each checkpoint...")

    for ck_epoch in CHECKPOINT_EPOCHS[::2]:   # every 20 epochs for speed
        remaining = EPOCHS - ck_epoch
        if remaining == 0:
            damage_results[ck_epoch] = {"delta": 0.0, "retire_acc": keep_acc, "keep_acc": keep_acc}
            continue

        # Branch: load snapshot, retire edges, continue training
        set_seed(SEED + ck_epoch)
        mt_branch = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
        mt_branch.load_state_dict(checkpoints[ck_epoch]["state_dict"])
        ot_branch = torch.optim.Adam(mt_branch.parameters(), lr=0.01, weight_decay=5e-4)

        bvv = bvt = 0.0
        for e in range(remaining):
            mt_branch.train()
            F.cross_entropy(mt_branch(x, ei_sparse)[tm], y[tm]).backward()
            ot_branch.step(); ot_branch.zero_grad()
            va = acc(mt_branch, ei_sparse, x, y, vm)
            ta = acc(mt_branch, ei_sparse, x, y, tsm)
            if va > bvv: bvv, bvt = va, ta

        delta = keep_acc - bvt
        damage_results[ck_epoch] = {
            "retire_acc": bvt,
            "keep_acc":   keep_acc,
            "delta":      delta,
            "ck_epoch":   ck_epoch,
        }
        print(f"    checkpoint={ck_epoch:3d}  retire_acc={bvt:.4f}  keep_acc={keep_acc:.4f}  Δ={delta:+.4f}")

    return damage_results


# ─── Component D: Temporal swap ───────────────────────────────────────────

def temporal_swap(retired_indices, ei, x, y, n, nc, tm, vm, tsm):
    """
    Same final edge set (TGS retired edges).
    Retire them ALL at epoch t_retire (instead of gradually).
    Vary t_retire ∈ SWAP_RETIRE_EPOCHS.
    Same total training budget (EPOCHS).
    """
    m0 = ei.shape[1]
    keep_mask = torch.ones(m0, dtype=torch.bool, device=DEVICE)
    if retired_indices:
        keep_mask[torch.tensor(retired_indices, dtype=torch.long)] = False
    ei_sparse = ei[:, keep_mask]

    swap_results = {}
    print("  Running temporal swap experiment...")

    for t_retire in SWAP_RETIRE_EPOCHS:
        set_seed(SEED)
        mt = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
        ot = torch.optim.Adam(mt.parameters(), lr=0.01, weight_decay=5e-4)
        bvv = bvt = 0.0

        for e in range(EPOCHS):
            ei_cur = ei if e < t_retire else ei_sparse
            mt.train()
            F.cross_entropy(mt(x, ei_cur)[tm], y[tm]).backward()
            ot.step(); ot.zero_grad()
            va = acc(mt, ei_cur, x, y, vm)
            ta = acc(mt, ei_cur, x, y, tsm)
            if va > bvv: bvv, bvt = va, ta

        swap_results[t_retire] = {"t_retire": t_retire, "final_acc": bvt}
        print(f"    t_retire={t_retire:3d}  final_acc={bvt:.4f}")

    return swap_results


# ─── Component E: Static-Final-Oracle ────────────────────────────────────

def static_final_oracle(retired_indices, ei, x, y, n, nc, tm, vm, tsm, tgs_acc):
    """
    Give static pruning the perfect final sparse graph (derived from TGS),
    then train from epoch 0 on that graph.
    Also run: dense, static ER (matched sparsity), TGS (already have).
    """
    m0 = ei.shape[1]
    n_retired = len(retired_indices)
    sparsity   = n_retired / m0

    # TGS final sparse graph (oracle knows this at time 0)
    keep_mask = torch.ones(m0, dtype=torch.bool, device=DEVICE)
    if retired_indices:
        keep_mask[torch.tensor(retired_indices, dtype=torch.long)] = False
    ei_oracle = ei[:, keep_mask]

    # Static ER baseline at matched sparsity
    src = ei[0].numpy(); dg = degree(ei[1], n, dtype=torch.float).numpy()
    er  = 1.0/dg[src].clip(1) + 1.0/dg[ei[1].numpy()].clip(1)
    _, sidx = torch.from_numpy(er).float().sort()
    rm_er = set(sidx[:n_retired].tolist())
    ei_er = ei[:, torch.tensor([i not in rm_er for i in range(m0)], dtype=torch.bool)]

    def train_static(ei_s, label):
        set_seed(SEED)
        ms  = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
        os_ = torch.optim.Adam(ms.parameters(), lr=0.01, weight_decay=5e-4)
        bvv = bvt = 0.0
        for e in range(EPOCHS):
            ms.train()
            F.cross_entropy(ms(x, ei_s)[tm], y[tm]).backward()
            os_.step(); os_.zero_grad()
            va = acc(ms, ei_s, x, y, vm)
            ta = acc(ms, ei_s, x, y, tsm)
            if va > bvv: bvv, bvt = va, ta
        print(f"    {label:25s}  final_acc={bvt:.4f}")
        return bvt

    print("  Running Static-Final-Oracle comparison...")

    # Dense baseline
    dense_acc = train_static(ei, "Dense (no pruning)")
    er_acc    = train_static(ei_er,     "Static ER (matched sp.)")
    oracle_acc= train_static(ei_oracle, "Static-Final-Oracle")

    print(f"    {'TGS':25s}  final_acc={tgs_acc:.4f}  ← learns the trajectory")
    print(f"\n    KEY RESULT: TGS > Oracle? {tgs_acc > oracle_acc}")
    print(f"    If yes: the model needs dense edges during early learning,")
    print(f"    even given the perfect final topology.")

    return {
        "dense_acc":   dense_acc,
        "static_er":   er_acc,
        "oracle_acc":  oracle_acc,
        "tgs_acc":     tgs_acc,
        "sparsity":    sparsity,
        "tgs_beats_oracle": tgs_acc > oracle_acc,
    }


# ─── Component C: Edge redundancy distribution ───────────────────────────

def edge_redundancy_distribution(checkpoints, ei, x, y, n, nc, tm, tsm, sample_indices):
    """
    At each checkpoint, compute R_e(t) = acc_with_e - acc_without_e
    for a sample of edges. Shows the distribution shifting toward 0.
    """
    print("  Computing edge redundancy distribution...")
    redundancy = {}
    m0 = ei.shape[1]

    for ck_epoch in [0, 20, 40, 80, 120, 160, 200]:
        if ck_epoch not in checkpoints:
            continue
        mt = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
        mt.load_state_dict(checkpoints[ck_epoch]["state_dict"])
        mt.eval()

        # Active mask at this checkpoint
        am = torch.tensor(checkpoints[ck_epoch]["active_mask"], dtype=torch.bool, device=DEVICE)
        ei_t = ei[:, am]
        base_acc = acc(mt, ei_t, x, y, tsm)

        R_vals = []
        # Use only edges active at this checkpoint
        active_sample = [i for i in sample_indices if am[i]][:REDUNDANCY_SAMPLE]
        for idx in active_sample:
            # Remove this one edge
            mask = am.clone(); mask[idx] = False
            ei_minus = ei[:, mask]
            a_minus   = acc(mt, ei_minus, x, y, tsm)
            R_vals.append(base_acc - a_minus)

        redundancy[ck_epoch] = {
            "epoch":   ck_epoch,
            "R_mean":  float(np.mean(R_vals)) if R_vals else 0.0,
            "R_std":   float(np.std(R_vals))  if R_vals else 0.0,
            "R_pct_near_zero": float(np.mean(np.abs(R_vals) < 0.01)) if R_vals else 0.0,
            "R_values": R_vals,
        }
        print(f"    epoch={ck_epoch:3d}  R_mean={np.mean(R_vals):+.4f}  "
              f"R_std={np.std(R_vals):.4f}  "
              f"near_zero={np.mean(np.abs(R_vals)<0.01):.1%}")

    return redundancy


# ─── Main ────────────────────────────────────────────────────────────────────

CHECKPOINT_FILE = "results/counterfactual_checkpoint.json"


def main():
    t0 = time.time()
    print("=" * 65)
    print("Counterfactual Edge-Retirement Experiment")
    print("=" * 65)

    ei, x, y, n, nc, tm, vm, tsm, score = setup_graph()
    m0 = ei.shape[1]

    # Sample indices placeholder — will be populated after TGS run
    # (we sample from the RETIRED edge set, which is most meaningful)
    sample_indices = list(range(min(REDUNDANCY_SAMPLE * 3, m0)))

    # ── A+B+C prep: Dense TGS run ─────────────────────────────────────
    print("\n[Phase 1] Dense TGS run with checkpoints")
    checkpoints, retired_indices, final_sparsity = run_dense_with_checkpoints(
        ei, x, y, n, nc, tm, vm, tsm)

    tgs_acc   = checkpoints[EPOCHS]["test_acc"]
    tgs_probe = checkpoints[EPOCHS]["probe_acc"]
    print(f"\n  TGS final: acc={tgs_acc:.4f}  sparsity={final_sparsity:.3f}")
    print(f"  Edges retired: {len(retired_indices)}/{m0}")

    # Sample from RETIRED edges for redundancy analysis
    rng_s = np.random.default_rng(42)
    if len(retired_indices) >= REDUNDANCY_SAMPLE:
        sample_indices = rng_s.choice(retired_indices, REDUNDANCY_SAMPLE, replace=False).tolist()
    else:
        sample_indices = retired_indices
    print(f"  Sampling {len(sample_indices)} retired edges for redundancy analysis")

    # ── B: Retirement damage ─────────────────────────────────────────
    print("\n[Phase 2] Counterfactual retirement damage")
    damage_results = measure_retirement_damage(
        checkpoints, retired_indices, ei, x, y, n, nc, tm, vm, tsm)

    # ── C: Redundancy distribution ───────────────────────────────────
    print("\n[Phase 3] Edge redundancy distribution")
    redundancy = edge_redundancy_distribution(
        checkpoints, ei, x, y, n, nc, tm, tsm, sample_indices)

    # ── D: Temporal swap ─────────────────────────────────────────────
    print("\n[Phase 4] Temporal swap (same edges, different timing)")
    swap_results = temporal_swap(retired_indices, ei, x, y, n, nc, tm, vm, tsm)

    # ── E: Static-Final-Oracle ───────────────────────────────────────
    print("\n[Phase 5] Static-Final-Oracle baseline")
    oracle_results = static_final_oracle(
        retired_indices, ei, x, y, n, nc, tm, vm, tsm, tgs_acc)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)

    print("\nA. Representation Maturity (linear probe accuracy):")
    for e in [0, 20, 40, 80, 120, 160, 200]:
        if e in checkpoints:
            print(f"   epoch={e:3d}  probe={checkpoints[e]['probe_acc']:.4f}  "
                  f"test={checkpoints[e]['test_acc']:.4f}")

    print("\nB. Retirement Damage falls with maturity:")
    for e in sorted(damage_results):
        d = damage_results[e]
        print(f"   checkpoint={e:3d}  Δ={d['delta']:+.4f}  retire_acc={d['retire_acc']:.4f}")

    print("\nC. Edge redundancy — fraction near-zero R_e(t):")
    for e, r in sorted(redundancy.items()):
        print(f"   epoch={e:3d}  near_zero={r['R_pct_near_zero']:.1%}  R_mean={r['R_mean']:+.4f}")

    print("\nD. Temporal swap — same edges, different retirement epoch:")
    for t_r, v in sorted(swap_results.items()):
        print(f"   t_retire={t_r:3d}  final_acc={v['final_acc']:.4f}")

    print("\nE. Static-Final-Oracle:")
    print(f"   Dense:                 {oracle_results['dense_acc']:.4f}")
    print(f"   Static ER:             {oracle_results['static_er']:.4f}")
    print(f"   Static-Final-Oracle:   {oracle_results['oracle_acc']:.4f}  ← knows perfect topology")
    print(f"   TGS:                   {oracle_results['tgs_acc']:.4f}  ← temporal trajectory")
    print(f"   TGS > Oracle:          {oracle_results['tgs_beats_oracle']}")
    if oracle_results["tgs_beats_oracle"]:
        gap = (oracle_results["tgs_acc"] - oracle_results["oracle_acc"]) * 100
        print(f"   → The model needs dense edges during early learning (+{gap:.2f}pp)")
        print(f"   → The trajectory, not the final topology, is the explanation.")

    runtime = (time.time() - t0) / 60
    print(f"\nTotal runtime: {runtime:.1f} min")

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    # Strip state_dicts from checkpoints for JSON (too large)
    ck_save = {}
    for e, ck in checkpoints.items():
        ck_save[e] = {k: v for k, v in ck.items() if k != "state_dict"}

    out = {
        "graph_config": CFG,
        "graph_score":  score,
        "n_nodes": n, "m0": m0, "nc": nc,
        "tgs_acc": tgs_acc, "final_sparsity": final_sparsity,
        "n_retired": len(retired_indices),
        "checkpoints": ck_save,
        "damage_results": damage_results,
        "redundancy": redundancy,
        "swap_results": swap_results,
        "oracle_results": oracle_results,
        "runtime_min": runtime,
    }
    with open("results/counterfactual_retirement.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → results/counterfactual_retirement.json")
    return out


if __name__ == "__main__":
    main()
