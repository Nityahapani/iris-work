"""
experiments/mechanism_ablation.py

Mechanism Ablation: What Actually Causes TGS's Advantage?
==========================================================

Six conditions, same final sparsity, same graph, same seeds:

  A. Static ER          — remove edges before training, ER pruning, no temporal
  B. Static Influence   — remove edges before training, influence-ranked pruning
  C. Random Temporal    — retire edges over time, random order (no influence)
  D. Reverse Temporal   — retire highest-influence edges FIRST (wrong order)
  E. TGS                — retire lowest-influence edges first (correct)
  F. Oracle Temporal    — retire edges in order of their final ER proxy score,
                          temporally (knows the answer statically, applied temporally)

Causal logic:
  If E > A,B           → temporality provides advantage beyond static selection
  If C ≈ E >> A        → TIMING matters, not ordering accuracy
  If D < C ≈ E         → ordering has some effect but timing dominates
  If F ≈ E             → TGS's influence estimator matches the oracle

  The existing Cora result (exp5): TGS≈Random>>Reverse>>Static
  confirms C≈E>>D>>A on one graph. We need this across multiple
  structural types (low/high score) to make the mechanistic claim.

Run on 8 graphs spanning the score range.
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

DEVICE = torch.device("cpu")
EPOCHS = 200
SEEDS  = [42, 99, 200, 301]

# 8 graphs: 4 low-score, 4 high-score
CONFIGS = [
    # (p_intra, p_inter, hub_pct, extra, label, expected_score)
    (0.011, 0.024, 0.00, 0,  "LOW-1",  0.034),
    (0.013, 0.022, 0.00, 0,  "LOW-2",  0.044),
    (0.015, 0.019, 0.00, 0,  "LOW-3",  0.055),
    (0.016, 0.019, 0.00, 0,  "LOW-4",  0.060),
    (0.045, 0.011, 0.04, 25, "HIGH-1", 0.230),
    (0.060, 0.008, 0.05, 30, "HIGH-2", 0.310),
    (0.075, 0.006, 0.08, 42, "HIGH-3", 0.450),
    (0.090, 0.004, 0.10, 52, "HIGH-4", 0.540),
]

CHECKPOINT = "results/mechanism_checkpoint.json"
CHUNK_SIZE = 2   # configs per chunk (each has 4 seeds × 6 methods = heavy)


def prep(cfg, seed):
    p_in, p_out, hub, extra, label, _ = cfg
    ei, x, y, n, nc = make_graph(p_in, p_out, hub, extra, seed)
    h, cv, score    = graph_stats(ei, y, n)
    ei = ei.to(DEVICE); x = x.to(DEVICE); y = y.to(DEVICE)
    g  = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    tm  = torch.zeros(n, dtype=torch.bool)
    vm  = torch.zeros(n, dtype=torch.bool)
    tsm = torch.zeros(n, dtype=torch.bool)
    tm[perm[:int(0.6*n)]]             = True
    vm[perm[int(0.6*n):int(0.8*n)]]  = True
    tsm[perm[int(0.8*n):]]            = True
    return ei, x, y, n, nc, tm, vm, tsm, score


def train_static(ei_pruned, x, y, n, nc, tm, vm, tsm, seed):
    set_seed(seed)
    ms  = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
    os_ = torch.optim.Adam(ms.parameters(), lr=0.01, weight_decay=5e-4)
    bvs = bts = 0.0
    for e in range(EPOCHS):
        ms.train()
        F.cross_entropy(ms(x, ei_pruned)[tm], y[tm]).backward()
        os_.step(); os_.zero_grad()
        ms.eval()
        with torch.no_grad(): out = ms(x, ei_pruned)
        p  = out.argmax(-1)
        va = (p[vm]  == y[vm]).float().mean().item()
        ta = (p[tsm] == y[tsm]).float().mean().item()
        if va > bvs: bvs, bts = va, ta
    return float(bts)


def run_tgs_capture(ei, x, y, n, nc, tm, vm, tsm, seed, max_sp=0.65):
    """Run TGS, return accuracy, sparsity, and retirement schedule."""
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
    retirement_log = []   # [(epoch, edge_idx), ...]

    for e in range(EPOCHS):
        mt.train(); am = tg.active_mask
        prev_active = set(am.nonzero(as_tuple=True)[0].tolist())
        F.cross_entropy(mt(x, tg.edge_index, est.edge_weights[am])[tm], y[tm]).backward()
        est.update_influence(am); ot.step(); ot.zero_grad()
        sc.update_val_acc(bvt)
        sc.step(est.influence_scores(am)); tg.step()
        curr_active = set(tg.active_mask.nonzero(as_tuple=True)[0].tolist())
        newly_retired = prev_active - curr_active
        for idx in newly_retired:
            retirement_log.append((e, idx))
        mt.eval()
        with torch.no_grad(): out = mt(x, tg.edge_index)
        p  = out.argmax(-1)
        va = (p[vm]  == y[vm]).float().mean().item()
        ta = (p[tsm] == y[tsm]).float().mean().item()
        if va > bvt: bvt, btt = va, ta

    retired_set    = set(tg.active_mask.logical_not().nonzero(as_tuple=True)[0].tolist())
    keep_mask      = torch.tensor([i not in retired_set for i in range(m0)], dtype=torch.bool)
    final_ei       = ei[:, keep_mask]
    actual_sp      = float(tg.sparsity)
    return float(btt), actual_sp, retirement_log, final_ei, keep_mask


def run_temporal_variant(ei, x, y, n, nc, tm, vm, tsm, seed,
                         retirement_log, order_type="random"):
    """
    Run a temporal variant with same retirement schedule TIMING but different ORDER.
    order_type: 'random' | 'reverse' | 'oracle'
    oracle uses ER proxy to order retirements (best static ranking applied temporally)
    """
    m0 = ei.shape[1]
    retired_edges = [idx for _, idx in retirement_log]
    retirement_epochs = {idx: ep for ep, idx in retirement_log}

    if order_type == "random":
        rng = np.random.default_rng(seed)
        reorder = rng.permutation(len(retired_edges)).tolist()
        new_order = [retired_edges[i] for i in reorder]
    elif order_type == "reverse":
        new_order = list(reversed(retired_edges))
    elif order_type == "oracle":
        # ER proxy: retire low-ER (hub-to-hub) edges first
        src = ei[0].numpy(); dst = ei[1].numpy()
        dega = degree(ei[1], n, dtype=torch.float).numpy()
        er = 1.0 / dega[src].clip(1) + 1.0 / dega[dst].clip(1)
        er_retired = [(er[idx], idx) for idx in retired_edges]
        new_order = [idx for _, idx in sorted(er_retired)]  # ascending = low ER first
    else:
        raise ValueError(order_type)

    # Remap: same epoch timings, new order
    sorted_epochs = sorted(set(retirement_epochs.values()))
    # Group original retirements by epoch count
    ep_groups = {}
    for ep, idx in retirement_log:
        ep_groups.setdefault(ep, []).append(idx)

    # Assign new_order to same epoch groups
    new_retirement_log = []
    pos = 0
    for ep in sorted(ep_groups.keys()):
        n_ret = len(ep_groups[ep])
        for idx in new_order[pos:pos + n_ret]:
            new_retirement_log.append((ep, idx))
        pos += n_ret

    new_schedule = {}  # edge_idx → epoch to retire
    for ep, idx in new_retirement_log:
        new_schedule[idx] = ep

    # Train with this schedule
    set_seed(seed)
    mt = TemporalGCN(x.shape[1], 40, nc, 2, 0.5).to(DEVICE)
    ot = torch.optim.Adam(mt.parameters(), lr=0.01, weight_decay=5e-4)
    active = torch.ones(m0, dtype=torch.bool)
    bvt = btt = 0.0

    for e in range(EPOCHS):
        # Retire edges scheduled for this epoch
        for idx, ret_ep in new_schedule.items():
            if ret_ep == e:
                active[idx] = False
        ei_cur = ei[:, active]
        mt.train()
        F.cross_entropy(mt(x, ei_cur)[tm], y[tm]).backward()
        ot.step(); ot.zero_grad()
        mt.eval()
        with torch.no_grad(): out = mt(x, ei_cur)
        p  = out.argmax(-1)
        va = (p[vm]  == y[vm]).float().mean().item()
        ta = (p[tsm] == y[tsm]).float().mean().item()
        if va > bvt: bvt, btt = va, ta

    return float(btt)


def run_config_all_methods(cfg, seed):
    p_in, p_out, hub, extra, label, _ = cfg
    ei, x, y, n, nc, tm, vm, tsm, score = prep(cfg, seed)
    m0 = ei.shape[1]

    # E. TGS
    tgs_acc, sp, ret_log, final_ei, keep_mask = run_tgs_capture(
        ei, x, y, n, nc, tm, vm, tsm, seed)

    n_retired = int(m0 * sp)

    # A. Static ER — remove same number of edges by ER proxy, before training
    src = ei[0].numpy(); dst = ei[1].numpy()
    dega = degree(ei[1], n, dtype=torch.float).numpy()
    er = 1.0 / dega[src].clip(1) + 1.0 / dega[dst].clip(1)
    _, sidx_er = torch.from_numpy(er).float().sort()
    rm_er = set(sidx_er[:n_retired].tolist())
    ei_er = ei[:, torch.tensor([i not in rm_er for i in range(m0)], dtype=torch.bool)]
    stat_er_acc = train_static(ei_er, x, y, n, nc, tm, vm, tsm, seed)

    # B. Static Influence — use structural score (deg product, same as TGS primary signal)
    deg_t = degree(ei[1], n, dtype=torch.float)
    inf_score = deg_t[ei[0]] * deg_t[ei[1]]   # high = safe to remove
    _, sidx_inf = inf_score.sort(descending=True)
    rm_inf = set(sidx_inf[:n_retired].tolist())
    ei_inf = ei[:, torch.tensor([i not in rm_inf for i in range(m0)], dtype=torch.bool)]
    stat_inf_acc = train_static(ei_inf, x, y, n, nc, tm, vm, tsm, seed)

    # C. Random Temporal
    rand_acc = run_temporal_variant(
        ei, x, y, n, nc, tm, vm, tsm, seed, ret_log, "random")

    # D. Reverse Temporal
    rev_acc = run_temporal_variant(
        ei, x, y, n, nc, tm, vm, tsm, seed, ret_log, "reverse")

    # F. Oracle Temporal
    oracle_acc = run_temporal_variant(
        ei, x, y, n, nc, tm, vm, tsm, seed, ret_log, "oracle")

    return {
        "label": label, "score": score, "seed": seed, "sparsity": sp,
        "A_static_er":     stat_er_acc,
        "B_static_inf":    stat_inf_acc,
        "C_random_temp":   rand_acc,
        "D_reverse_temp":  rev_acc,
        "E_tgs":           tgs_acc,
        "F_oracle_temp":   oracle_acc,
    }


def run_next_chunk():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f: state = json.load(f)
    else:
        state = {"results": []}

    done_labels = set(
        f"{r['label']}_{r['seed']}" for r in state["results"]
    )
    total = len(CONFIGS) * len(SEEDS)

    todo = [(cfg, seed)
            for cfg in CONFIGS
            for seed in SEEDS
            if f"{cfg[4]}_{seed}" not in done_labels]

    if not todo:
        print("All done."); return True

    t0 = time.time()
    chunk = todo[:CHUNK_SIZE * len(SEEDS)]   # one full config at a time
    chunk = todo[:len(SEEDS)]                # one config, all seeds

    cfg = CONFIGS[len(done_labels) // len(SEEDS)]  # current config
    remaining_seeds = [s for s in SEEDS if f"{cfg[4]}_{s}" not in done_labels]

    for seed in remaining_seeds:
        r = run_config_all_methods(cfg, seed)
        state["results"].append(r)
        print(
            f"  {cfg[4]} seed={seed}  score={r['score']:.4f}  "
            f"A={r['A_static_er']:.4f}  B={r['B_static_inf']:.4f}  "
            f"C={r['C_random_temp']:.4f}  D={r['D_reverse_temp']:.4f}  "
            f"E={r['E_tgs']:.4f}  F={r['F_oracle_temp']:.4f}  "
            f"{time.time()-t0:.0f}s"
        )
        with open(CHECKPOINT, "w") as f: json.dump(state, f, indent=2)

    done_now = len(set(r['label'] for r in state['results']))
    remaining_cfgs = len(CONFIGS) - done_now
    print(f"\nConfig {cfg[4]} done. {remaining_cfgs} configs remain.")
    return remaining_cfgs == 0


def finalise():
    with open(CHECKPOINT) as f: state = json.load(f)
    results = state["results"]

    # Aggregate by config label
    from collections import defaultdict
    by_label = defaultdict(list)
    for r in results:
        by_label[r["label"]].append(r)

    methods = ["A_static_er", "B_static_inf", "C_random_temp",
               "D_reverse_temp", "E_tgs", "F_oracle_temp"]
    method_labels = {
        "A_static_er":    "A. Static ER",
        "B_static_inf":   "B. Static Influence",
        "C_random_temp":  "C. Random Temporal",
        "D_reverse_temp": "D. Reverse Temporal",
        "E_tgs":          "E. TGS",
        "F_oracle_temp":  "F. Oracle Temporal",
    }

    print("\n" + "="*80)
    print("MECHANISM ABLATION RESULTS")
    print("="*80)
    print(f"\n{'Config':8s} {'Score':>6}", end="")
    for m in methods:
        print(f"  {method_labels[m][:14]:>14}", end="")
    print()
    print("─"*80)

    agg = []
    for label in [c[4] for c in CONFIGS]:
        group = by_label[label]
        if not group: continue
        score = np.mean([r["score"] for r in group])
        row = {"label": label, "score": score}
        print(f"{label:8s} {score:>6.3f}", end="")
        for m in methods:
            mean_acc = np.mean([r[m] for r in group])
            row[m] = float(mean_acc)
            tgs_acc = np.mean([r["E_tgs"] for r in group])
            delta = mean_acc - tgs_acc
            marker = " ◄" if m == "E_tgs" else ""
            print(f"  {mean_acc:>14.4f}{marker}", end="")
        print()
        agg.append(row)

    # Summary: low vs high score
    low  = [r for r in agg if "LOW"  in r["label"]]
    high = [r for r in agg if "HIGH" in r["label"]]

    print(f"\n{'─'*80}")
    print(f"{'Mean':8s} {'':>6}", end="")
    for m in methods:
        low_mean  = np.mean([r[m] for r in low])  if low  else float('nan')
        high_mean = np.mean([r[m] for r in high]) if high else float('nan')
        print(f"  L={low_mean:.3f}/H={high_mean:.3f}", end="")
    print()

    # Key finding
    print(f"\nKEY FINDINGS:")
    for grp_name, grp in [("LOW-score", low), ("HIGH-score", high)]:
        if not grp: continue
        a  = np.mean([r["A_static_er"]    for r in grp])
        b  = np.mean([r["B_static_inf"]   for r in grp])
        c  = np.mean([r["C_random_temp"]  for r in grp])
        d  = np.mean([r["D_reverse_temp"] for r in grp])
        e  = np.mean([r["E_tgs"]          for r in grp])
        f_ = np.mean([r["F_oracle_temp"]  for r in grp])
        print(f"  {grp_name}: "
              f"Static-ER={a:.4f}  Static-Inf={b:.4f}  "
              f"Random-T={c:.4f}  Reverse-T={d:.4f}  "
              f"TGS={e:.4f}  Oracle-T={f_:.4f}")
        print(f"    TGS vs Static-ER: {(e-a)*100:+.2f}pp  |  "
              f"TGS vs Random-T: {(e-c)*100:+.2f}pp  |  "
              f"Random-T vs Static-ER: {(c-a)*100:+.2f}pp")

    os.makedirs("results", exist_ok=True)
    out = {
        "configs": [c[4] for c in CONFIGS],
        "methods": methods,
        "method_labels": method_labels,
        "aggregated": agg,
        "low_score_configs": [r["label"] for r in low],
        "high_score_configs": [r["label"] for r in high],
    }
    with open("results/mechanism_ablation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → results/mechanism_ablation.json")
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
