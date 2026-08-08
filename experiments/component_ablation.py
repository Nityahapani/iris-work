"""
experiments/component_ablation.py

Ablation table: which parts of TGS actually matter?

Method                        | Ie(t) corr | Test acc | Sparsity | FLOPs
------------------------------|------------|----------|----------|-------
Random retirement             |            |          |          |
Degree-only                   |            |          |          |
Degree + gradient modulation  |            |          |          |
Degree + maturity discount    |            |          |          |
Full TGS (all three)          |            |          |          |

Each variant uses the same scheduler, warmup, and sparsity ceiling.
Only the influence_scores() logic changes.
"""

import sys, os, json, time
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
import numpy as np
import logging
from scipy.stats import spearmanr

logging.basicConfig(level=logging.WARNING)

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from tgs.core.temporal_graph import TemporalGraph
from tgs.core.influence import GradientNormEstimator, RepresentationDeltaEstimator
from tgs.models.gcn import TemporalGCN
from tgs.schedulers.adaptive_scheduler import AdaptiveRetirementScheduler
from tgs.evaluation.flops import FLOPsCounter
from tgs.utils.reproducibility import set_seed

DEVICE    = torch.device("cpu")
SEED      = 42
EPOCHS    = 300
DATASET   = "Cora"
MAX_SP    = 0.65
WARMUP    = 40
FRAC      = 0.10
EVERY     = 2
SAMPLE    = 200   # edges sampled for exact Ie(t) correlation check


# ── Variant influence-score functions ──────────────────────────────────────
# Each takes (estimator, active_mask) and returns a score tensor [m0].
# Low score = safe to retire.

def scores_random(est, active_mask):
    """Random scores — no information, pure chance."""
    s = torch.full((est.num_edges,), float("inf"), device=est.device)
    eligible = active_mask & (est._er_proxy_score < 1e9)  # all active
    eligible = active_mask
    n = int(active_mask.sum().item())
    if n == 0:
        return s
    rand = torch.rand(est.num_edges, device=est.device)
    rand[~active_mask] = float("inf")
    return rand


def scores_degree_only(est, active_mask):
    """Degree-product only — no gradient, no maturity."""
    scores = torch.full((est.num_edges,), float("inf"), device=est.device)
    if not active_mask.any():
        return scores

    # Hard gate: lock top-10% by er_proxy
    er = est._er_proxy_score.clone(); er[~active_mask] = -1.0
    n_active = int(active_mask.sum().item())
    n_locked = max(1, int(n_active * 0.10))
    _, top_idx = er.topk(n_locked)
    locked = torch.zeros(est.num_edges, dtype=torch.bool, device=est.device)
    locked[top_idx] = True
    eligible = active_mask & ~locked

    if not eligible.any():
        scores[~active_mask] = 0.0
        return scores

    struct = est._struct_retire_score.clone()
    vals = struct[eligible]; mn, mx = vals.min(), vals.max()
    norm = torch.zeros(est.num_edges, device=est.device)
    if (mx - mn) > 1e-12:
        norm[eligible] = (vals - mn) / (mx - mn)

    # Invert: high deg_product → low danger → low score → retire
    scores[eligible] = 1.0 - norm[eligible]
    scores[~active_mask] = 0.0
    return scores


def scores_degree_plus_gradient(est, active_mask):
    """Degree + gradient modulation — no maturity."""
    scores = torch.full((est.num_edges,), float("inf"), device=est.device)
    if not active_mask.any():
        return scores

    er = est._er_proxy_score.clone(); er[~active_mask] = -1.0
    n_active = int(active_mask.sum().item())
    n_locked = max(1, int(n_active * 0.10))
    _, top_idx = er.topk(n_locked)
    locked = torch.zeros(est.num_edges, dtype=torch.bool, device=est.device)
    locked[top_idx] = True
    eligible = active_mask & ~locked

    if not eligible.any():
        scores[~active_mask] = 0.0
        return scores

    def _norm(t, mask):
        vals = t[mask]; mn, mx = vals.min(), vals.max()
        out = torch.zeros(est.num_edges, device=est.device)
        if (mx - mn) > 1e-12:
            out[mask] = (vals - mn) / (mx - mn)
        return out

    struct_norm = _norm(est._struct_retire_score, eligible)
    grad_norm   = _norm(est._ema_influence,       eligible)

    danger  = 1.0 - struct_norm
    boosted = danger * (1.0 + 0.3 * grad_norm)
    # No maturity discount
    scores[eligible] = boosted[eligible]
    scores[~active_mask] = 0.0
    return scores


def scores_degree_plus_maturity(est, active_mask):
    """Degree + maturity discount — no gradient modulation."""
    scores = torch.full((est.num_edges,), float("inf"), device=est.device)
    if not active_mask.any():
        return scores

    er = est._er_proxy_score.clone(); er[~active_mask] = -1.0
    n_active = int(active_mask.sum().item())
    n_locked = max(1, int(n_active * 0.10))
    _, top_idx = er.topk(n_locked)
    locked = torch.zeros(est.num_edges, dtype=torch.bool, device=est.device)
    locked[top_idx] = True
    eligible = active_mask & ~locked

    if not eligible.any():
        scores[~active_mask] = 0.0
        return scores

    def _norm(t, mask):
        vals = t[mask]; mn, mx = vals.min(), vals.max()
        out = torch.zeros(est.num_edges, device=est.device)
        if (mx - mn) > 1e-12:
            out[mask] = (vals - mn) / (mx - mn)
        return out

    struct_norm   = _norm(est._struct_retire_score, eligible)
    maturity_norm = _norm(est._maturity_score,      eligible)

    danger    = 1.0 - struct_norm
    discounted = danger / (1.0 + 0.2 * maturity_norm)
    scores[eligible] = discounted[eligible]
    scores[~active_mask] = 0.0
    return scores


def scores_full_tgs(est, active_mask):
    """Full TGS: degree + gradient + maturity (calls estimator directly)."""
    return est.influence_scores(active_mask)


VARIANTS = [
    ("Random",                   scores_random),
    ("Degree only",              scores_degree_only),
    ("Degree + gradient",        scores_degree_plus_gradient),
    ("Degree + maturity",        scores_degree_plus_maturity),
    ("Full TGS",                 scores_full_tgs),
]


# ── Runner ─────────────────────────────────────────────────────────────────

def run_variant(data, num_features, num_classes, score_fn, label):
    set_seed(SEED)
    m0 = data.edge_index.shape[1]

    tg    = TemporalGraph(data.edge_index, data.num_nodes, device=DEVICE)
    est   = GradientNormEstimator(m0, DEVICE,
                edge_index=data.edge_index, num_nodes=data.num_nodes,
                alpha=0.3, gamma=0.2, hub_gate_pct=0.10)
    model = TemporalGCN(num_features, 64, num_classes, 2, 0.5).to(DEVICE)
    opt   = torch.optim.Adam(
                list(model.parameters()) + [est.edge_weights],
                lr=0.01, weight_decay=5e-4)
    sched = AdaptiveRetirementScheduler(tg,
                epsilon_max=5e-3, epsilon_min=1e-5,
                anneal_steps=100, warmup_steps=WARMUP,
                max_retire_frac=FRAC, max_sparsity=MAX_SP,
                retire_every=EVERY)
    flops     = FLOPsCounter(m0, 2, 64)
    exact_est = RepresentationDeltaEstimator(model, data.x, tg)

    best_val = best_test = 0.0
    corr_records = []

    CHECK_EPOCHS = {80, 150, 200}

    for epoch in range(EPOCHS):
        model.train(); am = tg.active_mask
        logits = model(data.x, tg.edge_index, est.edge_weights[am])
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        opt.zero_grad(); loss.backward(); est.update_influence(am); opt.step()

        model.eval()
        with torch.no_grad():
            el = model(data.x, tg.edge_index)
        preds    = el.argmax(-1)
        val_acc  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        # Exact Ie(t) correlation check
        if epoch in CHECK_EPOCHS:
            scores_t = score_fn(est, am)
            elig = (am & (scores_t < 1e9)).nonzero(as_tuple=False).squeeze(1)
            if len(elig) >= 10:
                perm  = torch.randperm(len(elig))[:SAMPLE]
                sidx  = elig[perm]
                exact = exact_est.compute_influence(sidx, batch_size=50)
                sc    = scores_t[sidx].detach().numpy()
                ex    = exact[sidx].numpy()
                r, _  = spearmanr(sc, ex)
                t20   = np.percentile(sc, 20)
                agr   = (len(set(np.where(sc <= t20)[0]) &
                             set(np.where(ex <= np.percentile(ex, 20))[0]))
                         / max(1, int(SAMPLE * 0.2)))
                corr_records.append({"epoch": epoch, "r": float(r), "agr": float(agr)})

        sched.update_val_acc(val_acc)
        sched.step(score_fn(est, am))
        flops.record_step(tg.mt); tg.step()
        if val_acc > best_val: best_val, best_test = val_acc, test_acc

    # Fresh GCN on selected edge set
    tgs_ei = tg.edge_index.clone()
    set_seed(SEED)
    m2  = TemporalGCN(num_features, 64, num_classes, 2, 0.5).to(DEVICE)
    o2  = torch.optim.Adam(m2.parameters(), lr=0.01, weight_decay=5e-4)
    bv2 = bt2 = 0.0
    for epoch in range(EPOCHS):
        m2.train()
        F.cross_entropy(m2(data.x, tgs_ei)[data.train_mask],
                        data.y[data.train_mask]).backward()
        o2.step(); o2.zero_grad()
        m2.eval()
        with torch.no_grad(): el2 = m2(data.x, tgs_ei)
        p2 = el2.argmax(-1)
        v2 = (p2[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        t2 = (p2[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        if v2 > bv2: bv2, bt2 = v2, t2

    f = flops.summary()
    mean_r   = float(np.mean([c["r"]   for c in corr_records])) if corr_records else 0.0
    mean_agr = float(np.mean([c["agr"] for c in corr_records])) if corr_records else 0.0

    return {
        "variant":       label,
        "test_acc":      best_test,
        "fresh_acc":     bt2,
        "sparsity":      tg.sparsity,
        "flops_red":     f["flops_reduction"],
        "mean_ie_corr":  mean_r,
        "mean_agr":      mean_agr,
        "corr_records":  corr_records,
    }


def main():
    dataset = Planetoid(root="./data", name=DATASET, transform=NormalizeFeatures())
    data    = dataset[0].to(DEVICE)

    print(f"\n{'Component Ablation — ' + DATASET}")
    print(f"{'='*85}")
    print(f"{'Variant':<30} {'Ie(t) r':>8} {'Agr20%':>8} {'TGS acc':>8} {'Fresh acc':>10} {'Sparsity':>9} {'FLOPs↓':>7}")
    print(f"{'-'*85}")

    all_results = []
    for label, score_fn in VARIANTS:
        r = run_variant(data, dataset.num_features, dataset.num_classes, score_fn, label)
        all_results.append(r)
        print(
            f"{label:<30} {r['mean_ie_corr']:>+8.4f} {r['mean_agr']:>8.3f} "
            f"{r['test_acc']:>8.4f} {r['fresh_acc']:>10.4f} "
            f"{r['sparsity']:>9.3f} {r['flops_red']:>7.3f}"
        )

    os.makedirs("results", exist_ok=True)
    with open("results/component_ablation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results/component_ablation.json")


if __name__ == "__main__":
    main()
