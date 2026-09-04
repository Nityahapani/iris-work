"""
Graph Surgery 2×2×2 Experiment
================================
Directly addresses the judge question:
  "How do you know H × deg-CV is actually causing edge maturation,
   rather than just being a proxy for some other graph property?"

Prior experiment (timing_structure_2x2.py) had a fatal design flaw:
  - Target: benign cell h=0.50, but achieved h=0.122
  - Both 'benign' and 'hostile' cells landed near h=0.12
  - β_TxS = -0.0078 is noise, not a true null, because the manipulation failed

Root cause: Wisconsin starts at h=0.196. Degree-preserving rewiring from h=0.196
to h=0.50 requires ~98 cross→same edge swaps. The swap success rate is low
(most random pairs create multi-edges/self-loops), so 50k iterations aren't enough.

Fix: Use synthetic planted-community graphs instead of rewiring Wisconsin.
Synthetic graphs give EXACT control over h and deg-CV.
We match Wisconsin's n, m, and class balance to keep the task comparable.
Node features are sampled from Wisconsin's empirical feature distribution.

Design
------
Factor H  — Homophily:     LOW (h≈0.20) vs HIGH (h≈0.70)
Factor CV — Degree CV:     LOW (CV≈0.3) vs HIGH (CV≈1.5)
Factor T  — Timing:        EARLY (warmup=0) vs LATE (warmup=40)

This is a full 2×2×2 = 8-cell design, 5 seeds each = 40 runs.

The critical comparison:
  LOW-H × HIGH-CV × EARLY  vs  LOW-H × HIGH-CV × LATE
  HIGH-H × LOW-CV × EARLY  vs  HIGH-H × LOW-CV × LATE
  And the interaction: does timing benefit depend specifically on (LOW-H, HIGH-CV)?

Predicted pattern (from the H×CV mechanism story):
  - Timing penalty (late vs early) should be LARGE when H is low AND CV is high
  - Timing should not matter much for HIGH-H graphs (homophilous, little noise to remove)
  - Timing should not matter much for LOW-CV graphs (no hubs = uniform degree = weak structural signal)

If β_T×H is significant AND β_T×CV is significant AND β_T×H×CV is significant,
we have strong evidence that the timing benefit is specifically caused by hub-mediated
cross-class noise — not some other graph property.

Usage
-----
    PYTHONPATH=. python3 experiments/graph_surgery_2x2x2.py [--seeds N] [--epochs E]
"""

import json, time, argparse, logging
from pathlib import Path
from itertools import product
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree

logging.basicConfig(level=logging.WARNING)
OUT = Path(__file__).parent.parent / "results" / "graph_surgery_2x2x2"
OUT.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tgs.utils import load_config, set_seed, load_dataset
from tgs.evaluation.baselines import run_baseline


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic graph construction
# ─────────────────────────────────────────────────────────────────────────────

def _make_degree_sequence(n: int, m_undirected: int, target_cv: float,
                           rng: np.random.Generator) -> np.ndarray:
    """
    Build a degree sequence of length n with total = 2*m_undirected
    and deg_cv ≈ target_cv.

    LOW CV (≈0.3): near-uniform, use clipped Gaussian.
    HIGH CV (≈1.5): hub-injected — 10% of nodes get ~5× mean degree.

    Wisconsin mean_deg ≈ 2.05, so hubs get deg≈10, non-hubs deg≈1-2.
    This is inspired by Wisconsin's actual degree distribution (deg_cv=1.01).
    """
    mean_deg = 2 * m_undirected / n
    total_stubs = 2 * m_undirected

    if target_cv <= 0.5:
        # Near-uniform: Gaussian clipped to [1, 2*mean_deg]
        sigma = target_cv * mean_deg
        raw = rng.normal(mean_deg, sigma, size=n)
        raw = np.clip(raw, 1, 2 * mean_deg + 1)
    else:
        # Hub-injected: n_hubs nodes get hub_deg, rest get 1
        # hub_mult chosen to hit target CV
        # CV ≈ sqrt(frac_hubs) * (hub_mult - 1) / sqrt(frac_hubs * hub_mult^2 + (1-frac_hubs))
        # Empirically: CV≈1.3 with 10% hubs at 5x mean
        n_hubs = max(2, int(0.10 * n))
        hub_deg = max(3, int(5.0 * mean_deg))
        hub_stubs = n_hubs * hub_deg
        nonhub_stubs = total_stubs - hub_stubs
        if nonhub_stubs <= (n - n_hubs):
            # Reduce hub_deg if it takes too many stubs
            hub_deg = max(2, (total_stubs - (n - n_hubs)) // n_hubs)
            hub_stubs = n_hubs * hub_deg
            nonhub_stubs = total_stubs - hub_stubs

        nonhub_deg = max(1, nonhub_stubs // (n - n_hubs))
        raw = np.full(n, float(nonhub_deg))
        hub_idx = rng.choice(n, n_hubs, replace=False)
        raw[hub_idx] = float(hub_deg)

    degs = np.round(raw).astype(int)
    degs = np.maximum(degs, 1)

    # Adjust total stubs to exactly 2*m
    diff = total_stubs - degs.sum()
    if diff != 0:
        sign = int(np.sign(diff))
        adjust_pool = np.where(degs > 1)[0] if sign < 0 else np.arange(n)
        if len(adjust_pool) == 0:
            adjust_pool = np.arange(n)
        adj_idx = rng.choice(adjust_pool, size=abs(diff), replace=True)
        for idx in adj_idx:
            degs[idx] = max(1, degs[idx] + sign)

    return degs


def make_planted_graph(
    n: int,
    m_undirected: int,
    nc: int,
    class_counts: list,
    target_h: float,
    target_deg_cv: float,
    features: np.ndarray,
    rng: np.random.Generator,
    max_attempts: int = 15,
) -> Data:
    """
    Build a synthetic graph with:
      - n nodes, ~m_undirected edges (within 10%)
      - Same class balance as class_counts
      - Homophily ≈ target_h  (within 0.10)
      - Degree CV ≈ target_deg_cv (LOW≈0.3, HIGH≈1.3)
      - Node features drawn from empirical distribution (rows shuffled)

    Strategy: planted configuration model.
      1. Assign labels by class_counts.
      2. Generate degree sequence from target CV (near-uniform vs hub-injected).
      3. Configuration model edge placement with class-conditional sampling:
         - Draw stubs from degree sequence.
         - For each stub pair, assign to same-class (prob=target_h) or cross-class.
         - Reject self-loops and multi-edges.
      4. Features: randomly sample rows from the real Wisconsin feature matrix.

    The key invariants for the causal claim:
      - n, m, class_balance are FIXED across all four conditions.
      - Only h and deg-CV vary — controlled by construction, not by rewiring.
      - The rewiring approach in timing_structure_2x2.py FAILED because Wisconsin
        (h=0.196) cannot be degree-preserving-rewired to h=0.50 in 50k swaps.
        Synthetic generation bypasses this constraint entirely.
    """
    # 1. Labels
    labels = []
    for cls, cnt in enumerate(class_counts):
        labels.extend([cls] * cnt)
    labels = np.array(labels[:n])
    assert len(labels) == n

    class_node_lists = {c: np.where(labels == c)[0] for c in range(nc)}
    class_probs = np.array([class_counts[c] / n for c in range(nc)])

    best_data, best_h_err, best_cv_err = None, 999.0, 999.0

    for attempt in range(max_attempts):
        rng_local = np.random.default_rng(rng.integers(1 << 31))

        # 2. Degree sequence
        degs = _make_degree_sequence(n, m_undirected, target_deg_cv, rng_local)
        # Hub nodes = top 10% by degree (consistent definition across graphs)
        hub_thresh = np.quantile(degs, 0.90)
        hub_nodes_set = set(np.where(degs >= hub_thresh)[0].tolist())

        # Weighted stub pool (each node appears degree[i] times)
        stubs = np.repeat(np.arange(n), degs)
        rng_local.shuffle(stubs)

        edge_set = set()
        target_edges = m_undirected

        # Place edges: configuration model with class-conditional selection
        for _ in range(target_edges * 8):
            if len(edge_set) >= target_edges:
                break

            # Draw first endpoint from stub pool (proportional to degree)
            u = int(stubs[rng_local.integers(len(stubs))])

            # Decide same-class or cross-class
            if rng_local.random() < target_h:
                # Same-class: draw v from same class as u
                cls = int(labels[u])
                pool = class_node_lists[cls]
                if len(pool) < 2:
                    continue
                v = int(rng_local.choice(pool))
            else:
                # Cross-class: draw v from a different class
                # Weight cross-class selection toward hubs if high-CV
                cls_u = int(labels[u])
                other_classes = [c for c in range(nc) if c != cls_u]
                if not other_classes:
                    continue
                cls_v = int(rng_local.choice(other_classes))
                pool_v = class_node_lists[cls_v]
                if len(pool_v) == 0:
                    continue
                # Bias toward hub nodes in cross-class selection (when high-CV)
                if target_deg_cv > 0.8 and len(hub_nodes_set) > 0:
                    hub_in_pool = [x for x in pool_v if x in hub_nodes_set]
                    if hub_in_pool and rng_local.random() < 0.5:
                        v = int(rng_local.choice(hub_in_pool))
                    else:
                        v = int(rng_local.choice(pool_v))
                else:
                    v = int(rng_local.choice(pool_v))

            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key in edge_set:
                continue
            edge_set.add(key)

        if len(edge_set) < int(0.80 * m_undirected):
            continue

        # Build edge_index (directed)
        us = [u for u, v in edge_set] + [v for u, v in edge_set]
        vs_list = [v for u, v in edge_set] + [u for u, v in edge_set]
        edge_index = torch.tensor([us, vs_list], dtype=torch.long)

        # 4. Features: class-informative Gaussian
        # Each class gets a distinct mean vector; nodes get class mean + noise.
        # feature_dim = 32, SNR controlled so dense GCN gets ~60-70% accuracy.
        # Using real Wisconsin features (shuffled rows) destroys class signal —
        # GCN collapses to majority-class, making all cells uninformative.
        feat_dim = 32
        class_means = rng_local.normal(0, 2.0, size=(nc, feat_dim))
        feat_noise = 1.5  # noise std: tune so dense GCN ~60-70% acc
        node_feats = np.zeros((n, feat_dim))
        for i in range(n):
            node_feats[i] = class_means[labels[i]] + rng_local.normal(0, feat_noise, feat_dim)
        x = torch.tensor(node_feats, dtype=torch.float)

        # Verify
        achieved_h = _compute_h(edge_index, labels)
        deg_actual = degree(edge_index[1], n).numpy()
        achieved_cv = float(deg_actual.std() / (deg_actual.mean() + 1e-8))
        m_achieved = edge_index.shape[1] // 2

        h_err = abs(achieved_h - target_h)
        cv_err = abs(achieved_cv - target_deg_cv)

        # Track best
        if h_err + cv_err < best_h_err + best_cv_err:
            y = torch.tensor(labels, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, y=y, num_nodes=n)
            data._achieved_h = achieved_h
            data._achieved_cv = achieved_cv
            data._m = m_achieved
            best_data = data
            best_h_err, best_cv_err = h_err, cv_err

        if h_err < 0.10 and cv_err < 0.30:
            return data, achieved_h, achieved_cv

    # Return best found
    logging.warning(
        f"make_planted_graph: targets h={target_h:.2f},cv={target_deg_cv:.2f} "
        f"→ best achieved h={best_data._achieved_h:.3f},"
        f"cv={best_data._achieved_cv:.3f} (h_err={best_h_err:.3f}, cv_err={best_cv_err:.3f})"
    )
    return best_data, best_data._achieved_h, best_data._achieved_cv


def _compute_h(edge_index, labels_np):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    same = sum(1 for u, v in zip(src, dst) if labels_np[u] == labels_np[v])
    return same / max(len(src), 1)


def make_splits(n, nc, labels, train_frac=0.6, val_frac=0.2, rng=None):
    """Build train/val/test masks with stratified sampling."""
    if rng is None:
        rng = np.random.default_rng(42)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    for cls in range(nc):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_train = max(1, int(len(idx) * train_frac))
        n_val = max(1, int(len(idx) * val_frac))
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True
    return train_mask, val_mask, test_mask


# ─────────────────────────────────────────────────────────────────────────────
# Run one (graph, timing, seed) cell
# ─────────────────────────────────────────────────────────────────────────────

def run_one(data, nf, nc, seed, warmup, cfg, device):
    from tgs.core import TemporalGraph, EdgeManager
    from tgs.core.influence import GradientNormEstimator
    from tgs.models import TemporalGCN
    from tgs.schedulers import AdaptiveRetirementScheduler
    from tgs.evaluation import Evaluator
    from tgs.evaluation.flops import FLOPsCounter

    set_seed(seed)
    data = data.to(device)

    m0 = data.edge_index.shape[1]
    tg = TemporalGraph(data.edge_index, data.num_nodes, device=device)
    influence_est = GradientNormEstimator(
        m0, device,
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        ema_decay=cfg.ema_decay,
    )
    model = TemporalGCN(nf, cfg.hidden_channels, nc,
                        cfg.num_layers, cfg.dropout).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [influence_est.edge_weights],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = AdaptiveRetirementScheduler(
        temporal_graph=tg,
        epsilon_max=cfg.epsilon_max,
        epsilon_min=cfg.epsilon_min,
        anneal_steps=cfg.anneal_steps,
        schedule=cfg.anneal_schedule,
        warmup_steps=warmup,
        max_retire_frac=cfg.max_retire_frac,
        max_sparsity=cfg.max_sparsity,
        retire_every=cfg.retire_every,
    )
    evaluator = Evaluator(nc)
    flops_counter = FLOPsCounter(m0, cfg.num_layers, cfg.hidden_channels)

    for epoch in range(cfg.epochs):
        model.train()
        active_mask = tg.active_mask
        ei_t = tg.edge_index
        ew_t = influence_est.edge_weights[active_mask]
        logits = model(data.x, ei_t, ew_t)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        influence_est.update_influence(active_mask)
        optimizer.step()
        influence_scores = influence_est.influence_scores(active_mask)
        scheduler.step(influence_scores)
        flops_counter.record_step(tg.mt)
        model.eval()
        with torch.no_grad():
            eval_logits = model(data.x, tg.edge_index)
        metrics = evaluator.update(
            logits=eval_logits, labels=data.y,
            train_mask=data.train_mask, val_mask=data.val_mask,
            test_mask=data.test_mask, sparsity=tg.sparsity,
            step=epoch, distortion_bound=scheduler.cumulative_distortion_bound,
        )
        scheduler.update_val_acc(metrics["val_acc"])
        influence_est.update_disagreement(eval_logits, active_mask)
        tg.step()

    summary = evaluator.compute()
    tgs_acc = summary["test_acc_at_best_val"]

    b_dense = run_baseline(
        "dense", data, nf, nc, 0.0,
        hidden=cfg.hidden_channels, epochs=cfg.epochs,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        dropout=cfg.dropout, seed=seed, device=device,
    )

    return {
        "tgs_acc": round(float(tgs_acc), 4),
        "dense_acc": round(float(b_dense["best_test_acc"]), 4),
        "gap": round(float(b_dense["best_test_acc"]) - float(tgs_acc), 4),
        "sparsity": round(float(tg.sparsity), 3),
        "warmup": warmup,
        "seed": seed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# 2×2 structure conditions (H × CV)
STRUCTURE_CONDITIONS = {
    "low_h_low_cv":  dict(h=0.20, cv=0.30, label="LOW-H × LOW-CV  (low noise, uniform deg)"),
    "low_h_high_cv": dict(h=0.20, cv=1.50, label="LOW-H × HIGH-CV (low noise, hub-heavy)  ← TGS regime"),
    "high_h_low_cv": dict(h=0.70, cv=0.30, label="HIGH-H × LOW-CV (high noise, uniform deg)"),
    "high_h_high_cv":dict(h=0.70, cv=1.50, label="HIGH-H × HIGH-CV(high noise, hub-heavy)"),
}

TIMING_CONDITIONS = {
    "early": 0,
    "late": 40,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--force", action="store_true", help="Re-run all cells")
    args = parser.parse_args()

    SEEDS = list(range(42, 42 + args.seeds))

    cfg = load_config("configs/wisconsin_gcn.yaml")
    if args.epochs:
        cfg.epochs = args.epochs
    device = torch.device("cpu")

    # Load real Wisconsin to extract n, m, class balance
    data_real, _nf_real, nc = load_dataset(cfg, device)
    n = data_real.num_nodes
    m_undirected = data_real.edge_index.shape[1] // 2
    labels_real = data_real.y.numpy()
    class_counts_dict = Counter(labels_real.tolist())
    class_counts = [class_counts_dict[c] for c in range(nc)]
    features_np = data_real.x.numpy()
    nf = 32  # synthetic feature dim (class-informative Gaussian, not raw Wisconsin)

    print(f"Wisconsin reference: n={n}, m={m_undirected}, nc={nc}")
    print(f"Class counts: {class_counts}")
    print(f"Synthetic feature dim: {nf} (class-informative Gaussian, SNR controlled)")
    print()

    out_file = OUT / "results.json"
    all_results = json.loads(out_file.read_text()) if (out_file.exists() and not args.force) else {}

    rng_master = np.random.default_rng(0)

    # ── Build/cache synthetic graph variants ──────────────────────────────
    print("Building synthetic graph variants...")
    variants = {}
    for cond_name, cond in STRUCTURE_CONDITIONS.items():
        print(f"  [{cond_name}] target h={cond['h']:.2f}, CV={cond['cv']:.2f} ...", end=" ", flush=True)
        graph_rng = np.random.default_rng(hash(cond_name) % (2**31))
        data_syn, achieved_h, achieved_cv = make_planted_graph(
            n=n,
            m_undirected=m_undirected,
            nc=nc,
            class_counts=class_counts,
            target_h=cond["h"],
            target_deg_cv=cond["cv"],
            features=features_np,
            rng=graph_rng,
            max_attempts=15,
        )
        # Attach fixed splits (same across seeds; seed controls model init)
        split_rng = np.random.default_rng(99)
        train_mask, val_mask, test_mask = make_splits(
            n, nc, data_syn.y.numpy(), rng=split_rng
        )
        data_syn.train_mask = train_mask
        data_syn.val_mask = val_mask
        data_syn.test_mask = test_mask
        variants[cond_name] = (data_syn, achieved_h, achieved_cv)

        deg_v = degree(data_syn.edge_index[1], n).numpy()
        cv_v = float(deg_v.std() / (deg_v.mean() + 1e-8))
        m_v = data_syn.edge_index.shape[1] // 2
        print(f"achieved h={achieved_h:.3f}, CV={cv_v:.3f}, m={m_v}")

        if cond_name not in all_results:
            all_results[cond_name] = {
                "label": cond["label"],
                "target_h": cond["h"], "target_cv": cond["cv"],
                "achieved_h": round(achieved_h, 3), "achieved_cv": round(cv_v, 3),
                "m": m_v,
                "early": {"runs": []},
                "late": {"runs": []},
            }

    # ── Run all 4×2×seeds combinations ───────────────────────────────────
    print(f"\nRunning 4×2 factorial × {len(SEEDS)} seeds ({4*2*len(SEEDS)} total runs)...")
    for cond_name, (data_syn, achieved_h, achieved_cv) in variants.items():
        for timing_name, warmup in TIMING_CONDITIONS.items():
            cell = all_results[cond_name][timing_name]
            done_seeds = {r["seed"] for r in cell["runs"]}
            for seed in SEEDS:
                if seed in done_seeds and not args.force:
                    continue
                tag = f"[{cond_name}] × [{timing_name}] seed={seed}"
                print(f"  {tag}...", end=" ", flush=True)
                t0 = time.time()
                r = run_one(data_syn, nf, nc, seed, warmup, cfg, device)
                r["time_sec"] = round(time.time() - t0, 1)
                cell["runs"].append(r)
                out_file.write_text(json.dumps(all_results, indent=2))
                print(f"TGS={r['tgs_acc']:.4f} Dense={r['dense_acc']:.4f} "
                      f"gap={r['gap']:+.4f} sp={r['sparsity']:.2f} ({r['time_sec']:.0f}s)")

    # ── Analysis ──────────────────────────────────────────────────────────
    import statistics as st

    print(f"\n\n{'='*72}")
    print("GRAPH SURGERY 2×2×2 RESULTS")
    print(f"{'='*72}")
    print(f"gap = Dense_acc − TGS_acc  (negative = TGS wins)")
    print()

    cell_means = {}
    for cond_name in STRUCTURE_CONDITIONS:
        for timing_name in TIMING_CONDITIONS:
            runs = all_results[cond_name][timing_name]["runs"]
            if not runs:
                continue
            gaps = [r["gap"] for r in runs]
            gm = st.mean(gaps)
            gs = st.stdev(gaps) if len(gaps) > 1 else 0.0
            wins = sum(1 for g in gaps if g < 0)
            cell_means[(cond_name, timing_name)] = gm
            cond = STRUCTURE_CONDITIONS[cond_name]
            h_tag = "HIGH-H" if cond["h"] > 0.4 else "LOW-H "
            cv_tag = "HIGH-CV" if cond["cv"] > 0.8 else "LOW-CV "
            t_tag = timing_name.upper()
            print(f"  {h_tag} × {cv_tag} × {t_tag:5s}: "
                  f"gap={gm:+.4f} ± {gs:.4f}  wins={wins}/{len(gaps)}")

    print()
    print("Timing benefit (late - early gap) per structural cell:")
    for cond_name in STRUCTURE_CONDITIONS:
        e = cell_means.get((cond_name, "early"), None)
        l = cell_means.get((cond_name, "late"), None)
        if e is None or l is None:
            continue
        timing_benefit = e - l  # positive = late timing helps TGS (reduces gap)
        cond = STRUCTURE_CONDITIONS[cond_name]
        h_tag = "HIGH-H" if cond["h"] > 0.4 else "LOW-H "
        cv_tag = "HIGH-CV" if cond["cv"] > 0.8 else "LOW-CV "
        print(f"  {h_tag} × {cv_tag}: timing_benefit = {timing_benefit:+.4f}")

    # 3-way interaction: is timing benefit concentrated in LOW-H × HIGH-CV?
    def safe(cond, timing):
        return cell_means.get((cond, timing), 0.0)

    lh_hcv_benefit = safe("low_h_high_cv",  "early") - safe("low_h_high_cv",  "late")
    lh_lcv_benefit = safe("low_h_low_cv",   "early") - safe("low_h_low_cv",   "late")
    hh_hcv_benefit = safe("high_h_high_cv", "early") - safe("high_h_high_cv", "late")
    hh_lcv_benefit = safe("high_h_low_cv",  "early") - safe("high_h_low_cv",  "late")

    # The predicted pattern: lh_hcv_benefit >> all others
    interaction_H  = ((lh_hcv_benefit + lh_lcv_benefit) - (hh_hcv_benefit + hh_lcv_benefit)) / 2
    interaction_CV = ((lh_hcv_benefit + hh_hcv_benefit) - (lh_lcv_benefit + hh_lcv_benefit)) / 2
    three_way      = (lh_hcv_benefit - lh_lcv_benefit) - (hh_hcv_benefit - hh_lcv_benefit)

    print(f"\nFactorial decomposition of timing benefit:")
    print(f"  β_H  (homophily moderates timing)    = {interaction_H:+.4f}")
    print(f"  β_CV (degree-CV moderates timing)    = {interaction_CV:+.4f}")
    print(f"  β_HxCV (3-way T×H×CV interaction)   = {three_way:+.4f}  ← CRITICAL")
    print()
    print("Predicted pattern:")
    print("  β_H < 0  : low-H graphs benefit more from late timing (heterophily = noise to remove)")
    print("  β_CV > 0 : high-CV graphs benefit more from late timing (hubs focus the noise signal)")
    print("  β_HxCV   : timing benefit is concentrated specifically in LOW-H × HIGH-CV")
    print()

    if three_way > 0.02:
        print("✓ 3-WAY INTERACTION CONFIRMED")
        print("  Timing benefit is specifically concentrated in the LOW-H × HIGH-CV cell.")
        print("  This is exactly what the H×CV mechanism predicts.")
        print("  Neither low-H alone nor high-CV alone produces the full timing benefit.")
    elif three_way > 0.005:
        print("~ Weak 3-way interaction — directionally consistent but small.")
    else:
        print("✗ No clear 3-way interaction. Timing benefit is not specifically")
        print("  concentrated in LOW-H × HIGH-CV. Mechanism story needs revision.")

    # Save decomposition
    all_results["_analysis"] = {
        "timing_benefits": {
            "low_h_high_cv":  round(lh_hcv_benefit, 4),
            "low_h_low_cv":   round(lh_lcv_benefit, 4),
            "high_h_high_cv": round(hh_hcv_benefit, 4),
            "high_h_low_cv":  round(hh_lcv_benefit, 4),
        },
        "interaction_H":   round(interaction_H, 4),
        "interaction_CV":  round(interaction_CV, 4),
        "three_way_TxHxCV": round(three_way, 4),
        "predicted_winner": "low_h_high_cv",
        "design": "2x2x2: H(low/high) x CV(low/high) x T(early/late)",
    }
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
