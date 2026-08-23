"""
Controlled Graph Intervention Experiment
=========================================
Tests the causal mechanism: "TGS wins because its influence estimator
preferentially removes cross-class edges when they concentrate at hubs."

Starting from Wisconsin (the strongest TGS win), we construct 6 matched
graph variants that each destroy exactly ONE structural property while
preserving everything else (n, m, degree sequence, class balance, overall
homophily where possible). Then run TGS vs. dense vs. random on every
variant and check whether the predicted direction of change is observed.

Variants
--------
1. ORIGINAL           — Wisconsin as-is (control)
2. CROSS→HUBS        — rewire so ALL cross-class edges touch hubs
                        (amplify the mechanism → TGS advantage should increase)
3. CROSS→NON-HUBS    — rewire so ALL cross-class edges avoid hubs
                        (remove the mechanism → advantage should collapse)
4. SAME→HUBS         — rewire so same-class edges touch hubs instead
                        (flip which edges are at hubs, homophily preserved)
5. LABEL_SHUFFLE      — shuffle class labels (preserve graph, destroy label structure)
                        (TGS advantage should disappear — no signal to exploit)
6. RANDOM_REWIRE      — random edge swap preserving degree sequence
                        (preserves homophily on average, destroys hub structure)

All variants use double-edge swaps to preserve the full degree sequence.

Usage
-----
    PYTHONPATH=. python3 experiments/graph_intervention.py
"""

import json, time, logging, copy, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_undirected

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
OUT = Path(__file__).parent.parent / "results" / "graph_intervention"
OUT.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# 1. Load Wisconsin
# ──────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from tgs.utils import load_config, set_seed, load_dataset
from tgs.evaluation.baselines import run_baseline
from scripts.train import train

BASE_CFG = "configs/wisconsin_gcn.yaml"
SEEDS    = [42, 43, 44, 45, 46]
N_REPEAT = len(SEEDS)


def load_base():
    cfg  = load_config(BASE_CFG)
    data, nf, nc = load_dataset(cfg, torch.device("cpu"))
    return data, nf, nc, cfg


# ──────────────────────────────────────────────
# 2. Edge-swap primitives
# ──────────────────────────────────────────────

def edges_to_set(src, dst):
    """Return set of (min, max) undirected edge pairs."""
    return {(min(u, v), max(u, v)) for u, v in zip(src.tolist(), dst.tolist())}


def set_to_tensor(edge_set, n):
    """Convert set of (u,v) pairs to undirected edge_index tensor."""
    us = [u for u, v in edge_set] + [v for u, v in edge_set]
    vs = [v for u, v in edge_set] + [u for u, v in edge_set]
    return torch.tensor([us, vs], dtype=torch.long)


def double_edge_swap(edge_set, n, rng, n_swaps=None):
    """
    Markov-chain double edge swap: pick two edges (a,b) and (c,d),
    rewire to (a,d) and (c,b) if neither new edge exists and no self-loop.
    Preserves the degree sequence exactly.
    """
    edge_list = list(edge_set)
    if n_swaps is None:
        n_swaps = len(edge_list) * 10
    swapped = 0
    attempts = 0
    while swapped < n_swaps and attempts < n_swaps * 20:
        attempts += 1
        idx1, idx2 = rng.choice(len(edge_list), 2, replace=False)
        a, b = edge_list[idx1]
        c, d = edge_list[idx2]
        # Two possible rewirings
        if rng.random() < 0.5:
            e1 = (min(a, d), max(a, d))
            e2 = (min(c, b), max(c, b))
        else:
            e1 = (min(a, c), max(a, c))
            e2 = (min(b, d), max(b, d))
        # Accept if no self-loops and neither edge already exists
        if e1[0] == e1[1] or e2[0] == e2[1]:
            continue
        if e1 in edge_set or e2 in edge_set:
            continue
        # Swap
        edge_set.discard(edge_list[idx1])
        edge_set.discard(edge_list[idx2])
        edge_set.add(e1)
        edge_set.add(e2)
        edge_list[idx1] = e1
        edge_list[idx2] = e2
        swapped += 1
    return edge_set


# ──────────────────────────────────────────────
# 3. Variant constructors
# ──────────────────────────────────────────────

def build_variant(data, nc, variant, rng):
    """
    Returns a new Data object with the same x, y, masks but a modified
    edge_index according to the variant specification.
    """
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    y   = data.y.numpy()
    n   = data.num_nodes
    deg = degree(data.edge_index[1], n).numpy()
    hub_thresh = np.quantile(deg, 0.90)
    hub_nodes  = set(np.where(deg >= hub_thresh)[0].tolist())

    # Work in undirected edge set (min, max pairs)
    edge_set = edges_to_set(src, dst)

    def is_cross(u, v):  return y[u] != y[v]
    def is_hub(u, v):    return u in hub_nodes or v in hub_nodes

    if variant == "original":
        new_ei = data.edge_index.clone()

    elif variant == "cross_to_hubs":
        # Target: cross-class edges should all touch at least one hub.
        # Move cross-non-hub edges to hub positions via swaps.
        cross_non_hub = [(u, v) for u, v in edge_set if is_cross(u, v) and not is_hub(u, v)]
        same_hub      = [(u, v) for u, v in edge_set if not is_cross(u, v) and is_hub(u, v)]
        n_swap = min(len(cross_non_hub), len(same_hub))
        rng.shuffle(cross_non_hub); rng.shuffle(same_hub)
        for (u1, v1), (u2, v2) in zip(cross_non_hub[:n_swap], same_hub[:n_swap]):
            # swap: (cross,non-hub) + (same,hub) → try (cross,hub)
            # Try u1-u2, v1-v2 or u1-v2, v1-u2
            candidates = [
                ((min(u1, u2), max(u1, u2)), (min(v1, v2), max(v1, v2))),
                ((min(u1, v2), max(u1, v2)), (min(v1, u2), max(v1, u2))),
            ]
            for e_new1, e_new2 in candidates:
                if e_new1[0] == e_new1[1] or e_new2[0] == e_new2[1]: continue
                if e_new1 in edge_set or e_new2 in edge_set: continue
                edge_set.discard((u1, v1)); edge_set.discard((u2, v2))
                edge_set.add(e_new1); edge_set.add(e_new2)
                break
        new_ei = set_to_tensor(edge_set, n)

    elif variant == "cross_to_non_hubs":
        # Move cross-hub edges to non-hub positions.
        cross_hub     = [(u, v) for u, v in edge_set if is_cross(u, v) and is_hub(u, v)]
        same_non_hub  = [(u, v) for u, v in edge_set if not is_cross(u, v) and not is_hub(u, v)]
        n_swap = min(len(cross_hub), len(same_non_hub))
        rng.shuffle(cross_hub); rng.shuffle(same_non_hub)
        for (u1, v1), (u2, v2) in zip(cross_hub[:n_swap], same_non_hub[:n_swap]):
            candidates = [
                ((min(u1, u2), max(u1, u2)), (min(v1, v2), max(v1, v2))),
                ((min(u1, v2), max(u1, v2)), (min(v1, u2), max(v1, u2))),
            ]
            for e_new1, e_new2 in candidates:
                if e_new1[0] == e_new1[1] or e_new2[0] == e_new2[1]: continue
                if e_new1 in edge_set or e_new2 in edge_set: continue
                edge_set.discard((u1, v1)); edge_set.discard((u2, v2))
                edge_set.add(e_new1); edge_set.add(e_new2)
                break
        new_ei = set_to_tensor(edge_set, n)

    elif variant == "same_to_hubs":
        # Move same-class edges to hub positions (cross-class edges move away).
        same_non_hub = [(u, v) for u, v in edge_set if not is_cross(u, v) and not is_hub(u, v)]
        cross_hub    = [(u, v) for u, v in edge_set if is_cross(u, v) and is_hub(u, v)]
        n_swap = min(len(same_non_hub), len(cross_hub))
        rng.shuffle(same_non_hub); rng.shuffle(cross_hub)
        for (u1, v1), (u2, v2) in zip(same_non_hub[:n_swap], cross_hub[:n_swap]):
            candidates = [
                ((min(u1, u2), max(u1, u2)), (min(v1, v2), max(v1, v2))),
                ((min(u1, v2), max(u1, v2)), (min(v1, u2), max(v1, u2))),
            ]
            for e_new1, e_new2 in candidates:
                if e_new1[0] == e_new1[1] or e_new2[0] == e_new2[1]: continue
                if e_new1 in edge_set or e_new2 in edge_set: continue
                edge_set.discard((u1, v1)); edge_set.discard((u2, v2))
                edge_set.add(e_new1); edge_set.add(e_new2)
                break
        new_ei = set_to_tensor(edge_set, n)

    elif variant == "label_shuffle":
        # Shuffle class labels → destroy all label-structure signal.
        new_y = torch.tensor(rng.permutation(data.y.numpy()))
        new_ei = data.edge_index.clone()
        d2 = Data(x=data.x, edge_index=new_ei, y=new_y, num_nodes=n,
                  train_mask=data.train_mask, val_mask=data.val_mask,
                  test_mask=data.test_mask)
        return d2

    elif variant == "random_rewire":
        # Double-edge swaps: preserves degree sequence, randomises label-edge alignment.
        edge_set = double_edge_swap(edge_set, n, rng, n_swaps=len(edge_set) * 5)
        new_ei = set_to_tensor(edge_set, n)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return Data(x=data.x, edge_index=new_ei, y=data.y, num_nodes=n,
                train_mask=data.train_mask, val_mask=data.val_mask,
                test_mask=data.test_mask)


# ──────────────────────────────────────────────
# 4. Fingerprint a variant (for verification)
# ──────────────────────────────────────────────

def fingerprint(data):
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    y   = data.y.numpy()
    n   = data.num_nodes
    deg = degree(data.edge_index[1], n).numpy()
    hub_thresh = np.quantile(deg, 0.90)
    hub_nodes  = set(np.where(deg >= hub_thresh)[0].tolist())
    same    = y[src] == y[dst]
    touches = np.isin(src, list(hub_nodes)) | np.isin(dst, list(hub_nodes))
    cross_hub = (~same) & touches
    return {
        "m":              int(data.edge_index.shape[1]),
        "homophily":      round(float(same.mean()), 3),
        "hub_cross_frac": round(float(cross_hub.sum() / max(touches.sum(), 1)), 3),
        "deg_cv":         round(float(deg.std() / max(deg.mean(), 1e-8)), 3),
    }


# ──────────────────────────────────────────────
# 5. Run one (variant, seed) combination
# ──────────────────────────────────────────────

def run_one(variant_data, nf, nc, seed, cfg):
    set_seed(seed)
    device = torch.device("cpu")
    variant_data = variant_data.to(device)

    # TGS
    cfg2 = load_config(BASE_CFG)
    cfg2.seed = seed

    # Patch the dataset into the training loop directly
    # by monkeypatching load_dataset for this call
    import tgs.utils.datasets as _ds
    _orig = _ds.load_dataset
    def _patched(c, d): return variant_data, nf, nc
    _ds.load_dataset = _patched
    try:
        r_tgs = train(cfg2)
    finally:
        _ds.load_dataset = _orig

    tgs_acc = r_tgs["test_acc_at_best_val"]

    # Dense baseline
    b_dense = run_baseline(
        "dense", variant_data, nf, nc, 0.0,
        hidden=64, epochs=300, lr=0.01, weight_decay=5e-4,
        dropout=0.5, seed=seed, device=device,
    )
    # Random baseline at TGS's actual sparsity
    b_rand = run_baseline(
        "random", variant_data, nf, nc, r_tgs["final_sparsity"],
        hidden=64, epochs=300, lr=0.01, weight_decay=5e-4,
        dropout=0.5, seed=seed, device=device,
    )

    return {
        "tgs_acc":      round(tgs_acc, 4),
        "dense_acc":    round(b_dense["best_test_acc"], 4),
        "random_acc":   round(b_rand["best_test_acc"], 4),
        "gap_vs_dense": round(b_dense["best_test_acc"] - tgs_acc, 4),
        "gap_vs_random":round(b_rand["best_test_acc"]  - tgs_acc, 4),
        "sparsity":     round(r_tgs["final_sparsity"], 3),
        "seed":         seed,
    }


# ──────────────────────────────────────────────
# 6. Main sweep
# ──────────────────────────────────────────────

VARIANTS = [
    ("original",         "Control: unmodified Wisconsin"),
    ("cross_to_hubs",    "Cross-class edges → hubs (amplify mechanism)"),
    ("cross_to_non_hubs","Cross-class edges → non-hubs (remove mechanism)"),
    ("same_to_hubs",     "Same-class edges → hubs (flip hub content)"),
    ("label_shuffle",    "Shuffle class labels (destroy label structure)"),
    ("random_rewire",    "Random degree-preserving rewire"),
]

PREDICTIONS = {
    "original":          "TGS wins (gap < 0)",
    "cross_to_hubs":     "TGS advantage INCREASES (gap more negative)",
    "cross_to_non_hubs": "TGS advantage COLLAPSES (gap near 0 or positive)",
    "same_to_hubs":      "TGS advantage DECREASES (useful signal at hubs, not noise)",
    "label_shuffle":     "TGS advantage DISAPPEARS (no label structure to exploit)",
    "random_rewire":     "TGS advantage DECREASES (hub-cross concentration reduced)",
}


def main():
    import statistics as st
    data_base, nf, nc, cfg = load_base()
    rng = np.random.default_rng(0)  # fixed graph-construction seed

    all_results = {}
    out_file = OUT / "results.json"
    if out_file.exists():
        all_results = json.loads(out_file.read_text())

    for variant, description in VARIANTS:
        print(f"\n{'='*60}")
        print(f"Variant: {variant}")
        print(f"  {description}")
        print(f"  Prediction: {PREDICTIONS[variant]}")

        variant_data = build_variant(data_base, nc, variant, rng)
        fp = fingerprint(variant_data)
        print(f"  Fingerprint: m={fp['m']} h={fp['homophily']:.3f} "
              f"hub_cross={fp['hub_cross_frac']:.3f} deg_cv={fp['deg_cv']:.3f}")

        if variant not in all_results:
            all_results[variant] = {
                "description": description,
                "prediction": PREDICTIONS[variant],
                "fingerprint": fp,
                "runs": [],
            }
        done_seeds = {r["seed"] for r in all_results[variant]["runs"]}

        for seed in SEEDS:
            if seed in done_seeds:
                print(f"  seed={seed} [cached]")
                continue
            print(f"  seed={seed} ...", end=" ", flush=True)
            t0 = time.time()
            r = run_one(variant_data, nf, nc, seed, cfg)
            r["time_sec"] = round(time.time() - t0, 1)
            all_results[variant]["runs"].append(r)
            out_file.write_text(json.dumps(all_results, indent=2))
            print(f"TGS={r['tgs_acc']:.4f} Dense={r['dense_acc']:.4f} "
                  f"gap={r['gap_vs_dense']:+.4f} ({r['time_sec']:.0f}s)")

    # ── Summary table ──
    print(f"\n\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Variant':22s} {'hub_cross':>9s} {'TGS':>7s} {'Dense':>7s} "
          f"{'gap(μ)':>8s} {'wins':>6s}  Prediction")
    print("-"*80)
    for variant, _ in VARIANTS:
        d = all_results[variant]
        runs = d["runs"]
        tgs   = [r["tgs_acc"]   for r in runs]
        dense = [r["dense_acc"] for r in runs]
        gaps  = [r["gap_vs_dense"] for r in runs]
        wins  = sum(1 for g in gaps if g < 0)
        hcf   = d["fingerprint"]["hub_cross_frac"]
        verdict = "✓" if (
            (variant == "original"          and st.mean(gaps) < -0.05) or
            (variant == "cross_to_hubs"     and st.mean(gaps) < all_results["original"]["runs"][0]["gap_vs_dense"]) or
            (variant == "cross_to_non_hubs" and st.mean(gaps) > -0.02) or
            (variant == "same_to_hubs"      and st.mean(gaps) > -0.10) or
            (variant == "label_shuffle"     and abs(st.mean(gaps)) < 0.03) or
            (variant == "random_rewire"     and st.mean(gaps) > -0.10)
        ) else "✗"
        print(f"{variant:22s} {hcf:>9.3f} {st.mean(tgs):>7.4f} {st.mean(dense):>7.4f} "
              f"{st.mean(gaps):>+8.4f} {wins:>2d}/{len(gaps)}   {verdict} {PREDICTIONS[variant][:40]}")

    print(f"\nFull results: {out_file}")


if __name__ == "__main__":
    main()
EOF