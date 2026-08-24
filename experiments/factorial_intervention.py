"""
2×2 Factorial Graph Intervention Experiment
=============================================

Addresses the confound in the previous intervention experiment:
changing hub_cross_frac also changed homophily, so we couldn't
cleanly attribute the TGS advantage change to hub localization.

This experiment constructs FOUR graph variants from Wisconsin's
degree sequence with independent control over:

  Factor 1 — Cross-class density:  LOW (h≈0.50) vs HIGH (h≈0.20)
  Factor 2 — Hub concentration:    LOW (hub_cross≈0.20) vs HIGH (hub_cross≈0.80)

Giving cells:
  A: LOW cross  × LOW hub   (neither mechanism)
  B: LOW cross  × HIGH hub  (hub structure alone)
  C: HIGH cross × LOW hub   (cross-class noise, diffuse)
  D: HIGH cross × HIGH hub  (both → Wisconsin regime, TGS should win most)

This lets us decompose the TGS advantage into additive components:
  β_cross = (C + D)/2 - (A + B)/2   (main effect of cross-class noise)
  β_hub   = (B + D)/2 - (A + C)/2   (main effect of hub concentration)
  β_int   = (D - C) - (B - A)       (interaction: the key coefficient)

β_int > 0 and significant means: hub localization and cross-class noise
interact synergistically — TGS's advantage emerges specifically when
both are present, not from either alone.

Repeated across Wisconsin, Texas, Chameleon, Squirrel-2k.

Usage
-----
    PYTHONPATH=. python3 experiments/factorial_intervention.py
"""

import json, time, logging, itertools
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

logging.basicConfig(level=logging.WARNING)
OUT = Path(__file__).parent.parent / "results" / "factorial_intervention"
OUT.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from tgs.utils import load_config, set_seed, load_dataset
from tgs.evaluation.baselines import run_baseline
from scripts.train import train


# ─────────────────────────────────────────────────────────────────────────────
# Graph generation: constrained rewiring to hit target (h, hub_cross)
# ─────────────────────────────────────────────────────────────────────────────

def build_factorial_cell(
    data: Data,
    nc: int,
    target_homophily: float,
    target_hub_cross: float,
    rng: np.random.Generator,
    max_iter: int = 50_000,
) -> Data:
    """
    Construct a graph variant with the same n, degree sequence, and class
    balance as `data`, but rewired to hit (target_homophily, target_hub_cross).

    Strategy: start from the original graph and perform targeted edge swaps
    that move in the direction of the target. Two swap types:

    Type 1 (homophily adjustment):
        Swap (cross,*) ↔ (same,*) without changing hub structure.
        Pick a cross-class edge and a same-class edge; rewire to get
        one cross + one same (no net change) or two same (decrease cross).

    Type 2 (hub_cross adjustment):
        Swap (cross,hub) ↔ (cross,nonhub) to move cross-class edges
        between hub and non-hub positions WITHOUT changing homophily.

    Both types use double-edge swaps that preserve the degree sequence.
    """
    src_np = data.edge_index[0].numpy()
    dst_np = data.edge_index[1].numpy()
    y_np = data.y.numpy()
    n = data.num_nodes

    # Identify hub nodes from ORIGINAL degree sequence (preserved throughout)
    deg = degree(data.edge_index[1], n).numpy()
    hub_thresh = np.quantile(deg, 0.90)
    hub_nodes = set(np.where(deg >= hub_thresh)[0].tolist())

    # Work in undirected edge set
    edge_set = {(min(int(s), int(d)), max(int(s), int(d)))
                for s, d in zip(src_np, dst_np)}

    def classify(u, v):
        sc = y_np[u] == y_np[v]
        ih = u in hub_nodes or v in hub_nodes
        return ('same' if sc else 'cross', 'hub' if ih else 'nonhub')

    def metrics(es):
        total = len(es)
        cross = sum(1 for u, v in es if y_np[u] != y_np[v])
        hub_e = [(u, v) for u, v in es if u in hub_nodes or v in hub_nodes]
        cross_hub = sum(1 for u, v in hub_e if y_np[u] != y_np[v])
        h = 1.0 - cross / max(total, 1)
        hcf = cross_hub / max(len(hub_e), 1)
        return h, hcf

    def try_swap_homophily(es, edge_list, increase_same: bool, rng):
        """Swap to move homophily toward target: increase_same=True → more same-class edges."""
        cross_edges = [(u, v) for u, v in edge_list if y_np[u] != y_np[v]]
        same_edges  = [(u, v) for u, v in edge_list if y_np[u] == y_np[v]]
        if not cross_edges or not same_edges:
            return False
        if increase_same:
            e1 = cross_edges[rng.integers(len(cross_edges))]
            e2 = same_edges[rng.integers(len(same_edges))]
        else:
            e1 = same_edges[rng.integers(len(same_edges))]
            e2 = cross_edges[rng.integers(len(cross_edges))]
        u1, v1 = e1; u2, v2 = e2
        for new1, new2 in [
            ((min(u1,u2), max(u1,u2)), (min(v1,v2), max(v1,v2))),
            ((min(u1,v2), max(u1,v2)), (min(v1,u2), max(v1,u2))),
        ]:
            if new1[0]==new1[1] or new2[0]==new2[1]: continue
            if new1 in es or new2 in es: continue
            es.discard(e1); es.discard(e2)
            es.add(new1); es.add(new2)
            return True
        return False

    def try_swap_hub_cross(es, edge_list, increase_hub_cross: bool, rng):
        """Move cross-class edges between hub and non-hub without changing homophily."""
        if increase_hub_cross:
            src_pool = [(u, v) for u, v in edge_list
                        if y_np[u] != y_np[v] and u not in hub_nodes and v not in hub_nodes]
            tgt_pool = [(u, v) for u, v in edge_list
                        if y_np[u] == y_np[v] and (u in hub_nodes or v in hub_nodes)]
        else:
            src_pool = [(u, v) for u, v in edge_list
                        if y_np[u] != y_np[v] and (u in hub_nodes or v in hub_nodes)]
            tgt_pool = [(u, v) for u, v in edge_list
                        if y_np[u] == y_np[v] and u not in hub_nodes and v not in hub_nodes]
        if not src_pool or not tgt_pool:
            return False
        e1 = src_pool[rng.integers(len(src_pool))]
        e2 = tgt_pool[rng.integers(len(tgt_pool))]
        u1, v1 = e1; u2, v2 = e2
        for new1, new2 in [
            ((min(u1,u2), max(u1,u2)), (min(v1,v2), max(v1,v2))),
            ((min(u1,v2), max(u1,v2)), (min(v1,u2), max(v1,u2))),
        ]:
            if new1[0]==new1[1] or new2[0]==new2[1]: continue
            if new1 in es or new2 in es: continue
            es.discard(e1); es.discard(e2)
            es.add(new1); es.add(new2)
            return True
        return False

    # Iterative targeted rewiring
    for iteration in range(max_iter):
        edge_list = list(edge_set)
        h, hcf = metrics(edge_set)
        h_err   = h   - target_homophily
        hcf_err = hcf - target_hub_cross

        # Prioritise whichever dimension is furthest from target
        if abs(h_err) >= abs(hcf_err):
            # Adjust homophily
            if h_err > 0.01:    # too homophilous → need more cross
                try_swap_homophily(edge_set, edge_list, increase_same=False, rng=rng)
            elif h_err < -0.01: # too heterophilous → need more same
                try_swap_homophily(edge_set, edge_list, increase_same=True,  rng=rng)
        else:
            # Adjust hub_cross_frac
            if hcf_err < -0.05:  # hub_cross too low → move cross to hubs
                try_swap_hub_cross(edge_set, edge_list, increase_hub_cross=True,  rng=rng)
            elif hcf_err > 0.05: # hub_cross too high → move cross away
                try_swap_hub_cross(edge_set, edge_list, increase_hub_cross=False, rng=rng)

        if iteration % 10_000 == 9_999:
            h, hcf = metrics(edge_set)
            logging.debug(f"  iter={iteration+1} h={h:.3f}(tgt={target_homophily:.2f}) hcf={hcf:.3f}(tgt={target_hub_cross:.2f})")

    # Build final edge_index
    us = [u for u, v in edge_set] + [v for u, v in edge_set]
    vs = [v for u, v in edge_set] + [u for u, v in edge_set]
    new_ei = torch.tensor([us, vs], dtype=torch.long)

    h_final, hcf_final = metrics(edge_set)
    return Data(
        x=data.x, edge_index=new_ei, y=data.y, num_nodes=data.num_nodes,
        train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
        _achieved_h=h_final, _achieved_hcf=hcf_final,
    ), h_final, hcf_final


def fingerprint_cell(data, hub_nodes):
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    y   = data.y.numpy()
    n   = data.num_nodes
    same = y[src] == y[dst]
    touches = np.array([s in hub_nodes or d in hub_nodes for s, d in zip(src, dst)])
    h   = float(same.mean())
    hcf = float((~same & touches).sum() / max(touches.sum(), 1))
    return h, hcf


# ─────────────────────────────────────────────────────────────────────────────
# Run TGS + baselines on a given graph
# ─────────────────────────────────────────────────────────────────────────────

def run_methods(variant_data, nf, nc, seed, cfg_name):
    set_seed(seed)
    device = torch.device("cpu")
    variant_data = variant_data.to(device)

    import tgs.utils.datasets as _ds
    _orig = _ds.load_dataset
    def _patched(c, d): return variant_data, nf, nc
    _ds.load_dataset = _patched
    cfg = load_config(cfg_name)
    cfg.seed = seed
    try:
        r_tgs = train(cfg)
    finally:
        _ds.load_dataset = _orig

    tgs_acc = r_tgs["test_acc_at_best_val"]
    sparsity = r_tgs["final_sparsity"]

    b_dense = run_baseline("dense", variant_data, nf, nc, 0.0,
        hidden=64, epochs=300, lr=0.01, weight_decay=5e-4,
        dropout=0.5, seed=seed, device=device)
    b_rand = run_baseline("random", variant_data, nf, nc, sparsity,
        hidden=64, epochs=300, lr=0.01, weight_decay=5e-4,
        dropout=0.5, seed=seed, device=device)

    return {
        "tgs":    round(tgs_acc, 4),
        "dense":  round(b_dense["best_test_acc"], 4),
        "random": round(b_rand["best_test_acc"], 4),
        "gap":    round(b_dense["best_test_acc"] - tgs_acc, 4),
        "sparsity": round(sparsity, 3),
        "seed":   seed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "Wisconsin":  ("configs/wisconsin_gcn.yaml",  0.90),
    "Texas":      ("configs/texas_gcn.yaml",      0.90),
    "Chameleon":  ("configs/chameleon_gcn.yaml",  0.90),
    "Squirrel-2k":("configs/squirrel_2k_gcn.yaml",0.90),
}

# 2×2 cell targets: (h_target, hub_cross_target, label)
CELLS = {
    "A": (0.50, 0.20, "LOW cross  × LOW hub"),
    "B": (0.50, 0.80, "LOW cross  × HIGH hub"),
    "C": (0.20, 0.20, "HIGH cross × LOW hub"),
    "D": (0.20, 0.80, "HIGH cross × HIGH hub"),
}

SEEDS = [42, 43, 44, 45, 46]


def main():
    import statistics as st

    all_results = {}
    out_file = OUT / "results.json"
    if out_file.exists():
        all_results = json.loads(out_file.read_text())

    for ds_name, (cfg_name, hub_pct) in DATASET_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        cfg = load_config(cfg_name)
        data_base, nf, nc = load_dataset(cfg, torch.device("cpu"))
        deg = degree(data_base.edge_index[1], data_base.num_nodes).numpy()
        hub_thresh = np.quantile(deg, hub_pct)
        hub_nodes = set(np.where(deg >= hub_thresh)[0].tolist())
        h0, hcf0 = fingerprint_cell(data_base, hub_nodes)
        print(f"  Base: n={data_base.num_nodes} m={data_base.edge_index.shape[1]} "
              f"h={h0:.3f} hub_cross={hcf0:.3f}")

        if ds_name not in all_results:
            all_results[ds_name] = {}

        rng = np.random.default_rng(0)  # fixed graph-construction seed

        for cell_id, (h_tgt, hcf_tgt, cell_label) in CELLS.items():
            print(f"\n  Cell {cell_id}: {cell_label}")
            print(f"    Target: h={h_tgt:.2f} hub_cross={hcf_tgt:.2f}")

            # Build the graph variant (fixed construction seed)
            t0 = time.time()
            variant, h_got, hcf_got = build_factorial_cell(
                data_base, nc, h_tgt, hcf_tgt, rng)
            print(f"    Achieved: h={h_got:.3f} hub_cross={hcf_got:.3f} "
                  f"({time.time()-t0:.0f}s to build)")

            cell_key = f"cell_{cell_id}"
            if cell_key not in all_results[ds_name]:
                all_results[ds_name][cell_key] = {
                    "label": cell_label,
                    "target": {"h": h_tgt, "hub_cross": hcf_tgt},
                    "achieved": {"h": round(h_got, 3), "hub_cross": round(hcf_got, 3)},
                    "runs": [],
                }
            done_seeds = {r["seed"] for r in all_results[ds_name][cell_key]["runs"]}

            for seed in SEEDS:
                if seed in done_seeds:
                    print(f"    seed={seed} [cached]")
                    continue
                print(f"    seed={seed} ...", end=" ", flush=True)
                t0 = time.time()
                r = run_methods(variant, nf, nc, seed, cfg_name)
                r["time_sec"] = round(time.time() - t0, 1)
                all_results[ds_name][cell_key]["runs"].append(r)
                out_file.write_text(json.dumps(all_results, indent=2))
                print(f"TGS={r['tgs']:.4f} Dense={r['dense']:.4f} gap={r['gap']:+.4f} ({r['time_sec']:.0f}s)")

        # Compute factorial decomposition for this dataset
        print(f"\n  Factorial decomposition for {ds_name}:")
        cell_gaps = {}
        for cell_id in ["A", "B", "C", "D"]:
            runs = all_results[ds_name][f"cell_{cell_id}"]["runs"]
            if runs:
                cell_gaps[cell_id] = st.mean(r["gap"] for r in runs)
                h_got  = all_results[ds_name][f"cell_{cell_id}"]["achieved"]["h"]
                hcf_got = all_results[ds_name][f"cell_{cell_id}"]["achieved"]["hub_cross"]
                print(f"    Cell {cell_id}: h={h_got:.3f} hcf={hcf_got:.3f}  gap={cell_gaps[cell_id]:+.4f}")

        if len(cell_gaps) == 4:
            A, B, C, D = cell_gaps["A"], cell_gaps["B"], cell_gaps["C"], cell_gaps["D"]
            beta_cross = ((C + D) - (A + B)) / 2
            beta_hub   = ((B + D) - (A + C)) / 2
            beta_int   = (D - C) - (B - A)
            print(f"\n    β_cross (main effect)  = {beta_cross:+.4f}  (negative = cross-class noise hurts)")
            print(f"    β_hub   (main effect)  = {beta_hub:+.4f}  (negative = hubs hurt dense more)")
            print(f"    β_int   (INTERACTION)  = {beta_int:+.4f}  ← KEY: negative = synergy")
            print(f"    Interpretation: TGS advantage from hub×cross = {-beta_int:.4f} accuracy points")
            all_results[ds_name]["factorial"] = {
                "beta_cross": round(beta_cross, 4),
                "beta_hub":   round(beta_hub,   4),
                "beta_int":   round(beta_int,   4),
            }
            out_file.write_text(json.dumps(all_results, indent=2))

    # Final summary across datasets
    print(f"\n\n{'='*70}")
    print("CROSS-DATASET FACTORIAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':14s} {'β_cross':>9s} {'β_hub':>8s} {'β_int':>8s}  Synergy confirmed?")
    print("-"*60)
    for ds_name in DATASET_CONFIGS:
        if "factorial" in all_results.get(ds_name, {}):
            f = all_results[ds_name]["factorial"]
            ok = "✓ YES" if f["beta_int"] < -0.02 else "~ WEAK" if f["beta_int"] < 0 else "✗ NO"
            print(f"{ds_name:14s} {f['beta_cross']:>+9.4f} {f['beta_hub']:>+8.4f} "
                  f"{f['beta_int']:>+8.4f}  {ok}")

    print(f"\nFull results: {out_file}")


if __name__ == "__main__":
    main()
