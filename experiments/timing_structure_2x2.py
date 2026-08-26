"""
2×2 Timing × Structure Intervention — Centerpiece Experiment
=============================================================

The cleanest possible causal test of TGS's mechanism.

Hypothesis: TGS's advantage requires BOTH:
  (1) Allowing representation learning before retirement (timing)
  (2) Graph structure where cross-class edges concentrate at hubs (structure)

Neither factor alone should produce the effect. Only their interaction.

Design
------
Everything fixed across all cells:
  - Same base graph (Wisconsin, Cell D from factorial: h≈0.20, hcf≈0.83)
  - Same final sparsity (65%)
  - Same set of edges eligible for retirement
  - Same model (2-layer GCN, hidden=64)
  - Same training budget (300 epochs)
  - Same seeds (42-46)
  - Same optimizer, lr, weight_decay

Vary ONLY:
  Factor T — Retirement timing:
    EARLY  (warmup=0):  retire edges immediately, before any learning
    LATE   (warmup=40): standard TGS, retire after representation learning

  Factor S — Graph structure:
    BENIGN (Cell A from factorial: h≈0.50, hcf≈0.20): low cross, low hub
    HOSTILE(Cell D from factorial: h≈0.20, hcf≈0.83): high cross, high hub

Predicted 2×2 outcome (gap = Dense_acc - TGS_acc, negative = TGS wins):

               BENIGN structure    HOSTILE structure
  EARLY timing    ≈ 0 (no benefit)  ≈ 0 (retires blind, can't exploit signal)
  LATE  timing    ≈ 0 (no benefit)  LARGE negative (full TGS advantage)

The interaction β_TxS is the key number. If large and negative:
TGS advantage emerges specifically from LATE retirement on HOSTILE structure —
exactly the mechanism story.

This ties together three prior experiments:
  - timing_sweep_cora.json: warmup=0 → warmup=40 shows +16pp on Cora
  - factorial_intervention: β_int=-0.067, confirming structure interaction
  - temporal_order_ablation: order matters, static is much worse

Usage
-----
    PYTHONPATH=. python3 experiments/timing_structure_2x2.py
"""

import json, time, logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.WARNING)
OUT = Path(__file__).parent.parent / "results" / "timing_structure_2x2"
OUT.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tgs.utils import load_config, set_seed, load_dataset
from tgs.evaluation.baselines import run_baseline
from experiments.factorial_intervention import (
    build_factorial_cell, fingerprint_cell, CELLS
)
from torch_geometric.utils import degree


# ─────────────────────────────────────────────────────────────────────────────
# The two structural cells we need
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURE_CELLS = {
    "benign":  CELLS["A"],   # (h=0.50, hcf=0.20, "LOW cross × LOW hub")
    "hostile": CELLS["D"],   # (h=0.20, hcf=0.80, "HIGH cross × HIGH hub")
}

TIMING_CONDITIONS = {
    "early": 0,    # warmup=0: retire immediately
    "late":  40,   # warmup=40: standard TGS (representation learning first)
}

SEEDS = [42, 43, 44, 45, 46]
BASE_CFG = "configs/wisconsin_gcn.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Run one (structure, timing, seed) cell
# ─────────────────────────────────────────────────────────────────────────────

def run_cell(variant_data, nf: int, nc: int, seed: int, warmup: int) -> dict:
    """
    Run TGS with a specific warmup on variant_data, plus dense baseline.
    Returns accuracy and gap for this (structure, timing, seed) combination.
    """
    from tgs.core import TemporalGraph, EdgeManager
    from tgs.core.influence import GradientNormEstimator
    from tgs.models import TemporalGCN
    from tgs.schedulers import AdaptiveRetirementScheduler
    from tgs.evaluation import Evaluator
    from tgs.evaluation.flops import FLOPsCounter
    import torch.nn.functional as F

    set_seed(seed)
    device = torch.device("cpu")
    data = variant_data.to(device)

    cfg = load_config(BASE_CFG)
    cfg.seed = seed
    cfg.warmup_steps = warmup

    m0 = data.edge_index.shape[1]
    tg = TemporalGraph(data.edge_index, data.num_nodes, device=device)
    influence_est = GradientNormEstimator(
        m0, device, edge_index=data.edge_index,
        num_nodes=data.num_nodes, ema_decay=cfg.ema_decay,
    )
    model = TemporalGCN(nf, cfg.hidden_channels, nc,
                        cfg.num_layers, cfg.dropout).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [influence_est.edge_weights],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = AdaptiveRetirementScheduler(
        temporal_graph=tg,
        epsilon_max=cfg.epsilon_max, epsilon_min=cfg.epsilon_min,
        anneal_steps=cfg.anneal_steps, schedule=cfg.anneal_schedule,
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

    # Dense baseline (same graph, no sparsification)
    b_dense = run_baseline(
        "dense", data, nf, nc, 0.0,
        hidden=cfg.hidden_channels, epochs=cfg.epochs,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        dropout=cfg.dropout, seed=seed, device=device,
    )

    return {
        "tgs_acc":    round(tgs_acc, 4),
        "dense_acc":  round(b_dense["best_test_acc"], 4),
        "gap":        round(b_dense["best_test_acc"] - tgs_acc, 4),
        "sparsity":   round(tg.sparsity, 3),
        "warmup":     warmup,
        "seed":       seed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import statistics as st

    cfg = load_config(BASE_CFG)
    data_base, nf, nc = load_dataset(cfg, torch.device("cpu"))
    deg = degree(data_base.edge_index[1], data_base.num_nodes).numpy()
    hub_nodes = set(np.where(deg >= np.quantile(deg, 0.90))[0].tolist())

    out_file = OUT / "results.json"
    all_results = json.loads(out_file.read_text()) if out_file.exists() else {}

    rng = np.random.default_rng(42)

    # Build the two structural variants once, with fixed graph seed
    print("Building structural variants...")
    variants = {}
    for struct_name, (h_tgt, hcf_tgt, label) in STRUCTURE_CELLS.items():
        if struct_name in all_results and "fingerprint" in all_results[struct_name]:
            print(f"  {struct_name}: loaded from cache")
            # Rebuild deterministically (same rng seed 42)
            var, h_got, hcf_got = build_factorial_cell(
                data_base, nc, h_tgt, hcf_tgt, np.random.default_rng(42))
        else:
            var, h_got, hcf_got = build_factorial_cell(
                data_base, nc, h_tgt, hcf_tgt, np.random.default_rng(42))
        variants[struct_name] = var
        h_v, hcf_v = fingerprint_cell(var, hub_nodes)
        print(f"  {struct_name}: h={h_v:.3f} hub_cross={hcf_v:.3f} (target h={h_tgt:.2f} hcf={hcf_tgt:.2f})")
        if struct_name not in all_results:
            all_results[struct_name] = {
                "label": label,
                "target_h": h_tgt, "target_hcf": hcf_tgt,
                "achieved_h": round(h_v, 3), "achieved_hcf": round(hcf_v, 3),
                "fingerprint": {"h": round(h_v, 3), "hcf": round(hcf_v, 3)},
            }
        for timing_name in TIMING_CONDITIONS:
            if timing_name not in all_results[struct_name]:
                all_results[struct_name][timing_name] = {"runs": []}

    # Run all 2×2×5 = 20 combinations
    print("\nRunning 2×2 factorial (structure × timing, 5 seeds each)...")
    for struct_name, variant in variants.items():
        for timing_name, warmup in TIMING_CONDITIONS.items():
            print(f"\n  [{struct_name}] × [{timing_name}] (warmup={warmup})")
            cell = all_results[struct_name][timing_name]
            done_seeds = {r["seed"] for r in cell["runs"]}
            for seed in SEEDS:
                if seed in done_seeds:
                    print(f"    seed={seed} [cached]")
                    continue
                print(f"    seed={seed}...", end=" ", flush=True)
                t0 = time.time()
                r = run_cell(variant, nf, nc, seed, warmup)
                r["time_sec"] = round(time.time() - t0, 1)
                cell["runs"].append(r)
                out_file.write_text(json.dumps(all_results, indent=2))
                print(f"TGS={r['tgs_acc']:.4f} Dense={r['dense_acc']:.4f} "
                      f"gap={r['gap']:+.4f} ({r['time_sec']:.0f}s)")

    # ── Results table ──────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("2×2 TIMING × STRUCTURE RESULTS")
    print(f"{'='*70}")
    print(f"gap = Dense acc − TGS acc  (negative = TGS wins)")
    print()

    cell_gaps = {}
    for struct_name in ["benign", "hostile"]:
        for timing_name in ["early", "late"]:
            runs = all_results[struct_name][timing_name]["runs"]
            gaps = [r["gap"] for r in runs]
            gm = st.mean(gaps)
            gs = st.stdev(gaps) if len(gaps) > 1 else 0
            wins = sum(1 for g in gaps if g < 0)
            cell_gaps[(struct_name, timing_name)] = gm
            label = f"{struct_name.upper():7s} × {timing_name.upper():5s}"
            print(f"  {label}  gap={gm:+.4f} ± {gs:.4f}  wins={wins}/{len(gaps)}")

    # Factorial decomposition
    E_B = cell_gaps[("benign",  "early")]
    L_B = cell_gaps[("benign",  "late")]
    E_H = cell_gaps[("hostile", "early")]
    L_H = cell_gaps[("hostile", "late")]

    beta_T   = ((L_B + L_H) - (E_B + E_H)) / 2
    beta_S   = ((E_H + L_H) - (E_B + L_B)) / 2
    beta_TxS = (L_H - E_H) - (L_B - E_B)

    print(f"\n  Factorial decomposition:")
    print(f"    β_T   (timing main effect)    = {beta_T:+.4f}")
    print(f"    β_S   (structure main effect) = {beta_S:+.4f}")
    print(f"    β_TxS (INTERACTION)           = {beta_TxS:+.4f}  ← KEY")
    print()
    print(f"  Predicted table:")
    print(f"                   BENIGN     HOSTILE")
    print(f"    EARLY timing   {E_B:+.4f}   {E_H:+.4f}")
    print(f"    LATE  timing   {L_B:+.4f}   {L_H:+.4f}")
    print()
    if beta_TxS < -0.03:
        print(f"  ✓ INTERACTION CONFIRMED (β_TxS={beta_TxS:+.4f})")
        print(f"    TGS advantage requires BOTH late timing AND hostile structure.")
        print(f"    Neither alone produces the effect.")
    else:
        print(f"  ~ Interaction weak or unexpected (β_TxS={beta_TxS:+.4f})")

    # Save decomposition
    all_results["_decomposition"] = {
        "beta_T": round(beta_T, 4),
        "beta_S": round(beta_S, 4),
        "beta_TxS": round(beta_TxS, 4),
        "cell_gaps": {f"{s}_{t}": round(v, 4)
                      for (s, t), v in cell_gaps.items()},
    }
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
