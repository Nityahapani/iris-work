"""
Sparsity sweep: run TGS and all static baselines at multiple sparsity
levels on a single dataset, multiple seeds. Produces a results JSON
suitable for plotting the full accuracy-vs-sparsity tradeoff curve.

Usage:
    PYTHONPATH=. python3 scripts/sparsity_sweep.py --config configs/wisconsin_gcn.yaml \
        --sparsities 0.0 0.2 0.35 0.5 0.65 0.8 --seeds 42 43 44 45 46
"""

import argparse, json, time, logging
from pathlib import Path
import torch
logging.basicConfig(level=logging.WARNING)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tgs.utils import load_config, set_seed, load_dataset
from tgs.evaluation.baselines import run_baseline
from scripts.train import train

OUT_DIR = Path(__file__).parent.parent / "results"

def run_tgs_at_sparsity(cfg, target_sparsity: float, seed: int) -> dict:
    """Run TGS with a given max_sparsity target."""
    cfg.seed = seed
    cfg.max_sparsity = target_sparsity
    if target_sparsity == 0.0:
        # No retirement at all — equivalent to dense
        cfg.max_sparsity = 0.0
        cfg.warmup_steps = cfg.epochs + 1  # warmup longer than training = no retirement
    t0 = time.time()
    r = train(cfg)
    return {
        "method": "tgs",
        "target_sparsity": target_sparsity,
        "actual_sparsity": r["final_sparsity"],
        "test_acc": r["test_acc_at_best_val"],
        "flops_reduction": r["flops"]["flops_reduction"],
        "time_sec": round(time.time() - t0, 1),
        "seed": seed,
    }


def run_baseline_at_sparsity(cfg, name: str, target_sparsity: float, seed: int) -> dict:
    cfg2 = load_config(cfg._path)
    cfg2.seed = seed
    device = torch.device(cfg2.device)
    data, nf, nc = load_dataset(cfg2, device)
    t0 = time.time()
    b = run_baseline(
        name=name, data=data, num_features=nf, num_classes=nc,
        target_sparsity=target_sparsity,
        hidden=cfg2.hidden_channels, epochs=cfg2.epochs,
        lr=cfg2.lr, weight_decay=cfg2.weight_decay,
        dropout=cfg2.dropout, seed=seed, device=device,
    )
    return {
        "method": name,
        "target_sparsity": target_sparsity,
        "actual_sparsity": b.get("actual_sparsity", target_sparsity),
        "test_acc": b["best_test_acc"],
        "time_sec": round(time.time() - t0, 1),
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sparsities", nargs="+", type=float,
                        default=[0.0, 0.20, 0.35, 0.50, 0.65, 0.80])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--baselines", nargs="+",
                        default=["dense", "random", "local_degree", "eff_resistance"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg._path = args.config  # stash for reloading

    out_path = OUT_DIR / f"sparsity_sweep_{cfg.dataset.lower()}.json"
    # Load existing partial results
    results = json.loads(out_path.read_text()) if out_path.exists() else []

    def already_done(method, sparsity, seed):
        return any(
            r["method"] == method and
            abs(r["target_sparsity"] - sparsity) < 1e-6 and
            r["seed"] == seed
            for r in results
        )

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    total = (len(args.sparsities) * (1 + len(args.baselines)) * len(args.seeds))
    done = 0

    for sparsity in args.sparsities:
        for seed in args.seeds:
            # TGS
            if not already_done("tgs", sparsity, seed):
                print(f"[TGS] sparsity={sparsity} seed={seed}", flush=True)
                r = run_tgs_at_sparsity(load_config(args.config), sparsity, seed)
                results.append(r)
                save()
                print(f"  -> acc={r['test_acc']:.4f} actual_sparsity={r['actual_sparsity']:.3f}", flush=True)
            done += 1

            # Baselines (skip dense at non-zero sparsity; dense baseline = sparsity 0.0)
            for bname in args.baselines:
                if bname == "dense" and sparsity > 0.0:
                    continue  # dense doesn't make sense at non-zero sparsity target
                if bname != "dense" and sparsity == 0.0:
                    continue  # no-sparsity baselines reduce to dense
                if not already_done(bname, sparsity, seed):
                    print(f"[{bname}] sparsity={sparsity} seed={seed}", flush=True)
                    r = run_baseline_at_sparsity(cfg, bname, sparsity, seed)
                    results.append(r)
                    save()
                    print(f"  -> acc={r['test_acc']:.4f}", flush=True)
                done += 1

    print(f"\nDone. {len(results)} results saved to {out_path}")


if __name__ == "__main__":
    main()
