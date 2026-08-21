"""Run TGS + dense baseline across multiple seeds for one dataset config."""
import json, sys, time, logging
from pathlib import Path
import torch
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from tgs.utils import load_config, load_dataset
from tgs.evaluation.baselines import run_baseline
from scripts.train import train

cfg_name = sys.argv[1]
seeds = [42, 43, 44, 45, 46]

cfg_path = Path(__file__).parent.parent / "configs" / cfg_name
results = {"config": cfg_name, "seeds": {}}

for seed in seeds:
    cfg = load_config(str(cfg_path))
    cfg.seed = seed
    device = torch.device(cfg.device)

    t0 = time.time()
    r = train(cfg)
    tgs_acc = r["test_acc_at_best_val"]
    tgs_sparsity = r["final_sparsity"]
    tgs_time = time.time() - t0

    cfg2 = load_config(str(cfg_path))
    cfg2.seed = seed
    data, nf, nc = load_dataset(cfg2, device)
    b = run_baseline(
        name="dense", data=data, num_features=nf, num_classes=nc, target_sparsity=0.0,
        hidden=cfg2.hidden_channels, epochs=cfg2.epochs, lr=cfg2.lr,
        weight_decay=cfg2.weight_decay, dropout=cfg2.dropout, seed=seed, device=device,
    )
    dense_acc = b["best_test_acc"]

    results["seeds"][seed] = {
        "tgs_acc": tgs_acc, "tgs_sparsity": tgs_sparsity,
        "dense_acc": dense_acc, "gap": dense_acc - tgs_acc,
        "time_sec": round(time.time() - t0, 1),
    }
    print(f"seed={seed} TGS={tgs_acc:.4f} Dense={dense_acc:.4f} gap={dense_acc-tgs_acc:+.4f}", flush=True)

out_path = Path(__file__).parent.parent / "results" / f"multiseed_{cfg_name.replace('.yaml','')}.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"Saved to {out_path}")
