"""
compare_baselines.py
Runs TGS and all static baselines at matched sparsity levels.
Outputs a comparison table and saves results/baselines_comparison.json.

Usage:
    python experiments/compare_baselines.py
"""

import sys, json, logging
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from tgs.core.temporal_graph import TemporalGraph
from tgs.core.influence import GradientNormEstimator
from tgs.models.gcn import TemporalGCN
from tgs.schedulers.adaptive_scheduler import AdaptiveRetirementScheduler
from tgs.evaluation.metrics import Evaluator
from tgs.evaluation.flops import FLOPsCounter
from tgs.evaluation.baselines import run_baseline
from tgs.utils.reproducibility import set_seed

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EPOCHS   = 300
SEED     = 42
DEVICE   = torch.device("cpu")
DATASETS = ["Cora", "CiteSeer"]

# Sparsity levels to test baselines at (to bracket TGS final sparsity)
SPARSITY_LEVELS = [0.10, 0.20, 0.30, 0.40]


def run_tgs(data, num_features, num_classes, dataset_name):
    set_seed(SEED)
    m0 = data.edge_index.shape[1]
    tg = TemporalGraph(data.edge_index, data.num_nodes, device=DEVICE)
    influence_est = GradientNormEstimator(m0, DEVICE)
    model = TemporalGCN(num_features, 64, num_classes, num_layers=2, dropout=0.5).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [influence_est.edge_weights],
        lr=0.01, weight_decay=5e-4
    )
    scheduler = AdaptiveRetirementScheduler(
        tg, epsilon_max=5e-3, epsilon_min=1e-5,
        anneal_steps=100, warmup_steps=40,
        max_retire_frac=0.02, max_sparsity=0.50, retire_every=5
    )
    evaluator = Evaluator(num_classes)
    flops = FLOPsCounter(m0, 2, 64)
    best_val = best_test = 0.0

    for epoch in range(EPOCHS):
        model.train()
        active_mask = tg.active_mask
        logits = model(data.x, tg.edge_index, influence_est.edge_weights[active_mask])
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad(); loss.backward()
        influence_est.update_influence(active_mask)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            el = model(data.x, tg.edge_index)
        metrics = evaluator.update(el, data.y, data.train_mask, data.val_mask, data.test_mask,
                                    tg.sparsity, epoch, scheduler.cumulative_distortion_bound)
        scheduler.update_val_acc(metrics["val_acc"])
        scheduler.step(influence_est.influence_scores(active_mask))
        flops.record_step(tg.mt)
        tg.step()

        if metrics["val_acc"] > best_val:
            best_val = metrics["val_acc"]
            best_test = metrics["test_acc"]

    rs = scheduler.summary()
    return {
        "method": "TGS (ours)",
        "dataset": dataset_name,
        "best_val_acc": best_val,
        "best_test_acc": best_test,
        "final_sparsity": tg.sparsity,
        "flops_reduction": flops.summary()["flops_reduction"],
        "distortion_bound": scheduler.cumulative_distortion_bound,
        "edges_retired": rs["total_retired"],
        "m0": m0,
    }


def main():
    all_results = []

    for dataset_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        dataset = Planetoid(root="./data", name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0].to(DEVICE)

        # --- TGS ---
        print(f"  Running TGS...", end=" ", flush=True)
        tgs_result = run_tgs(data, dataset.num_features, dataset.num_classes, dataset_name)
        tgs_result["dataset"] = dataset_name
        all_results.append(tgs_result)
        print(f"test={tgs_result['best_test_acc']:.4f} | sparsity={tgs_result['final_sparsity']:.3f}")

        # --- Static baselines at multiple sparsity levels ---
        for sp in SPARSITY_LEVELS:
            for baseline in ["dense", "random", "local_degree", "eff_resistance"]:
                actual_sp = 0.0 if baseline == "dense" else sp
                print(f"  Running {baseline:<16} sp={actual_sp:.2f}...", end=" ", flush=True)
                r = run_baseline(
                    name=baseline,
                    data=data,
                    num_features=dataset.num_features,
                    num_classes=dataset.num_classes,
                    target_sparsity=actual_sp,
                    epochs=EPOCHS, seed=SEED, device=DEVICE,
                )
                r["dataset"] = dataset_name
                r["method"] = {
                    "dense": "Dense GCN",
                    "random": "Random",
                    "local_degree": "Local Degree",
                    "eff_resistance": "Eff. Resistance",
                }[baseline]
                all_results.append(r)
                print(f"test={r['best_test_acc']:.4f}")

            # Only run dense once
            break  # remove this if you want dense at every sparsity level (it's the same)

    # --- Print comparison table ---
    print(f"\n{'='*70}")
    print(f"COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Dataset':<10} {'Test Acc':>9} {'Sparsity':>9} {'FLOPs↓':>8}")
    print(f"{'-'*70}")

    for r in all_results:
        sparsity = r.get("final_sparsity", r.get("actual_sparsity", 0.0))
        flops_red = r.get("flops_reduction", sparsity)  # for static: FLOPs reduction ≈ sparsity
        test_acc = r.get("best_test_acc", 0.0)
        dataset = r.get("dataset", "")
        method = r.get("method", r.get("baseline", "?"))

        marker = " ◄" if method == "TGS (ours)" else ""
        print(f"{method:<20} {dataset:<10} {test_acc:>9.4f} {sparsity:>9.3f} {flops_red:>8.3f}{marker}")

    # Save
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/baselines_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/baselines_comparison.json")


if __name__ == "__main__":
    main()
