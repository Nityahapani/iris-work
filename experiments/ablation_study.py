"""
ablation_study.py
Systematic ablation over TGS design choices on Cora.

Ablations:
  A1 — Epsilon schedule: cosine vs linear vs step
  A2 — Warmup length: 20, 40, 60, 80 steps
  A3 — Retirement frequency: every 2, 5, 10, 20 steps
  A4 — Max sparsity ceiling: 0.30, 0.40, 0.50, 0.60

Usage:
    python experiments/ablation_study.py
"""

import sys, json, os, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

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
from tgs.utils.reproducibility import set_seed

DEVICE = torch.device("cpu")
SEED   = 42
EPOCHS = 300


def run_config(data, num_features, num_classes,
               schedule="cosine", warmup=40, retire_every=5, max_sparsity=0.50):
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
        anneal_steps=100, schedule=schedule,
        warmup_steps=warmup, max_retire_frac=0.02,
        max_sparsity=max_sparsity, retire_every=retire_every
    )
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
        preds = el.argmax(dim=-1)
        val_acc  = (preds[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        scheduler.update_val_acc(val_acc)
        scheduler.step(influence_est.influence_scores(active_mask))
        flops.record_step(tg.mt)
        tg.step()

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc

    rs = scheduler.summary()
    return {
        "best_test_acc": best_test,
        "best_val_acc": best_val,
        "final_sparsity": tg.sparsity,
        "flops_reduction": flops.summary()["flops_reduction"],
        "edges_retired": rs["total_retired"],
        "distortion_bound": scheduler.cumulative_distortion_bound,
    }


def main():
    dataset = Planetoid(root="./data", name="Cora", transform=NormalizeFeatures())
    data = dataset[0].to(DEVICE)
    nf, nc = dataset.num_features, dataset.num_classes

    all_results = {}

    # ------------------------------------------------------------------
    # A1: Epsilon schedule
    # ------------------------------------------------------------------
    print("\nA1: Epsilon annealing schedule")
    print(f"  {'Schedule':<12} {'Test':>7} {'Sparsity':>9} {'FLOPs↓':>8}")
    a1 = {}
    for sched in ["cosine", "linear", "step"]:
        r = run_config(data, nf, nc, schedule=sched)
        a1[sched] = r
        print(f"  {sched:<12} {r['best_test_acc']:>7.4f} {r['final_sparsity']:>9.3f} {r['flops_reduction']:>8.3f}")
    all_results["A1_schedule"] = a1

    # ------------------------------------------------------------------
    # A2: Warmup length
    # ------------------------------------------------------------------
    print("\nA2: Warmup steps")
    print(f"  {'Warmup':>8} {'Test':>7} {'Sparsity':>9} {'FLOPs↓':>8}")
    a2 = {}
    for w in [20, 40, 60, 80]:
        r = run_config(data, nf, nc, warmup=w)
        a2[str(w)] = r
        print(f"  {w:>8} {r['best_test_acc']:>7.4f} {r['final_sparsity']:>9.3f} {r['flops_reduction']:>8.3f}")
    all_results["A2_warmup"] = a2

    # ------------------------------------------------------------------
    # A3: Retirement frequency
    # ------------------------------------------------------------------
    print("\nA3: Retirement frequency (retire every k steps)")
    print(f"  {'Every k':>8} {'Test':>7} {'Sparsity':>9} {'FLOPs↓':>8}")
    a3 = {}
    for k in [2, 5, 10, 20]:
        r = run_config(data, nf, nc, retire_every=k)
        a3[str(k)] = r
        print(f"  {k:>8} {r['best_test_acc']:>7.4f} {r['final_sparsity']:>9.3f} {r['flops_reduction']:>8.3f}")
    all_results["A3_retire_every"] = a3

    # ------------------------------------------------------------------
    # A4: Max sparsity ceiling
    # ------------------------------------------------------------------
    print("\nA4: Max sparsity ceiling")
    print(f"  {'Max sp':>8} {'Test':>7} {'Sparsity':>9} {'FLOPs↓':>8}")
    a4 = {}
    for max_sp in [0.20, 0.30, 0.40, 0.50, 0.60]:
        r = run_config(data, nf, nc, max_sparsity=max_sp)
        a4[str(max_sp)] = r
        print(f"  {max_sp:>8.2f} {r['best_test_acc']:>7.4f} {r['final_sparsity']:>9.3f} {r['flops_reduction']:>8.3f}")
    all_results["A4_max_sparsity"] = a4

    os.makedirs("results", exist_ok=True)
    with open("results/ablation_study.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nAblation results saved to results/ablation_study.json")


if __name__ == "__main__":
    main()
