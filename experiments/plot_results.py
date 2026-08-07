"""
plot_results.py
Generate all figures for the ISEF writeup from logged training data.

Figures produced:
  1. sparsity_over_training.png  — m_t/m_0 vs training step
  2. accuracy_vs_sparsity.png    — test acc vs sparsity (TGS + baselines)
  3. epsilon_schedule.png        — ε_t over training steps
  4. distortion_bound.png        — cumulative k*ε over training
  5. flops_savings.png           — bar chart of FLOPs reduction per method

Usage:
    python experiments/plot_results.py
"""

import sys, os, json
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from tgs.core.temporal_graph import TemporalGraph
from tgs.core.influence import GradientNormEstimator
from tgs.models.gcn import TemporalGCN
from tgs.schedulers.adaptive_scheduler import AdaptiveRetirementScheduler
from tgs.evaluation.metrics import Evaluator
from tgs.evaluation.flops import FLOPsCounter
from tgs.utils.reproducibility import set_seed

os.makedirs("results/figures", exist_ok=True)
DEVICE = torch.device("cpu")
SEED   = 42
STYLE  = {
    "TGS (ours)":      {"color": "#2563eb", "lw": 2.5, "ls": "-",  "marker": "o"},
    "Dense GCN":       {"color": "#16a34a", "lw": 2.0, "ls": "--", "marker": "s"},
    "Random":          {"color": "#dc2626", "lw": 1.5, "ls": ":",  "marker": "^"},
    "Local Degree":    {"color": "#d97706", "lw": 1.5, "ls": "-.", "marker": "D"},
    "Eff. Resistance": {"color": "#7c3aed", "lw": 1.5, "ls": "--", "marker": "v"},
}


def run_tgs_with_history(data, num_features, num_classes):
    """Run TGS and collect per-epoch histories for plotting."""
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

    history = {"epoch": [], "sparsity": [], "val_acc": [], "test_acc": [],
               "epsilon": [], "distortion_bound": [], "mt": []}

    best_val = best_test = 0.0
    for epoch in range(300):
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
        scheduler_eps = scheduler.epsilon_at(epoch)
        flops_step = tg.mt
        tg.step()

        history["epoch"].append(epoch)
        history["sparsity"].append(tg.sparsity)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        history["epsilon"].append(scheduler_eps)
        history["distortion_bound"].append(scheduler.cumulative_distortion_bound)
        history["mt"].append(flops_step)

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc

    return history, best_val, best_test, tg.sparsity


def fig1_sparsity_over_training(history):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = history["epoch"]
    sparsity = history["sparsity"]

    ax.plot(epochs, sparsity, color=STYLE["TGS (ours)"]["color"], lw=2.5, label="TGS sparsity")
    ax.axvline(x=40, color="gray", ls="--", lw=1, label="Warmup end (epoch 40)")

    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel("Sparsity  (1 − |E_t| / |E_0|)", fontsize=12)
    ax.set_title("Edge Sparsity over Training — Cora", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.02, 0.60)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/sparsity_over_training.png", dpi=150)
    plt.close()
    print("  Saved: sparsity_over_training.png")


def fig2_accuracy_vs_sparsity(history, baseline_results=None):
    fig, ax = plt.subplots(figsize=(7, 4))

    # TGS trajectory
    ax.plot(history["sparsity"], history["test_acc"],
            color=STYLE["TGS (ours)"]["color"], lw=2.5,
            label="TGS (ours) — trajectory", zorder=5)

    # Mark final TGS point
    final_sp = history["sparsity"][-1]
    final_acc = history["test_acc"][history["sparsity"].index(final_sp) if final_sp in history["sparsity"] else -1]
    ax.scatter([final_sp], [history["test_acc"][-1]],
               color=STYLE["TGS (ours)"]["color"], s=80, zorder=6)

    # Baselines (loaded from file if available)
    if baseline_results:
        plotted = set()
        for r in baseline_results:
            method = r.get("method", r.get("baseline", "?"))
            if method == "TGS (ours)" or method in plotted:
                continue
            sp = r.get("actual_sparsity", r.get("final_sparsity", 0.0))
            acc = r.get("best_test_acc", 0.0)
            st = STYLE.get(method, {"color": "gray", "marker": "x"})
            ax.scatter([sp], [acc], color=st["color"],
                       marker=st["marker"], s=80, label=method, zorder=4)
            plotted.add(method)

    ax.set_xlabel("Sparsity  (fraction of edges removed)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Test Accuracy vs Sparsity — Cora", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/accuracy_vs_sparsity.png", dpi=150)
    plt.close()
    print("  Saved: accuracy_vs_sparsity.png")


def fig3_epsilon_schedule(history):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = history["epoch"]
    epsilons = history["epsilon"]

    ax.semilogy(epochs, epsilons, color="#7c3aed", lw=2.5)
    ax.axvline(x=40, color="gray", ls="--", lw=1, label="Warmup end")
    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel("ε_t  (retirement threshold, log scale)", fontsize=12)
    ax.set_title("Adaptive ε Schedule (cosine annealing)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("results/figures/epsilon_schedule.png", dpi=150)
    plt.close()
    print("  Saved: epsilon_schedule.png")


def fig4_distortion_bound(history):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = history["epoch"]
    bounds = history["distortion_bound"]

    ax.plot(epochs, bounds, color="#dc2626", lw=2.5, label="Cumulative bound  k·ε")
    ax.fill_between(epochs, 0, bounds, alpha=0.15, color="#dc2626")
    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel("‖H_t − H_t^{−S}‖_F  ≤  k·ε", fontsize=12)
    ax.set_title("Theoretical Distortion Bound over Training", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/distortion_bound.png", dpi=150)
    plt.close()
    print("  Saved: distortion_bound.png")


def fig5_flops_bar(baseline_results, tgs_flops_red, tgs_sparsity):
    """Bar chart comparing FLOPs reduction across methods at similar sparsity."""
    # Pull one entry per method close to tgs_sparsity
    seen = {}
    for r in (baseline_results or []):
        method = r.get("method", r.get("baseline", "?"))
        sp = r.get("actual_sparsity", r.get("final_sparsity", 0.0))
        acc = r.get("best_test_acc", 0.0)
        if method not in seen:
            seen[method] = (sp, acc)

    methods = list(seen.keys()) + ["TGS (ours)"]
    sparsities = [seen[m][0] for m in seen] + [tgs_sparsity]
    accs = [seen[m][1] for m in seen] + [None]
    colors = [STYLE.get(m, {"color": "gray"})["color"] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(methods, sparsities, color=colors, edgecolor="white", linewidth=0.8)

    # Annotate with test acc
    for bar, m, sp in zip(bars, methods, sparsities):
        acc_val = seen.get(m, (None, None))[1]
        if m == "TGS (ours)":
            # load from history if needed
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"TGS", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"{sp:.2f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax.set_ylabel("Final Sparsity", fontsize=12)
    ax.set_title("Sparsity Achieved per Method — Cora", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.65)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("results/figures/flops_savings.png", dpi=150)
    plt.close()
    print("  Saved: flops_savings.png")


def main():
    print("Loading Cora...")
    dataset = Planetoid(root="./data", name="Cora", transform=NormalizeFeatures())
    data = dataset[0].to(DEVICE)

    print("Running TGS (collecting history)...")
    history, best_val, best_test, final_sp = run_tgs_with_history(
        data, dataset.num_features, dataset.num_classes
    )
    print(f"  TGS: test={best_test:.4f}, sparsity={final_sp:.3f}")

    # Load baseline results if available
    baseline_results = None
    if os.path.exists("results/baselines_comparison.json"):
        with open("results/baselines_comparison.json") as f:
            all_bl = json.load(f)
        baseline_results = [r for r in all_bl if r.get("dataset") == "Cora"]
        print(f"  Loaded {len(baseline_results)} baseline results from file")

    print("\nGenerating figures...")
    fig1_sparsity_over_training(history)
    fig2_accuracy_vs_sparsity(history, baseline_results)
    fig3_epsilon_schedule(history)
    fig4_distortion_bound(history)

    from tgs.evaluation.flops import FLOPsCounter
    fc = FLOPsCounter(data.edge_index.shape[1], 2, 64)
    for mt in history["mt"]:
        fc.record_step(mt)
    fig5_flops_bar(baseline_results, fc.summary()["flops_reduction"], final_sp)

    print(f"\nAll figures saved to results/figures/")


if __name__ == "__main__":
    main()
