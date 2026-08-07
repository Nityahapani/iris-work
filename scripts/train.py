"""
Main training script for Temporal Graph Sparsification.

Usage:
    python scripts/train.py --config configs/cora_gcn.yaml
    python scripts/train.py --config configs/cora_gcn.yaml --device cuda
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from tgs.core import TemporalGraph, EdgeManager, JacobianInfluenceEstimator
from tgs.core.influence import GradientNormEstimator
from tgs.models import TemporalGCN
from tgs.schedulers import AdaptiveRetirementScheduler, RetirementScheduler
from tgs.evaluation import Evaluator
from tgs.evaluation.flops import FLOPsCounter
from tgs.utils import load_config, setup_logging, set_seed, Config

logger = logging.getLogger(__name__)


def load_dataset(cfg: Config, device: torch.device):
    dataset = Planetoid(
        root=cfg.dataset_root,
        name=cfg.dataset,
        transform=NormalizeFeatures(),
    )
    data = dataset[0].to(device)
    return data, dataset.num_features, dataset.num_classes


def build_model(cfg: Config, in_channels: int, out_channels: int) -> TemporalGCN:
    return TemporalGCN(
        in_channels=in_channels,
        hidden_channels=cfg.hidden_channels,
        out_channels=out_channels,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )


def build_scheduler(cfg: Config, tg: TemporalGraph):
    if cfg.scheduler == "adaptive":
        return AdaptiveRetirementScheduler(
            temporal_graph=tg,
            epsilon_max=cfg.epsilon_max,
            epsilon_min=cfg.epsilon_min,
            anneal_steps=cfg.anneal_steps,
            schedule=cfg.anneal_schedule,
            warmup_steps=cfg.warmup_steps,
            max_retire_frac=cfg.max_retire_frac,
            max_sparsity=cfg.max_sparsity,
            retire_every=cfg.retire_every,
        )
    else:
        return RetirementScheduler(
            temporal_graph=tg,
            epsilon=cfg.epsilon,
            warmup_steps=cfg.warmup_steps,
            max_retire_frac=cfg.max_retire_frac,
            max_sparsity=cfg.max_sparsity,
            retire_every=cfg.retire_every,
        )


def train(cfg: Config) -> dict:
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    set_seed(cfg.seed)

    # ----------------------------------------------------------------
    # 1. Data
    # ----------------------------------------------------------------
    data, in_ch, num_classes = load_dataset(cfg, device)
    m0 = data.edge_index.shape[1]
    logger.info(f"Dataset: {cfg.dataset} | n={data.num_nodes}, m0={m0}, classes={num_classes}")

    # ----------------------------------------------------------------
    # 2. Core components
    # ----------------------------------------------------------------
    tg = TemporalGraph(data.edge_index, data.num_nodes, device=device)
    edge_manager = EdgeManager(data.edge_index, data.num_nodes)
    influence_est = GradientNormEstimator(
        m0, device,
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        ema_decay=cfg.ema_decay,
    )

    # ----------------------------------------------------------------
    # 3. Model
    # ----------------------------------------------------------------
    model = build_model(cfg, in_ch, num_classes).to(device)
    # Optimizer includes edge weights as parameters for gradient tracking
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [influence_est.edge_weights],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ----------------------------------------------------------------
    # 4. Scheduler
    # ----------------------------------------------------------------
    scheduler = build_scheduler(cfg, tg)

    # ----------------------------------------------------------------
    # 5. Evaluator + FLOPs counter
    # ----------------------------------------------------------------
    evaluator = Evaluator(num_classes)
    flops_counter = FLOPsCounter(m0, cfg.num_layers, cfg.hidden_channels)

    # ----------------------------------------------------------------
    # 6. Training loop
    # ----------------------------------------------------------------
    logger.info(f"Starting training | epochs={cfg.epochs} | scheduler={cfg.scheduler}")

    for epoch in range(cfg.epochs):
        model.train()

        # Get current active edge set and weights
        active_mask = tg.active_mask
        edge_index_t = tg.edge_index
        edge_weight_t = influence_est.edge_weights[active_mask]  # differentiable

        # Forward pass
        logits = model(data.x, edge_index_t, edge_weight_t)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update influence estimates from gradients (before optimizer step)
        influence_est.update_influence(active_mask)

        optimizer.step()

        # ---- Retirement decision ----
        influence_scores = influence_est.influence_scores(active_mask)
        n_retired = scheduler.step(influence_scores)

        # ---- FLOPs tracking ----
        flops_counter.record_step(tg.mt)

        # ---- Evaluation ----
        model.eval()
        with torch.no_grad():
            eval_edge_index = tg.edge_index
            eval_logits = model(data.x, eval_edge_index)

        metrics = evaluator.update(
            logits=eval_logits,
            labels=data.y,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            test_mask=data.test_mask,
            sparsity=tg.sparsity,
            step=epoch,
            distortion_bound=scheduler.cumulative_distortion_bound,
        )

        # ---- Advance step counter ----
        tg.step()

        if epoch % cfg.log_every == 0:
            logger.info(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"val={metrics['val_acc']:.4f} | test={metrics['test_acc']:.4f} | "
                f"sparsity={tg.sparsity:.3f} | retired_this_step={n_retired}"
            )

    # ----------------------------------------------------------------
    # 7. Results
    # ----------------------------------------------------------------
    results = evaluator.compute()
    results["flops"] = flops_counter.summary()
    results["retirement_summary"] = scheduler.summary()
    results["lipschitz_bound_CH"] = model.lipschitz_bound()
    results["config"] = cfg.__dict__

    save_dir = Path(cfg.save_dir) / cfg.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS — {cfg.dataset}")
    logger.info(f"  Test accuracy:     {results['test_acc_at_best_val']:.4f}")
    logger.info(f"  Final sparsity:    {results['final_sparsity']:.3f}")
    logger.info(f"  FLOPs reduction:   {results['flops']['flops_reduction']:.3f}")
    logger.info(f"  Distortion bound:  {results['final_distortion_bound']:.6f}")
    logger.info(f"  Results saved to:  {save_dir}")
    logger.info(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="TGS Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    setup_logging(log_file=f"{cfg.save_dir}/{cfg.run_name}/train.log")
    train(cfg)


if __name__ == "__main__":
    main()
