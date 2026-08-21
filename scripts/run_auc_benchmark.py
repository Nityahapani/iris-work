"""
TGS + static-baseline benchmark scored by ROC-AUC instead of accuracy.

Why this exists: Tolokers (~22% positive) and Questions (~3% positive)
are binary node classification tasks with heavy class imbalance.
Accuracy on these is dominated by the majority-class rate and can't
distinguish a good model from a trivial one (this is exactly what
happened with Minesweeper — see conversation history). The dataset's
own benchmark paper (Platonov et al.) scores these with ROC-AUC, so
this script mirrors scripts/train.py's training loop but selects the
best checkpoint by validation AUC and reports test AUC.

Usage:
    PYTHONPATH=. python3 scripts/run_auc_benchmark.py --config configs/questions_gcn.yaml
    PYTHONPATH=. python3 scripts/run_auc_benchmark.py --config configs/questions_gcn.yaml --baseline dense
    PYTHONPATH=. python3 scripts/run_auc_benchmark.py --config configs/questions_gcn.yaml --baseline random --target-sparsity 0.65
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from tgs.core import TemporalGraph, EdgeManager
from tgs.core.influence import GradientNormEstimator
from tgs.models import TemporalGCN
from tgs.schedulers import AdaptiveRetirementScheduler, RetirementScheduler
from tgs.evaluation.flops import FLOPsCounter
from tgs.evaluation.baselines import dense_edges, random_sparsify, local_degree_sparsify, effective_resistance_sparsify
from tgs.utils import load_config, setup_logging, set_seed, Config, load_dataset

logger = logging.getLogger(__name__)


def _auc(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """Binary ROC-AUC using the positive-class probability."""
    if mask.sum() == 0:
        return 0.0
    probs = F.softmax(logits[mask], dim=-1)[:, 1].detach().cpu().numpy()
    y = labels[mask].detach().cpu().numpy()
    if len(set(y.tolist())) < 2:
        return 0.0  # AUC undefined with only one class present
    return roc_auc_score(y, probs)


def build_scheduler(cfg: Config, tg: TemporalGraph):
    if cfg.scheduler == "adaptive":
        return AdaptiveRetirementScheduler(
            temporal_graph=tg, epsilon_max=cfg.epsilon_max, epsilon_min=cfg.epsilon_min,
            anneal_steps=cfg.anneal_steps, schedule=cfg.anneal_schedule,
            warmup_steps=cfg.warmup_steps, max_retire_frac=cfg.max_retire_frac,
            max_sparsity=cfg.max_sparsity, retire_every=cfg.retire_every,
        )
    return RetirementScheduler(
        temporal_graph=tg, epsilon=cfg.epsilon, warmup_steps=cfg.warmup_steps,
        max_retire_frac=cfg.max_retire_frac, max_sparsity=cfg.max_sparsity,
        retire_every=cfg.retire_every,
    )


def train_tgs_auc(cfg: Config) -> dict:
    """TGS training loop, scored by AUC. Mirrors scripts/train.py::train."""
    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    data, in_ch, num_classes = load_dataset(cfg, device)
    assert num_classes == 2, "run_auc_benchmark.py assumes binary classification"
    m0 = data.edge_index.shape[1]
    logger.info(f"[TGS] {cfg.dataset} | n={data.num_nodes}, m0={m0}")

    tg = TemporalGraph(data.edge_index, data.num_nodes, device=device)
    influence_est = GradientNormEstimator(
        m0, device, edge_index=data.edge_index, num_nodes=data.num_nodes, ema_decay=cfg.ema_decay,
    )
    model = TemporalGCN(in_ch, cfg.hidden_channels, num_classes, cfg.num_layers, cfg.dropout).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [influence_est.edge_weights], lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = build_scheduler(cfg, tg)
    flops_counter = FLOPsCounter(m0, cfg.num_layers, cfg.hidden_channels)

    best_val_auc, best_test_auc, best_sparsity = 0.0, 0.0, 0.0

    for epoch in range(cfg.epochs):
        model.train()
        active_mask = tg.active_mask
        edge_index_t = tg.edge_index
        edge_weight_t = influence_est.edge_weights[active_mask]

        logits = model(data.x, edge_index_t, edge_weight_t)
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
        val_auc = _auc(eval_logits, data.y, data.val_mask)
        test_auc = _auc(eval_logits, data.y, data.test_mask)

        if val_auc > best_val_auc:
            best_val_auc, best_test_auc = val_auc, test_auc
            best_sparsity = tg.sparsity

        # Update disagreement signal from current predictions
        influence_est.update_disagreement(eval_logits, active_mask)

        tg.step()

        if epoch % cfg.log_every == 0:
            logger.info(f"Epoch {epoch:03d} | loss={loss.item():.4f} | val_auc={val_auc:.4f} | test_auc={test_auc:.4f} | sparsity={tg.sparsity:.3f}")

    return {
        "test_auc_at_best_val": best_test_auc,
        "best_val_auc": best_val_auc,
        "final_sparsity": tg.sparsity,
        "sparsity_at_best_val": best_sparsity,
        "flops_reduction": flops_counter.summary()["flops_reduction"],
        "final_distortion_bound": scheduler.cumulative_distortion_bound,
    }


def train_baseline_auc(name: str, cfg: Config, target_sparsity: float) -> dict:
    """Static-sparsification baseline, scored by AUC."""
    device = torch.device(cfg.device)
    set_seed(cfg.seed)
    data, in_ch, num_classes = load_dataset(cfg, device)
    assert num_classes == 2
    m0 = data.edge_index.shape[1]

    if name == "dense":
        ei = dense_edges(data.edge_index)
    elif name == "random":
        ei = random_sparsify(data.edge_index, target_sparsity, cfg.seed)
    elif name == "local_degree":
        ei = local_degree_sparsify(data.edge_index, data.num_nodes, target_sparsity)
    elif name == "eff_resistance":
        ei = effective_resistance_sparsify(data.edge_index, data.num_nodes, target_sparsity, cfg.seed)
    else:
        raise ValueError(name)
    ei = ei.to(device)
    actual_sparsity = 1.0 - ei.shape[1] / m0

    model = TemporalGCN(in_ch, cfg.hidden_channels, num_classes, cfg.num_layers, cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_auc, best_test_auc = 0.0, 0.0
    for epoch in range(cfg.epochs):
        model.train()
        logits = model(data.x, ei)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, ei)
        val_auc = _auc(out, data.y, data.val_mask)
        test_auc = _auc(out, data.y, data.test_mask)
        if val_auc > best_val_auc:
            best_val_auc, best_test_auc = val_auc, test_auc

    return {
        "baseline": name,
        "target_sparsity": target_sparsity,
        "actual_sparsity": actual_sparsity,
        "test_auc_at_best_val": best_test_auc,
        "best_val_auc": best_val_auc,
        "m0": m0,
        "mt": ei.shape[1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", default=None, help="If set, run this static baseline instead of TGS")
    parser.add_argument("--target-sparsity", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    if args.baseline:
        result = train_baseline_auc(args.baseline, cfg, args.target_sparsity)
        logger.info(f"[{args.baseline}] test_auc={result['test_auc_at_best_val']:.4f} sparsity={result['actual_sparsity']:.3f}")
    else:
        result = train_tgs_auc(cfg)
        logger.info(f"[TGS] test_auc={result['test_auc_at_best_val']:.4f} sparsity={result['final_sparsity']:.3f}")

    out_path = args.out or f"results/{cfg.run_name}_{args.baseline or 'tgs'}_auc.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
