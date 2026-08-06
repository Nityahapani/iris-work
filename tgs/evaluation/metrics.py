"""
Evaluator: tracks all metrics needed for the ISEF evaluation suite.

Metrics (matching the project brief):
  - Node classification accuracy (train / val / test)
  - Final sparsity ratio: 1 - |E_T|/|E_0|
  - Average sparsity over training: (1/T) Σ (1 - m_t/m_0)
  - Convergence step: first epoch where val accuracy plateaus within δ
  - Cumulative theoretical distortion bound: k * ε (Corollary 6.5)
  - GPU memory peak
"""

import torch
from torch import Tensor
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Tracks and computes all evaluation metrics across a training run.

    Usage:
        evaluator = Evaluator(num_classes=7)
        for epoch in range(T):
            ...
            evaluator.update(
                logits=model(x, edge_index),
                labels=data.y,
                train_mask=data.train_mask,
                val_mask=data.val_mask,
                test_mask=data.test_mask,
                sparsity=tg.sparsity,
                step=epoch,
            )
        results = evaluator.compute()
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self._history: list[dict] = []

    def update(
        self,
        logits: Tensor,
        labels: Tensor,
        train_mask: Tensor,
        val_mask: Tensor,
        test_mask: Tensor,
        sparsity: float,
        step: int,
        distortion_bound: float = 0.0,
    ) -> dict:
        """
        Compute metrics for the current step and append to history.

        Returns:
            dict with keys: train_acc, val_acc, test_acc, sparsity, step
        """
        with torch.no_grad():
            preds = logits.argmax(dim=-1)

            train_acc = self._accuracy(preds, labels, train_mask)
            val_acc = self._accuracy(preds, labels, val_mask)
            test_acc = self._accuracy(preds, labels, test_mask)

        entry = {
            "step": step,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "sparsity": sparsity,
            "distortion_bound": distortion_bound,
        }
        self._history.append(entry)
        return entry

    @staticmethod
    def _accuracy(preds: Tensor, labels: Tensor, mask: Tensor) -> float:
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        return correct / total if total > 0 else 0.0

    def compute(self) -> dict:
        """
        Aggregate the full training history into a results dict.

        Returns:
            Dictionary with best/final metrics, convergence step, etc.
        """
        if not self._history:
            return {}

        val_accs = [h["val_acc"] for h in self._history]
        test_accs = [h["test_acc"] for h in self._history]
        sparsities = [h["step"] for h in self._history]

        best_val_idx = int(np.argmax(val_accs))

        return {
            "best_val_acc": val_accs[best_val_idx],
            "test_acc_at_best_val": test_accs[best_val_idx],
            "final_test_acc": test_accs[-1],
            "final_val_acc": val_accs[-1],
            "best_step": self._history[best_val_idx]["step"],
            "final_sparsity": self._history[-1]["sparsity"],
            "mean_sparsity": float(np.mean([h["sparsity"] for h in self._history])),
            "final_distortion_bound": self._history[-1]["distortion_bound"],
            "convergence_step": self._find_convergence_step(val_accs),
            "num_steps": len(self._history),
        }

    def _find_convergence_step(self, val_accs: list, window: int = 10, delta: float = 1e-3) -> int:
        """
        First step t where val accuracy changes by less than delta
        over the next `window` steps.
        """
        for i in range(len(val_accs) - window):
            window_range = max(val_accs[i:i+window]) - min(val_accs[i:i+window])
            if window_range < delta:
                return self._history[i]["step"]
        return self._history[-1]["step"]

    def history(self) -> list[dict]:
        return self._history

    def best_val_checkpoint(self) -> dict:
        """Return the history entry with the best validation accuracy."""
        return max(self._history, key=lambda h: h["val_acc"])
