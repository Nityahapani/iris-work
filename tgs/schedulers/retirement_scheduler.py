"""
RetirementScheduler: implements the safe retirement criterion (Definition 5.1)
and batch retirement with cumulative distortion control (Theorem 6.3).

The scheduler operates as follows each training step:
  1. Receive updated influence scores Î_e(t) from the estimator
  2. Identify edges satisfying Î_e(t) ≤ ε  (safe retirement criterion)
  3. Apply cooling schedule — suppress retirements in early training
  4. Retire the eligible edges via TemporalGraph.retire_edges()

Theory connections:
  Definition 5.1 — safe retirement threshold ε
  Theorem 5.2    — representation distortion ≤ ε at retirement
  Theorem 5.3    — prediction distortion ≤ K_f · ε at retirement
  Corollary 6.5  — k simultaneous retirements: cumulative distortion ≤ kε
  Theorem 7.4    — optimal retirement balances Be(t) = λ (marginal loss = marginal cost)
"""

import torch
from torch import Tensor
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RetirementScheduler:
    """
    Fixed-ε retirement scheduler.

    Retires all active edges satisfying Î_e(t) ≤ ε, subject to:
      - A warm-up period [0, warmup_steps] during which no retirements occur
      - A maximum retirement fraction per step (rate limiting)
      - A global sparsity ceiling (never retire more than max_sparsity of E_0)

    Args:
        temporal_graph:    the TemporalGraph instance to operate on
        epsilon:           safe retirement threshold ε (Definition 5.1)
        warmup_steps:      number of initial steps with no retirement (cooling schedule)
        max_retire_frac:   max fraction of active edges to retire per step
        max_sparsity:      global ceiling on sparsity (e.g., 0.9 = keep at least 10%)
        retire_every:      only attempt retirement every k steps (reduces overhead)
    """

    def __init__(
        self,
        temporal_graph,
        epsilon: float = 1e-3,
        warmup_steps: int = 50,
        max_retire_frac: float = 0.05,
        max_sparsity: float = 0.9,
        retire_every: int = 5,
    ):
        self.tg = temporal_graph
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps
        self.max_retire_frac = max_retire_frac
        self.max_sparsity = max_sparsity
        self.retire_every = retire_every

        # Cumulative distortion tracker (Corollary 6.5: bound = k * ε)
        self._cumulative_k: int = 0

        # History for logging
        self._retirement_log: list[dict] = []

        # Val accuracy guard: stop retiring if accuracy drops
        self._val_acc_history: list[float] = []
        self._best_val_acc: float = 0.0
        self._val_acc_guard_triggered: bool = False
        self._val_patience_delta: float = 0.04  # allow 4% drop below peak before halting
        self._val_guard_window: int = 15         # smooth over last 15 steps before checking

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def step(self, influence_scores: Tensor) -> int:
        """
        Attempt retirement at the current training step.

        Args:
            influence_scores: Tensor [m0] — Î_e(t) for all edges
                              (inactive edges should have score 0)

        Returns:
            Number of edges retired this step (0 if warm-up or no eligible edges)
        """
        t = self.tg.t

        # 1. Cooling schedule: no retirements during warm-up
        if t < self.warmup_steps:
            return 0

        # 2. Only run every retire_every steps
        if (t - self.warmup_steps) % self.retire_every != 0:
            return 0

        # 3. Sparsity ceiling check
        if self.tg.sparsity >= self.max_sparsity:
            logger.debug(f"Step {t}: sparsity ceiling {self.max_sparsity:.2f} reached, skipping")
            return 0

        # 3b. Accuracy guard: halt retirement if val acc dropped > patience_delta
        if self._val_acc_guard_triggered:
            return 0

        # 4. Find eligible edges: active AND influence below threshold
        active_mask = self.tg.active_mask
        eligible_mask = active_mask & (influence_scores <= self.epsilon)
        eligible_indices = eligible_mask.nonzero(as_tuple=False).squeeze(1)

        if len(eligible_indices) == 0:
            return 0

        # 5. Rate limiting: cap at max_retire_frac of currently active edges
        max_retire = max(1, int(self.tg.mt * self.max_retire_frac))
        if len(eligible_indices) > max_retire:
            # Retire the lowest-influence edges first (safest)
            scores_eligible = influence_scores[eligible_indices]
            _, sorted_order = scores_eligible.sort()
            eligible_indices = eligible_indices[sorted_order[:max_retire]]

        # 6. Execute retirement
        n_retired = self.tg.retire_edges(eligible_indices)
        self._cumulative_k += n_retired

        # 7. Log
        entry = {
            "step": t,
            "retired": n_retired,
            "cumulative_retired": self._cumulative_k,
            "sparsity": self.tg.sparsity,
            "epsilon": self.epsilon,
            "cumulative_distortion_bound": self._cumulative_k * self.epsilon,
        }
        self._retirement_log.append(entry)

        logger.info(
            f"Step {t}: retired {n_retired} edges | "
            f"sparsity={self.tg.sparsity:.3f} | "
            f"distortion bound={entry['cumulative_distortion_bound']:.4f}"
        )

        return n_retired

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def update_val_acc(self, val_acc: float) -> None:
        """
        Inform the scheduler of current validation accuracy.
        Guard only activates after warmup. Uses a rolling window average
        to avoid reacting to single noisy epochs.
        """
        self._val_acc_history.append(val_acc)

        # Don't trigger guard before warmup is done
        if self.tg.t < self.warmup_steps + self._val_guard_window:
            if val_acc > self._best_val_acc:
                self._best_val_acc = val_acc
            return

        # Smoothed current acc over last window steps
        window = self._val_acc_history[-self._val_guard_window:]
        smoothed = sum(window) / len(window)

        if smoothed > self._best_val_acc:
            self._best_val_acc = smoothed

        if self._best_val_acc - smoothed > self._val_patience_delta:
            if not self._val_acc_guard_triggered:
                logger.info(
                    f"Val acc guard triggered at step {self.tg.t}: "
                    f"best={self._best_val_acc:.4f}, smoothed={smoothed:.4f} — halting retirement"
                )
            self._val_acc_guard_triggered = True

    @property
    def cumulative_distortion_bound(self) -> float:
        """
        Upper bound on cumulative representation distortion from Corollary 6.5:
            ‖H_t - H_t^{-S}‖_F ≤ k * ε
        where k = total edges retired so far.
        """
        return self._cumulative_k * self.epsilon

    @property
    def retirement_log(self) -> list[dict]:
        return self._retirement_log

    def summary(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "warmup_steps": self.warmup_steps,
            "total_retired": self._cumulative_k,
            "final_sparsity": self.tg.sparsity,
            "cumulative_distortion_bound": self.cumulative_distortion_bound,
        }
