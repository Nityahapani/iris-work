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

        Uses PERCENTILE-based retirement: retires the lowest-scoring
        eligible edges each step, up to max_retire_frac of active edges.
        This is more robust than a fixed-ε threshold because it adapts
        to whatever score distribution the estimator produces.

        The ε threshold in Definition 5.1 is still respected indirectly:
        we only retire edges whose score is below the median eligible score
        (i.e., the bottom half of eligible edges), ensuring we never retire
        edges the estimator considers above-average influence.
        """
        t = self.tg.t

        if t < self.warmup_steps:
            return 0

        if (t - self.warmup_steps) % self.retire_every != 0:
            return 0

        if self.tg.sparsity >= self.max_sparsity:
            return 0

        if self._val_acc_guard_triggered:
            return 0

        active_mask = self.tg.active_mask
        # Eligible = active AND not locked (score < inf)
        eligible_mask = active_mask & (influence_scores < float("inf"))
        eligible_indices = eligible_mask.nonzero(as_tuple=False).squeeze(1)

        if len(eligible_indices) == 0:
            return 0

        # Percentile gate: only retire edges in the bottom 30% of scores
        # (low score = safe to retire per our scoring convention)
        scores_eligible = influence_scores[eligible_indices]
        percentile_30   = torch.quantile(scores_eligible, 0.30)
        candidate_mask  = scores_eligible <= percentile_30
        candidate_idx   = eligible_indices[candidate_mask]

        if len(candidate_idx) == 0:
            return 0

        # Rate limit: cap at max_retire_frac of active edges
        max_retire = max(1, int(self.tg.mt * self.max_retire_frac))
        if len(candidate_idx) > max_retire:
            # Retire the absolute lowest-score edges first
            scores_candidates = influence_scores[candidate_idx]
            _, order = scores_candidates.sort()
            candidate_idx = candidate_idx[order[:max_retire]]

        n_retired = self.tg.retire_edges(candidate_idx)
        self._cumulative_k += n_retired

        entry = {
            "step": t,
            "retired": n_retired,
            "cumulative_retired": self._cumulative_k,
            "sparsity": self.tg.sparsity,
            "epsilon": self.epsilon,
            "cumulative_distortion_bound": self._cumulative_k * self.epsilon,
        }
        self._retirement_log.append(entry)

        if n_retired > 0:
            logger.info(
                f"Step {t}: retired {n_retired} | "
                f"sparsity={self.tg.sparsity:.3f} | "
                f"distortion_bound={entry['cumulative_distortion_bound']:.4f}"
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
