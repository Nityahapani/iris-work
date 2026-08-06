"""
AdaptiveRetirementScheduler: extends RetirementScheduler with a dynamic ε
that adapts based on training progress.

Motivation (Theorem 7.4):
    Optimal retirement occurs when Be(t) = λ (marginal loss = marginal cost).
    Early in training, Be(t) is high for most edges (representations are still
    forming). Late in training, many edges have Be(t) → 0. An adaptive ε
    mirrors this: loose early (more permissive retirement), tight late
    (conservative retirement once representations have stabilised).

Schedule options:
  'cosine'  — ε_t = ε_max * cos²(π * t / (2 * T_anneal)), then ε_min
  'linear'  — ε_t decreases linearly from ε_max to ε_min over T_anneal steps
  'step'    — ε drops by a factor at fixed milestones

Connection to Proposition 8.2:
    As training converges, Ie(t) → 0 for redundant edges. Tightening ε
    ensures we only retire edges whose influence has genuinely decayed,
    not just edges that appear low-influence due to noisy gradients early
    in training.
"""

import math
import torch
from torch import Tensor
from typing import Literal
import logging

from .retirement_scheduler import RetirementScheduler

logger = logging.getLogger(__name__)


class AdaptiveRetirementScheduler(RetirementScheduler):
    """
    Adaptive ε scheduler.

    Args:
        temporal_graph:  TemporalGraph instance
        epsilon_max:     initial (loose) threshold — used during early training
        epsilon_min:     final (tight) threshold — used after annealing
        anneal_steps:    steps over which ε decays from max to min
        schedule:        'cosine', 'linear', or 'step'
        step_milestones: for schedule='step', list of (step, epsilon) pairs
        warmup_steps:    no retirements before this step
        max_retire_frac: rate limiting
        max_sparsity:    global sparsity ceiling
        retire_every:    retirement frequency
    """

    def __init__(
        self,
        temporal_graph,
        epsilon_max: float = 1e-2,
        epsilon_min: float = 1e-4,
        anneal_steps: int = 200,
        schedule: Literal["cosine", "linear", "step"] = "cosine",
        step_milestones: list[tuple[int, float]] | None = None,
        warmup_steps: int = 50,
        max_retire_frac: float = 0.05,
        max_sparsity: float = 0.9,
        retire_every: int = 5,
    ):
        super().__init__(
            temporal_graph=temporal_graph,
            epsilon=epsilon_max,  # base class uses self.epsilon; we override dynamically
            warmup_steps=warmup_steps,
            max_retire_frac=max_retire_frac,
            max_sparsity=max_sparsity,
            retire_every=retire_every,
        )
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.anneal_steps = anneal_steps
        self.schedule = schedule
        self.step_milestones = step_milestones or []

        self._epsilon_history: list[float] = []

    # ------------------------------------------------------------------
    # ε schedule
    # ------------------------------------------------------------------

    def _compute_epsilon(self, t: int) -> float:
        """Compute ε_t according to the chosen annealing schedule."""
        # Steps elapsed since warm-up ends
        effective_t = max(0, t - self.warmup_steps)

        if self.schedule == "cosine":
            if effective_t >= self.anneal_steps:
                return self.epsilon_min
            progress = effective_t / self.anneal_steps
            return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * (
                math.cos(math.pi * progress / 2) ** 2
            )

        elif self.schedule == "linear":
            if effective_t >= self.anneal_steps:
                return self.epsilon_min
            progress = effective_t / self.anneal_steps
            return self.epsilon_max - progress * (self.epsilon_max - self.epsilon_min)

        elif self.schedule == "step":
            eps = self.epsilon_max
            for milestone_t, milestone_eps in sorted(self.step_milestones):
                if t >= milestone_t:
                    eps = milestone_eps
            return eps

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    # ------------------------------------------------------------------
    # Override step to update ε dynamically
    # ------------------------------------------------------------------

    def step(self, influence_scores: Tensor) -> int:
        """
        Same as RetirementScheduler.step() but updates self.epsilon
        dynamically before checking the retirement criterion.
        """
        t = self.tg.t
        self.epsilon = self._compute_epsilon(t)
        self._epsilon_history.append(self.epsilon)

        n_retired = super().step(influence_scores)

        if n_retired > 0:
            logger.info(f"Step {t}: ε_t={self.epsilon:.6f}, retired {n_retired}")

        return n_retired

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def epsilon_at(self, t: int) -> float:
        """Query what ε would be at any step t."""
        return self._compute_epsilon(t)

    def epsilon_history(self) -> list[float]:
        return self._epsilon_history

    def summary(self) -> dict:
        base = super().summary()
        base.update({
            "epsilon_max": self.epsilon_max,
            "epsilon_min": self.epsilon_min,
            "epsilon_final": self.epsilon,
            "anneal_steps": self.anneal_steps,
            "schedule": self.schedule,
        })
        return base
