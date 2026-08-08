"""
ImprovedRetirementScheduler: two algorithmic improvements over AdaptiveRetirementScheduler.

Improvement 1 — Distortion Budget (motivated by Corollary 6.5):
    The cumulative distortion bound is k·ε. Instead of a fixed sparsity ceiling,
    we cap retirement so the theoretical distortion bound stays below a user-set
    budget δ. This gives a principled stopping criterion grounded directly in
    the theory: retire freely until k·ε ≥ δ, then stop.

    This is strictly better than a sparsity ceiling because:
      - The ceiling is an empirical heuristic with no theoretical grounding
      - The distortion budget is derived from Corollary 6.5 directly
      - It adapts: if ε is tight (late in training), more edges can be retired
        before the budget is exhausted

Improvement 2 — Representation Stability Auto-Warmup (motivated by Proposition 8.2):
    Instead of a fixed warmup_steps, we detect when representations have
    stabilised: compute ||H_t - H_{t-1}||_F over a rolling window and start
    retiring only once this drops below a stability threshold.

    This is motivated by Proposition 8.2: redundant edges only have Ie(t)→0
    once the overall representations stabilise. Starting retirement before
    stability means we retire edges that may still be contributing.

    Advantage: automatically adapts to dataset/architecture without tuning warmup.
"""

import torch
from torch import Tensor
from typing import Optional
import logging

from .adaptive_scheduler import AdaptiveRetirementScheduler

logger = logging.getLogger(__name__)


class ImprovedRetirementScheduler(AdaptiveRetirementScheduler):
    """
    Adaptive scheduler with distortion budget and auto-warmup.

    Args:
        temporal_graph:       TemporalGraph instance
        epsilon_max/min:      ε annealing bounds
        anneal_steps:         annealing duration
        schedule:             'cosine', 'linear', 'step'
        distortion_budget:    δ — max allowed k·ε (Corollary 6.5).
                              None = no budget constraint (fallback to sparsity ceiling)
        auto_warmup:          if True, ignore warmup_steps and detect stability
        stability_threshold:  ||ΔH||_F below which warmup ends
        stability_window:     epochs over which to smooth ||ΔH||_F
        warmup_steps:         fallback fixed warmup if auto_warmup=False
        max_retire_frac:      rate limiting per step
        max_sparsity:         absolute sparsity ceiling (secondary guard)
        retire_every:         retirement frequency
    """

    def __init__(
        self,
        temporal_graph,
        epsilon_max: float = 5e-3,
        epsilon_min: float = 1e-5,
        anneal_steps: int = 100,
        schedule: str = "cosine",
        distortion_budget: Optional[float] = 0.10,
        auto_warmup: bool = True,
        stability_threshold: float = 1e-3,
        stability_window: int = 10,
        warmup_steps: int = 40,
        max_retire_frac: float = 0.10,
        max_sparsity: float = 0.65,
        retire_every: int = 2,
    ):
        super().__init__(
            temporal_graph=temporal_graph,
            epsilon_max=epsilon_max,
            epsilon_min=epsilon_min,
            anneal_steps=anneal_steps,
            schedule=schedule,
            warmup_steps=warmup_steps,
            max_retire_frac=max_retire_frac,
            max_sparsity=max_sparsity,
            retire_every=retire_every,
        )
        self.distortion_budget = distortion_budget
        self.auto_warmup = auto_warmup
        self.stability_threshold = stability_threshold
        self.stability_window = stability_window

        # Auto-warmup state
        self._stability_ready: bool = False
        self._repr_delta_history: list[float] = []
        self._prev_repr: Optional[Tensor] = None
        self._auto_warmup_ended_at: Optional[int] = None

    # ------------------------------------------------------------------
    # Representation stability tracking (Improvement 2)
    # ------------------------------------------------------------------

    def update_representations(self, H: Tensor) -> float:
        """
        Track representation stability using RELATIVE delta: ||ΔH||_F / ||H||_F.
        Also detect val-accuracy plateau as a secondary stability signal.

        Returns:
            Current relative stability delta (lower = more stable)
        """
        h_norm = H.detach().norm(p='fro').item()

        if self._prev_repr is None:
            self._prev_repr = H.detach().clone()
            return float('inf')

        with torch.no_grad():
            abs_delta = (H.detach() - self._prev_repr).norm(p='fro').item()

        # Relative delta: normalised by current H norm
        rel_delta = abs_delta / max(h_norm, 1e-8)
        self._repr_delta_history.append(rel_delta)
        self._prev_repr = H.detach().clone()

        if len(self._repr_delta_history) >= self.stability_window:
            smoothed = sum(self._repr_delta_history[-self.stability_window:]) / self.stability_window
        else:
            smoothed = rel_delta

        # Stability declared when relative change drops below threshold
        if not self._stability_ready and smoothed < self.stability_threshold:
            self._stability_ready = True
            self._auto_warmup_ended_at = self.tg.t
            logger.info(
                f"Auto-warmup ended at step {self.tg.t} (relative stability): "
                f"smoothed ||ΔH||/||H|| = {smoothed:.6f} < {self.stability_threshold}"
            )

        return smoothed

    def _check_val_acc_plateau(self) -> bool:
        """
        Secondary stability signal: val accuracy has plateaued for stability_window steps.
        Used as fallback if relative representation delta never drops below threshold.
        """
        if len(self._val_acc_history) < self.stability_window + 5:
            return False
        window = self._val_acc_history[-self.stability_window:]
        plateau_range = max(window) - min(window)
        return plateau_range < 0.01  # within 1% over the window

    # ------------------------------------------------------------------
    # Override step — add distortion budget check and auto-warmup
    # ------------------------------------------------------------------

    def step(self, influence_scores: Tensor) -> int:
        t = self.tg.t

        # Auto-warmup: block retirement until stability is detected
        if self.auto_warmup and not self._stability_ready:
            # Fallback: if val acc has plateaued, declare stability anyway
            if self._check_val_acc_plateau():
                self._stability_ready = True
                self._auto_warmup_ended_at = t
                logger.info(f"Auto-warmup ended at step {t} (val acc plateau fallback)")
            else:
                return 0

        # Distortion budget: stop if k·ε ≥ δ (Corollary 6.5)
        if self.distortion_budget is not None:
            remaining = self.distortion_budget - self.cumulative_distortion_bound
            if remaining <= 0:
                return 0
            # Cap max edges this step so we don't overshoot the budget
            # k_new * ε_t ≤ remaining  →  k_new ≤ remaining / ε_t
            eps_t = max(self.epsilon, 1e-10)
            budget_max = max(0, int(remaining / eps_t))
            if budget_max == 0:
                return 0
            # Temporarily override max_retire_frac to enforce budget cap
            original_frac = self.max_retire_frac
            budget_frac = budget_max / max(self.tg.mt, 1)
            self.max_retire_frac = min(original_frac, budget_frac)
        else:
            original_frac = self.max_retire_frac
            budget_max = None

        # Override warmup_steps to 0 when auto_warmup is active (parent uses it)
        original_warmup = self.warmup_steps
        if self.auto_warmup:
            self.warmup_steps = 0

        n_retired = super().step(influence_scores)

        if self.auto_warmup:
            self.warmup_steps = original_warmup

        # Restore max_retire_frac after budget-constrained step
        if self.distortion_budget is not None or True:
            self.max_retire_frac = original_frac

        return n_retired

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stability_history(self) -> list[float]:
        return self._repr_delta_history

    def summary(self) -> dict:
        base = super().summary()
        base.update({
            "auto_warmup": self.auto_warmup,
            "stability_ready_at": self._auto_warmup_ended_at,
            "distortion_budget": self.distortion_budget,
            "distortion_used": self.cumulative_distortion_bound,
            "budget_utilisation": (
                self.cumulative_distortion_bound / self.distortion_budget
                if self.distortion_budget else None
            ),
        })
        return base
