"""
FLOPsCounter: estimates FLOPs per training step for GCN-style models.

Theory connection (Proposition 9.1):
    Standard message-passing GNNs cost O(m_t * d) per layer per step.
    Total cost: C_temp = Σ_{t=0}^{T} O(m_t * d) ≤ T * O(m_0 * d) = C_dense
    Savings ratio: 1 - (1/T) Σ m_t/m_0

We estimate FLOPs as the number of edge-feature multiplications:
    FLOPs_per_step ≈ L * m_t * d_hidden * 2  (multiply-add = 2 ops)

This is a lower bound on actual FLOPs but gives a clean theoretical comparison.
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FLOPsCounter:
    """
    Tracks theoretical FLOPs across training steps.

    Args:
        m0:          initial number of edges |E_0|
        num_layers:  L, number of GNN layers
        hidden_dim:  d, hidden feature dimension
    """

    def __init__(self, m0: int, num_layers: int, hidden_dim: int):
        self.m0 = m0
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self._step_flops: list[int] = []
        self._step_mt: list[int] = []

    def record_step(self, mt: int) -> int:
        """
        Record the FLOPs for one training step with mt active edges.

        Args:
            mt: |E_t|, number of active edges at this step

        Returns:
            Estimated FLOPs for this step
        """
        # L layers * m_t edges * d features * 2 (multiply-add)
        flops = self.num_layers * mt * self.hidden_dim * 2
        self._step_flops.append(flops)
        self._step_mt.append(mt)
        return flops

    # ------------------------------------------------------------------
    # Aggregated statistics
    # ------------------------------------------------------------------

    @property
    def total_flops(self) -> int:
        return sum(self._step_flops)

    @property
    def dense_flops(self) -> int:
        """FLOPs if we had trained on the full graph all T steps."""
        T = len(self._step_flops)
        return T * self.num_layers * self.m0 * self.hidden_dim * 2

    @property
    def savings_ratio(self) -> float:
        """
        Proposition 9.1 savings ratio:
            1 - (1/T) Σ m_t/m_0
        """
        if not self._step_mt:
            return 0.0
        mean_mt = np.mean(self._step_mt)
        return 1.0 - mean_mt / self.m0

    @property
    def flops_reduction(self) -> float:
        """Actual FLOPs reduction vs dense baseline."""
        if self.dense_flops == 0:
            return 0.0
        return 1.0 - self.total_flops / self.dense_flops

    def summary(self) -> dict:
        return {
            "total_flops": self.total_flops,
            "dense_flops": self.dense_flops,
            "flops_reduction": self.flops_reduction,
            "savings_ratio_theoretical": self.savings_ratio,
            "mean_mt": float(np.mean(self._step_mt)) if self._step_mt else 0.0,
            "m0": self.m0,
            "num_steps": len(self._step_flops),
        }

    def step_history(self) -> list[dict]:
        return [
            {"step": i, "mt": self._step_mt[i], "flops": self._step_flops[i]}
            for i in range(len(self._step_flops))
        ]
