"""
BaseTemporalGNN: abstract base class for GNN models compatible with
the temporal graph sparsification framework.

All models must accept:
  - x:           node features [n, d_0]
  - edge_index:  active edge index [2, m_t]  (changes each step)
  - edge_weight: optional edge weights [m_t]

The edge_weight argument is what connects the model to the
GradientNormEstimator — by making weights differentiable and
passing them through message passing, ∂L/∂w_e becomes computable.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from abc import ABC, abstractmethod


class BaseTemporalGNN(ABC, nn.Module):
    """
    Abstract base for temporal GNNs.

    Subclasses implement forward() accepting (x, edge_index, edge_weight).
    The base class provides:
      - get_representations(): returns intermediate H^(ℓ) for each layer
      - reset_parameters(): re-initialises all weight matrices
      - lipschitz_constants(): returns (K_ℓ, M_ℓ) for each layer
        (used to compute the theoretical CH bound from Theorem 4.4)
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        # Storage for intermediate representations (populated during forward)
        self._layer_representations: list[Tensor] = []

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:           [n, d_0]
            edge_index:  [2, m_t]
            edge_weight: [m_t] or None

        Returns:
            H_t = H^(L): final node representations [n, d_L]
        """
        ...

    def get_representations(self) -> list[Tensor]:
        """
        Return stored intermediate representations [H^(0), H^(1), ..., H^(L)].
        Populated after each forward() call.
        """
        return self._layer_representations

    def lipschitz_bound(self) -> float:
        """
        Compute C_H = K_1 * prod_{ℓ=2}^{L} M_ℓ from Theorem 4.4.

        Uses spectral norms of weight matrices as Lipschitz estimates.
        This gives a tractable upper bound on the theoretical influence bound.
        """
        norms = []
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                # Spectral norm ≈ largest singular value
                sv = torch.linalg.matrix_norm(param.data, ord=2).item()
                norms.append(sv)

        if not norms:
            return float("inf")

        # C_H = product of all layer Lipschitz constants
        ch = 1.0
        for sv in norms:
            ch *= sv
        return ch

    @abstractmethod
    def reset_parameters(self) -> None:
        """Re-initialise all learnable parameters."""
        ...
