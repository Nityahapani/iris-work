"""
TemporalGCN: GCN implementation compatible with the temporal graph framework.

Architecture follows Kipf & Welling (2017):
    H^(ℓ+1) = σ(Â H^(ℓ) W_ℓ)
where Â = D^{-1/2}(A+I)D^{-1/2} is recomputed each step from the
current active edge set E_t.

Key difference from standard GCN: edge_weight is a differentiable parameter
passed in from GradientNormEstimator, enabling ∂L/∂w_e computation.

Theory connection (Remark 3.5):
    Assumption 3.2 holds with K_ℓ = ‖H W_ℓ‖_F / ‖H‖_F
    Assumption 3.3 holds with M_ℓ = ‖Â‖_F ‖W_ℓ‖_F
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree
from typing import Optional

from .base import BaseTemporalGNN


class TemporalGCN(BaseTemporalGNN):
    """
    GCN with temporal (dynamic) edge support.

    Args:
        in_channels:     d_0, input feature dimension
        hidden_channels: d_ℓ, hidden dimension (same for all layers)
        out_channels:    number of classes (d_L)
        num_layers:      L, number of message-passing layers
        dropout:         dropout probability applied between layers
        normalize:       if True, apply symmetric normalization (Â)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        normalize: bool = True,
    ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers)
        self.dropout = dropout
        self.normalize = normalize

        # Build L message-passing layers
        self.convs = nn.ModuleList()
        for ℓ in range(num_layers):
            in_ch = in_channels if ℓ == 0 else hidden_channels
            out_ch = out_channels if ℓ == num_layers - 1 else hidden_channels
            # normalize=False: we handle normalization ourselves to support
            # differentiable edge weights passed in from the influence estimator
            self.convs.append(
                GCNConv(in_ch, out_ch, normalize=False, add_self_loops=False)
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with optional differentiable edge weights.

        Args:
            x:           node features [n, d_0]
            edge_index:  active edges [2, m_t]
            edge_weight: [m_t] — if provided, these are the differentiable
                         weights from GradientNormEstimator

        Returns:
            H_t: node representations [n, d_L]
        """
        self._layer_representations = [x]

        # Compute normalized adjacency for current E_t
        norm_edge_index, norm_weight = self._normalize(edge_index, edge_weight, x.device)

        h = x
        for ℓ, conv in enumerate(self.convs):
            h = conv(h, norm_edge_index, norm_weight)

            is_last = (ℓ == self.num_layers - 1)
            if not is_last:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            self._layer_representations.append(h)

        return h  # H_t = H^(L), shape [n, d_L]

    def predict(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        """Return softmax probabilities Ŷ_t."""
        logits = self.forward(x, edge_index, edge_weight)
        return F.log_softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Internal: symmetric normalization supporting edge weights
    # ------------------------------------------------------------------

    def _normalize(
        self,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        device: torch.device,
    ):
        """
        Compute Â = D^{-1/2}(A+I)D^{-1/2} with optional edge weights.
        Self-loops added with weight 1.
        """
        num_nodes = self._infer_num_nodes(edge_index)
        n = num_nodes

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)

        # Add self-loops (weight 1)
        edge_index_sl, edge_weight_sl = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=n
        )

        # Degree with self-loops
        row, col = edge_index_sl
        deg = degree(col, n, dtype=torch.float32)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

        # Normalized weights
        norm_w = deg_inv_sqrt[row] * edge_weight_sl * deg_inv_sqrt[col]

        return edge_index_sl, norm_w

    def _infer_num_nodes(self, edge_index: Tensor) -> int:
        return int(edge_index.max().item()) + 1
