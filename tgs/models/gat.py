"""
tgs/models/gat.py

Graph Attention Network compatible with the temporal graph framework.
Architecture: Veličković et al. (2018), GAT with multi-head attention.

Key difference from GCN: attention weights are learned per-edge, per-head.
The influence estimator works identically — edge weights w_e are passed
through the attention mechanism, making ∂L/∂w_e computable.

Theory connection (Remark 3.5):
    GAT satisfies Assumptions 3.2-3.4 with K_ℓ = max attention weight * ||W_ℓ||.
    The Lipschitz constants are bounded because attention coefficients sum to 1
    (softmax normalisation), so ||Â||_F ≤ sqrt(n) for n-node graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv
from typing import Optional

from .base import BaseTemporalGNN


class TemporalGAT(BaseTemporalGNN):
    """
    GAT with temporal (dynamic) edge support.

    Args:
        in_channels:     input feature dimension
        hidden_channels: hidden dimension per head
        out_channels:    number of output classes
        num_layers:      number of GAT message-passing layers
        heads:           number of attention heads (intermediate layers)
        dropout:         dropout on features and attention weights
        concat:          if True, concatenate heads; if False, average them
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.6,
        concat: bool = True,
    ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers)
        self.heads   = heads
        self.dropout = dropout
        self.concat  = concat

        self.convs = nn.ModuleList()
        for ℓ in range(num_layers):
            is_last = (ℓ == num_layers - 1)
            in_ch   = in_channels if ℓ == 0 else hidden_channels * (heads if concat else 1)
            out_ch  = out_channels if is_last else hidden_channels
            h       = 1 if is_last else heads
            cat     = False if is_last else concat
            self.convs.append(
                GATConv(in_ch, out_ch, heads=h, dropout=dropout,
                        concat=cat, add_self_loops=True)
            )

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
        edge_weight is passed as edge_attr to GATConv.
        GATConv supports scalar edge features which modulate attention.
        """
        self._layer_representations = [x]
        h = F.dropout(x, p=self.dropout, training=self.training)

        # Reshape edge_weight for GATConv: needs [m, 1] or None
        ea = edge_weight.unsqueeze(-1) if edge_weight is not None else None

        for ℓ, conv in enumerate(self.convs):
            is_last = (ℓ == self.num_layers - 1)
            h = conv(h, edge_index, edge_attr=ea)
            if not is_last:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            self._layer_representations.append(h)

        return h

    def predict(self, x: Tensor, edge_index: Tensor,
                edge_weight: Optional[Tensor] = None) -> Tensor:
        return F.log_softmax(self.forward(x, edge_index, edge_weight), dim=-1)
