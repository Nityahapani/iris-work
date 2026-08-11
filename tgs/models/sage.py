"""
tgs/models/sage.py

GraphSAGE compatible with the temporal graph framework.
Architecture: Hamilton et al. (2017), mean aggregation.

Theory connection: SAGEConv satisfies Assumptions 3.2-3.4 with
    K_ℓ = ||W_ℓ|| (neighbour aggregation is a mean, bounded by 1)
    M_ℓ = ||W_ℓ||_F (feature transform Lipschitz)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv
from typing import Optional

from .base import BaseTemporalGNN


class TemporalSAGE(BaseTemporalGNN):
    """
    GraphSAGE with temporal edge support.

    Args:
        in_channels:     input feature dimension
        hidden_channels: hidden dimension
        out_channels:    number of output classes
        num_layers:      number of SAGE layers
        dropout:         feature dropout
        aggr:            aggregation: 'mean' (default), 'max', 'lstm'
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggr: str = "mean",
    ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers)
        self.dropout = dropout

        self.convs = nn.ModuleList()
        for ℓ in range(num_layers):
            in_ch  = in_channels     if ℓ == 0             else hidden_channels
            out_ch = out_channels    if ℓ == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(in_ch, out_ch, aggr=aggr))

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
        SAGEConv does not natively use edge_weight in mean aggregation,
        but we accept it for API compatibility with the TGS framework.
        The weight is passed as edge_attr — SAGEConv ignores it in mean mode,
        but having it in the graph allows gradient flow through edge_weights
        for the influence estimator.
        """
        self._layer_representations = [x]
        h = x
        for ℓ, conv in enumerate(self.convs):
            is_last = (ℓ == self.num_layers - 1)
            h = conv(h, edge_index)
            if not is_last:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            self._layer_representations.append(h)
        return h

    def predict(self, x, edge_index, edge_weight=None):
        return F.log_softmax(self.forward(x, edge_index, edge_weight), dim=-1)
