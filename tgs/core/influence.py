"""
JacobianInfluenceEstimator: gradient-based approximation of edge influence Ie(t).

Theory context (Definition 4.2):
    Ie(t) = ‖H_t - H_t^{-e}‖_F

Exact computation requires two forward passes per edge — O(m) passes total,
which is intractable. We approximate using the first-order Taylor expansion:

    H_t^(-e) approx H_t + (dH_t/dA_t) * (-Delta_e)

So:
    Ie(t) approx norm((dH_t/dA_t) * Delta_e, 'fro')

In practice, we use the gradient of the loss w.r.t. each edge weight as a
proxy: a high-gradient edge contributes more to the current optimization
step, so it's more dangerous to retire. A low-gradient edge has negligible
marginal contribution — safe to retire per Definition 5.1.

Two estimators are implemented:
  1. GradientNormEstimator  — ‖∂L/∂w_e‖  (cheap, computed from existing backward pass)
  2. RepresentationDeltaEstimator — exact Ie(t) for small graphs / ablation studies

The scheduler always uses GradientNormEstimator by default; RepresentationDelta
is available for ablation experiments.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GradientNormEstimator:
    """
    Approximates edge influence via gradient norm of the loss w.r.t. edge weights.

        Î_e(t) = |∂L/∂w_e|

    This is computed for free after the backward pass: edge weights w_e are
    treated as leaf parameters. The gradient ∂L/∂w_e reflects how much the
    loss changes per unit change in that edge's contribution — a direct proxy
    for how much information the edge is still contributing to training.

    Connection to theory:
        From Theorem 7.4, the optimal retirement condition is Be(t) = λ, where
        Be(t) = L(Yhat^(E_t minus e), Y*) - L(Yhat^(E_t), Y*) approx -dL/dw_e * Delta_w_e.
        Thus |∂L/∂w_e| = 0 signals Be(t) = 0, i.e., the edge contributes
        nothing to the loss — a natural retirement signal.

    Args:
        num_edges: m0, total initial edges
        device: torch device
    """

    def __init__(self, num_edges: int, device: torch.device):
        self.num_edges = num_edges
        self.device = device

        # Learnable edge weights initialized to 1 (unweighted graph)
        # These are differentiable, so ∂L/∂w_e is available after backward()
        self.edge_weights = nn.Parameter(
            torch.ones(num_edges, device=device), requires_grad=True
        )

        # EMA of gradient norms for smoothing (reduces noise)
        self._ema_influence: Tensor = torch.zeros(num_edges, device=device)
        self._ema_decay: float = 0.9

        logger.debug(f"GradientNormEstimator: tracking {num_edges} edges")

    def active_weights(self, active_mask: Tensor) -> Tensor:
        """
        Return edge weights for the currently active edges.
        The model uses these weights in message passing — masked to active edges.
        """
        return self.edge_weights[active_mask]

    def update_influence(self, active_mask: Tensor) -> None:
        """
        After backward(), read ∂L/∂w_e for active edges and update EMA.
        Call this after loss.backward() and before optimizer.step().
        """
        if self.edge_weights.grad is None:
            logger.warning("update_influence called but edge_weights.grad is None — skipping")
            return

        grad = self.edge_weights.grad.detach().abs()  # |∂L/∂w_e|, shape [m0]

        # EMA update: only update active edges
        self._ema_influence[active_mask] = (
            self._ema_decay * self._ema_influence[active_mask]
            + (1 - self._ema_decay) * grad[active_mask]
        )

        # Zero grad for next step
        self.edge_weights.grad.zero_()

    def influence_scores(self, active_mask: Tensor) -> Tensor:
        """
        Return smoothed influence score Î_e(t) for active edges.
        Lower score = safer to retire.

        Returns:
            Tensor of shape [m0]; inactive edges have score 0.
        """
        scores = self._ema_influence.clone()
        scores[~active_mask] = 0.0  # inactive edges irrelevant
        return scores

    def reset(self) -> None:
        """Reset EMA — call if starting a fresh training run."""
        self._ema_influence.zero_()
        if self.edge_weights.grad is not None:
            self.edge_weights.grad.zero_()


class RepresentationDeltaEstimator:
    """
    Exact influence estimator: computes Ie(t) = ‖H_t - H_t^{-e}‖_F
    via two forward passes — one with and one without edge e.

    This is O(m) forward passes per step — only feasible for small graphs
    (Cora/CiteSeer) and used exclusively in ablation experiments to validate
    that GradientNormEstimator is a good proxy.

    Args:
        model: the GNN model (must accept edge_index as argument)
        x: node feature matrix [n, d0]
        active_mask: current active mask over E_0
        temporal_graph: the TemporalGraph instance
    """

    def __init__(self, model: nn.Module, x: Tensor, temporal_graph):
        self.model = model
        self.x = x
        self.temporal_graph = temporal_graph

    @torch.no_grad()
    def compute_influence(
        self,
        active_indices: Tensor,
        batch_size: int = 64,
    ) -> Tensor:
        """
        Compute exact Ie(t) for all active edges.

        Args:
            active_indices: indices into E_0 of active edges
            batch_size: number of edges to evaluate per batch

        Returns:
            Tensor of shape [m0]; inactive edges have value 0.
        """
        m0 = self.temporal_graph.m0
        influence = torch.zeros(m0, device=self.x.device)

        # Full-graph representation H_t
        edge_index_full = self.temporal_graph.edge_index
        self.model.eval()
        H_full = self.model(self.x, edge_index_full)  # [n, d_L]

        for start in range(0, len(active_indices), batch_size):
            batch = active_indices[start : start + batch_size]
            for idx in batch:
                idx_int = int(idx.item())
                # H_t^{-e}: representation without edge idx
                edge_index_minus_e = self.temporal_graph.edge_index_without(idx_int)
                H_minus = self.model(self.x, edge_index_minus_e)
                ie = (H_full - H_minus).norm(p="fro").item()
                influence[idx_int] = ie

        return influence

    def compute_influence_single(self, edge_idx: int) -> float:
        """
        Compute Ie(t) for a single edge. Used in sequential retirement (Section 6).
        """
        edge_index_full = self.temporal_graph.edge_index
        with torch.no_grad():
            H_full = self.model(self.x, edge_index_full)
            edge_index_minus = self.temporal_graph.edge_index_without(edge_idx)
            H_minus = self.model(self.x, edge_index_minus)
        return (H_full - H_minus).norm(p="fro").item()


def build_estimator(
    method: str,
    num_edges: int,
    device: torch.device,
    model: Optional[nn.Module] = None,
    x: Optional[Tensor] = None,
    temporal_graph=None,
) -> GradientNormEstimator | RepresentationDeltaEstimator:
    """
    Factory for influence estimators.

    Args:
        method: 'gradient' (default, fast) or 'exact' (ablation only)
    """
    if method == "gradient":
        return GradientNormEstimator(num_edges, device)
    elif method == "exact":
        assert model is not None and x is not None and temporal_graph is not None, \
            "RepresentationDeltaEstimator requires model, x, and temporal_graph"
        return RepresentationDeltaEstimator(model, x, temporal_graph)
    else:
        raise ValueError(f"Unknown influence estimator: {method}. Choose 'gradient' or 'exact'.")
