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
    Approximates edge influence via a composite score combining:

      1. Gradient signal   — |∂L/∂w_e|  (how much edge affects current loss)
      2. Structural score  — degree-weighted bridge importance (protects hub edges)
      3. Representation maturity — variance of gradient over recent steps
                                   (stable = mature = safe to retire)

    This implements the three-score system described in the project brief,
    grounded in Theorem 7.4: optimal retirement when Be(t) = λ.

    Final composite score (lower = safer to retire):
        S_e(t) = α * grad_ema  +  β * structural  +  γ * maturity

    Args:
        num_edges:        m0, total initial edges
        device:           torch device
        edge_index:       [2, m0] — used to compute structural scores
        num_nodes:        n — for degree computation
        alpha:            weight for gradient signal  (default 0.5)
        beta:             weight for structural score (default 0.3)
        gamma:            weight for maturity signal  (default 0.2)
        ema_decay:        EMA smoothing for gradient signal
        maturity_window:  steps over which to measure gradient variance
    """

    def __init__(
        self,
        num_edges: int,
        device: torch.device,
        edge_index: Optional[Tensor] = None,
        num_nodes: int = 0,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        ema_decay: float = 0.9,
        maturity_window: int = 20,
    ):
        self.num_edges = num_edges
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._ema_decay = ema_decay
        self.maturity_window = maturity_window

        # Differentiable edge weights
        self.edge_weights = nn.Parameter(
            torch.ones(num_edges, device=device), requires_grad=True
        )

        # Score components
        self._ema_influence: Tensor = torch.zeros(num_edges, device=device)
        self._structural_score: Tensor = torch.zeros(num_edges, device=device)
        self._grad_history: list[Tensor] = []   # rolling window for maturity
        self._maturity_score: Tensor = torch.zeros(num_edges, device=device)

        # Precompute structural scores if edge_index provided
        if edge_index is not None and num_nodes > 0:
            self._compute_structural_scores(edge_index, num_nodes)

        logger.debug(f"GradientNormEstimator (composite): tracking {num_edges} edges | α={alpha} β={beta} γ={gamma}")

    def _compute_structural_scores(self, edge_index: Tensor, num_nodes: int) -> None:
        """
        Structural importance: normalised endpoint degree sum.
        High-degree edges (hubs) get a HIGH score → harder to retire.
        This protects bridge edges that maintain global connectivity.
        Scores are normalised to [0, 1].
        """
        from torch_geometric.utils import degree
        src, dst = edge_index[0], edge_index[1]
        deg = degree(dst, num_nodes, dtype=torch.float).to(self.device)

        # Score = (deg[u] + deg[v]) / max_possible — higher means more important
        raw = (deg[src] + deg[dst])
        max_deg = raw.max().clamp(min=1.0)
        self._structural_score = (raw / max_deg).to(self.device)
        logger.debug(f"Structural scores: mean={self._structural_score.mean():.3f} std={self._structural_score.std():.3f}")

    def update_influence(self, active_mask: Tensor) -> None:
        """
        After backward(): update all three score components.
        Call after loss.backward(), before optimizer.step().
        """
        if self.edge_weights.grad is None:
            logger.warning("update_influence: grad is None — skipping")
            return

        grad = self.edge_weights.grad.detach().abs()  # [m0]

        # 1. Gradient EMA
        self._ema_influence[active_mask] = (
            self._ema_decay * self._ema_influence[active_mask]
            + (1 - self._ema_decay) * grad[active_mask]
        )

        # 2. Maturity: low variance in recent gradients = stable = mature = safe
        self._grad_history.append(grad.clone())
        if len(self._grad_history) > self.maturity_window:
            self._grad_history.pop(0)

        if len(self._grad_history) >= 5:
            stacked = torch.stack(self._grad_history, dim=0)  # [W, m0]
            grad_var = stacked.var(dim=0)                     # [m0]
            # Normalise to [0,1]; low var (mature) → low maturity score
            max_var = grad_var.max().clamp(min=1e-10)
            self._maturity_score = (grad_var / max_var)

        self.edge_weights.grad.zero_()

    def influence_scores(self, active_mask: Tensor) -> Tensor:
        """
        Composite influence score S_e(t) for active edges.
        Lower = safer to retire.

        Strategy:
          - Structural score acts as a HARD GATE: edges in the top-k% by
            degree sum are never retired (score set to infinity).
          - For remaining edges, score = α * grad_ema + γ * maturity.
          - This avoids the normalisation conflict where structural weight
            inadvertently penalises low-degree edges that should be retired.

        Hub protection threshold: top 15% of edges by degree sum are locked.
        """
        scores = torch.full((self.num_edges,), float("inf"), device=self.device)

        if not active_mask.any():
            return scores

        # --- Hard gate: lock top-15% highest-degree active edges ---
        active_struct = self._structural_score.clone()
        active_struct[~active_mask] = -1.0  # exclude inactive

        n_active = active_mask.sum().item()
        n_locked = max(1, int(n_active * 0.15))
        _, top_idx = active_struct.topk(n_locked)
        locked_mask = torch.zeros(self.num_edges, dtype=torch.bool, device=self.device)
        locked_mask[top_idx] = True

        # Eligible = active AND not locked
        eligible_mask = active_mask & ~locked_mask

        if not eligible_mask.any():
            return scores

        # --- Soft score for eligible edges ---
        def _norm(t: Tensor, mask: Tensor) -> Tensor:
            vals = t[mask]
            mn, mx = vals.min(), vals.max()
            if (mx - mn) < 1e-12:
                out = torch.zeros(self.num_edges, device=self.device)
                out[mask] = 0.0
                return out
            out = torch.zeros(self.num_edges, device=self.device)
            out[mask] = (vals - mn) / (mx - mn)
            return out

        grad_norm     = _norm(self._ema_influence, eligible_mask)
        maturity_norm = _norm(self._maturity_score, eligible_mask)

        composite = self.alpha * grad_norm + self.gamma * maturity_norm
        scores[eligible_mask] = composite[eligible_mask]

        # Inactive edges: score = 0 (already retired, irrelevant)
        scores[~active_mask] = 0.0

        return scores

    def reset(self) -> None:
        self._ema_influence.zero_()
        self._maturity_score.zero_()
        self._grad_history.clear()
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
