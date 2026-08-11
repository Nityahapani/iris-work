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
    Composite influence estimator grounded in empirical correlation with exact Ie(t).

    Diagnostic finding (see results/estimator_comparison.json):
        Spearman correlation between structural features and exact Ie(t) at epoch 200:
          deg_product:  r = -0.648  (high product → low influence → safe to retire)
          er_proxy:     r = +0.621  (low degree → high influence → protect)
          grad_ema:     r ≈ 0.00    (gradient proxy is uncorrelated with Ie(t))

    Design (three components, all grounded in correlation analysis):

      1. Structural retirement score  [PRIMARY — static, precomputed]
         score_struct(e) = deg(u) * deg(v)
         Normalised to [0,1]. High = safe to retire (low Ie(t)).
         Derived from r=-0.648 correlation with exact Ie(t).

      2. Gradient momentum modulation [DYNAMIC — computed each step]
         Uses exponential moving average of |∂L/∂w_e|.
         High gradient → edge still actively used → INCREASE score (harder to retire).
         This is the only place gradients appear, and they modulate not determine.

      3. Maturity gate  [DYNAMIC — rolling variance]
         Low gradient variance = training has stabilised for this edge.
         Edges that stabilise early get a score REDUCTION (easier to retire).

    Final score (lower = safer to retire):
        S_e(t) = struct_score(e) * (1 + α * grad_ema_norm) * (1 - γ * maturity_norm)

    Multiplicative form ensures:
      - struct_score dominates (it has the strongest Ie(t) correlation)
      - gradient signal can only raise scores, never lower them below structural baseline
      - maturity can only lower scores once verified stable
    """

    def __init__(
        self,
        num_edges: int,
        device: torch.device,
        edge_index: Optional[Tensor] = None,
        num_nodes: int = 0,
        alpha: float = 0.3,       # gradient modulation strength
        beta: float = 0.0,        # unused (kept for API compat)
        gamma: float = 0.2,       # maturity discount strength
        ema_decay: float = 0.9,
        maturity_window: int = 20,
        hub_gate_pct: float = 0.10,   # top-k% by er_proxy are hard-locked
    ):
        self.num_edges = num_edges
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._ema_decay = ema_decay
        self.maturity_window = maturity_window
        self.hub_gate_pct = hub_gate_pct

        # Differentiable edge weights (needed for gradient flow)
        self.edge_weights = nn.Parameter(
            torch.ones(num_edges, device=device), requires_grad=True
        )

        # Structural retirement score: high deg_product = safe to retire
        self._struct_retire_score: Tensor = torch.zeros(num_edges, device=device)
        # ER-proxy score: high er_proxy = important bridge = protect
        self._er_proxy_score: Tensor = torch.zeros(num_edges, device=device)

        # Dynamic components
        self._ema_influence: Tensor = torch.zeros(num_edges, device=device)
        self._grad_history: list[Tensor] = []
        self._maturity_score: Tensor = torch.zeros(num_edges, device=device)

        if edge_index is not None and num_nodes > 0:
            self._compute_structural_scores(edge_index, num_nodes)

        logger.debug(
            f"GradientNormEstimator: {num_edges} edges | "
            f"α={alpha} γ={gamma} hub_gate={hub_gate_pct:.0%}"
        )

    def _compute_structural_scores(self, edge_index: Tensor, num_nodes: int) -> None:
        """
        Precompute two structural scores per edge:
          struct_retire_score: deg(u)*deg(v), normalised to [0,1]
                               High = safe to retire (empirical r=-0.648 with Ie(t))
          er_proxy_score:      1/deg(u) + 1/deg(v), normalised to [0,1]
                               High = bridge edge = protect (r=+0.621 with Ie(t))
        """
        from torch_geometric.utils import degree
        src, dst = edge_index[0], edge_index[1]
        deg = degree(dst, num_nodes, dtype=torch.float).to(self.device)

        # Retirement attractiveness: high deg_product = safe to retire
        deg_prod = deg[src] * deg[dst]
        mx = deg_prod.max().clamp(min=1.0)
        self._struct_retire_score = (deg_prod / mx)

        # Bridge importance: high er_proxy = dangerous to remove
        er = 1.0 / deg[src].clamp(min=1.0) + 1.0 / deg[dst].clamp(min=1.0)
        mx_er = er.max().clamp(min=1.0)
        self._er_proxy_score = (er / mx_er)

        logger.debug(
            f"Structural scores: retire_mean={self._struct_retire_score.mean():.3f} "
            f"er_mean={self._er_proxy_score.mean():.3f}"
        )

    def update_influence(self, active_mask: Tensor) -> None:
        """Update gradient EMA and maturity after backward().
        For GAT/SAGE where edge_weights may not be in the computation graph,
        we skip gradient update but still update maturity from structural scores."""
        if self.edge_weights.grad is None:
            # Edge weights not in computation graph (e.g. SAGE ignores them).
            # Still update maturity from gradient history if available.
            if len(self._grad_history) >= 5:
                stacked = torch.stack(self._grad_history, dim=0)
                grad_var = stacked.var(dim=0)
                mx_var = grad_var.max().clamp(min=1e-10)
                self._maturity_score = 1.0 - (grad_var / mx_var)
            return

        grad = self.edge_weights.grad.detach().abs()

        # EMA of gradient signal
        self._ema_influence[active_mask] = (
            self._ema_decay * self._ema_influence[active_mask]
            + (1 - self._ema_decay) * grad[active_mask]
        )

        # Rolling window for maturity
        self._grad_history.append(grad.clone())
        if len(self._grad_history) > self.maturity_window:
            self._grad_history.pop(0)

        if len(self._grad_history) >= 5:
            stacked = torch.stack(self._grad_history, dim=0)
            grad_var = stacked.var(dim=0)
            mx_var = grad_var.max().clamp(min=1e-10)
            # Low variance = mature (stable). Invert: low var → high maturity discount.
            self._maturity_score = 1.0 - (grad_var / mx_var)

        self.edge_weights.grad.zero_()

    def influence_scores(self, active_mask: Tensor) -> Tensor:
        """
        Composite score S_e(t). Lower = safer to retire.

        Hard gate: top hub_gate_pct of edges by er_proxy are locked (score=inf).
        These are the bridges with highest empirical Ie(t).

        For eligible edges:
            S_e = struct_retire_score * (1 + α * grad_norm) * (1 - γ * maturity_norm)

        Note that struct_retire_score is HIGH for safe edges (high deg_product).
        We INVERT at the end: retirement_priority = 1 - S_e_norm so that
        low-S edges (dangerous) get high scores that exceed ε threshold.

        Wait — scheduler retires edges with score BELOW ε.
        So we want: low score = safe to retire.
        struct_retire_score is already: high = safe.
        We want score = 1 - struct_retire_score_normalised (invert).
        But then gradient modulation should RAISE score for active-gradient edges.

        Final: score = (1 - struct_norm) * (1 + α*grad_norm) / (1 + γ*maturity_norm)
        - low struct (low deg_product = dangerous bridge) → high score → not retired
        - high grad (edge actively used) → higher score → not retired
        - high maturity (stable, not needed) → lower score → retired sooner
        """
        scores = torch.full((self.num_edges,), float("inf"), device=self.device)

        if not active_mask.any():
            return scores

        # --- Hard gate: lock top hub_gate_pct by er_proxy ---
        er_active = self._er_proxy_score.clone()
        er_active[~active_mask] = -1.0
        n_active = int(active_mask.sum().item())
        n_locked = max(1, int(n_active * self.hub_gate_pct))
        _, top_idx = er_active.topk(n_locked)
        locked = torch.zeros(self.num_edges, dtype=torch.bool, device=self.device)
        locked[top_idx] = True
        eligible = active_mask & ~locked

        if not eligible.any():
            scores[~active_mask] = 0.0
            return scores

        # --- Normalise each component over eligible edges ---
        def _norm(t: Tensor, mask: Tensor) -> Tensor:
            vals = t[mask]
            mn, mx = vals.min(), vals.max()
            out = torch.zeros(self.num_edges, device=self.device)
            if (mx - mn) > 1e-12:
                out[mask] = (vals - mn) / (mx - mn)
            return out

        struct_norm   = _norm(self._struct_retire_score, eligible)
        grad_norm     = _norm(self._ema_influence,       eligible)
        maturity_norm = _norm(self._maturity_score,      eligible)

        # Composite: invert struct so low-deg-product edges get HIGH score
        # (high score = dangerous = don't retire)
        danger  = (1.0 - struct_norm)                    # low deg_prod → high danger
        boosted = danger * (1.0 + self.alpha * grad_norm) # active gradient → more dangerous
        # Maturity discounts danger: stable edge → lower score → retire sooner
        final   = boosted / (1.0 + self.gamma * maturity_norm)

        scores[eligible] = final[eligible]
        scores[~active_mask] = 0.0
        return scores

    def reset(self) -> None:
        self._ema_influence.zero_()
        self._maturity_score.zero_()
        self._grad_history.clear()
        if self.edge_weights.grad is not None:
            self.edge_weights.grad.zero_()


class AdjacencySensitivityEstimator:
    """
    Computes edge influence via sensitivity of node representations
    to perturbation of the normalised adjacency entry A_uv.

    For GCN:  H^(1) = sigma(A_hat H^(0) W)
    The sensitivity of H_v to A_uv is:
        dH_v / dA_uv = H_u^(0) W  (for 1-layer; propagated for deeper)

    This is a CLOSED-FORM approximation of Ie(t) = ||H - H^{-e}||_F
    for linear/first-order perturbation, computed without any extra
    forward passes. Far more correlated with true Ie(t) than raw gradient.

    Connection to Theorem 4.4:
        The perturbation bound ||H - H^{-e}||_F <= CH * ||Delta_e||_F
        motivates using the Jacobian dH/dA as the influence proxy.
        We estimate the Jacobian cheaply from the weight matrices.

    Args:
        model:      TemporalGCN (we read weight matrices from it)
        num_edges:  m0
        device:     torch device
        ema_decay:  smoothing
    """

    def __init__(self, model, num_edges: int, device: torch.device, ema_decay: float = 0.95):
        self.model = model
        self.num_edges = num_edges
        self.device = device
        self._ema_decay = ema_decay

        # Differentiable edge weights (still needed for gradient tracking)
        self.edge_weights = nn.Parameter(
            torch.ones(num_edges, device=device), requires_grad=True
        )
        self._ema_sensitivity: Tensor = torch.zeros(num_edges, device=device)
        self._structural_score: Tensor = torch.zeros(num_edges, device=device)
        logger.debug(f'AdjacencySensitivityEstimator: {num_edges} edges')

    def init_structural(self, edge_index: Tensor, num_nodes: int) -> None:
        from torch_geometric.utils import degree
        src, dst = edge_index[0], edge_index[1]
        deg = degree(dst, num_nodes, dtype=torch.float).to(self.device)
        raw = deg[src] + deg[dst]
        self._structural_score = (raw / raw.max().clamp(min=1.0)).to(self.device)

    @torch.no_grad()
    def update_sensitivity(self, x: Tensor, edge_index: Tensor, active_mask: Tensor) -> None:
        """
        Compute closed-form adjacency sensitivity for all active edges.

        For each edge (u,v): sensitivity = ||H_u|| * ||W_1|| (first layer proxy).
        This approximates how much representation H_v would change if A_uv changed.
        Averaged across the L layers via the C_H product structure from Theorem 4.4.
        """
        src = edge_index[0]  # active edge sources
        full_src = self.model._edge_index_full_src if hasattr(self.model, '_edge_index_full_src') else None

        # Get weight norms from each GCN layer
        layer_norms = []
        for conv in self.model.convs:
            W = conv.lin.weight if hasattr(conv, 'lin') else conv.weight
            layer_norms.append(torch.linalg.matrix_norm(W, ord=2).item())

        ch = 1.0
        for n in layer_norms:
            ch *= n

        # Node feature norms ||h_u|| for source nodes
        # Use current representation H^(0) = x (input features)
        x_norm = x.norm(dim=-1)  # [n]

        # For each active edge (u,v): sensitivity ~ ||x_u|| * C_H
        # (first-order approximation of the multi-layer perturbation bound)
        active_idx = active_mask.nonzero(as_tuple=False).squeeze(1)

        # Map active edges back to full edge_index
        # active edge_index is tg.edge_index; full src is tg._edge_index_full[0]
        # We approximate: use the active edge src nodes
        sensitivity = torch.zeros(self.num_edges, device=self.device)
        sensitivity[active_idx] = x_norm[src] * ch

        # EMA update
        self._ema_sensitivity[active_mask] = (
            self._ema_decay * self._ema_sensitivity[active_mask]
            + (1 - self._ema_decay) * sensitivity[active_mask]
        )

        # Zero gradient if present
        if self.edge_weights.grad is not None:
            self.edge_weights.grad.zero_()

    def influence_scores(self, active_mask: Tensor) -> Tensor:
        """
        Composite score: sensitivity (Jacobian proxy) + structural gate.
        Lower = safer to retire.
        """
        scores = torch.full((self.num_edges,), float('inf'), device=self.device)

        # Hard gate: lock top 15% by degree
        struct = self._structural_score.clone()
        struct[~active_mask] = -1.0
        n_active = active_mask.sum().item()
        n_locked = max(1, int(n_active * 0.15))
        _, top_idx = struct.topk(n_locked)
        locked = torch.zeros(self.num_edges, dtype=torch.bool, device=self.device)
        locked[top_idx] = True

        eligible = active_mask & ~locked
        if not eligible.any():
            scores[~active_mask] = 0.0
            return scores

        # Normalise sensitivity for eligible edges
        sens = self._ema_sensitivity.clone()
        vals = sens[eligible]
        mn, mx = vals.min(), vals.max()
        if (mx - mn) > 1e-12:
            sens_norm = torch.zeros(self.num_edges, device=self.device)
            sens_norm[eligible] = (vals - mn) / (mx - mn)
        else:
            sens_norm = torch.zeros(self.num_edges, device=self.device)

        scores[eligible] = sens_norm[eligible]
        scores[~active_mask] = 0.0
        return scores


class RepresentationDeltaEstimator:
    """
    Exact influence estimator: Ie(t) = ||H_t - H_t^{-e}||_F via two forward passes.
    O(m) forward passes per step — only for ablation studies on small graphs.
    """

    def __init__(self, model, x: Tensor, temporal_graph):
        self.model = model
        self.x = x
        self.temporal_graph = temporal_graph

        # Dummy edge_weights parameter (not used in scoring, but needed for API compat)
        self.edge_weights = nn.Parameter(
            torch.ones(temporal_graph.m0, device=x.device), requires_grad=False
        )

    @torch.no_grad()
    def compute_influence(self, active_indices: Tensor, batch_size: int = 64) -> Tensor:
        m0 = self.temporal_graph.m0
        influence = torch.zeros(m0, device=self.x.device)
        edge_index_full = self.temporal_graph.edge_index
        self.model.eval()
        H_full = self.model(self.x, edge_index_full)
        for start in range(0, len(active_indices), batch_size):
            batch = active_indices[start: start + batch_size]
            for idx in batch:
                idx_int = int(idx.item())
                ei_minus = self.temporal_graph.edge_index_without(idx_int)
                H_minus = self.model(self.x, ei_minus)
                influence[idx_int] = (H_full - H_minus).norm(p='fro').item()
        return influence

    def compute_influence_single(self, edge_idx: int) -> float:
        with torch.no_grad():
            H_full = self.model(self.x, self.temporal_graph.edge_index)
            ei_minus = self.temporal_graph.edge_index_without(edge_idx)
            H_minus = self.model(self.x, ei_minus)
        return (H_full - H_minus).norm(p='fro').item()


def build_estimator(
    method: str,
    num_edges: int,
    device: torch.device,
    model: Optional[nn.Module] = None,
    x: Optional[Tensor] = None,
    temporal_graph=None,
) -> "GradientNormEstimator | AdjacencySensitivityEstimator | RepresentationDeltaEstimator":
    if method == "gradient":
        return GradientNormEstimator(num_edges, device)
    elif method == "sensitivity":
        assert model is not None, "AdjacencySensitivityEstimator requires model"
        return AdjacencySensitivityEstimator(model, num_edges, device)
    elif method == "exact":
        assert model is not None and x is not None and temporal_graph is not None
        return RepresentationDeltaEstimator(model, x, temporal_graph)
    else:
        raise ValueError(f"Unknown estimator: {method}. Choose gradient/sensitivity/exact.")
