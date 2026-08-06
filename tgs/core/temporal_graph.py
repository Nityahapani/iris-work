"""
TemporalGraph: implements the temporal graph sequence {G_t}_{t=0}^T
and retirement schedule τ : E_0 → {0,1,...,T,+∞}.

Theory refs:
  Definition 2.1  — Temporal Graph Sequence (monotone retirement)
  Definition 2.2  — Retirement Schedule τ(e)
  Remark 2.3      — S strictly contains static sparsification decisions
"""

import torch
from torch import Tensor
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TemporalGraph:
    """
    Maintains the active edge set E_t across training steps.

    The core invariant is monotone retirement:
        E_{t+1} ⊆ E_t  for all t  (Definition 2.1)

    Edges are never reintroduced once retired. The retirement schedule
    τ(e) records, for each edge, the first step at which it was removed.
    τ(e) = +∞  (represented as -1 here) means the edge is never retired.

    Args:
        edge_index: LongTensor of shape [2, m] — initial edge set E_0
        num_nodes:  number of nodes n
        device:     torch device
    """

    INF = -1  # sentinel for "never retired"

    def __init__(
        self,
        edge_index: Tensor,
        num_nodes: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_nodes = num_nodes
        self.device = device

        # E_0: initial edge set, stored as [2, m] LongTensor
        self._edge_index_full: Tensor = edge_index.to(device)
        self._m0: int = edge_index.shape[1]  # |E_0|

        # Active mask: True if edge is still in E_t
        self._active: Tensor = torch.ones(self._m0, dtype=torch.bool, device=device)

        # Retirement schedule τ : edge_idx → step retired (INF if never)
        self._retirement_step: Tensor = torch.full(
            (self._m0,), fill_value=self.INF, dtype=torch.long, device=device
        )

        # Current training step t
        self._t: int = 0

        # Running count of retirements
        self._total_retired: int = 0

        logger.info(
            f"TemporalGraph initialised | n={num_nodes}, m0={self._m0}, device={device}"
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the training step counter."""
        self._t += 1

    @property
    def t(self) -> int:
        return self._t

    @property
    def m0(self) -> int:
        """Initial number of edges |E_0|."""
        return self._m0

    @property
    def mt(self) -> int:
        """Current number of active edges |E_t|."""
        return int(self._active.sum().item())

    @property
    def sparsity(self) -> float:
        """Fraction of edges retired so far: 1 - |E_t|/|E_0|."""
        return 1.0 - self.mt / self._m0

    # ------------------------------------------------------------------
    # Active edge access
    # ------------------------------------------------------------------

    @property
    def edge_index(self) -> Tensor:
        """Return the active edge index E_t as [2, mt] LongTensor."""
        return self._edge_index_full[:, self._active]

    @property
    def active_mask(self) -> Tensor:
        """Boolean mask over E_0; True = edge is active."""
        return self._active

    @property
    def active_indices(self) -> Tensor:
        """Indices into E_0 of currently active edges."""
        return self._active.nonzero(as_tuple=False).squeeze(1)

    # ------------------------------------------------------------------
    # Retirement
    # ------------------------------------------------------------------

    def retire_edges(self, edge_indices: Tensor) -> int:
        """
        Retire a batch of edges (by their index in E_0).

        Implements the monotone retirement condition: once retired,
        an edge is never reactivated (Definition 2.1).

        Args:
            edge_indices: 1-D LongTensor of indices into E_0

        Returns:
            Number of edges actually retired (excludes already-retired edges)
        """
        # Only retire edges that are still active
        newly_retired = edge_indices[self._active[edge_indices]]

        if len(newly_retired) == 0:
            return 0

        self._active[newly_retired] = False
        self._retirement_step[newly_retired] = self._t
        self._total_retired += len(newly_retired)

        logger.debug(
            f"Step {self._t}: retired {len(newly_retired)} edges "
            f"| total retired: {self._total_retired}/{self._m0} "
            f"| sparsity: {self.sparsity:.3f}"
        )
        return len(newly_retired)

    def retire_by_mask(self, mask: Tensor) -> int:
        """
        Retire edges identified by a boolean mask over E_0.

        Args:
            mask: BoolTensor of shape [m0]; True = retire this edge

        Returns:
            Number of edges actually retired
        """
        indices = (mask & self._active).nonzero(as_tuple=False).squeeze(1)
        return self.retire_edges(indices)

    # ------------------------------------------------------------------
    # Retirement schedule queries
    # ------------------------------------------------------------------

    def retirement_time(self, edge_idx: int) -> int:
        """
        Return τ(e) for edge at position edge_idx in E_0.
        Returns TemporalGraph.INF (-1) if never retired.
        """
        return int(self._retirement_step[edge_idx].item())

    def get_schedule(self) -> Tensor:
        """
        Return the full retirement schedule τ as a LongTensor of shape [m0].
        Unretired edges have value INF = -1.
        """
        return self._retirement_step.clone()

    # ------------------------------------------------------------------
    # Counterfactual graph (for influence computation)
    # ------------------------------------------------------------------

    def edge_index_without(self, edge_idx: int) -> Tensor:
        """
        Return active edge index with edge edge_idx additionally removed.
        Used to compute H_t^{-e} in influence estimation (Definition 4.2).
        """
        mask = self._active.clone()
        mask[edge_idx] = False
        return self._edge_index_full[:, mask]

    def edge_index_without_set(self, edge_indices: Tensor) -> Tensor:
        """
        Return active edge index with a set S of edges removed.
        Used in multiple-edge retirement analysis (Section 6).
        """
        mask = self._active.clone()
        mask[edge_indices] = False
        return self._edge_index_full[:, mask]

    # ------------------------------------------------------------------
    # Stats / logging
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "step": self._t,
            "m0": self._m0,
            "mt": self.mt,
            "retired": self._total_retired,
            "sparsity": self.sparsity,
        }

    def __repr__(self) -> str:
        return (
            f"TemporalGraph(n={self.num_nodes}, m0={self._m0}, "
            f"mt={self.mt}, t={self._t}, sparsity={self.sparsity:.3f})"
        )
