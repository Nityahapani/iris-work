"""
EdgeManager: utilities for edge bookkeeping, normalization, and
efficient sparse adjacency construction from an active edge set.

Handles the normalized adjacency Â = D^{-1/2} A D^{-1/2} used in
GCN-style message passing (referenced in Remark 3.5 of the theory).
"""

import torch
from torch import Tensor
from torch_geometric.utils import add_self_loops, degree
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EdgeManager:
    """
    Thin wrapper that provides:
      1. Edge-to-index lookups (fast O(1) with a hash map)
      2. Normalized adjacency computation from an active edge index
      3. Edge weight initialization (uniform = 1 for unweighted graphs)

    Args:
        edge_index: initial edge index [2, m0]
        num_nodes:  n
        add_self_loops_flag: whether to include self-loops in normalization
    """

    def __init__(
        self,
        edge_index: Tensor,
        num_nodes: int,
        add_self_loops_flag: bool = True,
    ):
        self.num_nodes = num_nodes
        self.add_self_loops_flag = add_self_loops_flag

        # Build (u, v) -> edge_index lookup
        src, dst = edge_index[0], edge_index[1]
        self._edge_to_idx: dict = {}
        for idx in range(edge_index.shape[1]):
            u, v = int(src[idx].item()), int(dst[idx].item())
            self._edge_to_idx[(u, v)] = idx

        logger.debug(f"EdgeManager: indexed {edge_index.shape[1]} edges for {num_nodes} nodes")

    # ------------------------------------------------------------------
    # Index lookups
    # ------------------------------------------------------------------

    def edge_idx(self, u: int, v: int) -> Optional[int]:
        """Return the index in E_0 for directed edge (u, v), or None."""
        return self._edge_to_idx.get((u, v), None)

    def edge_exists(self, u: int, v: int) -> bool:
        return (u, v) in self._edge_to_idx

    # ------------------------------------------------------------------
    # Normalized adjacency
    # ------------------------------------------------------------------

    def normalized_adjacency(
        self,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the symmetrically normalized adjacency:
            Â = D^{-1/2} (A + I) D^{-1/2}

        This is what GCN-style Φ_ℓ(A, H) = σ(Â H W_ℓ) uses
        (see Remark 3.5 of theory document).

        Args:
            edge_index:  [2, mt] active edge index
            edge_weight: optional [mt] weights (defaults to all-ones)
            dtype:       float precision

        Returns:
            (edge_index_norm, edge_weight_norm) — normalized edge index and weights
        """
        num_nodes = self.num_nodes

        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.shape[1], dtype=dtype, device=edge_index.device
            )

        if self.add_self_loops_flag:
            edge_index, edge_weight = add_self_loops(
                edge_index,
                edge_weight,
                fill_value=1.0,
                num_nodes=num_nodes,
            )

        # D^{-1/2}
        row, col = edge_index[0], edge_index[1]
        deg = degree(col, num_nodes, dtype=dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

        # Â_{ij} = D^{-1/2}_{ii} * A_{ij} * D^{-1/2}_{jj}
        norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, norm_weight

    # ------------------------------------------------------------------
    # Batch normalization (for efficiency during training)
    # ------------------------------------------------------------------

    def normalized_adjacency_cached(
        self,
        edge_index: Tensor,
        cache_key: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Cached version of normalized_adjacency.
        cache_key is typically the training step t; if edge_index didn't
        change since last call, returns the cached result.
        """
        if not hasattr(self, "_cache"):
            self._cache = {}

        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]

        result = self.normalized_adjacency(edge_index)

        if cache_key is not None:
            # Keep only the last 2 steps in cache to limit memory
            if len(self._cache) >= 2:
                oldest = min(self._cache.keys())
                del self._cache[oldest]
            self._cache[cache_key] = result

        return result
