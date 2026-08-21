"""
Structural predictor for TGS advantage.

Given a graph's edge_index, node labels, and node count, predicts whether
TGS (adaptive edge retirement) will outperform a matched-sparsity static
sparsification baseline, using the two-condition rule derived empirically
across 12 valid datasets (see analysis/predictive_rule.md).

Rule (F1=1.00 on 12 valid datasets):
    TGS wins when ALL of:
      (1) homophily  < 0.25
      (2) n          <= 5,000

An earlier three-condition version included deg_cv >= 1.00, which was
introduced to explain Cornell's single-seed tie. Multi-seed testing
showed Cornell's dense baseline is degenerate (GCN std=0 across seeds),
invalidating that datapoint. Once Cornell is excluded, deg_cv provides
no additional separating power and the rule simplifies to two conditions.
deg_cv is retained in StructuralFingerprint for diagnostics only.

Validated on (Precision=1.00, Recall=1.00, F1=1.00):
    Wisconsin (WIN 5/5), Chameleon (WIN 5/5), Texas (WIN 4/5),
    Squirrel-2k (WIN 3/5 weak), Squirrel (TIE), Actor (TIE, noise),
    Minesweeper (TIE), Cora (TIE), PubMed (LOSS), Photo (LOSS),
    CiteSeer (LOSS), Cora_ML (LOSS)
    [Cornell excluded: degenerate dense baseline]

See analysis/predictive_rule.md for full derivation and caveats.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch_geometric.utils import degree


# Empirically derived thresholds (see analysis/predictive_rule.md).
# An earlier version included deg_cv >= 1.00 as a third condition, but
# multi-seed testing invalidated the only datapoint it was needed for
# (Cornell's dense baseline is degenerate). The 2-condition rule achieves
# F1=1.00 on all 12 valid datasets without it. deg_cv is still computed
# in StructuralFingerprint for diagnostic use.
HOMOPHILY_THRESHOLD = 0.25   # dominant predictor; Pearson r=+0.597 with gap
N_THRESHOLD         = 5_000  # proxy for model-fit headroom above chance


@dataclass
class StructuralFingerprint:
    homophily:  float
    deg_cv:     float   # retained for diagnostics; not used in the rule
    n:          int
    m:          int

    def __str__(self):
        return (f"homophily={self.homophily:.3f}, deg_cv={self.deg_cv:.3f}, "
                f"n={self.n:,}, m={self.m:,}")


def compute_fingerprint(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    num_nodes: int,
) -> StructuralFingerprint:
    """Compute the structural fingerprint of a graph for TGS prediction."""
    src, dst = edge_index[0].cpu(), edge_index[1].cpu()
    y_cpu = y.cpu()
    homophily = float((y_cpu[src] == y_cpu[dst]).float().mean())
    deg = degree(dst, num_nodes).numpy()
    deg_cv = float(deg.std() / max(deg.mean(), 1e-8))
    return StructuralFingerprint(
        homophily=homophily,
        deg_cv=deg_cv,
        n=num_nodes,
        m=int(edge_index.shape[1]),
    )


def predict_tgs_advantage(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    num_nodes: int,
    *,
    homophily_threshold: float = HOMOPHILY_THRESHOLD,
    n_threshold:         int   = N_THRESHOLD,
) -> Tuple[bool, str, StructuralFingerprint]:
    """
    Predict whether TGS will outperform a matched-sparsity static baseline.

    Parameters
    ----------
    edge_index : torch.Tensor  [2, m]
    y          : torch.Tensor  [n]  — node labels (integers)
    num_nodes  : int

    Returns
    -------
    (predicted_win, reason, fingerprint)
        predicted_win : bool    — True if TGS is expected to beat static baseline
        reason        : str     — which condition was decisive
        fingerprint   : StructuralFingerprint — computed metrics

    Rule (F1=1.00 on 12 valid datasets):
        WIN when: homophily < 0.25  AND  n <= 5,000
    """
    fp = compute_fingerprint(edge_index, y, num_nodes)

    if fp.homophily >= homophily_threshold:
        return (
            False,
            f"LOSS predicted: homophily={fp.homophily:.3f} >= {homophily_threshold} "
            f"(edges are mostly same-class; sparsification removes signal, not noise)",
            fp,
        )

    if fp.n > n_threshold:
        return (
            False,
            f"TIE predicted: homophily={fp.homophily:.3f} passes, but "
            f"n={fp.n:,} > {n_threshold:,} "
            f"(graph likely too large for 2-layer GCN to achieve meaningful accuracy; "
            f"influence signal degrades near chance-level — verify dense accuracy > "
            f"chance + 0.15 before using TGS)",
            fp,
        )

    return (
        True,
        f"WIN predicted: both conditions met — "
        f"homophily={fp.homophily:.3f} < {homophily_threshold}, "
        f"n={fp.n:,} <= {n_threshold:,}",
        fp,
    )
