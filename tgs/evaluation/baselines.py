"""
Baselines for comparison against TGS.

Implemented:
  1. DenseGCN          — no sparsification (upper bound)
  2. RandomSparsify    — randomly remove edges to target sparsity
  3. LocalDegree       — remove edges incident to high-degree nodes first
  4. EffectiveResistance — sample edges proportional to effective resistance

All baselines train a standard GCN on a statically sparsified graph.
They represent the best static sparsification can do — TGS should match
or beat them at equivalent final sparsity (Proposition 7.2).
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Sparsification strategies (return a pruned edge_index)
# ------------------------------------------------------------------

def dense_edges(edge_index: Tensor, **kwargs) -> Tensor:
    """No sparsification — return full graph."""
    return edge_index


def random_sparsify(edge_index: Tensor, target_sparsity: float, seed: int = 42) -> Tensor:
    """
    Randomly remove edges to reach target_sparsity fraction removed.
    Baseline: no structure-awareness whatsoever.
    """
    rng = np.random.default_rng(seed)
    m = edge_index.shape[1]
    n_keep = max(1, int(m * (1 - target_sparsity)))
    keep_idx = rng.choice(m, size=n_keep, replace=False)
    keep_idx = torch.tensor(np.sort(keep_idx), dtype=torch.long)
    return edge_index[:, keep_idx]


def local_degree_sparsify(edge_index: Tensor, num_nodes: int, target_sparsity: float) -> Tensor:
    """
    Remove edges incident to highest-degree nodes first.
    Heuristic: high-degree edges are often redundant for message passing.
    """
    m = edge_index.shape[1]
    n_remove = int(m * target_sparsity)
    if n_remove == 0:
        return edge_index

    src, dst = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, dtype=torch.float)
    deg.scatter_add_(0, src, torch.ones(m, dtype=torch.float))
    deg.scatter_add_(0, dst, torch.ones(m, dtype=torch.float))

    # Score each edge by sum of endpoint degrees (higher = more redundant)
    edge_score = deg[src] + deg[dst]
    _, sorted_idx = edge_score.sort(descending=True)

    remove_set = set(sorted_idx[:n_remove].tolist())
    keep_mask = torch.tensor([i not in remove_set for i in range(m)], dtype=torch.bool)
    return edge_index[:, keep_mask]


def effective_resistance_sparsify(
    edge_index: Tensor,
    num_nodes: int,
    target_sparsity: float,
    seed: int = 42,
) -> Tensor:
    """
    Sample edges with probability proportional to effective resistance.
    High-resistance edges bridge sparse regions — more important to keep.
    Low-resistance edges have many parallel paths — safe to drop.

    Uses the Spielman-Srivastava approximation via truncated SVD of the
    graph Laplacian for tractability on CPU.
    """
    m = edge_index.shape[1]
    n_keep = max(1, int(m * (1 - target_sparsity)))

    try:
        # Build Laplacian
        L = _build_laplacian(edge_index, num_nodes)

        # Approximate effective resistance via pseudoinverse (truncated SVD)
        # R_e ≈ (b_e)^T L^+ b_e where b_e is the incidence vector
        k = min(64, num_nodes - 1)  # rank for approximation
        L_dense = L.toarray().astype(np.float32)
        U, S, Vt = np.linalg.svd(L_dense, full_matrices=False)

        # Pseudoinverse: only keep nonzero singular values
        tol = 1e-6
        S_inv = np.where(S > tol, 1.0 / S, 0.0)
        L_pinv = (Vt.T * S_inv) @ U.T  # [n, n]

        # Effective resistance for each edge
        src_np = edge_index[0].numpy()
        dst_np = edge_index[1].numpy()
        er = np.array([
            max(0.0, L_pinv[src_np[i], src_np[i]]
                   - 2 * L_pinv[src_np[i], dst_np[i]]
                   + L_pinv[dst_np[i], dst_np[i]])
            for i in range(m)
        ])

        # Sample proportional to effective resistance
        er_sum = er.sum()
        if er_sum < 1e-10:
            probs = np.ones(m) / m
        else:
            probs = er / er_sum

        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(m, size=n_keep, replace=False, p=probs)
        keep_idx = torch.tensor(np.sort(keep_idx), dtype=torch.long)
        logger.debug(f"EffResistance: kept {n_keep}/{m} edges")
        return edge_index[:, keep_idx]

    except Exception as e:
        logger.warning(f"Effective resistance failed ({e}), falling back to random")
        return random_sparsify(edge_index, target_sparsity, seed)


def _build_laplacian(edge_index: Tensor, num_nodes: int):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    m = len(src)
    data = np.ones(m)
    A = sp.csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
    A = (A + A.T) / 2  # symmetrise
    D = sp.diags(np.array(A.sum(axis=1)).flatten())
    return D - A


# ------------------------------------------------------------------
# Unified baseline runner
# ------------------------------------------------------------------

def run_baseline(
    name: str,
    data,
    num_features: int,
    num_classes: int,
    target_sparsity: float,
    hidden: int = 64,
    epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    dropout: float = 0.5,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Train a GCN on a statically sparsified graph and return results.

    Args:
        name: one of 'dense', 'random', 'local_degree', 'eff_resistance'
        target_sparsity: fraction of edges to remove (0 = keep all)
    """
    from tgs.models.gcn import TemporalGCN
    from tgs.utils.reproducibility import set_seed
    set_seed(seed)

    num_nodes = data.num_nodes
    m0 = data.edge_index.shape[1]

    # Sparsify
    if name == "dense":
        ei = dense_edges(data.edge_index)
    elif name == "random":
        ei = random_sparsify(data.edge_index, target_sparsity, seed)
    elif name == "local_degree":
        ei = local_degree_sparsify(data.edge_index, num_nodes, target_sparsity)
    elif name == "eff_resistance":
        ei = effective_resistance_sparsify(data.edge_index, num_nodes, target_sparsity, seed)
    else:
        raise ValueError(f"Unknown baseline: {name}")

    ei = ei.to(device)
    actual_sparsity = 1.0 - ei.shape[1] / m0
    logger.info(f"Baseline [{name}] | target_sp={target_sparsity:.2f} | actual_sp={actual_sparsity:.3f} | m={ei.shape[1]}")

    # Model + optimiser
    model = TemporalGCN(num_features, hidden, num_classes,
                        num_layers=2, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = best_test = 0.0

    for epoch in range(epochs):
        model.train()
        logits = model(data.x, ei)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, ei)
        preds = out.argmax(dim=-1)

        val_acc = (preds[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc

    return {
        "baseline": name,
        "target_sparsity": target_sparsity,
        "actual_sparsity": actual_sparsity,
        "best_val_acc": best_val,
        "best_test_acc": best_test,
        "m0": m0,
        "mt": ei.shape[1],
    }
