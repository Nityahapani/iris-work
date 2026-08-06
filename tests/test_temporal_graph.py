"""Unit tests for TemporalGraph — verifying Definition 2.1 and 2.2 invariants."""

import pytest
import torch
from tgs.core.temporal_graph import TemporalGraph


def make_graph():
    edge_index = torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]], dtype=torch.long)
    return TemporalGraph(edge_index, num_nodes=4)


def test_initial_state():
    tg = make_graph()
    assert tg.mt == 6
    assert tg.m0 == 6
    assert tg.sparsity == 0.0
    assert tg.t == 0


def test_monotone_retirement():
    """Definition 2.1: E_{t+1} ⊆ E_t — retired edges never return."""
    tg = make_graph()
    tg.retire_edges(torch.tensor([0, 1]))
    assert tg.mt == 4
    mt_before = tg.mt
    # Retiring already-retired edges should be a no-op
    tg.retire_edges(torch.tensor([0, 1]))
    assert tg.mt == mt_before


def test_retirement_schedule():
    """Definition 2.2: τ(e) recorded correctly."""
    tg = make_graph()
    tg.step()  # t=1
    tg.step()  # t=2
    tg.retire_edges(torch.tensor([2]))
    assert tg.retirement_time(2) == 2


def test_sparsity():
    tg = make_graph()
    tg.retire_edges(torch.tensor([0, 1, 2]))
    assert abs(tg.sparsity - 0.5) < 1e-6


def test_edge_index_without():
    tg = make_graph()
    ei_full = tg.edge_index
    ei_minus = tg.edge_index_without(0)
    assert ei_minus.shape[1] == ei_full.shape[1] - 1


def test_stats():
    tg = make_graph()
    s = tg.stats()
    assert "step" in s and "sparsity" in s and "mt" in s
