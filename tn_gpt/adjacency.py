"""
Growable tensor network graph backed by nn.ModuleDict.

Each edge (i, j) owns:
  - W: (D x D) bond tensor, initialized to zero
  - gate: L0Gate scalar controlling whether the edge is active

New nodes are added as sequence length grows. Initial topology is a causal
linear chain (MPS-like): edge (t-1, t) is added when position t is first seen.
Long-range edges can be added explicitly via add_edge(i, j).

nn.ModuleDict auto-registers all edge parameters for gradient tracking,
regardless of when edges are added relative to optimizer construction.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .l0_gate import L0Gate


class TNEdge(nn.Module):
    def __init__(self, bond_dim: int):
        super().__init__()
        self.W    = nn.Parameter(torch.zeros(bond_dim, bond_dim))
        self.gate = L0Gate()


class GrowableAdjacency(nn.Module):
    def __init__(self, bond_dim: int):
        super().__init__()
        self.bond_dim = bond_dim
        self.n_nodes  = 0
        self.edges    = nn.ModuleDict()   # key: "i_j" -> TNEdge

    # ------------------------------------------------------------------
    # Graph mutation
    # ------------------------------------------------------------------

    def ensure_nodes(self, seq_len: int):
        """Grow graph to cover seq_len positions, adding causal chain edges."""
        while self.n_nodes < seq_len:
            nid = self.n_nodes
            if nid > 0:
                self._add_edge(nid - 1, nid)
            self.n_nodes += 1

    def add_edge(self, i: int, j: int):
        """Explicitly add a (potentially long-range) edge."""
        assert 0 <= i < j < self.n_nodes, f"invalid edge ({i}, {j}) for {self.n_nodes} nodes"
        self._add_edge(i, j)

    def _add_edge(self, i: int, j: int):
        key = f"{i}_{j}"
        if key not in self.edges:
            self.edges[key] = TNEdge(self.bond_dim)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def active_edges_to(self, pos: int) -> list[tuple[int, TNEdge]]:
        """All edges (j -> pos) with j <= pos (causal). Includes self-loop."""
        result = []
        for j in range(pos + 1):
            key = f"{j}_{pos}"
            if key in self.edges:
                result.append((j, self.edges[key]))
        return result

    def expected_L0(self) -> Tensor:
        if not self.edges:
            return torch.tensor(0.0)
        return sum(e.gate.expected_L0() for e in self.edges.values())

    def active_edge_count(self) -> int:
        return sum(1 for e in self.edges.values() if e.gate.is_active)

    def __repr__(self) -> str:
        return (
            f"GrowableAdjacency(nodes={self.n_nodes}, "
            f"edges={len(self.edges)}, "
            f"active={self.active_edge_count()}, "
            f"bond_dim={self.bond_dim})"
        )
