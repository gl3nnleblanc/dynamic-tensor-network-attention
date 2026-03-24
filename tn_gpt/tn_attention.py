"""
L0-gated tensor network attention module.

Replaces multi-head self-attention in the transformer block. Instead of
computing QKV projections and a softmax attention matrix, this module:

  1. Projects each input token embedding to bond dimension D.
  2. Stores projected embeddings in a per-sequence hidden_cache.
  3. For the current position, aggregates information from all causally
     connected prior positions via 1-hop message passing:

       out = h_pos + sum_{j <= pos, active(j,pos)}  z_j_pos * W_j_pos @ h_j

  4. Projects aggregated output back to n_embd.

The graph topology (which edges exist) grows dynamically as new sequence
positions are encountered. Gates z_j_pos are learned via Hard Concrete and
an L0 regularization term in the training loss.

hidden_cache is passed in and mutated in-place, analogous to Karpathy's
keys/values lists. Clear it between documents.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .adjacency import GrowableAdjacency


class TNAttention(nn.Module):
    def __init__(self, n_embd: int, bond_dim: int):
        super().__init__()
        self.n_embd      = n_embd
        self.bond_dim    = bond_dim
        self.input_proj  = nn.Linear(n_embd, bond_dim, bias=False)
        self.output_proj = nn.Linear(bond_dim, n_embd, bias=False)
        self.adjacency   = GrowableAdjacency(bond_dim)

    def forward(self, x: Tensor, pos: int, hidden_cache: list) -> Tensor:
        """
        Args:
            x:            (n_embd,)  token embedding at current position (post-rmsnorm)
            pos:          int        current position index (0-based)
            hidden_cache: list       projected bond-dim embeddings from positions 0..pos-1
                                     mutated in-place: h for current pos is appended

        Returns:
            (n_embd,) output embedding
        """
        self.adjacency.ensure_nodes(pos + 1)

        h = self.input_proj(x)          # (D,)
        hidden_cache.append(h)          # store for future positions to attend to

        # 1-hop message passing over active causal edges
        out = h.clone()
        for j, edge in self.adjacency.active_edges_to(pos):
            z = edge.gate()             # Hard Concrete scalar in [0, 1]
            if z.item() > 0.0:
                out = out + z * (edge.W @ hidden_cache[j])

        return self.output_proj(out)    # (n_embd,)

    def expected_L0(self) -> Tensor:
        return self.adjacency.expected_L0()

    def __repr__(self) -> str:
        return (
            f"TNAttention(n_embd={self.n_embd}, bond_dim={self.bond_dim}, "
            f"adjacency={self.adjacency})"
        )
