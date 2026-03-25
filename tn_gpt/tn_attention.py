"""
MPS Tensor Network Attention.

Replaces multi-head self-attention with a Matrix Product State (MPS) acting as a
linear recurrence over the sequence.

Nodes  = site tensors A[t], one per sequence position, shape (D, n_embd, D)
Edges  = contracted bond indices between site tensors

Chain bonds (t-1 <-> t) are always active — implicit in the recurrence.
Long-range bonds (i <-> j, i < j-1) are L0-gated via log_alpha.

Forward pass:
    M_t[l,r] = sum_s  A[t][l,s,r] * x_t[s]          contract physical leg with input
    v_t      = v_{t-1} @ M_t                          chain contraction (always on)
             + sum_{i<t-1} z_it * (v_i @ B[i,t])     long-range contributions
    y_t      = output_proj(v_t)

log_alpha[i,j] is THE adjacency matrix for long-range edges only.
It is a real (N,N) parameter tensor — not a dict, not indexed by attention roles.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

from .l0_gate import hard_concrete_sample, expected_l0


class TNAttention(nn.Module):
    def __init__(self, n_embd: int, bond_dim: int, block_size: int):
        super().__init__()
        self.n_embd     = n_embd
        self.bond_dim   = bond_dim
        self.block_size = block_size
        N, D = block_size, bond_dim

        # Site tensors: A[t] has shape (D, n_embd, D).
        # M_t = I_D + einsum(A[t], x[t]) — residual parameterization.
        # With A small at init, M_t ≈ I so sv(M_t) ≈ 1 and the recurrence is stable.
        # The model learns perturbations away from identity.
        self.A = nn.Parameter(torch.randn(N, D, n_embd, D) * 0.01)

        # Long-range bond matrices: B[i,j] has shape (D, D)
        # Contracts v_i (right bond of site i) with the state update at site j.
        # Only meaningful for i < j-1.
        self.B = nn.Parameter(torch.randn(N, N, D, D) * 0.1)

        # THE adjacency matrix. log_alpha[i,j] = logit for long-range bond (i->j).
        # Chain bonds are NOT here — they are always on via the recurrence.
        # Initialize to 0: P(z>0) ≈ 83%, gates mostly open so gradients flow freely.
        # L0 penalty will prune edges that don't help; task gradient keeps useful ones.
        self.log_alpha = nn.Parameter(torch.full((N, N), 0.0))

        # Learned left boundary state (initial v before position 0)
        self.v0 = nn.Parameter(torch.randn(D) * (1.0 / math.sqrt(D)))

        self.output_proj = nn.Linear(D, n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (T, n_embd) — full sequence of input embeddings
        returns: (T, n_embd) — output embeddings
        """
        T = x.shape[0]
        N, D = self.block_size, self.bond_dim

        # --- Step 1: build transfer matrices M_t = I + einsum(A[t], x[t]) ---
        # Residual parameterization keeps sv(M_t) ≈ 1 at init and throughout training.
        I_D = torch.eye(D, device=x.device)
        M = I_D + torch.einsum('tlpr,tp->tlr', self.A[:T], x)   # (T, D, D)

        # --- Step 2: sample gate matrix for long-range edges ---
        if self.training:
            Z = hard_concrete_sample(self.log_alpha[:T, :T])  # (T, T)
        else:
            Z = torch.sigmoid(self.log_alpha[:T, :T])          # (T, T)

        # --- Step 3: left-to-right MPS recurrence ---
        v_states = []
        v = self.v0   # (D,) left boundary state

        for t in range(T):
            # Chain: contract right bond of v_{t-1} with left bond of M_t
            v_t = v @ M[t]                                     # (D,)@(D,D) -> (D,)

            # Long-range: i < t-1 only (i = t-1 is the chain, already handled)
            for i in range(t - 1):
                z = Z[i, t]
                if z.item() > 0.0:
                    v_t = v_t + z * (v_states[i] @ self.B[i, t])  # (D,)

            v_t = torch.tanh(v_t)
            v_states.append(v_t)
            v = v_t

        # --- Step 4: project bond states to output embeddings ---
        v_stack = torch.stack(v_states)          # (T, D)
        return self.output_proj(v_stack)          # (T, n_embd)

    def l0_loss(self) -> Tensor:
        """Expected *fraction* of active long-range bonds (chain excluded), in [0, 1].
        Normalised by number of potential edges so lambda_l0 is scale-invariant."""
        mask = torch.ones(self.block_size, self.block_size,
                          device=self.log_alpha.device).tril(diagonal=-2)
        n_potential = mask.sum()
        return (expected_l0(self.log_alpha) * mask).sum() / n_potential

    def gate_matrix(self) -> Tensor:
        """Adjacency matrix as gate probabilities. Shape (N, N). For visualization."""
        return torch.sigmoid(self.log_alpha).detach().cpu()

    def active_edge_count(self) -> int:
        mask = torch.ones(self.block_size, self.block_size).tril(diagonal=-2)
        return int((expected_l0(self.log_alpha.cpu()) * mask > 0.5).sum().item())

    def __repr__(self) -> str:
        return (f"TNAttention(n_embd={self.n_embd}, bond_dim={self.bond_dim}, "
                f"block_size={self.block_size}, "
                f"active_long_range_edges={self.active_edge_count()})")
