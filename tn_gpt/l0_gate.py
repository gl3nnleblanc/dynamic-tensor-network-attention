"""
Hard Concrete L0 gate (Louizos et al. 2018, https://arxiv.org/abs/1712.01312).

Each gate has a single learned logit `log_alpha`. During training, samples from
the Hard Concrete distribution: a stretched + clamped Gumbel-softmax that places
actual probability mass at exactly 0 and 1. At eval, returns sigmoid(log_alpha).

expected_L0() gives a differentiable estimate of P(z > 0), used to penalize the
expected number of active edges in the loss.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class L0Gate(nn.Module):
    BETA  = 2 / 3   # temperature
    GAMMA = -0.1    # stretch lower bound (must be < 0)
    ZETA  = 1.1     # stretch upper bound (must be > 1)

    def __init__(self, init_log_alpha: float = -10.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha))

    def forward(self) -> Tensor:
        if self.training:
            u = torch.zeros(1, device=self.log_alpha.device).uniform_().clamp(1e-8, 1 - 1e-8)
            s = torch.sigmoid((u.log() - (1 - u).log() + self.log_alpha) / self.BETA)
            s_bar = s * (self.ZETA - self.GAMMA) + self.GAMMA
            return s_bar.clamp(0.0, 1.0)
        else:
            return torch.sigmoid(self.log_alpha).clamp(0.0, 1.0)

    def expected_L0(self) -> Tensor:
        """Differentiable P(z > 0) — use this in the regularization loss."""
        return torch.sigmoid(
            self.log_alpha - self.BETA * math.log(-self.GAMMA / self.ZETA)
        )

    @property
    def is_active(self) -> bool:
        return self.expected_L0().item() > 0.5
