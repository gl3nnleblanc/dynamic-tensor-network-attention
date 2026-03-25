"""
Hard Concrete L0 gate functions (Louizos et al. 2018).

Functional, vectorized — apply element-wise to a tensor of any shape.
"""

import math
import torch
from torch import Tensor

BETA  = 2 / 3
GAMMA = -0.1
ZETA  = 1.1


def hard_concrete_sample(log_alpha: Tensor) -> Tensor:
    """Element-wise Hard Concrete sample. Returns tensor in [0, 1], same shape as log_alpha."""
    u     = torch.zeros_like(log_alpha).uniform_().clamp(1e-8, 1 - 1e-8)
    s     = torch.sigmoid((u.log() - (1 - u).log() + log_alpha) / BETA)
    s_bar = s * (ZETA - GAMMA) + GAMMA
    return s_bar.clamp(0.0, 1.0)


def expected_l0(log_alpha: Tensor) -> Tensor:
    """Element-wise P(z > 0). Differentiable. Same shape as log_alpha."""
    return torch.sigmoid(log_alpha - BETA * math.log(-GAMMA / ZETA))
