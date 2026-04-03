"""
Position sizing head (Section 3.2).

Maps the fused representation to a bounded bet fraction in [0, f_max].
The upper bound f_max is configurable per asset class to reflect
differing leverage and liquidity constraints:

    - Equities: f_max = 0.40
    - Crypto:   f_max = 0.25
    - Forex:    f_max = 0.30
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Recommended f_max defaults from Table 2 of the paper.
FMAX_DEFAULTS: dict[str, float] = {
    "equities": 0.40,
    "crypto": 0.25,
    "forex": 0.30,
}


class SizingHead(nn.Module):
    """Output head producing a bounded bet fraction.

    Architecture (Section 3.2, Eq. 4):
        Linear(d_model -> 32) -> ReLU -> Dropout(p) -> Linear(32 -> 1) -> Sigmoid * f_max

    The Sigmoid activation guarantees output lies in [0, f_max], avoiding the
    need for post-hoc clipping that would create zero-gradient regions.

    Args:
        d_model: Input dimension from the fusion module. Default 64.
        f_max:   Maximum bet fraction. Default 0.40 (equities).
        dropout: Dropout probability. Default 0.1.
    """

    def __init__(
        self,
        d_model: int = 64,
        f_max: float = 0.40,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.f_max = f_max
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_fused: (batch, d_model) fused representation.

        Returns:
            bet_fraction: (batch, 1) in [0, f_max].
        """
        return self.head(h_fused) * self.f_max
