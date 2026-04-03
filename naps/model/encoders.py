"""
NAPS encoder modules (Section 3.2).

Three parallel encoders extract fixed-dimensional representations from
heterogeneous input streams:

- TemporalTransformerEncoder: multi-day OHLCV + technical indicators
- SignalUncertaintyEncoder:   point signal with calibrated uncertainty
- RiskStateEncoder:           portfolio state and regime features
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalTransformerEncoder(nn.Module):
    """Encode a lookback window of daily market features via Transformer.

    Architecture (Section 3.2, Eq. 1):
        x_proj = Linear(n_features -> d_model)
        x_pos  = x_proj + PE            # learnable positional embeddings
        h_seq  = TransformerEncoder(x_pos)
        h_temp = MeanPool(h_seq)         # R^d_model

    Args:
        d_model:    Hidden / embedding dimension.  Default 64.
        n_heads:    Number of attention heads.      Default 4.
        n_layers:   Number of Transformer encoder layers. Default 2.
        lookback:   Sequence length L (trading days). Default 60.
        n_features: Raw features per day (5 OHLCV + 15 technicals). Default 20.
        dropout:    Dropout probability inside Transformer layers.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        lookback: int = 60,
        n_features: int = 20,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)

        # Learnable positional embeddings for each position in the lookback window.
        self.pos_embedding = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 7,  # ~180K total model params
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, market_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            market_history: (batch, L, n_features) daily market features.

        Returns:
            h_temp: (batch, d_model) mean-pooled temporal representation.
        """
        x = self.input_proj(market_history)          # (B, L, d_model)
        x = x + self.pos_embedding[:, : x.size(1)]  # add positional embeddings
        h_seq = self.transformer(x)                  # (B, L, d_model)
        h_temp = h_seq.mean(dim=1)                   # (B, d_model)
        return h_temp


class SignalUncertaintyEncoder(nn.Module):
    """Encode the trading signal together with its calibrated uncertainty.

    Input vector (Section 3.2):
        [dir_t, mag_t, u_t] in R^3
        - dir_t:  signal direction  (+1 / -1 / 0)
        - mag_t:  signal magnitude  (continuous)
        - u_t:    MC-dropout uncertainty estimate (std)

    Architecture:
        Linear(3 -> 64) -> ReLU -> Linear(64 -> 64) -> ReLU

    Output:
        h_sig in R^64
    """

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, signal_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal_input: (batch, 3) concatenation of [dir_t, mag_t, u_t].

        Returns:
            h_sig: (batch, d_model).
        """
        return self.net(signal_input)


class RiskStateEncoder(nn.Module):
    """Encode portfolio risk state and regime features.

    Input vector (Section 3.2):
        [p_t ; r_t] in R^8
        - p_t in R^5: (drawdown, P&L, peak_days, exposure, serial_corr)
        - r_t in R^3: (realized_vol_20d, mkt_correlation_60d, trend_strength_20d)

    Architecture:
        Linear(8 -> 64) -> ReLU -> Linear(64 -> 64) -> ReLU

    Output:
        h_risk in R^64
    """

    def __init__(self, d_model: int = 64, n_risk: int = 5, n_regime: int = 3) -> None:
        super().__init__()
        input_dim = n_risk + n_regime
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, risk_state: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        """
        Args:
            risk_state: (batch, 5) portfolio state vector p_t.
            regime:     (batch, 3) regime feature vector r_t.

        Returns:
            h_risk: (batch, d_model).
        """
        x = torch.cat([risk_state, regime], dim=-1)  # (B, 8)
        return self.net(x)
