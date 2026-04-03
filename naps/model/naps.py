"""
NAPS end-to-end models (Section 3).

Provides:
    NAPSModel     - Full model with Transformer temporal encoder + cross-attention fusion.
    NAPSMLPModel  - Ablation variant replacing the Transformer with an MLP encoder
                    (Table 4, ablation study).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from naps.model.encoders import (
    TemporalTransformerEncoder,
    SignalUncertaintyEncoder,
    RiskStateEncoder,
)
from naps.model.fusion import CrossAttentionFusion
from naps.model.sizing_head import SizingHead


class NAPSModel(nn.Module):
    """Neural Adaptive Position Sizing (full model).

    Combines three parallel encoders, cross-attention fusion, and a bounded
    sizing head to produce a bet fraction f_t in [0, f_max] for each time step.

    Architecture overview (Figure 2):
        market_history ---> TemporalTransformerEncoder ---> h_temp ---|
        signal+uncert  ---> SignalUncertaintyEncoder   ---> h_sig  ---+--> CrossAttentionFusion ---> SizingHead ---> f_t
        risk+regime    ---> RiskStateEncoder            ---> h_risk --|

    Args:
        d_model:    Hidden dimension throughout the model. Default 64.
        n_heads:    Attention heads (Transformer + cross-attention). Default 4.
        n_layers:   Transformer encoder layers. Default 2.
        lookback:   Number of trading days in the input window. Default 60.
        n_features: Features per day (5 OHLCV + 15 technicals). Default 20.
        f_max:      Maximum bet fraction (asset-class dependent). Default 0.40.
        dropout:    Dropout probability. Default 0.1.

    Total parameter count: ~180K (Section 3.2).
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        lookback: int = 60,
        n_features: int = 20,
        f_max: float = 0.40,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --- Encoders (Section 3.2) ---
        self.temporal_encoder = TemporalTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            lookback=lookback,
            n_features=n_features,
            dropout=dropout,
        )
        self.signal_encoder = SignalUncertaintyEncoder(d_model=d_model)
        self.risk_encoder = RiskStateEncoder(d_model=d_model)

        # --- Fusion (Section 3.2) ---
        self.fusion = CrossAttentionFusion(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # --- Sizing head (Section 3.2) ---
        self.sizing_head = SizingHead(
            d_model=d_model,
            f_max=f_max,
            dropout=dropout,
        )

    def forward(
        self,
        market_history: torch.Tensor,
        signal: torch.Tensor,
        uncertainty: torch.Tensor,
        risk_state: torch.Tensor,
        regime: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass of the NAPS model.

        Args:
            market_history: (batch, 60, 20) lookback window of daily features.
            signal:         (batch, 2) trading signal [direction, magnitude].
            uncertainty:    (batch, 1) MC-dropout uncertainty estimate.
            risk_state:     (batch, 5) portfolio state
                            [drawdown, pnl, peak_days, exposure, serial_corr].
            regime:         (batch, 3) regime features
                            [realized_vol_20d, mkt_correlation_60d, trend_strength_20d].

        Returns:
            bet_fraction: (batch, 1) position size in [0, f_max].
        """
        # Encode each input stream independently.
        h_temp = self.temporal_encoder(market_history)                # (B, d)
        signal_input = torch.cat([signal, uncertainty], dim=-1)      # (B, 3)
        h_sig = self.signal_encoder(signal_input)                    # (B, d)
        h_risk = self.risk_encoder(risk_state, regime)               # (B, d)

        # Fuse via cross-attention with risk as query.
        h_fused = self.fusion(h_temp, h_sig, h_risk)                 # (B, d)

        # Produce bounded bet fraction.
        bet_fraction = self.sizing_head(h_fused)                     # (B, 1)
        return bet_fraction


class NAPSMLPModel(nn.Module):
    """NAPS ablation variant with MLP temporal encoder (Table 4).

    Replaces the Transformer encoder with a two-layer MLP that flattens the
    full lookback window.  Used in the ablation study to quantify the benefit
    of temporal attention over a simple feed-forward baseline.

    MLP encoder:
        Flatten(60 * 20 = 1200) -> Linear(1200 -> 256) -> ReLU -> Linear(256 -> d_model) -> ReLU

    All other components (signal encoder, risk encoder, fusion, sizing head)
    are identical to :class:`NAPSModel`.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        lookback: int = 60,
        n_features: int = 20,
        f_max: float = 0.40,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        flat_dim = lookback * n_features  # 60 * 20 = 1200

        # --- MLP temporal encoder (ablation replacement) ---
        self.temporal_mlp = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, d_model),
            nn.ReLU(inplace=True),
        )

        # --- Shared components (identical to NAPSModel) ---
        self.signal_encoder = SignalUncertaintyEncoder(d_model=d_model)
        self.risk_encoder = RiskStateEncoder(d_model=d_model)
        self.fusion = CrossAttentionFusion(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.sizing_head = SizingHead(
            d_model=d_model,
            f_max=f_max,
            dropout=dropout,
        )

    def forward(
        self,
        market_history: torch.Tensor,
        signal: torch.Tensor,
        uncertainty: torch.Tensor,
        risk_state: torch.Tensor,
        regime: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with MLP temporal encoder.

        Args:
            market_history: (batch, 60, 20) lookback window of daily features.
            signal:         (batch, 2) trading signal [direction, magnitude].
            uncertainty:    (batch, 1) MC-dropout uncertainty estimate.
            risk_state:     (batch, 5) portfolio state.
            regime:         (batch, 3) regime features.

        Returns:
            bet_fraction: (batch, 1) position size in [0, f_max].
        """
        B = market_history.size(0)
        h_temp = self.temporal_mlp(market_history.reshape(B, -1))    # (B, d)

        signal_input = torch.cat([signal, uncertainty], dim=-1)      # (B, 3)
        h_sig = self.signal_encoder(signal_input)                    # (B, d)
        h_risk = self.risk_encoder(risk_state, regime)               # (B, d)

        h_fused = self.fusion(h_temp, h_sig, h_risk)                 # (B, d)
        bet_fraction = self.sizing_head(h_fused)                     # (B, 1)
        return bet_fraction
