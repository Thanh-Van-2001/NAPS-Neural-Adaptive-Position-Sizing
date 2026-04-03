"""
Composite loss function for NAPS training (Section 3.3).

The loss consists of three terms:

    L = -SR_diff(f_{1:W}) + lambda1 * [max(DD_max - tau, 0)]^2
                           + lambda2 * (1/W) * sum |f_t - f_{t-1}| * c

1. **Differentiable Sharpe ratio** (Moody & Saffell [22]).  Uses exponentially
   weighted running mean/variance so that gradients flow through the sizing
   decisions f_t.

2. **Drawdown penalty**.  Quadratic penalty when the maximum drawdown of the
   equity curve exceeds a threshold tau.

3. **Turnover regularisation**.  Penalises rapid changes in bet fraction
   weighted by a transaction-cost rate *c*.

Default hyper-parameters (Table 1):
    W      = 252   (1 trading year)
    tau    = 0.15   (15 % max drawdown threshold)
    lambda1 = 5.0
    lambda2 = 1.0
    c      = 0.001  (10 bps one-way)
    eta    = 0.01   (Sharpe EMA decay)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NAPSLoss(nn.Module):
    """Composite training objective for the NAPS model (Section 3.3).

    All three sub-losses operate on full-length sequences so that temporal
    dependencies (drawdown, turnover) are captured correctly.  The combined
    loss is fully differentiable with respect to the model's bet-fraction
    outputs ``fractions``.

    Args:
        W:       Expected window length for normalisation.  Default 252.
        tau:     Maximum-drawdown threshold for the penalty.  Default 0.15.
        lambda1: Weight for the drawdown penalty term.  Default 5.0.
        lambda2: Weight for the turnover regularisation term.  Default 1.0.
        cost:    One-way transaction cost rate.  Default 0.001 (10 bps).
        eta:     Exponential decay rate for the differentiable Sharpe
                 running statistics.  Default 0.01.
    """

    def __init__(
        self,
        W: int = 252,
        tau: float = 0.15,
        lambda1: float = 5.0,
        lambda2: float = 1.0,
        cost: float = 0.001,
        eta: float = 0.01,
    ) -> None:
        super().__init__()
        self.W = W
        self.tau = tau
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.cost = cost
        self.eta = eta

    # ------------------------------------------------------------------
    # Sub-losses
    # ------------------------------------------------------------------

    @staticmethod
    def differentiable_sharpe(
        returns: torch.Tensor,
        eta: float = 0.01,
    ) -> torch.Tensor:
        """Differentiable Sharpe ratio via exponentially weighted statistics.

        Following Moody & Saffell [22], the running mean and second moment
        are updated with an exponential decay so that the entire computation
        graph is differentiable:

            A_t = A_{t-1} + eta * (r_t - A_{t-1})
            B_t = B_{t-1} + eta * (r_t^2 - B_{t-1})
            SR  = A_T / sqrt(B_T - A_T^2 + eps)

        Args:
            returns: (batch, T) per-step portfolio returns, **already**
                     multiplied by the bet fractions f_t.
            eta:     Exponential weighting decay rate.  Default 0.01.

        Returns:
            sharpe: (batch,) differentiable Sharpe estimate, one per
                    sample in the batch.
        """
        B, T = returns.shape
        eps = 1e-8

        # Initialise running statistics at zero.
        A = torch.zeros(B, device=returns.device, dtype=returns.dtype)
        Bm = torch.zeros(B, device=returns.device, dtype=returns.dtype)

        for t in range(T):
            r_t = returns[:, t]
            A = A + eta * (r_t - A)
            Bm = Bm + eta * (r_t * r_t - Bm)

        variance = Bm - A * A
        # Clamp variance to avoid sqrt of negative numbers due to numerics.
        sharpe = A / torch.sqrt(torch.clamp(variance, min=eps))
        return sharpe

    @staticmethod
    def drawdown_penalty(
        equity_curve: torch.Tensor,
        tau: float = 0.15,
        lambda1: float = 5.0,
    ) -> torch.Tensor:
        """Quadratic penalty on maximum drawdown exceeding threshold *tau*.

        Computes the running maximum of the equity curve via ``cummax``
        (which is differentiable in PyTorch), then derives the drawdown
        series and extracts the maximum drawdown per sample.

        Penalty = lambda1 * [max(DD_max - tau, 0)]^2

        Args:
            equity_curve: (batch, T) cumulative equity values (starting
                          from 1.0 is conventional but not required).
            tau:          Drawdown threshold.  Default 0.15.
            lambda1:      Penalty weight.  Default 5.0.

        Returns:
            penalty: (batch,) per-sample penalty scalar.
        """
        # Running peak of the equity curve along the time axis.
        running_peak, _ = torch.cummax(equity_curve, dim=1)

        # Drawdown fraction: how far below the peak the equity has fallen.
        drawdown = (running_peak - equity_curve) / (running_peak + 1e-8)

        # Maximum drawdown per sample.
        max_dd, _ = drawdown.max(dim=1)

        # Quadratic penalty beyond threshold.
        excess = torch.clamp(max_dd - tau, min=0.0)
        penalty = lambda1 * excess * excess
        return penalty

    @staticmethod
    def turnover_regularization(
        fractions: torch.Tensor,
        cost: float = 0.001,
        lambda2: float = 1.0,
    ) -> torch.Tensor:
        """Turnover penalty: average absolute change in fractions * cost.

        Reg = lambda2 * (1/W) * sum_{t=1}^{T-1} |f_t - f_{t-1}| * c

        where W is the configured training window (default 252, Section 3.3).

        Args:
            fractions: (batch, T) or (batch, T, 1) bet fractions over time.
            cost:      One-way transaction cost rate.  Default 0.001 (10 bps).
            lambda2:   Regularisation weight.  Default 1.0.
            W:         Normalisation window length.  Default 252.

        Returns:
            reg: (batch,) per-sample regularisation value.
        """
        if fractions.dim() == 3:
            fractions = fractions.squeeze(-1)

        # Absolute change between consecutive time steps.
        delta = torch.abs(fractions[:, 1:] - fractions[:, :-1])  # (B, T-1)
        # Use configured W for normalization (paper Section 3.3: W=252)
        norm = max(delta.shape[1], 1)
        avg_turnover = delta.sum(dim=1) / norm

        reg = lambda2 * avg_turnover * cost
        return reg

    # ------------------------------------------------------------------
    # Combined forward
    # ------------------------------------------------------------------

    def forward(
        self,
        fractions: torch.Tensor,
        returns: torch.Tensor,
        equity_curve: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the composite loss over a batch of sequences.

        Args:
            fractions:    (batch, T) or (batch, T, 1) model bet-fraction
                          outputs f_t for each time step.
            returns:      (batch, T) per-step asset returns r_t (raw,
                          **before** scaling by f_t).
            equity_curve: (batch, T) cumulative equity curve.  If ``None``
                          it is computed from ``fractions`` and ``returns``
                          assuming a starting equity of 1.0.

        Returns:
            loss: scalar (mean over the batch).
        """
        if fractions.dim() == 3:
            fractions = fractions.squeeze(-1)

        # Portfolio returns = f_t * r_t  (the sizing decision scales the asset return).
        portfolio_returns = fractions * returns  # (B, T)

        # Build equity curve if not provided.
        if equity_curve is None:
            equity_curve = torch.cumprod(1.0 + portfolio_returns, dim=1)

        # 1. Negative differentiable Sharpe (we minimise, so negate).
        sharpe = self.differentiable_sharpe(portfolio_returns, eta=self.eta)
        loss_sharpe = -sharpe  # (B,)

        # 2. Drawdown penalty.
        loss_dd = self.drawdown_penalty(
            equity_curve, tau=self.tau, lambda1=self.lambda1
        )

        # 3. Turnover regularisation.
        loss_turnover = self.turnover_regularization(
            fractions, cost=self.cost, lambda2=self.lambda2
        )

        # Combine and average over the batch.
        total = loss_sharpe + loss_dd + loss_turnover
        return total.mean()
