"""
Performance metrics for NAPS evaluation (Section 4.3, Table 2).

All metrics are computed on numpy arrays and match the definitions used in
the paper.  Transaction costs of 10 bps are applied when ``fractions`` are
provided.

Metrics computed:
    - Annualised return
    - Annualised volatility
    - Sharpe ratio (annualised)
    - Maximum drawdown
    - Calmar ratio (annualised return / |max drawdown|)
    - CVaR at 5 % (Conditional Value at Risk, a.k.a. Expected Shortfall)
    - Turnover: mean |f_t - f_{t-1}|
"""

from __future__ import annotations

import numpy as np


def compute_metrics(
    returns: np.ndarray,
    fractions: np.ndarray | None = None,
    risk_free: float = 0.0,
    cost: float = 0.001,
    trading_days: int = 252,
) -> dict[str, float]:
    """Compute the full suite of performance metrics.

    If ``fractions`` are supplied, transaction costs are deducted from
    returns before metric computation:

        net_return_t = return_t - cost * |f_t - f_{t-1}|

    Args:
        returns:      1-D array of daily portfolio returns (already scaled
                      by f_t if coming from the walk-forward pipeline).
        fractions:    1-D array of daily bet fractions f_t.  If provided,
                      transaction costs are subtracted and turnover is
                      reported.  May be ``None`` for raw return analysis.
        risk_free:    Daily risk-free rate for Sharpe computation.
                      Default 0.0.
        cost:         One-way transaction cost rate.  Default 0.001 (10 bps).
        trading_days: Trading days per year for annualisation.  Default 252.

    Returns:
        Dictionary with the following keys:

        - ``annualized_return``    : float
        - ``annualized_volatility``: float
        - ``sharpe_ratio``         : float
        - ``max_drawdown``         : float  (positive number, e.g. 0.12 = 12 %)
        - ``calmar_ratio``         : float
        - ``cvar_5``               : float  (expected shortfall at 5 %)
        - ``turnover``             : float  (mean |f_t - f_{t-1}|, or NaN)
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()

    # ---- Transaction costs ------------------------------------------------
    turnover_val = np.nan
    if fractions is not None:
        fractions = np.asarray(fractions, dtype=np.float64).ravel()
        assert len(fractions) == len(returns), (
            f"fractions length {len(fractions)} != returns length {len(returns)}"
        )
        delta_f = np.abs(np.diff(fractions))
        turnover_val = float(np.mean(delta_f))

        # Subtract transaction costs from returns (costs apply from day 1 onward).
        tc = np.zeros_like(returns)
        tc[1:] = cost * delta_f
        returns = returns - tc

    N = len(returns)
    if N == 0:
        return _empty_metrics()

    # ---- Annualised return ------------------------------------------------
    cumulative = np.prod(1.0 + returns)
    annualized_return = float(cumulative ** (trading_days / N) - 1.0)

    # ---- Annualised volatility --------------------------------------------
    daily_vol = float(np.std(returns, ddof=1)) if N > 1 else 0.0
    annualized_volatility = daily_vol * np.sqrt(trading_days)

    # ---- Sharpe ratio (annualised) ----------------------------------------
    excess = returns - risk_free
    mean_excess = float(np.mean(excess))
    std_excess = float(np.std(excess, ddof=1)) if N > 1 else 1e-8
    sharpe_ratio = (mean_excess / max(std_excess, 1e-8)) * np.sqrt(trading_days)

    # ---- Maximum drawdown -------------------------------------------------
    equity = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = (running_peak - equity) / running_peak
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # ---- Calmar ratio -----------------------------------------------------
    calmar_ratio = (
        annualized_return / max(abs(max_drawdown), 1e-8)
    )

    # ---- CVaR at 5 % (Expected Shortfall) ---------------------------------
    cvar_5 = _compute_cvar(returns, alpha=0.05)

    # ---- Turnover ---------------------------------------------------------
    # Already computed above; remains NaN if fractions were not provided.

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "cvar_5": cvar_5,
        "turnover": turnover_val,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Conditional Value at Risk (Expected Shortfall) at level *alpha*.

    CVaR_alpha = E[r | r <= VaR_alpha]

    This is the mean of returns falling at or below the alpha-quantile.

    Args:
        returns: 1-D array of returns.
        alpha:   Tail probability.  Default 0.05 (5 %).

    Returns:
        CVaR as a float (negative number indicates losses).
    """
    if len(returns) == 0:
        return 0.0
    sorted_returns = np.sort(returns)
    n_tail = max(int(np.floor(alpha * len(sorted_returns))), 1)
    cvar = float(np.mean(sorted_returns[:n_tail]))
    return cvar


def _empty_metrics() -> dict[str, float]:
    """Return a metrics dict with all values set to zero / NaN."""
    return {
        "annualized_return": 0.0,
        "annualized_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "calmar_ratio": 0.0,
        "cvar_5": 0.0,
        "turnover": np.nan,
    }
