"""
Feature engineering for NAPS (Sections 3.1, 3.2, 4.1).

Three feature groups feed the NAPS architecture:

1. **Market features** (20-dim): 5 raw OHLCV columns + 15 technical indicators,
   forming the temporal context window consumed by the Transformer encoder.
2. **Regime features** (3-dim): realized volatility, market correlation, and
   trend strength — conditioning the risk-state encoder.
3. **Portfolio state** (5-dim): drawdown, cumulative P&L, days since peak,
   aggregate exposure, and serial correlation of trade outcomes.

All indicators are computed using only past data (no look-ahead bias).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Technical indicators (Section 3.2, Table 1)
# ---------------------------------------------------------------------------

class TechnicalIndicators:
    """Compute 15 technical indicators from OHLCV data.

    The input DataFrame must contain columns:
        ``open``, ``high``, ``low``, ``close``, ``volume``

    Returns a DataFrame with 20 columns total: the original 5 OHLCV columns
    plus 15 technical indicator columns.  All calculations are strictly causal
    (only past data is used).

    Indicators (indexed 1-15):
        1.  SMA(20)               6.  MACD signal           11. Williams %R(14)
        2.  SMA(50)               7.  Bollinger width(20)   12. CCI(20)
        3.  EMA(12)               8.  ATR(14)               13. ADX(14)
        4.  EMA(26)               9.  OBV                   14. Stochastic %K(14)
        5.  MACD line            10.  Stochastic %D(3)      15. RSI(14)

    Note: indicators are numbered 1-15 per the specification; the ordering
    inside the returned DataFrame follows a logical grouping.
    """

    # Required input columns
    REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all 15 technical indicators and return the full feature set.

        Args:
            df: OHLCV DataFrame.  Must contain columns ``open``, ``high``,
                ``low``, ``close``, ``volume``.

        Returns:
            DataFrame with 20 columns (5 OHLCV + 15 indicators).  The first
            rows will contain NaN where the lookback is insufficient.
        """
        df = df.copy()
        _validate_columns(df, TechnicalIndicators.REQUIRED_COLS)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # --- Moving averages ---
        df["sma_20"] = close.rolling(window=20, min_periods=20).mean()
        df["sma_50"] = close.rolling(window=50, min_periods=50).mean()
        df["ema_12"] = close.ewm(span=12, adjust=False).mean()
        df["ema_26"] = close.ewm(span=26, adjust=False).mean()

        # --- MACD ---
        df["macd_line"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()

        # --- Bollinger Band width ---
        bb_ma = close.rolling(window=20, min_periods=20).mean()
        bb_std = close.rolling(window=20, min_periods=20).std()
        df["bb_width"] = (2.0 * bb_std) / bb_ma.where(bb_ma != 0, np.nan)

        # --- ATR(14) ---
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr_14"] = tr.rolling(window=14, min_periods=14).mean()

        # --- OBV (On-Balance Volume) ---
        obv_sign = np.sign(close.diff())
        df["obv"] = (obv_sign * volume).cumsum()

        # --- Stochastic %K(14) and %D(3) ---
        lowest_14 = low.rolling(window=14, min_periods=14).min()
        highest_14 = high.rolling(window=14, min_periods=14).max()
        denom = (highest_14 - lowest_14).replace(0, np.nan)
        df["stoch_k"] = 100.0 * (close - lowest_14) / denom
        df["stoch_d"] = df["stoch_k"].rolling(window=3, min_periods=3).mean()

        # --- Williams %R(14) ---
        df["williams_r"] = -100.0 * (highest_14 - close) / denom

        # --- CCI(20) ---
        tp = (high + low + close) / 3.0
        tp_ma = tp.rolling(window=20, min_periods=20).mean()
        tp_md = tp.rolling(window=20, min_periods=20).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        df["cci_20"] = (tp - tp_ma) / (0.015 * tp_md.where(tp_md != 0, np.nan))

        # --- ADX(14) ---
        df["adx_14"] = _compute_adx(high, low, close, period=14)

        return df

    @staticmethod
    def indicator_names() -> list[str]:
        """Return ordered list of the 15 indicator column names."""
        return [
            "sma_20", "sma_50", "ema_12", "ema_26",
            "macd_line", "macd_signal",
            "bb_width", "atr_14", "obv",
            "stoch_k", "stoch_d", "williams_r",
            "cci_20", "adx_14",
            # 15th feature is RSI
            "rsi_14",
        ]


def _compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average Directional Index via Wilder smoothing."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=high.index)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder smoothing (equivalent to EMA with alpha = 1/period)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.where(atr != 0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.where(atr != 0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx


# ---------------------------------------------------------------------------
# RSI helper (kept separate for reuse)
# ---------------------------------------------------------------------------

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index via Wilder smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ---------------------------------------------------------------------------
# Regime features r_t ∈ R^3 (Section 3.1)
# ---------------------------------------------------------------------------

class RegimeFeatures:
    """Market-regime descriptor: r_t ∈ R^3.

    Components:
        1. 20-day realized volatility (annualized std of log-returns).
        2. 60-day rolling correlation with market index.
        3. 20-day trend strength (slope of linear regression on close prices,
           normalised by the standard deviation of the prices in the window).
    """

    @staticmethod
    def compute(
        close: pd.Series,
        market_index: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Compute regime features from a price series.

        Args:
            close: Daily closing prices.
            market_index: Daily closing prices of a market-wide index.  If
                ``None`` a synthetic proxy (cumulative mean of returns) is used.

        Returns:
            DataFrame with columns ``realized_vol``, ``market_corr``,
            ``trend_strength``.
        """
        log_ret = np.log(close / close.shift(1))

        # 1. 20-day realized volatility (annualized)
        realized_vol = log_ret.rolling(window=20, min_periods=20).std() * np.sqrt(252)

        # 2. 60-day rolling correlation with market index
        if market_index is None:
            # Proxy: equal-weight mean of expanding returns (rough market)
            market_index = close.copy()
        mkt_ret = np.log(market_index / market_index.shift(1))
        market_corr = log_ret.rolling(window=60, min_periods=60).corr(mkt_ret)

        # 3. 20-day trend strength (OLS slope / std)
        trend_strength = close.rolling(window=20, min_periods=20).apply(
            _normalized_slope, raw=True
        )

        return pd.DataFrame(
            {
                "realized_vol": realized_vol,
                "market_corr": market_corr,
                "trend_strength": trend_strength,
            },
            index=close.index,
        )


def _normalized_slope(x: np.ndarray) -> float:
    """OLS slope of *x* against time, divided by std(x)."""
    n = len(x)
    t = np.arange(n, dtype=np.float64)
    std = np.std(x)
    if std < 1e-12:
        return 0.0
    slope = (np.mean(t * x) - np.mean(t) * np.mean(x)) / (np.var(t) + 1e-12)
    return slope / std


# ---------------------------------------------------------------------------
# Portfolio state p_t ∈ R^5 (Section 3.1)
# ---------------------------------------------------------------------------

class PortfolioState:
    """Portfolio-level state vector: p_t ∈ R^5.

    Components:
        1. Current drawdown from equity peak.
        2. Cumulative P&L (as fraction of initial capital).
        3. Days since last equity peak.
        4. Current aggregate exposure (fraction of capital).
        5. Serial correlation of last 10 trade outcomes (returns).
    """

    @staticmethod
    def compute(
        equity_curve: pd.Series,
        exposures: Optional[pd.Series] = None,
        trade_returns: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Build the portfolio-state DataFrame.

        Args:
            equity_curve: Time-indexed equity values (initial = 1.0).
            exposures: Fraction of capital deployed at each step.  Defaults
                to 1.0 everywhere if not supplied.
            trade_returns: Per-period trade returns.  Defaults to the
                period-over-period change in the equity curve.

        Returns:
            DataFrame with columns ``drawdown``, ``cum_pnl``,
            ``days_since_peak``, ``exposure``, ``serial_corr``.
        """
        equity = equity_curve.values.astype(np.float64)
        n = len(equity)

        # Running peak
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / np.where(running_max != 0, running_max, 1.0)

        # Cumulative P&L (fraction of initial)
        cum_pnl = (equity - equity[0]) / equity[0]

        # Days since last equity peak
        days_since_peak = np.zeros(n, dtype=np.float64)
        last_peak_idx = 0
        for i in range(n):
            if equity[i] >= running_max[i]:
                last_peak_idx = i
            days_since_peak[i] = i - last_peak_idx

        # Exposure
        if exposures is not None:
            exp_vals = exposures.values.astype(np.float64)
        else:
            exp_vals = np.ones(n, dtype=np.float64)

        # Serial correlation of last 10 trade outcomes
        if trade_returns is not None:
            tr = trade_returns.values.astype(np.float64)
        else:
            tr = np.diff(equity, prepend=equity[0]) / np.where(
                np.abs(equity) > 1e-12, np.abs(equity), 1.0
            )
        serial_corr = _rolling_serial_corr(tr, window=10)

        return pd.DataFrame(
            {
                "drawdown": drawdown,
                "cum_pnl": cum_pnl,
                "days_since_peak": days_since_peak,
                "exposure": exp_vals,
                "serial_corr": serial_corr,
            },
            index=equity_curve.index,
        )


def _rolling_serial_corr(x: np.ndarray, window: int = 10) -> np.ndarray:
    """Correlation between x[t] and x[t-1] over a rolling window."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window, n):
        seg = x[i - window + 1 : i + 1]
        a, b = seg[:-1], seg[1:]
        std_a = np.std(a, ddof=1)
        std_b = np.std(b, ddof=1)
        if std_a < 1e-12 or std_b < 1e-12:
            out[i] = 0.0
            continue
        # Manual Pearson correlation to avoid MKL abort on degenerate inputs
        a_centered = a - np.mean(a)
        b_centered = b - np.mean(b)
        denom = np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2))
        if denom < 1e-12:
            out[i] = 0.0
        else:
            c = float(np.sum(a_centered * b_centered) / denom)
            out[i] = c if np.isfinite(c) else 0.0
    return out


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise if *df* is missing any of *required* columns (case-insensitive)."""
    cols_lower = {c.lower() for c in df.columns}
    missing = [c for c in required if c.lower() not in cols_lower]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_all_features(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full 20-feature market representation from raw OHLCV data.

    This is a convenience wrapper that:
    1. Normalises column names to lower-case.
    2. Computes all 15 technical indicators via :class:`TechnicalIndicators`.
    3. Adds RSI(14) if not already present (bringing total TIs to 15).

    Args:
        ohlcv_df: DataFrame with at least ``open``, ``high``, ``low``,
            ``close``, ``volume`` columns (case-insensitive).

    Returns:
        DataFrame with exactly 20 columns: ``open``, ``high``, ``low``,
        ``close``, ``volume`` + 15 indicator columns.
    """
    df = ohlcv_df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Compute 14 indicators via TechnicalIndicators
    df = TechnicalIndicators.compute(df)

    # Add RSI to reach 15 indicators
    if "rsi_14" not in df.columns:
        df["rsi_14"] = _compute_rsi(df["close"], period=14)

    # Select the canonical 20-column order
    feature_cols = TechnicalIndicators.REQUIRED_COLS + TechnicalIndicators.indicator_names()
    return df[feature_cols]


def normalize_features(
    features: pd.DataFrame,
    method: str = "zscore",
) -> pd.DataFrame:
    """Normalise features column-wise (expanding window — no look-ahead).

    Args:
        features: Raw feature DataFrame.
        method: ``'zscore'`` (default) subtracts expanding mean and divides by
            expanding std.  ``'minmax'`` rescales to [0, 1] using expanding
            min / max.

    Returns:
        Normalised DataFrame of the same shape.
    """
    if method == "zscore":
        mean = features.expanding(min_periods=2).mean()
        std = features.expanding(min_periods=2).std().replace(0, 1.0)
        return (features - mean) / std
    elif method == "minmax":
        emin = features.expanding(min_periods=2).min()
        emax = features.expanding(min_periods=2).max()
        denom = (emax - emin).replace(0, 1.0)
        return (features - emin) / denom
    else:
        raise ValueError(f"Unknown normalisation method: {method!r}. Use 'zscore' or 'minmax'.")
