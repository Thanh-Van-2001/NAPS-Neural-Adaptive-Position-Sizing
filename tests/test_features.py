"""
Unit tests for feature computation (Sections 3.1, 3.2).

Tests cover:
    - Technical indicators on known data
    - Regime features dimensions
    - Portfolio state dimensions
    - No look-ahead in feature computation
"""

import pytest
import numpy as np
import pandas as pd

from naps.data.features import (
    TechnicalIndicators,
    RegimeFeatures,
    PortfolioState,
    compute_all_features,
    normalize_features,
)


SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, seed: int = SEED) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing.

    Creates a random-walk price series with realistic OHLCV structure:
    close follows a random walk, high >= close, low <= close, open ~ prev close.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, size=n))
    # Ensure prices stay positive
    close = np.maximum(close, 1.0)
    high = close + np.abs(rng.normal(0, 0.3, size=n))
    low = close - np.abs(rng.normal(0, 0.3, size=n))
    low = np.maximum(low, 0.5)
    opn = np.roll(close, 1)
    opn[0] = close[0]
    volume = rng.exponential(1e6, size=n)

    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# TechnicalIndicators
# ---------------------------------------------------------------------------

class TestTechnicalIndicators:
    """Test technical indicator computation."""

    def test_output_has_expected_columns(self):
        """Result should contain all OHLCV + indicator columns."""
        df = _make_ohlcv(200)
        result = TechnicalIndicators.compute(df)
        # Must contain all required OHLCV columns
        for col in TechnicalIndicators.REQUIRED_COLS:
            assert col in result.columns, f"Missing column: {col}"
        # Must contain indicator columns
        for col in ["sma_20", "sma_50", "ema_12", "ema_26", "macd_line",
                     "macd_signal", "bb_width", "atr_14"]:
            assert col in result.columns, f"Missing indicator: {col}"

    def test_sma_20_correctness(self):
        """SMA(20) should match a manual rolling mean calculation."""
        df = _make_ohlcv(100)
        result = TechnicalIndicators.compute(df)
        # Compute expected SMA manually for the last point
        expected = df["close"].iloc[-20:].mean()
        actual = result["sma_20"].iloc[-1]
        assert abs(actual - expected) < 1e-6, f"SMA(20): expected {expected}, got {actual}"

    def test_ema_12_present(self):
        """EMA(12) should be present and finite after warmup."""
        df = _make_ohlcv(100)
        result = TechnicalIndicators.compute(df)
        ema_vals = result["ema_12"].dropna()
        assert len(ema_vals) > 0
        assert np.isfinite(ema_vals.values).all()

    def test_atr_14_positive(self):
        """ATR(14) should be non-negative."""
        df = _make_ohlcv(100)
        result = TechnicalIndicators.compute(df)
        atr_vals = result["atr_14"].dropna()
        assert (atr_vals >= 0).all()

    def test_on_known_constant_data(self):
        """On constant prices, SMA should equal the price and ATR should be ~0."""
        n = 100
        dates = pd.bdate_range("2020-01-01", periods=n)
        df = pd.DataFrame({
            "open": 100.0, "high": 100.0, "low": 100.0,
            "close": 100.0, "volume": 1e6,
        }, index=dates)
        result = TechnicalIndicators.compute(df)
        sma = result["sma_20"].dropna()
        assert np.allclose(sma.values, 100.0, atol=1e-6)
        atr = result["atr_14"].dropna()
        assert np.allclose(atr.values, 0.0, atol=1e-6)

    def test_indicator_names_returns_list(self):
        """indicator_names() should return a list of 15 strings."""
        names = TechnicalIndicators.indicator_names()
        assert isinstance(names, list)
        assert len(names) == 15
        assert all(isinstance(n, str) for n in names)

    def test_missing_column_raises(self):
        """Missing a required column should raise ValueError."""
        df = pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1]})
        # Missing "volume"
        with pytest.raises(ValueError, match="[Mm]issing"):
            TechnicalIndicators.compute(df)


# ---------------------------------------------------------------------------
# RegimeFeatures
# ---------------------------------------------------------------------------

class TestRegimeFeatures:
    """Test regime feature dimensions and values."""

    def test_output_dimensions(self):
        """Should produce a DataFrame with 3 columns."""
        df = _make_ohlcv(200)
        regime = RegimeFeatures.compute(df["close"])
        assert regime.shape[1] == 3
        assert set(regime.columns) == {"realized_vol", "market_corr", "trend_strength"}

    def test_output_length_matches_input(self):
        """Output should have the same number of rows as input."""
        df = _make_ohlcv(200)
        regime = RegimeFeatures.compute(df["close"])
        assert len(regime) == len(df)

    def test_realized_vol_positive_after_warmup(self):
        """Realized vol should be non-negative where defined."""
        df = _make_ohlcv(200)
        regime = RegimeFeatures.compute(df["close"])
        vol = regime["realized_vol"].dropna()
        assert (vol >= 0).all()

    def test_with_market_index(self):
        """Should accept an external market index for correlation."""
        df = _make_ohlcv(200)
        rng = np.random.default_rng(SEED + 1)
        market = pd.Series(
            100.0 + np.cumsum(rng.normal(0, 0.5, 200)),
            index=df.index,
        )
        regime = RegimeFeatures.compute(df["close"], market_index=market)
        assert regime.shape[1] == 3


# ---------------------------------------------------------------------------
# PortfolioState
# ---------------------------------------------------------------------------

class TestPortfolioState:
    """Test portfolio state feature dimensions and values."""

    def test_output_dimensions(self):
        """Should produce a DataFrame with 5 columns."""
        n = 100
        dates = pd.bdate_range("2020-01-01", periods=n)
        # Use a random walk with enough variance to avoid degenerate corrcoef
        rng = np.random.default_rng(SEED)
        equity = pd.Series(
            np.maximum(1.0 + np.cumsum(rng.normal(0.001, 0.02, n)), 0.5),
            index=dates,
        )
        state = PortfolioState.compute(equity)
        assert state.shape[1] == 5
        expected_cols = {"drawdown", "cum_pnl", "days_since_peak", "exposure", "serial_corr"}
        assert set(state.columns) == expected_cols

    def test_output_length_matches_input(self):
        """Output should have the same number of rows as input."""
        n = 100
        dates = pd.bdate_range("2020-01-01", periods=n)
        equity = pd.Series(np.linspace(1.0, 1.5, n), index=dates)
        state = PortfolioState.compute(equity)
        assert len(state) == n

    def test_monotonic_equity_zero_drawdown(self):
        """Monotonically increasing equity should have zero drawdown."""
        n = 100
        dates = pd.bdate_range("2020-01-01", periods=n)
        equity = pd.Series(np.linspace(1.0, 2.0, n), index=dates)
        state = PortfolioState.compute(equity)
        assert np.allclose(state["drawdown"].values, 0.0, atol=1e-10)

    def test_drawdown_after_drop(self):
        """A price drop should produce positive drawdown."""
        n = 50
        dates = pd.bdate_range("2020-01-01", periods=n)
        equity_vals = np.ones(n)
        equity_vals[25:] = 0.80  # 20% drop
        equity = pd.Series(equity_vals, index=dates)
        state = PortfolioState.compute(equity)
        assert state["drawdown"].iloc[-1] > 0.15  # should be ~0.20

    def test_days_since_peak(self):
        """days_since_peak should increase after the peak."""
        n = 50
        dates = pd.bdate_range("2020-01-01", periods=n)
        equity_vals = np.ones(n)
        equity_vals[:20] = np.linspace(1.0, 1.5, 20)  # rising to peak at t=19
        equity_vals[20:] = 1.2  # below peak
        equity = pd.Series(equity_vals, index=dates)
        state = PortfolioState.compute(equity)
        # At the last time step, days since peak should be > 0
        assert state["days_since_peak"].iloc[-1] > 0


# ---------------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------------

class TestNoLookahead:
    """Verify features do not use future data (no look-ahead bias)."""

    def test_technical_indicators_no_lookahead(self):
        """Changing future data should not affect past indicator values.

        Strategy: compute indicators on the full series, then truncate
        the last 50 rows and recompute. The overlapping portion should
        be identical.
        """
        df_full = _make_ohlcv(200)
        df_short = df_full.iloc[:150].copy()

        result_full = TechnicalIndicators.compute(df_full)
        result_short = TechnicalIndicators.compute(df_short)

        # Compare overlapping rows (first 150) for a few indicators
        for col in ["sma_20", "ema_12", "atr_14"]:
            full_vals = result_full[col].iloc[:150].dropna()
            short_vals = result_short[col].dropna()
            # Align indices
            common_idx = full_vals.index.intersection(short_vals.index)
            if len(common_idx) > 0:
                np.testing.assert_allclose(
                    full_vals.loc[common_idx].values,
                    short_vals.loc[common_idx].values,
                    atol=1e-10,
                    err_msg=f"Look-ahead detected in {col}",
                )

    def test_normalize_features_no_lookahead(self):
        """Expanding-window normalisation should not leak future data."""
        df = _make_ohlcv(200)
        result = TechnicalIndicators.compute(df)
        # Select a subset of numeric columns for normalisation
        features = result[["sma_20", "ema_12", "atr_14"]].dropna()
        normed = normalize_features(features, method="zscore")

        # The normalised value at row i should depend only on rows <= i.
        # Check by recomputing on a truncated frame.
        n = len(features)
        mid = n // 2
        features_short = features.iloc[:mid]
        normed_short = normalize_features(features_short, method="zscore")

        # Overlapping portion should be identical
        common_idx = normed.index[:mid]
        pd.testing.assert_frame_equal(
            normed.loc[common_idx],
            normed_short.loc[common_idx],
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# compute_all_features convenience function
# ---------------------------------------------------------------------------

class TestComputeAllFeatures:
    """Test the convenience wrapper that produces 20 features."""

    def test_output_has_20_columns(self):
        """Should produce exactly 20 columns (5 OHLCV + 15 indicators)."""
        df = _make_ohlcv(200)
        result = compute_all_features(df)
        assert result.shape[1] == 20, f"Expected 20 columns, got {result.shape[1]}"

    def test_case_insensitive_columns(self):
        """Should handle upper-case column names."""
        df = _make_ohlcv(200)
        df.columns = [c.upper() for c in df.columns]
        result = compute_all_features(df)
        assert result.shape[1] == 20
