"""
Unit tests for baseline position sizers (Section 4.3).

Tests cover:
    - EqualWeightSizer returns fixed value
    - FullKellySizer: f* = (b*p - q) / b
    - HalfKellySizer: 0.5 * FullKelly
    - ATRSizer: inversely proportional to ATR
    - RiskParitySizer: inversely proportional to volatility
    - All outputs bounded in [0, f_max]
"""

import pytest
import numpy as np

from naps.baselines.equal_weight import EqualWeightSizer
from naps.baselines.kelly import FullKellySizer, HalfKellySizer
from naps.baselines.atr import ATRSizer
from naps.baselines.risk_parity import RiskParitySizer


SEED = 42


# ---------------------------------------------------------------------------
# EqualWeight
# ---------------------------------------------------------------------------

class TestEqualWeightSizer:
    """Test the fixed equal-weight baseline."""

    def test_returns_fixed_value(self):
        """compute_size should always return the configured fraction."""
        sizer = EqualWeightSizer(fraction=0.10)
        assert sizer.compute_size() == 0.10
        assert sizer.compute_size(signal=1.0, volatility=0.5) == 0.10

    def test_different_fractions(self):
        """Various fraction values should be returned exactly."""
        for frac in [0.05, 0.10, 0.20, 0.50]:
            sizer = EqualWeightSizer(fraction=frac, f_max=1.0)
            assert sizer.compute_size() == frac

    def test_respects_fmax(self):
        """Fraction should be clipped to f_max."""
        sizer = EqualWeightSizer(fraction=0.30, f_max=0.30)
        assert sizer.compute_size() == 0.30

    def test_invalid_fraction_raises(self):
        """Fraction <= 0 or > f_max should raise ValueError."""
        with pytest.raises(ValueError):
            EqualWeightSizer(fraction=0.0)
        with pytest.raises(ValueError):
            EqualWeightSizer(fraction=0.50, f_max=0.40)

    def test_output_bounded(self):
        """Output should always be in [0, f_max]."""
        sizer = EqualWeightSizer(fraction=0.10, f_max=0.40)
        result = sizer.compute_size()
        assert 0.0 <= result <= 0.40


# ---------------------------------------------------------------------------
# FullKelly
# ---------------------------------------------------------------------------

class TestFullKellySizer:
    """Test the Full Kelly criterion sizer."""

    def test_kelly_formula_direct(self):
        """f* = (b*p - q) / b with known p and b."""
        sizer = FullKellySizer(f_max=1.0)
        # p=0.6, b=1.5 -> f* = (1.5*0.6 - 0.4)/1.5 = (0.9-0.4)/1.5 = 0.333...
        f = sizer.compute_size(p=0.6, b=1.5)
        expected = (1.5 * 0.6 - 0.4) / 1.5
        assert abs(f - expected) < 1e-6, f"Expected {expected}, got {f}"

    def test_kelly_with_even_odds(self):
        """50/50 with 1:1 payoff -> f* = 0."""
        sizer = FullKellySizer(f_max=1.0)
        f = sizer.compute_size(p=0.5, b=1.0)
        expected = (1.0 * 0.5 - 0.5) / 1.0  # = 0.0
        assert abs(f - expected) < 1e-6

    def test_kelly_negative_edge_returns_zero(self):
        """When edge is negative (p*b < q), f* should be 0."""
        sizer = FullKellySizer(f_max=1.0)
        f = sizer.compute_size(p=0.3, b=1.0)
        # f* = (1.0*0.3 - 0.7)/1.0 = -0.4 -> clipped to 0
        assert f == 0.0

    def test_kelly_from_returns(self):
        """Estimate p and b from a return series."""
        np.random.seed(SEED)
        # Create returns with positive edge
        returns = np.random.normal(0.01, 0.02, size=200)
        sizer = FullKellySizer(window=60, f_max=1.0)
        f = sizer.compute_size(returns=returns)
        assert 0.0 <= f <= 1.0

    def test_output_bounded(self):
        """Output should never exceed f_max."""
        sizer = FullKellySizer(f_max=0.40)
        # Very high edge: p=0.9, b=3.0 -> f* would be huge
        f = sizer.compute_size(p=0.9, b=3.0)
        assert 0.0 <= f <= 0.40

    def test_zero_b_returns_zero(self):
        """b=0 (no winning trades) should return 0."""
        sizer = FullKellySizer(f_max=1.0)
        f = sizer.compute_size(p=0.5, b=0.0)
        assert f == 0.0

    def test_short_returns_array(self):
        """Very short returns (<2) should return 0."""
        sizer = FullKellySizer(f_max=1.0)
        f = sizer.compute_size(returns=np.array([0.01]))
        assert f == 0.0


# ---------------------------------------------------------------------------
# HalfKelly
# ---------------------------------------------------------------------------

class TestHalfKellySizer:
    """Test that Half Kelly = 0.5 * Full Kelly."""

    def test_half_of_full(self):
        """HalfKelly should be exactly 0.5 * FullKelly."""
        full = FullKellySizer(f_max=1.0)
        half = HalfKellySizer(f_max=1.0)

        for p, b in [(0.6, 1.5), (0.7, 2.0), (0.55, 1.2)]:
            f_full = full.compute_size(p=p, b=b)
            f_half = half.compute_size(p=p, b=b)
            assert abs(f_half - 0.5 * f_full) < 1e-6, (
                f"p={p}, b={b}: half={f_half}, 0.5*full={0.5*f_full}"
            )

    def test_output_bounded(self):
        """Output should be in [0, f_max]."""
        sizer = HalfKellySizer(f_max=0.40)
        f = sizer.compute_size(p=0.9, b=3.0)
        assert 0.0 <= f <= 0.40

    def test_from_returns(self):
        """HalfKelly should work from returns array."""
        np.random.seed(SEED)
        returns = np.random.normal(0.01, 0.02, size=200)
        sizer = HalfKellySizer(window=60, f_max=0.40)
        f = sizer.compute_size(returns=returns)
        assert 0.0 <= f <= 0.40


# ---------------------------------------------------------------------------
# ATR-based sizing
# ---------------------------------------------------------------------------

class TestATRSizer:
    """Test the ATR-based inverse-volatility sizer."""

    def test_inversely_proportional_to_atr(self):
        """Higher ATR should produce smaller bet fraction."""
        sizer = ATRSizer(risk_target=0.02, f_max=1.0)

        f_low_vol = sizer.compute_size(atr=1.0, price=100.0)
        f_high_vol = sizer.compute_size(atr=3.0, price=100.0)

        assert f_low_vol > f_high_vol, (
            f"Higher ATR should give smaller size: "
            f"low_vol={f_low_vol}, high_vol={f_high_vol}"
        )

    def test_formula_atr_over_price(self):
        """f = risk_target / (ATR / price)."""
        sizer = ATRSizer(risk_target=0.02, f_max=1.0)
        f = sizer.compute_size(atr=2.0, price=100.0)
        expected = 0.02 / (2.0 / 100.0)  # = 0.02 / 0.02 = 1.0
        assert abs(f - expected) < 1e-6

    def test_atr_computation_from_ohlc(self):
        """ATR should be computable from raw OHLC data."""
        np.random.seed(SEED)
        n = 30
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n)) * 0.5
        low = close - np.abs(np.random.randn(n)) * 0.5

        sizer = ATRSizer(atr_period=14, risk_target=0.02, f_max=1.0)
        f = sizer.compute_size(high=high, low=low, close=close)
        assert 0.0 <= f <= 1.0

    def test_output_bounded(self):
        """Output should be bounded in [0, f_max]."""
        sizer = ATRSizer(risk_target=0.02, f_max=0.40)
        # Very low ATR -> high fraction, but should be clipped
        f = sizer.compute_size(atr=0.01, price=100.0)
        assert 0.0 <= f <= 0.40

    def test_zero_price_returns_zero(self):
        """Zero price should return 0 (degenerate case)."""
        sizer = ATRSizer(risk_target=0.02, f_max=1.0)
        f = sizer.compute_size(atr=1.0, price=0.0)
        assert f == 0.0

    def test_zero_atr_returns_zero(self):
        """Zero ATR should return 0."""
        sizer = ATRSizer(risk_target=0.02, f_max=1.0)
        f = sizer.compute_size(atr=0.0, price=100.0)
        assert f == 0.0


# ---------------------------------------------------------------------------
# Risk Parity
# ---------------------------------------------------------------------------

class TestRiskParitySizer:
    """Test the inverse-volatility risk parity sizer."""

    def test_inversely_proportional_to_vol(self):
        """Higher vol should produce smaller bet fraction."""
        sizer = RiskParitySizer(target_risk=0.10, f_max=1.0)

        f_low_vol = sizer.compute_size(realized_vol=0.10)
        f_high_vol = sizer.compute_size(realized_vol=0.30)

        assert f_low_vol > f_high_vol, (
            f"Higher vol should give smaller size: "
            f"low_vol={f_low_vol}, high_vol={f_high_vol}"
        )

    def test_formula_target_over_vol(self):
        """f = target_risk / realized_vol."""
        sizer = RiskParitySizer(target_risk=0.10, f_max=1.0)
        f = sizer.compute_size(realized_vol=0.20)
        expected = 0.10 / 0.20  # = 0.5
        assert abs(f - expected) < 1e-6

    def test_from_returns(self):
        """Should compute vol from return series."""
        np.random.seed(SEED)
        returns = np.random.normal(0.001, 0.01, size=100)
        sizer = RiskParitySizer(vol_window=20, target_risk=0.10, f_max=1.0)
        f = sizer.compute_size(returns=returns)
        assert 0.0 <= f <= 1.0

    def test_output_bounded(self):
        """Output should be in [0, f_max]."""
        sizer = RiskParitySizer(target_risk=0.10, f_max=0.40)
        # Very low vol -> large fraction, clipped to f_max
        f = sizer.compute_size(realized_vol=0.01)
        assert 0.0 <= f <= 0.40

    def test_zero_vol_returns_zero(self):
        """Zero volatility should return 0."""
        sizer = RiskParitySizer(target_risk=0.10, f_max=1.0)
        f = sizer.compute_size(realized_vol=0.0)
        assert f == 0.0


# ---------------------------------------------------------------------------
# Cross-cutting: all sizers bounded in [0, f_max]
# ---------------------------------------------------------------------------

class TestAllSizersBounded:
    """Verify every sizer respects the [0, f_max] bound."""

    F_MAX = 0.40

    def test_equal_weight_bounded(self):
        sizer = EqualWeightSizer(fraction=0.10, f_max=self.F_MAX)
        assert 0.0 <= sizer.compute_size() <= self.F_MAX

    def test_full_kelly_bounded(self):
        sizer = FullKellySizer(f_max=self.F_MAX)
        for p, b in [(0.3, 0.5), (0.5, 1.0), (0.7, 2.0), (0.9, 5.0)]:
            f = sizer.compute_size(p=p, b=b)
            assert 0.0 <= f <= self.F_MAX, f"p={p}, b={b} -> f={f}"

    def test_half_kelly_bounded(self):
        sizer = HalfKellySizer(f_max=self.F_MAX)
        for p, b in [(0.3, 0.5), (0.5, 1.0), (0.7, 2.0), (0.9, 5.0)]:
            f = sizer.compute_size(p=p, b=b)
            assert 0.0 <= f <= self.F_MAX, f"p={p}, b={b} -> f={f}"

    def test_atr_bounded(self):
        sizer = ATRSizer(risk_target=0.02, f_max=self.F_MAX)
        for atr, price in [(0.5, 100), (2.0, 100), (5.0, 50), (0.01, 100)]:
            f = sizer.compute_size(atr=atr, price=price)
            assert 0.0 <= f <= self.F_MAX, f"atr={atr}, price={price} -> f={f}"

    def test_risk_parity_bounded(self):
        sizer = RiskParitySizer(target_risk=0.10, f_max=self.F_MAX)
        for vol in [0.05, 0.10, 0.20, 0.50, 1.0]:
            f = sizer.compute_size(realized_vol=vol)
            assert 0.0 <= f <= self.F_MAX, f"vol={vol} -> f={f}"
