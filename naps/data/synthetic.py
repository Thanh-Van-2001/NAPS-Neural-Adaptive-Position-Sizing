"""
Synthetic data generation for NAPS experiments (Section 4.1).

Generates realistic daily OHLCV data for three asset classes:
    - **Equity** (50 assets, 1260 days ≈ 5 years): fat tails, volatility
      clustering, mean reversion, and regime structure matching S&P 500
      characteristics.
    - **Crypto** (20 assets, 1260 days): higher volatility, 24/7 dynamics,
      momentum-driven with occasional crashes.
    - **Forex** (7 G7 pairs, 1260 days): low volatility, mean-reverting,
      carrying typical FX correlation structure.

Also provides helpers to generate synthetic signal predictions and MC-dropout
uncertainty estimates calibrated to specified directional accuracy targets
(Table 2 in the paper).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


class SyntheticMarketGenerator:
    """Generate realistic synthetic market data for backtesting NAPS.

    All methods accept a ``seed`` argument for full reproducibility.
    """

    # -----------------------------------------------------------------
    # Equity
    # -----------------------------------------------------------------

    @staticmethod
    def generate_equity_data(
        n_assets: int = 50,
        n_days: int = 1260,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Simulate daily OHLCV data with S&P 500-like characteristics.

        Properties:
            - Annual drift ~8%, annual vol ~16% (per asset, with spread).
            - GARCH(1,1) volatility clustering.
            - Student-t innovations (df=5) for fat tails.
            - Three-state Markov regime structure (bull / bear / sideways).

        Args:
            n_assets: Number of synthetic equity instruments.
            n_days: Number of trading days to simulate.
            seed: Random seed.

        Returns:
            Multi-indexed ``DataFrame`` with index levels ``(asset, date)``
            and columns ``open``, ``high``, ``low``, ``close``, ``volume``.
        """
        rng = np.random.default_rng(seed)
        frames = []

        dates = pd.bdate_range(start="2018-01-02", periods=n_days, freq="B")

        for i in range(n_assets):
            # Per-asset parameters with slight randomisation
            annual_drift = 0.08 + rng.normal(0, 0.03)
            base_vol = 0.16 + rng.normal(0, 0.03)
            daily_drift = annual_drift / 252
            daily_vol = base_vol / np.sqrt(252)

            # Regime sequence (bull=0, bear=1, sideways=2)
            regimes = _generate_regime_sequence(n_days, rng)

            # Regime-dependent drift adjustments
            drift_adj = np.where(regimes == 0, 0.0005, np.where(regimes == 1, -0.0008, 0.0))

            # GARCH(1,1) volatility
            sigma = _garch_volatility(n_days, daily_vol, rng)

            # Fat-tailed innovations (Student-t df=5)
            z = rng.standard_t(df=5, size=n_days) / np.sqrt(5 / 3)  # normalise variance

            log_returns = (daily_drift + drift_adj) + sigma * z

            # Build price series
            close = 100.0 * np.exp(np.cumsum(log_returns))
            ohlcv = _close_to_ohlcv(close, sigma, rng, base_volume=5e6)

            df = pd.DataFrame(ohlcv, columns=["open", "high", "low", "close", "volume"])
            df["date"] = dates[:n_days]
            df["asset"] = f"EQ_{i:03d}"
            frames.append(df)

        result = pd.concat(frames, ignore_index=True)
        result = result.set_index(["asset", "date"])
        return result

    # -----------------------------------------------------------------
    # Crypto
    # -----------------------------------------------------------------

    @staticmethod
    def generate_crypto_data(
        n_assets: int = 20,
        n_days: int = 1260,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Simulate crypto OHLCV with higher vol and momentum dynamics.

        Properties:
            - Annual drift ~15% (wide spread), annual vol ~60%.
            - Heavier tails (Student-t df=4).
            - Occasional crash events (Poisson-driven drawdowns).
            - Momentum persistence via AR(1) returns.

        Args:
            n_assets: Number of crypto assets.
            n_days: Number of calendar days (crypto trades 24/7).
            seed: Random seed.

        Returns:
            Multi-indexed DataFrame ``(asset, date)`` with OHLCV columns.
        """
        rng = np.random.default_rng(seed)
        frames = []

        # Calendar days (crypto = 7 days/week)
        dates = pd.date_range(start="2018-01-02", periods=n_days, freq="D")

        for i in range(n_assets):
            annual_drift = 0.15 + rng.normal(0, 0.10)
            base_vol = 0.60 + rng.normal(0, 0.15)
            daily_drift = annual_drift / 365
            daily_vol = base_vol / np.sqrt(365)

            sigma = _garch_volatility(n_days, daily_vol, rng, alpha=0.15, beta=0.80)

            # Fat tails + AR(1) momentum
            z = rng.standard_t(df=4, size=n_days) / np.sqrt(4 / 2)
            raw_ret = daily_drift + sigma * z

            # AR(1) persistence
            ar_ret = np.empty(n_days)
            ar_ret[0] = raw_ret[0]
            phi = 0.05 + rng.uniform(0, 0.10)  # mild momentum
            for t in range(1, n_days):
                ar_ret[t] = phi * ar_ret[t - 1] + raw_ret[t]

            # Poisson crash events (mean ≈2 per 1260 days)
            n_crashes = rng.poisson(2)
            crash_days = rng.integers(60, n_days, size=n_crashes)
            for cd in crash_days:
                crash_size = rng.uniform(0.10, 0.35)
                ar_ret[cd] = -crash_size

            close = 1000.0 * np.exp(np.cumsum(ar_ret))
            ohlcv = _close_to_ohlcv(close, sigma, rng, base_volume=2e7)

            df = pd.DataFrame(ohlcv, columns=["open", "high", "low", "close", "volume"])
            df["date"] = dates[:n_days]
            df["asset"] = f"CRYPTO_{i:03d}"
            frames.append(df)

        result = pd.concat(frames, ignore_index=True)
        result = result.set_index(["asset", "date"])
        return result

    # -----------------------------------------------------------------
    # Forex
    # -----------------------------------------------------------------

    @staticmethod
    def generate_forex_data(
        n_pairs: int = 7,
        n_days: int = 1260,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Simulate G7 forex pair OHLCV with mean-reversion dynamics.

        Properties:
            - Near-zero drift, annual vol ~8%.
            - Ornstein-Uhlenbeck mean reversion.
            - Lighter tails than equities (df=8).

        Pairs: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD.

        Args:
            n_pairs: Number of pairs (max 7; extras get synthetic names).
            n_days: Trading days.
            seed: Random seed.

        Returns:
            Multi-indexed DataFrame ``(asset, date)`` with OHLCV columns.
        """
        rng = np.random.default_rng(seed)
        frames = []

        pair_names = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
            "AUDUSD", "USDCAD", "NZDUSD",
        ]
        # Starting levels (approximate real-world)
        start_prices = [1.20, 1.35, 110.0, 0.92, 0.75, 1.25, 0.70]

        dates = pd.bdate_range(start="2018-01-02", periods=n_days, freq="B")

        for i in range(n_pairs):
            name = pair_names[i] if i < len(pair_names) else f"FX_{i:03d}"
            p0 = start_prices[i] if i < len(start_prices) else 1.0

            base_vol = 0.08 + rng.normal(0, 0.02)
            daily_vol = base_vol / np.sqrt(252)

            # Ornstein-Uhlenbeck (mean-reverting log-price)
            theta = 0.02 + rng.uniform(0, 0.02)  # reversion speed
            log_price = np.empty(n_days)
            log_price[0] = np.log(p0)
            mu = np.log(p0)

            sigma = _garch_volatility(n_days, daily_vol, rng, alpha=0.04, beta=0.92)
            z = rng.standard_t(df=8, size=n_days) / np.sqrt(8 / 6)

            for t in range(1, n_days):
                log_price[t] = (
                    log_price[t - 1]
                    + theta * (mu - log_price[t - 1])
                    + sigma[t] * z[t]
                )

            close = np.exp(log_price)
            ohlcv = _close_to_ohlcv(close, sigma, rng, base_volume=1e8)

            df = pd.DataFrame(ohlcv, columns=["open", "high", "low", "close", "volume"])
            df["date"] = dates[:n_days]
            df["asset"] = name
            frames.append(df)

        result = pd.concat(frames, ignore_index=True)
        result = result.set_index(["asset", "date"])
        return result

    # -----------------------------------------------------------------
    # Signal + uncertainty generation
    # -----------------------------------------------------------------

    @staticmethod
    def generate_signal_data(
        prices: pd.Series | np.ndarray,
        accuracy: float = 0.542,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic directional signals with calibrated accuracy.

        The signal model mimics a 2-layer LSTM that predicts next-day return
        direction and magnitude.  The *accuracy* parameter controls the
        fraction of days on which the predicted sign matches the realised sign
        (Section 4.2, Table 2).

        MC-dropout uncertainty is higher during volatile periods, matching the
        behaviour of a properly calibrated Bayesian signal model.

        Args:
            prices: 1-D array or Series of daily closing prices.
            accuracy: Target directional accuracy in (0.5, 1.0).  Default
                0.542 (equity target from Table 2).
            seed: Random seed.

        Returns:
            Tuple of:
                signals      (N, 2) -- column 0 = direction in {-1, 0, +1},
                                        column 1 = predicted magnitude.
                uncertainties (N,)  -- MC-dropout standard deviation.
        """
        rng = np.random.default_rng(seed)

        if isinstance(prices, pd.Series):
            prices = prices.values.astype(np.float64)
        else:
            prices = np.asarray(prices, dtype=np.float64)

        n = len(prices)
        returns = np.diff(prices) / prices[:-1]
        returns = np.concatenate([[0.0], returns])  # align length

        # True direction
        true_dir = np.sign(returns)

        # Noisy prediction: flip sign with probability (1 - accuracy)
        flip_mask = rng.random(n) > accuracy
        pred_dir = true_dir.copy()
        pred_dir[flip_mask] = -pred_dir[flip_mask]

        # Small fraction mapped to zero (flat prediction)
        flat_mask = rng.random(n) < 0.05
        pred_dir[flat_mask] = 0.0

        # Predicted magnitude: noisy version of |return|
        abs_ret = np.abs(returns)
        pred_mag = abs_ret * (1.0 + rng.normal(0, 0.3, size=n))
        pred_mag = np.clip(pred_mag, 0.0, None)

        signals = np.column_stack([pred_dir, pred_mag])

        # MC-dropout uncertainty: base level + scaled by local volatility
        rolling_vol = pd.Series(returns).rolling(window=20, min_periods=1).std().values
        base_uncertainty = 0.01
        uncertainties = base_uncertainty + 2.0 * rolling_vol + rng.exponential(0.005, size=n)

        return signals, uncertainties


# ======================================================================
# Internal helpers
# ======================================================================

def _generate_regime_sequence(
    n_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Three-state Markov chain: bull(0), bear(1), sideways(2).

    Transition matrix calibrated so that bull and sideways regimes are more
    persistent than bear regimes (matching empirical S&P data).
    """
    # Transition matrix (rows = current state, cols = next state)
    P = np.array(
        [
            [0.98, 0.01, 0.01],  # bull    -> {bull, bear, side}
            [0.02, 0.95, 0.03],  # bear    -> ...
            [0.02, 0.02, 0.96],  # sideways-> ...
        ]
    )
    states = np.empty(n_days, dtype=np.int64)
    states[0] = rng.choice(3, p=[0.5, 0.2, 0.3])
    for t in range(1, n_days):
        states[t] = rng.choice(3, p=P[states[t - 1]])
    return states


def _garch_volatility(
    n: int,
    base_vol: float,
    rng: np.random.Generator,
    alpha: float = 0.08,
    beta: float = 0.88,
) -> np.ndarray:
    """Simulate GARCH(1,1) conditional volatility path.

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    where ω = base_vol² · (1 - α - β) ensures the unconditional variance
    equals base_vol².
    """
    omega = base_vol**2 * (1 - alpha - beta)
    sigma2 = np.empty(n)
    sigma2[0] = base_vol**2

    # Pre-generate shocks (standard normal; caller applies fat tails separately)
    eps = rng.standard_normal(n)

    for t in range(1, n):
        sigma2[t] = omega + alpha * (base_vol * eps[t - 1]) ** 2 + beta * sigma2[t - 1]
        # Floor to avoid numerical issues
        sigma2[t] = max(sigma2[t], 1e-10)

    return np.sqrt(sigma2)


def _close_to_ohlcv(
    close: np.ndarray,
    sigma: np.ndarray,
    rng: np.random.Generator,
    base_volume: float = 5e6,
) -> np.ndarray:
    """Expand a close-price series into full OHLCV rows.

    Open/High/Low are constructed from Close ± intraday noise scaled by
    the local volatility.  Volume is log-normal with a slight inverse
    correlation to returns (higher volume on down days).
    """
    n = len(close)
    intraday_scale = sigma * 0.5  # intraday range roughly half of daily vol

    noise_open = rng.normal(0, 1, n) * intraday_scale
    noise_hl = np.abs(rng.normal(0, 1, (n, 2))) * intraday_scale[:, None]

    open_price = close * (1 + noise_open * 0.3)
    high = np.maximum(close, open_price) * (1 + noise_hl[:, 0] * 0.5)
    low = np.minimum(close, open_price) * (1 - noise_hl[:, 1] * 0.5)

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(close, open_price))
    low = np.minimum(low, np.minimum(close, open_price))
    low = np.clip(low, 1e-4, None)

    # Volume: log-normal with negative return correlation
    ret = np.diff(close, prepend=close[0]) / np.where(close > 0, close, 1.0)
    vol_shock = -0.3 * np.abs(ret) + rng.normal(0, 0.3, n)
    volume = base_volume * np.exp(vol_shock)
    volume = np.round(volume).astype(np.float64)

    return np.column_stack([open_price, high, low, close, volume])
