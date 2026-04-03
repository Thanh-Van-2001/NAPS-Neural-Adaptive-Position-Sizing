"""
ATR-based position sizer (Section 4.3).

Sizes positions inversely proportional to the Average True Range (ATR)
so that each trade risks approximately the same dollar amount:

    f = risk_target / (ATR_14 / price)

The fraction is capped at ``f_max`` to prevent excessive concentration.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ATRSizer:
    """Inverse 14-day ATR sizer with a fixed risk target.

    For each trade the bet fraction is set so that one ATR-sized adverse
    move costs approximately ``risk_target`` fraction of equity:

        f = risk_target / (ATR / price)

    Args:
        atr_period:   Number of periods for ATR calculation. Default 14.
        risk_target:  Fraction of equity risked per trade. Default 0.02 (2 %).
        f_max:        Maximum allowed bet fraction. Default 1.0.
    """

    def __init__(
        self,
        atr_period: int = 14,
        risk_target: float = 0.02,
        f_max: float = 1.0,
    ) -> None:
        self.atr_period = atr_period
        self.risk_target = risk_target
        self.f_max = f_max

    @staticmethod
    def compute_atr(
        high: NDArray[np.floating],
        low: NDArray[np.floating],
        close: NDArray[np.floating],
        period: int = 14,
    ) -> float:
        """Compute the Average True Range over the most recent *period* bars.

        True Range for bar *i* is::

            TR_i = max(high_i - low_i,
                       |high_i - close_{i-1}|,
                       |low_i  - close_{i-1}|)

        ATR is the simple mean of the last *period* true ranges.

        Args:
            high:   1-D array of high prices.
            low:    1-D array of low prices.
            close:  1-D array of close prices.
            period: Lookback window. Default 14.

        Returns:
            ATR value (scalar float).
        """
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)

        n = len(close)
        if n < 2:
            return float(high[-1] - low[-1]) if n == 1 else 0.0

        prev_close = close[:-1]
        h = high[1:]
        lo = low[1:]

        tr = np.maximum(
            h - lo,
            np.maximum(np.abs(h - prev_close), np.abs(lo - prev_close)),
        )

        # Use the most recent `period` true ranges.
        recent_tr = tr[-period:]
        return float(np.mean(recent_tr))

    def compute_size(
        self,
        high: NDArray[np.floating] | None = None,
        low: NDArray[np.floating] | None = None,
        close: NDArray[np.floating] | None = None,
        atr: float | None = None,
        price: float | None = None,
        **kwargs,
    ) -> float:
        """Compute the ATR-based bet fraction.

        Either supply raw OHLC arrays (``high``, ``low``, ``close``) to
        have ATR computed internally, or provide a pre-computed ``atr``
        value together with the current ``price``.

        Args:
            high:   1-D array of high prices.
            low:    1-D array of low prices.
            close:  1-D array of close prices.
            atr:    Pre-computed ATR value (overrides OHLC computation).
            price:  Current price (required if ``atr`` is given; otherwise
                    inferred as ``close[-1]``).

        Returns:
            Bet fraction in [0, f_max].
        """
        if atr is None:
            if high is None or low is None or close is None:
                raise ValueError(
                    "Must provide (high, low, close) arrays or (atr, price)."
                )
            high = np.asarray(high, dtype=np.float64)
            low = np.asarray(low, dtype=np.float64)
            close = np.asarray(close, dtype=np.float64)
            atr = self.compute_atr(high, low, close, period=self.atr_period)
            price = float(close[-1])
        else:
            if price is None:
                raise ValueError("`price` is required when `atr` is provided.")

        if price <= 0 or atr <= 0:
            return 0.0

        atr_frac = atr / price
        if atr_frac <= 0:
            return 0.0

        f = self.risk_target / atr_frac
        return float(np.clip(f, 0.0, self.f_max))
