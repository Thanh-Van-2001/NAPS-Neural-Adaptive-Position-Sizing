"""
Kelly Criterion position sizers (Section 2.1, 4.3).

Implements both Full Kelly and Half Kelly sizing using rolling-window
estimates of win rate (p) and payoff ratio (b = avg_win / avg_loss).

The optimal Kelly fraction is:

    f* = (b * p - q) / b

where q = 1 - p.  In practice the full Kelly fraction is too aggressive,
so Half Kelly (0.5 * f*) is the more common choice.

Rolling-window estimation uses the most recent ``window`` trade returns
to compute p and b adaptively.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class FullKellySizer:
    """Full Kelly Criterion sizer with rolling parameter estimation.

    Uses a rolling window of recent trade returns to estimate the win
    rate *p* and the payoff ratio *b*, then computes the Kelly fraction:

        f* = (b * p - q) / b

    Args:
        window:  Rolling lookback window (number of past trades / returns)
                 for estimating p and b. Default 60.
        f_max:   Maximum allowed bet fraction. Default 1.0.
    """

    def __init__(self, window: int = 60, f_max: float = 1.0) -> None:
        self.window = window
        self.f_max = f_max

    def _estimate_params(
        self, returns: NDArray[np.floating]
    ) -> tuple[float, float]:
        """Estimate win rate and payoff ratio from recent returns.

        Args:
            returns: 1-D array of recent trade returns (most recent last).

        Returns:
            (p, b) -- win rate and average-win / average-loss ratio.
        """
        recent = returns[-self.window:]

        wins = recent[recent > 0]
        losses = recent[recent < 0]

        p = len(wins) / len(recent) if len(recent) > 0 else 0.0

        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 1.0

        b = avg_win / avg_loss if avg_loss > 0 else 0.0

        return p, b

    def compute_size(
        self,
        returns: NDArray[np.floating] | None = None,
        p: float | None = None,
        b: float | None = None,
        **kwargs,
    ) -> float:
        """Compute the Full Kelly bet fraction.

        Either provide ``returns`` (a 1-D array of past trade returns) to
        let the sizer estimate p and b from a rolling window, or supply
        ``p`` and ``b`` directly.

        Args:
            returns: 1-D array of recent trade returns.
            p:       Win probability (override rolling estimate).
            b:       Payoff ratio avg_win / avg_loss (override).

        Returns:
            Bet fraction in [0, f_max].
        """
        if p is None or b is None:
            if returns is None:
                raise ValueError(
                    "Must provide either `returns` or both `p` and `b`."
                )
            returns = np.asarray(returns, dtype=np.float64)
            if len(returns) < 2:
                return 0.0
            p, b = self._estimate_params(returns)

        q = 1.0 - p

        if b <= 0.0:
            return 0.0

        f_star = (b * p - q) / b

        # Kelly can be negative (don't bet) or > 1 (leverage).
        return float(np.clip(f_star, 0.0, self.f_max))


class HalfKellySizer:
    """Half Kelly sizer -- 0.5 * f* (Section 4.3).

    A more conservative variant that halves the full Kelly fraction,
    reducing variance at the cost of slightly lower geometric growth.

    Args:
        window:  Rolling lookback window for parameter estimation. Default 60.
        f_max:   Maximum allowed bet fraction. Default 1.0.
    """

    def __init__(self, window: int = 60, f_max: float = 1.0) -> None:
        self._full_kelly = FullKellySizer(window=window, f_max=f_max)
        self.f_max = f_max

    def compute_size(
        self,
        returns: NDArray[np.floating] | None = None,
        p: float | None = None,
        b: float | None = None,
        **kwargs,
    ) -> float:
        """Compute the Half Kelly bet fraction (0.5 * f*).

        Accepts the same arguments as :meth:`FullKellySizer.compute_size`.

        Returns:
            Bet fraction in [0, f_max], equal to half the full Kelly value.
        """
        full = self._full_kelly.compute_size(returns=returns, p=p, b=b)
        half = 0.5 * full
        return float(np.clip(half, 0.0, self.f_max))
