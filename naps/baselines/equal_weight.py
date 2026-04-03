"""
Fixed Equal-Weight position sizer (Section 4.3).

The simplest baseline: every trade receives the same fixed fraction of
equity regardless of signal strength, volatility, or portfolio state.
"""

from __future__ import annotations


class EqualWeightSizer:
    """Always return a fixed bet fraction.

    This is the naive baseline where every position is sized identically,
    providing a lower-bound benchmark for more sophisticated methods.

    Args:
        fraction: Fixed bet fraction of equity per trade. Default 0.10
                  (10 %).
        f_max:    Maximum allowed bet fraction. Default 1.0.
    """

    def __init__(self, fraction: float = 0.10, f_max: float = 1.0) -> None:
        if not 0.0 < fraction <= f_max:
            raise ValueError(
                f"fraction must be in (0, f_max], got {fraction} with f_max={f_max}"
            )
        self.fraction = fraction
        self.f_max = f_max

    def compute_size(self, **kwargs) -> float:
        """Return the fixed bet fraction.

        Any keyword arguments are accepted (and ignored) so the interface
        is compatible with other sizers.

        Returns:
            Bet fraction in [0, f_max], always equal to ``self.fraction``.
        """
        return min(self.fraction, self.f_max)
