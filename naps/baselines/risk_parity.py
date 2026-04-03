"""
Risk Parity position sizer (Section 4.3).

Sizes positions inversely proportional to realised volatility so that
each instrument contributes equally to total portfolio risk:

    f_i = target_risk / realized_vol_i

The raw fractions are then normalised so that the portfolio-level risk
matches the target (assuming uncorrelated positions as a first
approximation, or a single-asset simplification).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class RiskParitySizer:
    """Inverse-volatility (risk parity) position sizer.

    For a single asset the bet fraction is:

        f = target_risk / realized_vol_20d

    When sizing across multiple assets simultaneously, the raw fractions
    are scaled so that the sum of (f_i * vol_i) equals the portfolio-level
    target risk.

    Args:
        vol_window:   Lookback window (trading days) for annualised
                      realised volatility. Default 20.
        target_risk:  Target annualised volatility contribution per
                      position. Default 0.10 (10 %).
        f_max:        Maximum allowed bet fraction per asset. Default 1.0.
        annualise:    If True (default), scale daily vol by sqrt(252).
    """

    def __init__(
        self,
        vol_window: int = 20,
        target_risk: float = 0.10,
        f_max: float = 1.0,
        annualise: bool = True,
    ) -> None:
        self.vol_window = vol_window
        self.target_risk = target_risk
        self.f_max = f_max
        self.annualise = annualise

    def _realised_vol(self, returns: NDArray[np.floating]) -> float:
        """Compute realised volatility from a 1-D return series.

        Args:
            returns: 1-D array of (log or simple) daily returns.

        Returns:
            Annualised realised volatility (scalar).
        """
        recent = np.asarray(returns, dtype=np.float64)[-self.vol_window:]
        if len(recent) < 2:
            return 0.0
        vol = float(np.std(recent, ddof=1))
        if self.annualise:
            vol *= np.sqrt(252)
        return vol

    def compute_size(
        self,
        returns: NDArray[np.floating] | None = None,
        realized_vol: float | None = None,
        **kwargs,
    ) -> float:
        """Compute the risk-parity bet fraction for a single asset.

        Either supply a ``returns`` array (daily returns) to have
        realised volatility estimated internally, or provide a
        pre-computed ``realized_vol``.

        Args:
            returns:       1-D array of daily returns.
            realized_vol:  Pre-computed annualised realised volatility
                           (overrides ``returns``-based computation).

        Returns:
            Bet fraction in [0, f_max].
        """
        if realized_vol is None:
            if returns is None:
                raise ValueError(
                    "Must provide either `returns` or `realized_vol`."
                )
            realized_vol = self._realised_vol(returns)

        if realized_vol <= 0:
            return 0.0

        f = self.target_risk / realized_vol
        return float(np.clip(f, 0.0, self.f_max))

    def compute_sizes_multi(
        self,
        returns_dict: dict[str, NDArray[np.floating]],
    ) -> dict[str, float]:
        """Compute normalised risk-parity fractions for multiple assets.

        Raw inverse-vol weights are rescaled so the portfolio's total risk
        contribution (sum of f_i * vol_i) equals ``target_risk``.  Each
        individual fraction is still capped at ``f_max``.

        Args:
            returns_dict: Mapping from asset name to 1-D return arrays.

        Returns:
            Dict mapping asset name to bet fraction in [0, f_max].
        """
        vols: dict[str, float] = {}
        for name, rets in returns_dict.items():
            vols[name] = self._realised_vol(rets)

        # Inverse-vol raw weights.
        inv_vols = {
            name: (1.0 / v if v > 0 else 0.0) for name, v in vols.items()
        }
        total_inv = sum(inv_vols.values())
        if total_inv <= 0:
            return {name: 0.0 for name in returns_dict}

        # Normalise so sum(f_i * vol_i) = target_risk.
        result: dict[str, float] = {}
        for name in returns_dict:
            w_i = inv_vols[name] / total_inv         # portfolio weight
            vol_i = vols[name]
            if vol_i > 0:
                f_i = self.target_risk * w_i / vol_i
            else:
                f_i = 0.0
            result[name] = float(np.clip(f_i, 0.0, self.f_max))

        return result
