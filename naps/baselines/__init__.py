"""
Baseline position-sizing methods (Section 4.3).

Each sizer exposes a uniform ``compute_size(**kwargs) -> float`` interface
that returns a bet fraction in [0, f_max].  These serve as comparisons
for the learned NAPS sizing model.

Baselines implemented:
    - EqualWeightSizer   (fixed equal weight)
    - FullKellySizer     (full Kelly criterion)
    - HalfKellySizer     (half Kelly criterion)
    - ATRSizer           (inverse-ATR with risk target)
    - RiskParitySizer    (inverse-volatility risk parity)
"""

from naps.baselines.equal_weight import EqualWeightSizer
from naps.baselines.kelly import FullKellySizer, HalfKellySizer
from naps.baselines.atr import ATRSizer
from naps.baselines.risk_parity import RiskParitySizer

__all__ = [
    "EqualWeightSizer",
    "FullKellySizer",
    "HalfKellySizer",
    "ATRSizer",
    "RiskParitySizer",
]
