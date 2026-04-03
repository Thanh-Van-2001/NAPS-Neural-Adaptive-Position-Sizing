"""
Data module for NAPS: feature engineering, datasets, and synthetic generation.

Provides:
    TechnicalIndicators   - 15 technical indicators from OHLCV (Section 3.2).
    RegimeFeatures        - 3 market-regime descriptors (Section 3.1).
    PortfolioState        - 5 portfolio-state variables (Section 3.1).
    NAPSDataset           - PyTorch Dataset with walk-forward support.
    SyntheticMarketGenerator - Realistic synthetic data for three asset classes.
"""

from naps.data.features import (
    TechnicalIndicators,
    RegimeFeatures,
    PortfolioState,
    compute_all_features,
    normalize_features,
)
from naps.data.dataset import NAPSDataset, create_walk_forward_splits
from naps.data.synthetic import SyntheticMarketGenerator

__all__ = [
    "TechnicalIndicators",
    "RegimeFeatures",
    "PortfolioState",
    "compute_all_features",
    "normalize_features",
    "NAPSDataset",
    "create_walk_forward_splits",
    "SyntheticMarketGenerator",
]
