"""
NAPS: Neural Adaptive Position Sizing

PyTorch implementation of the Neural Adaptive Position Sizing model
from "Neural Adaptive Position Sizing for Systematic Trading" (NeurIPS 2026).

The NAPS model learns to dynamically size positions by fusing temporal market
context, signal uncertainty, and portfolio risk state through cross-attention,
producing bounded bet fractions suitable for real-world deployment.
"""

__version__ = "1.0.0"

from naps.model.naps import NAPSModel, NAPSMLPModel

__all__ = ["NAPSModel", "NAPSMLPModel", "__version__"]
