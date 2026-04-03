"""
NAPS training pipeline (Sections 3.3 -- 3.4).

Provides:
    NAPSLoss            - Composite loss: differentiable Sharpe + drawdown penalty
                          + turnover regularisation (Section 3.3).
    WalkForwardTrainer  - Expanding-window walk-forward trainer with signal-model
                          isolation (Section 3.4).
    compute_metrics     - Full performance metric suite matching Table 2.
"""

from naps.training.objective import NAPSLoss
from naps.training.trainer import WalkForwardTrainer
from naps.training.metrics import compute_metrics

__all__ = [
    "NAPSLoss",
    "WalkForwardTrainer",
    "compute_metrics",
]
