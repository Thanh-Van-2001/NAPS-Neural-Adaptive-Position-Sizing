"""
NAPS signal model components (Section 4.2).

Provides the LSTM-based signal model for predicting next-day return direction
and magnitude, along with MC Dropout uncertainty estimation.
"""

from naps.signal.lstm_signal import SignalModel, train_signal_model
from naps.signal.mc_dropout import MCDropoutEstimator

__all__ = [
    "SignalModel",
    "train_signal_model",
    "MCDropoutEstimator",
]
