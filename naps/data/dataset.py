"""
PyTorch Dataset and walk-forward splitting for NAPS (Section 4.1).

Provides:
    NAPSDataset                - Indexed dataset returning six-tuple tensors.
    create_walk_forward_splits - Anchored walk-forward train/test index lists.

Each sample corresponds to one trading day *t* and contains:
    market_history  (L, 20)  -- past L days of OHLCV + technical indicators
    signal          (2,)     -- predicted direction and magnitude
    uncertainty     (1,)     -- MC-dropout standard deviation of the signal
    risk_state      (5,)     -- portfolio state at time t
    regime          (3,)     -- regime features at time t
    returns         (1,)     -- realised next-day return (used for loss only)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NAPSDataset(Dataset):
    """PyTorch Dataset for the NAPS position-sizing model.

    Args:
        market_features: (N, 20) array of daily market features (OHLCV +
            15 technical indicators).
        signals: (N, 2) array of signal predictions (direction, magnitude).
        uncertainties: (N,) or (N, 1) array of MC-dropout uncertainty.
        risk_states: (N, 5) array of portfolio-state vectors.
        regimes: (N, 3) array of regime features.
        returns: (N,) or (N, 1) array of realised next-day returns.
        lookback: Number of past days forming each context window. Default 60.

    The dataset is valid for indices ``lookback`` through ``N - 1`` (inclusive),
    so ``len(dataset) == N - lookback``.
    """

    def __init__(
        self,
        market_features: np.ndarray | pd.DataFrame,
        signals: np.ndarray | pd.DataFrame,
        uncertainties: np.ndarray | pd.Series,
        risk_states: np.ndarray | pd.DataFrame,
        regimes: np.ndarray | pd.DataFrame,
        returns: np.ndarray | pd.Series,
        lookback: int = 60,
    ) -> None:
        # Convert everything to float32 tensors
        self.market_features = _to_tensor(market_features)   # (N, 20)
        self.signals = _to_tensor(signals)                    # (N, 2)
        self.uncertainties = _to_tensor(uncertainties)         # (N,) or (N, 1)
        self.risk_states = _to_tensor(risk_states)            # (N, 5)
        self.regimes = _to_tensor(regimes)                    # (N, 3)
        self.returns = _to_tensor(returns)                    # (N,) or (N, 1)

        # Ensure trailing singleton dims
        if self.uncertainties.dim() == 1:
            self.uncertainties = self.uncertainties.unsqueeze(-1)
        if self.returns.dim() == 1:
            self.returns = self.returns.unsqueeze(-1)

        self.lookback = lookback
        self._n_total = self.market_features.shape[0]

        if self._n_total <= lookback:
            raise ValueError(
                f"Dataset has {self._n_total} rows but lookback={lookback}; "
                f"need at least {lookback + 1} rows."
            )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n_total - self.lookback

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a single training sample.

        Returns:
            Tuple of:
                market_history  (L, 20)
                signal          (2,)
                uncertainty     (1,)
                risk_state      (5,)
                regime          (3,)
                returns         (1,)
        """
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        t = idx + self.lookback  # absolute position in the arrays

        market_history = self.market_features[t - self.lookback : t]  # (L, 20)
        signal = self.signals[t]                                      # (2,)
        uncertainty = self.uncertainties[t]                            # (1,)
        risk_state = self.risk_states[t]                              # (5,)
        regime = self.regimes[t]                                      # (3,)
        ret = self.returns[t]                                         # (1,)

        return market_history, signal, uncertainty, risk_state, regime, ret


# ---------------------------------------------------------------------------
# Walk-forward splitting (Section 4.1)
# ---------------------------------------------------------------------------

def create_walk_forward_splits(
    dataset: NAPSDataset,
    initial_train: int = 252,
    step: int = 63,
    test: int = 63,
) -> List[Tuple[List[int], List[int]]]:
    """Generate anchored walk-forward train/test index splits.

    The scheme mirrors Section 4.1:
        - **Initial training window**: first *initial_train* samples.
        - **Test window**: next *test* samples.
        - Each subsequent fold advances the test window by *step* days while
          the training window grows (anchored start).

    ::

        Fold 0:  train [0 .. initial_train)   test [initial_train .. initial_train+test)
        Fold 1:  train [0 .. initial_train+step)  test [initial_train+step .. initial_train+step+test)
        ...

    Args:
        dataset: A :class:`NAPSDataset` instance (used only for ``len``).
        initial_train: Number of samples in the first training fold.
            Default 252 (one trading year).
        step: Number of samples the window advances per fold.
            Default 63 (one quarter).
        test: Number of samples in each test fold.
            Default 63 (one quarter).

    Returns:
        List of ``(train_indices, test_indices)`` tuples.  Each element is a
        plain Python list of integer indices valid for ``dataset[i]``.
    """
    n = len(dataset)
    splits: List[Tuple[List[int], List[int]]] = []

    fold = 0
    while True:
        train_end = initial_train + fold * step
        test_start = train_end
        test_end = test_start + test

        if test_end > n:
            break

        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))
        splits.append((train_idx, test_idx))
        fold += 1

    if not splits:
        raise ValueError(
            f"Cannot create any walk-forward folds: dataset length={n}, "
            f"initial_train={initial_train}, test={test}."
        )

    return splits


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_tensor(x: np.ndarray | pd.DataFrame | pd.Series | torch.Tensor) -> torch.Tensor:
    """Convert various array-like inputs to a float32 torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values
    return torch.tensor(np.asarray(x, dtype=np.float32))
