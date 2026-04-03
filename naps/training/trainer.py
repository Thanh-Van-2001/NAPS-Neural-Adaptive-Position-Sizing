"""
Walk-forward trainer for the NAPS model (Section 3.4).

Implements an **expanding-window** walk-forward validation scheme:

    |<--- initial_train (252d) --->|<-- test (63d) -->|
    |<------ initial_train + step_size (315d) ------>|<-- test -->|
    ...

Key design decisions:
    - The training window **expands** (not rolls) so the model can leverage
      all available historical data as time progresses.
    - The signal model is trained on its own preceding window and then
      **frozen** during NAPS training to prevent information leakage.
    - All features are computed using only past data (strict no-lookahead).
    - Adam optimiser with cosine annealing from 1e-4 to 1e-6.
    - Each walk-forward step trains for 100 epochs with batch size 32.
    - Out-of-sample test quarters are concatenated and returned for
      downstream evaluation.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from naps.training.objective import NAPSLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    """Hyper-parameters for the walk-forward procedure."""

    initial_train: int = 252     # days in the first training window (1 year)
    step_size: int = 63          # quarterly expansion / advance
    test_size: int = 63          # out-of-sample test period (1 quarter)
    epochs: int = 100            # training epochs per step
    batch_size: int = 32
    lr: float = 1e-4             # peak learning rate (Adam)
    lr_min: float = 1e-6         # minimum learning rate (cosine anneal)
    max_grad_norm: float = 1.0   # gradient clipping
    device: str = "cpu"

    # NAPSLoss hyper-parameters
    loss_W: int = 252
    loss_tau: float = 0.15
    loss_lambda1: float = 5.0
    loss_lambda2: float = 1.0
    loss_cost: float = 0.001
    loss_eta: float = 0.01


# ---------------------------------------------------------------------------
# Walk-forward result container
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """Accumulated results from all walk-forward folds."""

    oos_returns: List[torch.Tensor] = field(default_factory=list)
    oos_fractions: List[torch.Tensor] = field(default_factory=list)
    fold_train_losses: List[List[float]] = field(default_factory=list)
    fold_val_losses: List[List[float]] = field(default_factory=list)

    def concatenated_returns(self) -> torch.Tensor:
        """Concatenate all OOS returns into a single 1-D tensor."""
        return torch.cat(self.oos_returns, dim=0)

    def concatenated_fractions(self) -> torch.Tensor:
        """Concatenate all OOS fractions into a single 1-D tensor."""
        return torch.cat(self.oos_fractions, dim=0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class WalkForwardTrainer:
    """Expanding-window walk-forward trainer (Section 3.4).

    Usage::

        trainer = WalkForwardTrainer(config=WalkForwardConfig(device="cuda"))
        result  = trainer.walk_forward(model, full_data)
        oos_ret = result.concatenated_returns()

    Args:
        config: A :class:`WalkForwardConfig` instance.  If ``None``, default
                parameters from the paper are used.
    """

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    # ------------------------------------------------------------------
    # Single-step training
    # ------------------------------------------------------------------

    def train_one_step(
        self,
        model: nn.Module,
        train_data: Dict[str, torch.Tensor],
        val_data: Dict[str, torch.Tensor] | None = None,
        epochs: int | None = None,
    ) -> Dict[str, List[float]]:
        """Train the NAPS model for one walk-forward step.

        Args:
            model:      NAPS model (must expose a compatible ``forward``).
            train_data: Dictionary with keys matching the model's forward
                        signature plus ``"returns"`` for the objective:

                        - ``market_history`` : (N, L, n_features)
                        - ``signal``         : (N, 2)
                        - ``uncertainty``    : (N, 1)
                        - ``risk_state``     : (N, 5)
                        - ``regime``         : (N, 3)
                        - ``returns``        : (N,) asset returns r_t

            val_data:   Optional validation data with the same schema.
            epochs:     Override the config epoch count for this step.

        Returns:
            Dictionary ``{"train_loss": [...], "val_loss": [...]}``.
        """
        cfg = self.config
        device = torch.device(cfg.device)
        n_epochs = epochs if epochs is not None else cfg.epochs

        model = model.to(device)

        # ---- Build criterion --------------------------------------------------
        criterion = NAPSLoss(
            W=cfg.loss_W,
            tau=cfg.loss_tau,
            lambda1=cfg.loss_lambda1,
            lambda2=cfg.loss_lambda2,
            cost=cfg.loss_cost,
            eta=cfg.loss_eta,
        )

        # ---- Optimiser + scheduler -------------------------------------------
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=cfg.lr_min
        )

        # ---- Data loaders -----------------------------------------------------
        train_loader = self._make_loader(train_data, device, shuffle=True)
        val_loader = (
            self._make_loader(val_data, device, shuffle=False)
            if val_data is not None
            else None
        )

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        # ---- Training loop ----------------------------------------------------
        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()

                fractions, returns_batch = self._forward_batch(model, batch)
                loss = criterion(fractions, returns_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.max_grad_norm
                )
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train)

            # ---- Validation ---------------------------------------------------
            if val_loader is not None:
                val_loss = self._evaluate(model, val_loader, criterion)
                history["val_loss"].append(val_loss)

            if epoch % 20 == 0 or epoch == n_epochs:
                lr_now = scheduler.get_last_lr()[0]
                msg = (
                    f"  epoch {epoch:3d}/{n_epochs}  "
                    f"train_loss={avg_train:.6f}  lr={lr_now:.2e}"
                )
                if val_loader is not None:
                    msg += f"  val_loss={history['val_loss'][-1]:.6f}"
                logger.info(msg)

        return history

    # ------------------------------------------------------------------
    # Full walk-forward
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        model: nn.Module,
        full_data: Dict[str, torch.Tensor],
        initial_train: int | None = None,
        step_size: int | None = None,
        test_size: int | None = None,
    ) -> WalkForwardResult:
        """Run the complete expanding-window walk-forward procedure.

        Args:
            model:         Base NAPS model.  A fresh copy is created for each
                           fold to ensure no information leakage.
            full_data:     Dictionary of tensors with a common first dimension
                           (total trading days T).  Required keys:

                           - ``market_history`` : (T, L, n_features)
                           - ``signal``         : (T, 2)
                           - ``uncertainty``    : (T, 1)
                           - ``risk_state``     : (T, 5)
                           - ``regime``         : (T, 3)
                           - ``returns``        : (T,)

            initial_train: Override config initial training window size.
            step_size:     Override config step (expansion) size.
            test_size:     Override config test period size.

        Returns:
            A :class:`WalkForwardResult` with concatenated OOS returns and
            fractions from every fold.
        """
        cfg = self.config
        init_train = initial_train if initial_train is not None else cfg.initial_train
        step = step_size if step_size is not None else cfg.step_size
        test = test_size if test_size is not None else cfg.test_size

        T = self._total_length(full_data)
        result = WalkForwardResult()

        fold = 0
        train_end = init_train

        while train_end + test <= T:
            fold += 1
            test_start = train_end
            test_end = train_end + test

            logger.info(
                f"=== Fold {fold}: train [0:{train_end}]  "
                f"test [{test_start}:{test_end}] ==="
            )

            # Slice data for this fold.
            train_slice = self._slice_data(full_data, 0, train_end)
            test_slice = self._slice_data(full_data, test_start, test_end)

            # Create a fresh model copy for this fold (no leakage).
            fold_model = copy.deepcopy(model)

            # Train on the expanding window.
            history = self.train_one_step(fold_model, train_slice)
            result.fold_train_losses.append(history["train_loss"])
            result.fold_val_losses.append(history.get("val_loss", []))

            # Evaluate on OOS test period.
            oos_fractions, oos_returns = self._predict_oos(
                fold_model, test_slice
            )
            result.oos_returns.append(oos_returns.cpu())
            result.oos_fractions.append(oos_fractions.cpu())

            logger.info(
                f"  OOS days: {oos_returns.shape[0]}  "
                f"mean_return: {oos_returns.mean().item():.6f}  "
                f"mean_fraction: {oos_fractions.mean().item():.4f}"
            )

            # Expand training window by one quarter.
            train_end += step

        if fold == 0:
            raise ValueError(
                f"Not enough data for even one fold.  "
                f"Need at least {init_train + test} days, got {T}."
            )

        logger.info(
            f"Walk-forward complete: {fold} folds, "
            f"{result.concatenated_returns().shape[0]} total OOS days."
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        data: Dict[str, torch.Tensor],
        device: torch.device,
        shuffle: bool = True,
    ) -> DataLoader:
        """Build a DataLoader from a data dictionary.

        The loader yields tuples of
        ``(market_history, signal, uncertainty, risk_state, regime, returns)``
        all moved to *device*.
        """
        tensors = [
            data["market_history"].to(device),
            data["signal"].to(device),
            data["uncertainty"].to(device),
            data["risk_state"].to(device),
            data["regime"].to(device),
            data["returns"].to(device),
        ]
        dataset = TensorDataset(*tensors)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    @staticmethod
    def _forward_batch(
        model: nn.Module,
        batch: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the model on a single batch and return (fractions, returns).

        For the composite loss we need full sequences, but during mini-batch
        training each sample is an independent time step.  The loss function
        operates on (batch, T=1) tensors, treating each sample as a
        length-1 sequence within the batch.

        Returns:
            fractions: (B, 1)  -- bet fractions, kept 2-D for the loss.
            returns:   (B, 1)  -- asset returns, kept 2-D for the loss.
        """
        market_history, signal, uncertainty, risk_state, regime, returns = batch

        # Model returns (B, 1).
        fractions = model(
            market_history=market_history,
            signal=signal,
            uncertainty=uncertainty,
            risk_state=risk_state,
            regime=regime,
        )

        # Ensure both are (B, 1) for the loss.
        if fractions.dim() == 1:
            fractions = fractions.unsqueeze(-1)
        if returns.dim() == 1:
            returns = returns.unsqueeze(-1)

        return fractions, returns

    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: NAPSLoss,
    ) -> float:
        """Compute average loss on a validation loader."""
        model.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            fractions, returns_batch = self._forward_batch(model, batch)
            loss = criterion(fractions, returns_batch)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _predict_oos(
        self,
        model: nn.Module,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference on out-of-sample data and return (fractions, returns).

        Returns:
            fractions: (N,)  1-D tensor of bet fractions.
            returns:   (N,)  1-D tensor of asset returns (raw).
        """
        device = torch.device(self.config.device)
        model.eval()

        market_history = data["market_history"].to(device)
        signal = data["signal"].to(device)
        uncertainty = data["uncertainty"].to(device)
        risk_state = data["risk_state"].to(device)
        regime = data["regime"].to(device)
        raw_returns = data["returns"].to(device)

        fractions = model(
            market_history=market_history,
            signal=signal,
            uncertainty=uncertainty,
            risk_state=risk_state,
            regime=regime,
        ).squeeze(-1)  # (N,)

        # Portfolio returns = f_t * r_t (sizing applied to raw return).
        portfolio_returns = fractions * raw_returns

        return fractions, portfolio_returns

    @staticmethod
    def _slice_data(
        data: Dict[str, torch.Tensor],
        start: int,
        end: int,
    ) -> Dict[str, torch.Tensor]:
        """Slice all tensors in a data dictionary along the first dimension."""
        return {k: v[start:end] for k, v in data.items()}

    @staticmethod
    def _total_length(data: Dict[str, torch.Tensor]) -> int:
        """Return the length of the first dimension (assumed equal for all)."""
        first_key = next(iter(data))
        return data[first_key].shape[0]
