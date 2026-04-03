"""
LSTM signal model for next-day return prediction (Section 4.2).

A 2-layer LSTM encodes a lookback window of daily features and produces two
outputs:

- **direction** -- sign of next-day return in {-1, 0, +1} (3-class softmax).
- **magnitude** -- absolute magnitude of next-day return (scalar, ReLU output).

Out-of-sample directional accuracy targets (Table 2):
    Equity  54.2 %
    Crypto  52.8 %
    Forex   53.5 %
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SignalModel(nn.Module):
    """2-layer LSTM signal model (Section 4.2).

    Architecture::

        LSTM(n_features, hidden=128, layers=2, dropout=0.2)
            -> last hidden state h[-1]  (batch, 128)
        Direction head: Linear(128, 3) -> softmax  -> {-1, 0, +1}
        Magnitude head: Linear(128, 1) -> ReLU     -> |r_{t+1}|

    Args:
        n_features: Number of input features per time step (5 OHLCV +
                     15 technicals). Default 20.
        hidden_size: LSTM hidden dimension. Default 128.
        n_layers:    Number of stacked LSTM layers. Default 2.
        dropout:     Dropout probability applied between LSTM layers and
                     before the output heads. Default 0.2.
    """

    # Direction class indices -> signed labels
    DIRECTION_MAP = {0: -1, 1: 0, 2: 1}

    def __init__(
        self,
        n_features: int = 20,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout

        # --- Encoder --------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # --- Heads -----------------------------------------------------------
        self.dropout = nn.Dropout(p=dropout)

        # Direction head: 3 classes -> {-1, 0, +1}
        self.direction_head = nn.Linear(hidden_size, 3)

        # Magnitude head: scalar >= 0
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, n_features) daily market feature sequences.

        Returns:
            direction_logits: (batch, 3) raw logits for {-1, 0, +1}.
            magnitude:        (batch, 1) predicted absolute return magnitude.
        """
        # LSTM encodes the full sequence; we use the last time-step output.
        lstm_out, _ = self.lstm(x)              # (B, T, hidden)
        h_last = lstm_out[:, -1, :]             # (B, hidden)
        h_last = self.dropout(h_last)

        direction_logits = self.direction_head(h_last)   # (B, 3)
        magnitude = self.magnitude_head(h_last)          # (B, 1)

        return direction_logits, magnitude

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience inference method.

        Args:
            x: (batch, seq_len, n_features).

        Returns:
            direction: (batch,) signed direction in {-1, 0, +1}.
            magnitude: (batch,) predicted absolute return.
        """
        self.eval()
        logits, mag = self.forward(x)
        dir_idx = logits.argmax(dim=-1)                      # (B,)
        # Map class index {0,1,2} -> {-1,0,+1}
        direction = dir_idx - 1                               # (B,)
        return direction, mag.squeeze(-1)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_signal_model(
    model: SignalModel,
    train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    val_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    """Train the LSTM signal model (Section 4.2).

    The loss is a weighted combination of:
        L = L_dir (CrossEntropy on direction) + L_mag (MSE on magnitude)

    Args:
        model:      A :class:`SignalModel` instance.
        train_data: Tuple of (X, y_dir, y_mag) tensors where
                    X has shape (N, seq_len, n_features),
                    y_dir has shape (N,) with class labels in {0, 1, 2}
                        mapping to direction {-1, 0, +1},
                    y_mag has shape (N,) with absolute return magnitudes.
        epochs:     Number of training epochs. Default 50.
        lr:         Learning rate for Adam optimiser. Default 1e-3.
        batch_size: Mini-batch size. Default 64.
        val_data:   Optional validation tuple with same format as train_data.
        device:     Torch device string or object. Default ``"cpu"``.

    Returns:
        Dictionary with training history::

            {
                "train_loss": [float, ...],
                "val_loss":   [float, ...],   # empty if val_data is None
                "val_dir_acc": [float, ...],  # directional accuracy
            }
    """
    model = model.to(device)
    model.train()

    X_train, y_dir_train, y_mag_train = (t.to(device) for t in train_data)
    dataset = TensorDataset(X_train, y_dir_train, y_mag_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion_dir = nn.CrossEntropyLoss()
    criterion_mag = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: dict = {"train_loss": [], "val_loss": [], "val_dir_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_b, yd_b, ym_b in loader:
            optimizer.zero_grad()
            logits, mag_pred = model(X_b)

            loss_dir = criterion_dir(logits, yd_b.long())
            loss_mag = criterion_mag(mag_pred.squeeze(-1), ym_b.float())
            loss = loss_dir + loss_mag

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # ----- Validation ---------------------------------------------------
        if val_data is not None:
            model.eval()
            X_val, yd_val, ym_val = (t.to(device) for t in val_data)
            with torch.no_grad():
                logits_v, mag_v = model(X_val)
                vloss_dir = criterion_dir(logits_v, yd_val.long())
                vloss_mag = criterion_mag(mag_v.squeeze(-1), ym_val.float())
                vloss = (vloss_dir + vloss_mag).item()

                pred_dir = logits_v.argmax(dim=-1)
                acc = (pred_dir == yd_val.long()).float().mean().item()

            history["val_loss"].append(vloss)
            history["val_dir_acc"].append(acc)

    return history
