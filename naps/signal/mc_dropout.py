"""
MC Dropout uncertainty estimation (Section 3.1, 4.2).

Implements the dropout-as-approximate-Bayesian-inference technique of
Gal & Ghahramani (2016) [21].  At inference time, dropout is kept active
and K stochastic forward passes are performed.  The standard deviation of
the resulting predictions serves as the calibrated uncertainty estimate:

    u_t = std({y_t^(k)}_{k=1..K})

This uncertainty feeds into the NAPS signal-uncertainty encoder and
allows the sizing model to scale positions inversely with prediction
confidence.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _enable_dropout(module: nn.Module) -> None:
    """Recursively switch all :class:`nn.Dropout` layers to train mode.

    This keeps the rest of the model in eval mode (e.g. BatchNorm stays
    deterministic) while activating dropout for stochastic forward passes.
    """
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.train()


class MCDropoutEstimator:
    """Monte Carlo Dropout uncertainty estimator.

    Performs K stochastic forward passes through a model with dropout
    enabled at inference time and returns the predictive standard deviation
    as a calibrated uncertainty measure.

    References:
        Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian
        Approximation: Representing Model Uncertainty in Deep Learning*.
        ICML 2016. [21]

    Args:
        K:         Number of stochastic forward passes. Default 20.
        dropout_p: Dropout probability used during MC sampling. If the
                   model already has the desired dropout rate baked in,
                   this can be left at its default; it is recorded for
                   documentation purposes. Default 0.2.
    """

    def __init__(self, K: int = 20, dropout_p: float = 0.2) -> None:
        self.K = K
        self.dropout_p = dropout_p

    @torch.no_grad()
    def estimate_uncertainty(
        self,
        model: nn.Module,
        x: torch.Tensor,
        K: int | None = None,
        dropout_p: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run K stochastic forward passes and compute predictive uncertainty.

        The model is first set to eval mode (so that layers like BatchNorm
        behave deterministically), then only the Dropout layers are
        re-enabled to train mode for stochastic sampling.

        Args:
            model:     A signal model whose ``forward`` returns
                       ``(direction_logits, magnitude)``.
            x:         Input tensor of shape ``(batch, seq_len, n_features)``.
            K:         Override the default number of forward passes.
            dropout_p: Documented only -- the model's own dropout rate is
                       used. Retained for API compatibility.

        Returns:
            A tuple of three tensors:

            - **u_t** -- ``(batch,)`` predictive uncertainty (std of magnitude
              predictions across K passes).
            - **mean_direction_probs** -- ``(batch, 3)`` averaged softmax
              probabilities over K passes.
            - **mean_magnitude** -- ``(batch,)`` mean predicted magnitude
              over K passes.
        """
        K = K if K is not None else self.K

        # Put the full model in eval, then selectively re-enable dropout.
        model.eval()
        _enable_dropout(model)

        mag_samples = []    # list of (batch,) tensors
        dir_samples = []    # list of (batch, 3) softmax tensors

        for _ in range(K):
            logits, mag = model(x)                         # forward pass
            dir_probs = torch.softmax(logits, dim=-1)      # (B, 3)
            dir_samples.append(dir_probs)
            mag_samples.append(mag.squeeze(-1))            # (B,)

        # Stack and compute statistics across the K dimension.
        mag_stack = torch.stack(mag_samples, dim=0)        # (K, B)
        dir_stack = torch.stack(dir_samples, dim=0)        # (K, B, 3)

        u_t = mag_stack.std(dim=0)                         # (B,)
        mean_magnitude = mag_stack.mean(dim=0)             # (B,)
        mean_direction_probs = dir_stack.mean(dim=0)       # (B, 3)

        # Restore full eval mode.
        model.eval()

        return u_t, mean_direction_probs, mean_magnitude
