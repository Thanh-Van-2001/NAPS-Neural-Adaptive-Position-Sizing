"""
Cross-attention fusion module (Section 3.2).

The key design choice is that the *risk state* serves as the query while
market context and signal embeddings serve as keys/values.  This forces the
model to attend to market information *conditioned on* the current risk
posture, naturally implementing adaptive risk-awareness.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Fuse encoder representations via multi-head cross-attention.

    Architecture (Section 3.2, Eq. 3):
        Q = h_risk                          # (B, 1, d_model)
        K = V = [h_temp ; h_sig]            # (B, 2, d_model)
        h_fused = MHA(Q, K, V) + h_risk     # residual connection
        output  = h_fused.squeeze(1)        # (B, d_model)

    The residual connection ensures that the risk signal is never completely
    overridden by market/signal information, preserving a strong risk-aware
    baseline in the fused representation.

    Args:
        d_model: Embedding dimension. Default 64.
        n_heads: Number of attention heads. Default 4.
        dropout: Dropout on attention weights.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h_temp: torch.Tensor,
        h_sig: torch.Tensor,
        h_risk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_temp: (batch, d_model) temporal market encoding.
            h_sig:  (batch, d_model) signal+uncertainty encoding.
            h_risk: (batch, d_model) risk state encoding.

        Returns:
            h_fused: (batch, d_model) fused representation.
        """
        # Reshape to sequence dimension for attention:
        #   Q: risk  -> (B, 1, d)
        #   K/V: concat market + signal -> (B, 2, d)
        query = h_risk.unsqueeze(1)                              # (B, 1, d)
        kv = torch.stack([h_temp, h_sig], dim=1)                 # (B, 2, d)

        attn_out, _ = self.cross_attn(query, kv, kv)             # (B, 1, d)

        # Residual connection + layer norm (Section 3.2)
        h_fused = self.layer_norm(attn_out + query)              # (B, 1, d)
        return h_fused.squeeze(1)                                # (B, d)
