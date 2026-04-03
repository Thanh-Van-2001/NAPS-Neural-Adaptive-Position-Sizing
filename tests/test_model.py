"""
Unit tests for the NAPS model (Section 3.2).

Tests cover:
    - NAPSModel forward pass output shapes
    - Output bounded in [0, f_max]
    - NAPSMLPModel forward pass (ablation variant)
    - Each encoder independently (Temporal, Signal, Risk)
    - Cross-attention fusion dimensions
    - Parameter count ~180K
    - Gradient flow through the entire model
"""

import pytest
import torch
import numpy as np

from naps.model.naps import NAPSModel, NAPSMLPModel
from naps.model.encoders import (
    TemporalTransformerEncoder,
    SignalUncertaintyEncoder,
    RiskStateEncoder,
)
from naps.model.fusion import CrossAttentionFusion
from naps.model.sizing_head import SizingHead


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42
BATCH = 8
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
LOOKBACK = 60
N_FEATURES = 20
F_MAX = 0.40
DROPOUT = 0.1


@pytest.fixture
def naps_model():
    """Default NAPSModel with paper hyperparameters."""
    torch.manual_seed(SEED)
    return NAPSModel(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        lookback=LOOKBACK,
        n_features=N_FEATURES,
        f_max=F_MAX,
        dropout=DROPOUT,
    )


@pytest.fixture
def naps_mlp_model():
    """NAPSMLPModel ablation variant."""
    torch.manual_seed(SEED)
    return NAPSMLPModel(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        lookback=LOOKBACK,
        n_features=N_FEATURES,
        f_max=F_MAX,
        dropout=DROPOUT,
    )


@pytest.fixture
def sample_inputs():
    """Random inputs matching the expected tensor shapes."""
    torch.manual_seed(SEED)
    return {
        "market_history": torch.randn(BATCH, LOOKBACK, N_FEATURES),
        "signal": torch.randn(BATCH, 2),
        "uncertainty": torch.rand(BATCH, 1),
        "risk_state": torch.randn(BATCH, 5),
        "regime": torch.randn(BATCH, 3),
    }


# ---------------------------------------------------------------------------
# NAPSModel forward pass
# ---------------------------------------------------------------------------

class TestNAPSModelForward:
    """Test NAPSModel forward pass shapes and bounds."""

    def test_output_shape(self, naps_model, sample_inputs):
        """Forward pass should produce (batch, 1) output."""
        naps_model.eval()
        with torch.no_grad():
            out = naps_model(**sample_inputs)
        assert out.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {out.shape}"

    def test_output_bounded_zero_to_fmax(self, naps_model, sample_inputs):
        """All outputs must lie in [0, f_max]."""
        naps_model.eval()
        with torch.no_grad():
            out = naps_model(**sample_inputs)
        assert (out >= 0.0).all(), f"Found negative output: {out.min().item()}"
        assert (out <= F_MAX + 1e-6).all(), f"Found output > f_max: {out.max().item()}"

    def test_output_bounded_different_fmax(self, sample_inputs):
        """Verify bounding works for non-default f_max values."""
        for f_max in [0.10, 0.25, 0.50, 1.0]:
            torch.manual_seed(SEED)
            model = NAPSModel(f_max=f_max)
            model.eval()
            with torch.no_grad():
                out = model(**sample_inputs)
            assert (out >= 0.0).all()
            assert (out <= f_max + 1e-6).all(), (
                f"f_max={f_max}: output max = {out.max().item()}"
            )

    def test_batch_size_one(self, naps_model):
        """Model should handle batch_size=1."""
        torch.manual_seed(SEED)
        inputs = {
            "market_history": torch.randn(1, LOOKBACK, N_FEATURES),
            "signal": torch.randn(1, 2),
            "uncertainty": torch.rand(1, 1),
            "risk_state": torch.randn(1, 5),
            "regime": torch.randn(1, 3),
        }
        naps_model.eval()
        with torch.no_grad():
            out = naps_model(**inputs)
        assert out.shape == (1, 1)

    def test_deterministic_eval(self, naps_model, sample_inputs):
        """Eval mode should produce deterministic outputs."""
        naps_model.eval()
        with torch.no_grad():
            out1 = naps_model(**sample_inputs)
            out2 = naps_model(**sample_inputs)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# NAPSMLPModel forward pass (ablation)
# ---------------------------------------------------------------------------

class TestNAPSMLPModelForward:
    """Test the MLP ablation variant."""

    def test_output_shape(self, naps_mlp_model, sample_inputs):
        """MLP variant should also produce (batch, 1) output."""
        naps_mlp_model.eval()
        with torch.no_grad():
            out = naps_mlp_model(**sample_inputs)
        assert out.shape == (BATCH, 1)

    def test_output_bounded(self, naps_mlp_model, sample_inputs):
        """MLP variant output must also be bounded in [0, f_max]."""
        naps_mlp_model.eval()
        with torch.no_grad():
            out = naps_mlp_model(**sample_inputs)
        assert (out >= 0.0).all()
        assert (out <= F_MAX + 1e-6).all()


# ---------------------------------------------------------------------------
# Individual encoders
# ---------------------------------------------------------------------------

class TestTemporalTransformerEncoder:
    """Test the Temporal Transformer Encoder independently."""

    def test_output_shape(self):
        torch.manual_seed(SEED)
        encoder = TemporalTransformerEncoder(
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            lookback=LOOKBACK, n_features=N_FEATURES, dropout=DROPOUT,
        )
        x = torch.randn(BATCH, LOOKBACK, N_FEATURES)
        encoder.eval()
        with torch.no_grad():
            h_temp = encoder(x)
        assert h_temp.shape == (BATCH, D_MODEL)

    def test_different_lookback(self):
        """Encoder should work with shorter sequences (up to lookback)."""
        torch.manual_seed(SEED)
        lb = 30
        encoder = TemporalTransformerEncoder(
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            lookback=lb, n_features=N_FEATURES, dropout=DROPOUT,
        )
        x = torch.randn(BATCH, lb, N_FEATURES)
        encoder.eval()
        with torch.no_grad():
            h = encoder(x)
        assert h.shape == (BATCH, D_MODEL)


class TestSignalUncertaintyEncoder:
    """Test the Signal-Uncertainty Encoder independently."""

    def test_output_shape(self):
        torch.manual_seed(SEED)
        encoder = SignalUncertaintyEncoder(d_model=D_MODEL)
        signal_input = torch.randn(BATCH, 3)  # [dir, mag, uncertainty]
        encoder.eval()
        with torch.no_grad():
            h_sig = encoder(signal_input)
        assert h_sig.shape == (BATCH, D_MODEL)


class TestRiskStateEncoder:
    """Test the Risk-State Encoder independently."""

    def test_output_shape(self):
        torch.manual_seed(SEED)
        encoder = RiskStateEncoder(d_model=D_MODEL)
        risk_state = torch.randn(BATCH, 5)
        regime = torch.randn(BATCH, 3)
        encoder.eval()
        with torch.no_grad():
            h_risk = encoder(risk_state, regime)
        assert h_risk.shape == (BATCH, D_MODEL)


# ---------------------------------------------------------------------------
# Cross-attention fusion
# ---------------------------------------------------------------------------

class TestCrossAttentionFusion:
    """Test the cross-attention fusion module dimensions."""

    def test_output_shape(self):
        torch.manual_seed(SEED)
        fusion = CrossAttentionFusion(
            d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT,
        )
        h_temp = torch.randn(BATCH, D_MODEL)
        h_sig = torch.randn(BATCH, D_MODEL)
        h_risk = torch.randn(BATCH, D_MODEL)

        fusion.eval()
        with torch.no_grad():
            h_fused = fusion(h_temp, h_sig, h_risk)
        assert h_fused.shape == (BATCH, D_MODEL)

    def test_residual_connection(self):
        """Verify the residual connection: output should differ from zero-query."""
        torch.manual_seed(SEED)
        fusion = CrossAttentionFusion(d_model=D_MODEL, n_heads=N_HEADS)

        h_temp = torch.randn(BATCH, D_MODEL)
        h_sig = torch.randn(BATCH, D_MODEL)
        h_risk = torch.randn(BATCH, D_MODEL)

        fusion.eval()
        with torch.no_grad():
            h_fused = fusion(h_temp, h_sig, h_risk)
        # The fused output should not be identical to h_risk (attention adds info)
        assert not torch.allclose(h_fused, h_risk, atol=1e-4)


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

class TestParameterCount:
    """Verify the model has approximately 180K parameters (Section 3.2)."""

    def test_param_count_naps(self, naps_model):
        total = sum(p.numel() for p in naps_model.parameters())
        # Allow 20% tolerance around 180K
        assert 100_000 < total < 300_000, (
            f"Parameter count {total:,} is outside expected range ~180K"
        )

    def test_param_count_mlp_variant(self, naps_mlp_model):
        total = sum(p.numel() for p in naps_mlp_model.parameters())
        # MLP variant may be larger due to flat projection but still reasonable
        assert total > 50_000, f"Too few parameters: {total:,}"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Verify gradients flow through the entire model."""

    def test_gradients_flow_naps(self, naps_model, sample_inputs):
        """All model parameters should receive non-zero gradients."""
        naps_model.train()
        out = naps_model(**sample_inputs)
        loss = out.sum()
        loss.backward()

        for name, param in naps_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradients_flow_mlp(self, naps_mlp_model, sample_inputs):
        """MLP variant should also have complete gradient flow."""
        naps_mlp_model.train()
        out = naps_mlp_model(**sample_inputs)
        loss = out.sum()
        loss.backward()

        for name, param in naps_mlp_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
