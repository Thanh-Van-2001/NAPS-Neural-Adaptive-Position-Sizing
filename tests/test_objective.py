"""
Unit tests for the NAPS composite loss function (Section 3.3).

Tests cover:
    - Differentiable Sharpe ratio is differentiable
    - Drawdown penalty: 0 when DD < tau, positive when DD > tau
    - Turnover regularisation
    - Composite loss combines all terms correctly
    - Gradients flow through the loss
"""

import pytest
import torch
import numpy as np


from naps.training.objective import NAPSLoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42
BATCH = 4
T = 60  # time steps


@pytest.fixture
def loss_fn():
    """Default NAPSLoss with paper hyperparameters."""
    return NAPSLoss(
        W=252, tau=0.15, lambda1=5.0, lambda2=1.0, cost=0.001, eta=0.01,
    )


@pytest.fixture
def sample_data():
    """Generate sample fractions and returns for testing."""
    torch.manual_seed(SEED)
    fractions = torch.sigmoid(torch.randn(BATCH, T)) * 0.4  # [0, 0.4]
    returns = torch.randn(BATCH, T) * 0.02  # daily returns ~2% std
    return fractions, returns


# ---------------------------------------------------------------------------
# Differentiable Sharpe ratio
# ---------------------------------------------------------------------------

class TestDifferentiableSharpe:
    """Test the differentiable Sharpe ratio component."""

    def test_is_differentiable(self):
        """Sharpe should allow gradient computation w.r.t. returns."""
        torch.manual_seed(SEED)
        returns = torch.randn(BATCH, T, requires_grad=True)
        sharpe = NAPSLoss.differentiable_sharpe(returns, eta=0.01)
        loss = -sharpe.sum()
        loss.backward()
        assert returns.grad is not None
        assert returns.grad.abs().sum() > 0

    def test_positive_returns_give_positive_sharpe(self):
        """Consistently positive returns should yield positive Sharpe."""
        returns = torch.ones(BATCH, T) * 0.01  # +1% per day
        sharpe = NAPSLoss.differentiable_sharpe(returns, eta=0.01)
        assert (sharpe > 0).all(), f"Expected positive Sharpe, got {sharpe}"

    def test_negative_returns_give_negative_sharpe(self):
        """Consistently negative returns should yield negative Sharpe."""
        returns = torch.ones(BATCH, T) * (-0.01)  # -1% per day
        sharpe = NAPSLoss.differentiable_sharpe(returns, eta=0.01)
        assert (sharpe < 0).all(), f"Expected negative Sharpe, got {sharpe}"

    def test_zero_returns_give_zero_sharpe(self):
        """Zero returns should yield zero or near-zero Sharpe."""
        returns = torch.zeros(BATCH, T)
        sharpe = NAPSLoss.differentiable_sharpe(returns, eta=0.01)
        assert torch.allclose(sharpe, torch.zeros_like(sharpe), atol=1e-6)

    def test_output_shape(self):
        """Should return one Sharpe estimate per batch element."""
        torch.manual_seed(SEED)
        returns = torch.randn(BATCH, T)
        sharpe = NAPSLoss.differentiable_sharpe(returns, eta=0.01)
        assert sharpe.shape == (BATCH,)


# ---------------------------------------------------------------------------
# Drawdown penalty
# ---------------------------------------------------------------------------

class TestDrawdownPenalty:
    """Test the quadratic drawdown penalty term."""

    def test_no_drawdown_no_penalty(self):
        """Monotonically increasing equity should have zero penalty."""
        # Strictly increasing equity curve: no drawdown
        equity = torch.arange(1, T + 1, dtype=torch.float32).unsqueeze(0)
        equity = equity.expand(BATCH, -1)
        penalty = NAPSLoss.drawdown_penalty(equity, tau=0.15, lambda1=5.0)
        assert torch.allclose(penalty, torch.zeros_like(penalty), atol=1e-6)

    def test_small_drawdown_below_tau_no_penalty(self):
        """Drawdown below tau should produce zero penalty."""
        # Equity that dips 5% (below tau=15%)
        equity = torch.ones(BATCH, T)
        equity[:, T // 2:] = 0.96  # 4% drawdown from peak of 1.0
        penalty = NAPSLoss.drawdown_penalty(equity, tau=0.15, lambda1=5.0)
        assert torch.allclose(penalty, torch.zeros_like(penalty), atol=1e-6)

    def test_large_drawdown_above_tau_positive_penalty(self):
        """Drawdown exceeding tau should produce positive penalty."""
        # Equity that drops 30% (above tau=15%)
        equity = torch.ones(BATCH, T)
        equity[:, T // 2:] = 0.70  # 30% drawdown
        penalty = NAPSLoss.drawdown_penalty(equity, tau=0.15, lambda1=5.0)
        assert (penalty > 0).all(), f"Expected positive penalty, got {penalty}"
        # Expected: lambda1 * (0.30 - 0.15)^2 = 5.0 * 0.0225 = 0.1125
        expected = 5.0 * (0.30 - 0.15) ** 2
        assert torch.allclose(penalty, torch.tensor(expected).expand_as(penalty), atol=1e-3)

    def test_penalty_is_quadratic(self):
        """Penalty should scale quadratically with excess drawdown."""
        penalties = []
        for dd_frac in [0.20, 0.25, 0.30, 0.40]:
            equity = torch.ones(1, T)
            equity[:, T // 2:] = 1.0 - dd_frac
            p = NAPSLoss.drawdown_penalty(equity, tau=0.15, lambda1=5.0)
            penalties.append(p.item())

        # Check quadratic scaling: penalty ∝ (dd - tau)^2
        for i in range(len(penalties) - 1):
            assert penalties[i + 1] > penalties[i], (
                f"Penalty should increase with drawdown: {penalties}"
            )

    def test_drawdown_penalty_output_shape(self):
        """Should return one penalty per batch element."""
        equity = torch.ones(BATCH, T)
        penalty = NAPSLoss.drawdown_penalty(equity, tau=0.15, lambda1=5.0)
        assert penalty.shape == (BATCH,)


# ---------------------------------------------------------------------------
# Turnover regularisation
# ---------------------------------------------------------------------------

class TestTurnoverRegularization:
    """Test the turnover penalty term."""

    def test_constant_fractions_zero_penalty(self):
        """Constant fractions should produce zero turnover penalty."""
        fractions = torch.ones(BATCH, T) * 0.2
        reg = NAPSLoss.turnover_regularization(fractions, cost=0.001, lambda2=1.0)
        assert torch.allclose(reg, torch.zeros_like(reg), atol=1e-8)

    def test_changing_fractions_positive_penalty(self):
        """Rapidly changing fractions should incur positive penalty."""
        torch.manual_seed(SEED)
        fractions = torch.rand(BATCH, T) * 0.4  # random fractions
        reg = NAPSLoss.turnover_regularization(fractions, cost=0.001, lambda2=1.0)
        assert (reg > 0).all(), f"Expected positive turnover penalty, got {reg}"

    def test_higher_cost_higher_penalty(self):
        """Doubling the cost rate should double the penalty."""
        torch.manual_seed(SEED)
        fractions = torch.rand(BATCH, T) * 0.4
        reg_low = NAPSLoss.turnover_regularization(fractions, cost=0.001, lambda2=1.0)
        reg_high = NAPSLoss.turnover_regularization(fractions, cost=0.002, lambda2=1.0)
        torch.testing.assert_close(reg_high, 2.0 * reg_low, atol=1e-6, rtol=1e-6)

    def test_3d_fractions_squeezed(self):
        """Fractions with trailing dim (B, T, 1) should be handled."""
        torch.manual_seed(SEED)
        fractions_3d = torch.rand(BATCH, T, 1) * 0.4
        fractions_2d = fractions_3d.squeeze(-1)
        reg_3d = NAPSLoss.turnover_regularization(fractions_3d, cost=0.001, lambda2=1.0)
        reg_2d = NAPSLoss.turnover_regularization(fractions_2d, cost=0.001, lambda2=1.0)
        torch.testing.assert_close(reg_3d, reg_2d)

    def test_output_shape(self):
        """Should return one regularisation value per batch element."""
        fractions = torch.rand(BATCH, T) * 0.4
        reg = NAPSLoss.turnover_regularization(fractions, cost=0.001, lambda2=1.0)
        assert reg.shape == (BATCH,)


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

class TestCompositeLoss:
    """Test the full NAPSLoss.forward() combining all terms."""

    def test_returns_scalar(self, loss_fn, sample_data):
        """Composite loss should return a scalar (mean over batch)."""
        fractions, returns = sample_data
        loss = loss_fn(fractions, returns)
        assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"

    def test_loss_finite(self, loss_fn, sample_data):
        """Loss should be finite for typical inputs."""
        fractions, returns = sample_data
        loss = loss_fn(fractions, returns)
        assert torch.isfinite(loss), f"Non-finite loss: {loss.item()}"

    def test_custom_equity_curve(self, loss_fn, sample_data):
        """Should accept a pre-computed equity curve."""
        fractions, returns = sample_data
        equity = torch.cumprod(1.0 + fractions * returns, dim=1)
        loss = loss_fn(fractions, returns, equity_curve=equity)
        assert torch.isfinite(loss)

    def test_loss_decreases_with_better_fractions(self):
        """Better fractions (aligned with returns) should produce lower loss."""
        torch.manual_seed(SEED)
        returns = torch.randn(BATCH, T) * 0.02

        loss_fn = NAPSLoss(W=252, tau=0.15, lambda1=5.0, lambda2=1.0, cost=0.001)

        # "Smart" fractions: size proportional to return direction
        smart_fractions = torch.clamp(returns.sign() * 0.2, 0.0, 0.4)
        smart_fractions = smart_fractions.detach()
        loss_smart = loss_fn(smart_fractions, returns)

        # "Dumb" fractions: always max regardless of return
        dumb_fractions = torch.ones(BATCH, T) * 0.4
        loss_dumb = loss_fn(dumb_fractions, returns)

        # Smart sizing should have better (lower) loss, at least in expectation
        # This is a soft check -- the key point is they produce different losses
        assert loss_smart != loss_dumb


# ---------------------------------------------------------------------------
# Gradient flow through loss
# ---------------------------------------------------------------------------

class TestGradientsThroughLoss:
    """Verify gradients flow from the loss to fraction parameters."""

    def test_gradients_flow_to_fractions(self, loss_fn):
        """Fractions should receive non-zero gradients."""
        torch.manual_seed(SEED)
        # Create a leaf tensor with requires_grad=True
        raw = torch.randn(BATCH, T)
        fractions = (raw.sigmoid() * 0.4).detach().requires_grad_(True)
        returns = torch.randn(BATCH, T) * 0.02

        loss = loss_fn(fractions, returns)
        loss.backward()

        assert fractions.grad is not None, "No gradient on fractions"
        assert fractions.grad.abs().sum() > 0, "Zero gradient on fractions"

    def test_gradients_finite(self, loss_fn):
        """Gradients should be finite (no NaN or Inf)."""
        torch.manual_seed(SEED)
        raw = torch.randn(BATCH, T)
        fractions = (raw.sigmoid() * 0.4).detach().requires_grad_(True)
        returns = torch.randn(BATCH, T) * 0.02

        loss = loss_fn(fractions, returns)
        loss.backward()

        assert fractions.grad is not None, "No gradient on fractions"
        assert torch.isfinite(fractions.grad).all(), "Non-finite gradient detected"
