# NAPS: Neural Adaptive Position Sizing for Risk-Aware Algorithmic Trading

**Preprint (2026)**

> Position sizing -- deciding *how much* to allocate to each trade -- is a
> critical yet under-studied component of systematic trading strategies.
> Traditional approaches (Kelly criterion, fixed-fraction, inverse-volatility)
> rely on strong distributional assumptions and ignore the joint dynamics of
> market context, signal uncertainty, and portfolio risk.  We propose **NAPS**,
> a neural architecture that learns to adaptively size positions by fusing
> temporal market representations, calibrated signal uncertainty, and portfolio
> risk state through cross-attention.  Trained end-to-end with a composite
> differentiable Sharpe / drawdown-penalised objective, NAPS produces bounded
> bet fractions that outperform all baselines across equities, crypto, and
> forex in walk-forward evaluation, achieving a 23 % higher Sharpe ratio while
> reducing maximum drawdown by 31 % relative to the best classical method.

---

## Architecture (Figure 2)

```
  Market History                   Signal + Uncertainty         Risk State + Regime
  (B, 60, 20)                       (B, 2) + (B, 1)           (B, 5) + (B, 3)
       |                                  |                          |
       v                                  v                          v
 +-----------------------+   +-------------------------+   +---------------------+
 | Temporal Transformer  |   | Signal-Uncertainty      |   | Risk-State          |
 | Encoder               |   | Encoder                 |   | Encoder             |
 | Linear -> PE -> TF(2) |   | MLP(3 -> 64 -> 64)     |   | MLP(8 -> 64 -> 64) |
 | -> MeanPool           |   |                         |   |                     |
 +-----------------------+   +-------------------------+   +---------------------+
       |                                  |                          |
       | h_temp (B, 64)                   | h_sig (B, 64)           | h_risk (B, 64)
       |                                  |                          |
       +----------------------------------+----------+               |
                                                     |               |
                                                     v               v
                                            +----------------------------+
                                            | Cross-Attention Fusion     |
                                            | Q = h_risk                 |
                                            | K = V = [h_temp ; h_sig]   |
                                            | MHA + Residual + LayerNorm |
                                            +----------------------------+
                                                         |
                                                         | h_fused (B, 64)
                                                         v
                                              +----------------------+
                                              | Sizing Head          |
                                              | Linear(64->32)->ReLU |
                                              | ->Linear(32->1)     |
                                              | ->Sigmoid * f_max    |
                                              +----------------------+
                                                         |
                                                         v
                                                  f_t in [0, f_max]
```

**Total parameters: ~180K** (Section 3.2)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/naps-authors/naps.git
cd naps

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install pinned versions
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

```python
import torch
from naps import NAPSModel

# Instantiate with default hyperparameters (Table 1)
model = NAPSModel(
    d_model=64,
    n_heads=4,
    n_layers=2,
    lookback=60,
    n_features=20,
    f_max=0.40,
    dropout=0.1,
)

# Random inputs (batch of 8)
market_history = torch.randn(8, 60, 20)   # (B, lookback, n_features)
signal         = torch.randn(8, 2)        # (B, 2) direction + magnitude
uncertainty    = torch.rand(8, 1)         # (B, 1) MC-dropout uncertainty
risk_state     = torch.randn(8, 5)        # (B, 5) portfolio state
regime         = torch.randn(8, 3)        # (B, 3) regime features

# Forward pass
f_t = model(market_history, signal, uncertainty, risk_state, regime)
print(f_t.shape)    # torch.Size([8, 1])
print(f_t.min(), f_t.max())  # bounded in [0, 0.40]
```

---

## Module Overview

| Module | Description |
|---|---|
| `naps/model/encoders.py` | Temporal Transformer, Signal-Uncertainty, and Risk-State encoders (Section 3.2) |
| `naps/model/fusion.py` | Cross-attention fusion with risk-conditioned query (Section 3.2) |
| `naps/model/sizing_head.py` | Bounded sizing head with Sigmoid * f_max output (Section 3.2) |
| `naps/model/naps.py` | End-to-end NAPSModel and NAPSMLPModel ablation variant |
| `naps/training/objective.py` | Composite loss: differentiable Sharpe + drawdown penalty + turnover (Section 3.3) |
| `naps/signal/lstm_signal.py` | 2-layer LSTM for direction/magnitude prediction (Section 4.2) |
| `naps/signal/mc_dropout.py` | MC Dropout uncertainty estimation (Section 3.1) |
| `naps/baselines/` | Equal-weight, Full/Half Kelly, ATR, Risk Parity sizers (Section 4.3) |
| `naps/data/features.py` | Technical indicators, regime features, portfolio state (Section 3.1) |
| `naps/utils/bootstrap.py` | Paired bootstrap significance testing (Section 5.7) |
| `configs/default.yaml` | All paper hyperparameters (Table 1) |

---

## Running Experiments

```bash
# Run the full main experiment (Table 2)
make run-main

# Run ablation study (Table 4)
make run-ablation

# Run baseline comparisons only
make run-baselines

# Run everything
make run-all
```

---

## Results Summary (Table 2)

| Method | Equities Sharpe | Crypto Sharpe | Forex Sharpe | Avg. MDD |
|---|---|---|---|---|
| Equal Weight | 0.82 | 0.61 | 0.73 | 22.4% |
| Full Kelly | 0.91 | 0.58 | 0.79 | 28.7% |
| Half Kelly | 0.95 | 0.67 | 0.84 | 19.1% |
| ATR-Based | 0.98 | 0.71 | 0.86 | 17.3% |
| Risk Parity | 1.01 | 0.74 | 0.88 | 16.8% |
| **NAPS (ours)** | **1.24** | **0.91** | **1.08** | **11.6%** |

NAPS achieves a **23% higher Sharpe ratio** and **31% lower maximum drawdown** compared to the best classical baseline (Risk Parity), with statistical significance at p < 0.01 via paired bootstrap testing.

---

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run a specific test file
pytest tests/test_model.py -v
```

---

## Citation

```bibtex
@misc{naps2026,
  title  = {Neural Adaptive Position Sizing for Risk-Aware Algorithmic Trading},
  author = {NAPS Authors},
  year   = {2026},
  note   = {Preprint},
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
