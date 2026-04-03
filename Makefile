.PHONY: install install-dev test test-cov lint format typecheck run-main run-ablation clean help

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

install:  ## Install package in editable mode
	pip install -e .

install-dev:  ## Install with dev dependencies
	pip install -e ".[dev]"

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

test:  ## Run unit tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage report
	pytest tests/ -v --cov=naps --cov-report=term-missing --cov-report=html

lint:  ## Lint with ruff
	ruff check naps/ tests/

format:  ## Auto-format with black and ruff
	black naps/ tests/
	ruff check --fix naps/ tests/

typecheck:  ## Run mypy type checking
	mypy naps/ --ignore-missing-imports

# ---------------------------------------------------------------------------
# Experiments (Section 4)
# ---------------------------------------------------------------------------

run-main:  ## Run main experiment (Table 2)
	python -m experiments.run_main --config configs/default.yaml

run-ablation:  ## Run ablation study (Table 4)
	python -m experiments.run_ablation --config configs/default.yaml

run-baselines:  ## Run baseline comparisons (Section 4.3)
	python -m experiments.run_baselines --config configs/default.yaml

run-all:  ## Run all experiments
	$(MAKE) run-baselines
	$(MAKE) run-main
	$(MAKE) run-ablation

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

clean:  ## Remove build artefacts and caches
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
