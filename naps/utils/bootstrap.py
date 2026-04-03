"""
Paired bootstrap significance testing (Sections 4.3, 5.7).

Implements the paired bootstrap test of Efron & Tibshirani (1993) to assess
whether the difference in performance between two trading strategies is
statistically significant.  The test resamples *pairs* of per-period returns
(preserving temporal alignment) and computes the probability that the observed
performance gap arose by chance.

Also provides a convenience function for building a full pairwise significance
matrix across multiple strategies.

Significance levels reported in the paper:
    ***  p < 0.01
    **   p < 0.05
    *    p < 0.10
    n.s. otherwise
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Built-in metric functions (self-contained, no external dependencies)
# ---------------------------------------------------------------------------

def _sharpe_ratio(returns: np.ndarray) -> float:
    """Annualised Sharpe ratio from daily returns."""
    if len(returns) < 2 or np.std(returns, ddof=1) < 1e-12:
        return 0.0
    return float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252))


def _annualized_return(returns: np.ndarray) -> float:
    """Annualised compound return from daily returns."""
    cum = np.prod(1.0 + returns)
    n_years = len(returns) / 252.0
    if n_years <= 0:
        return 0.0
    return float(cum ** (1.0 / n_years) - 1.0)


def _max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from daily returns (returned as positive fraction)."""
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    dd = (running_max - equity) / np.where(running_max > 0, running_max, 1.0)
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def _mean_return(returns: np.ndarray) -> float:
    """Simple mean of daily returns."""
    return float(np.mean(returns))


_METRIC_FNS: Dict[str, Callable[[np.ndarray], float]] = {
    "sharpe": _sharpe_ratio,
    "annualized_return": _annualized_return,
    "max_drawdown": _max_drawdown,
    "mean": _mean_return,
}


def _get_metric_fn(metric: str) -> Callable[[np.ndarray], float]:
    """Resolve a metric name to its computation function."""
    if metric not in _METRIC_FNS:
        raise ValueError(
            f"Unknown metric '{metric}'.  "
            f"Available: {list(_METRIC_FNS.keys())}"
        )
    return _METRIC_FNS[metric]


# ---------------------------------------------------------------------------
# Significance label
# ---------------------------------------------------------------------------

def significance_label(p_value: float) -> str:
    """Convert a p-value to a significance label.

    Returns:
        ``"***"`` for p < 0.01, ``"**"`` for p < 0.05, ``"*"`` for p < 0.10,
        ``"n.s."`` otherwise.
    """
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.10:
        return "*"
    else:
        return "n.s."


# ---------------------------------------------------------------------------
# PairedBootstrap
# ---------------------------------------------------------------------------

class PairedBootstrap:
    """Paired bootstrap hypothesis test for strategy comparison.

    The test resamples *paired* daily returns (preserving the temporal
    correspondence between strategies) and computes the difference in the
    chosen metric on each resample.  The p-value is the fraction of
    resamples where the observed advantage disappears (two-sided test).

    Usage::

        pb = PairedBootstrap(n_resamples=10000, seed=42)
        p  = pb.test(returns_naps, returns_baseline, metric="sharpe")
        print(f"p = {p:.4f}  {significance_label(p)}")

    Args:
        n_resamples: Number of bootstrap resamples.  Default 10 000.
        seed:        Random seed for reproducibility.  Default ``None``.
    """

    def __init__(
        self,
        n_resamples: int = 10_000,
        seed: int | None = None,
    ) -> None:
        self.n_resamples = n_resamples
        self.rng = np.random.default_rng(seed)

    def test(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        metric: str = "sharpe",
    ) -> float:
        """Run the paired bootstrap test.

        H0: metric(A) == metric(B)
        H1: metric(A) != metric(B)   (two-sided)

        Args:
            returns_a: (N,) daily returns of strategy A.
            returns_b: (N,) daily returns of strategy B.
            metric:    One of ``"sharpe"``, ``"annualized_return"``,
                       ``"max_drawdown"``, ``"mean"``.

        Returns:
            Two-sided p-value.
        """
        returns_a = np.asarray(returns_a, dtype=np.float64).ravel()
        returns_b = np.asarray(returns_b, dtype=np.float64).ravel()

        if len(returns_a) != len(returns_b):
            raise ValueError(
                f"Return arrays must have equal length.  "
                f"Got {len(returns_a)} vs {len(returns_b)}."
            )

        N = len(returns_a)
        metric_fn = _get_metric_fn(metric)

        # Observed difference.
        observed_diff = metric_fn(returns_a) - metric_fn(returns_b)

        # Bootstrap resampling.
        count_extreme = 0
        for _ in range(self.n_resamples):
            idx = self.rng.integers(0, N, size=N)
            diff_i = metric_fn(returns_a[idx]) - metric_fn(returns_b[idx])

            # Two-sided: count resamples where the sign of the difference
            # flips relative to the observed difference.
            if observed_diff >= 0 and diff_i < 0:
                count_extreme += 1
            elif observed_diff < 0 and diff_i >= 0:
                count_extreme += 1

        p_value = count_extreme / self.n_resamples
        return float(p_value)

    def pairwise_significance_matrix(
        self,
        all_returns_dict: Dict[str, np.ndarray],
        metric: str = "sharpe",
    ) -> Dict[str, Dict[str, Dict[str, float | str]]]:
        """Compute pairwise p-values for all strategy pairs.

        Args:
            all_returns_dict: Mapping from strategy name to (N,) daily
                              returns array.  All arrays must have the
                              same length.
            metric:           Metric to compare.  Default ``"sharpe"``.

        Returns:
            Nested dictionary::

                {
                    "NAPS": {
                        "Kelly": {"p_value": 0.003, "significance": "***"},
                        "EqualWeight": {"p_value": 0.041, "significance": "**"},
                    },
                    ...
                }

            Only upper-triangle pairs are computed (A vs B, not B vs A);
            symmetric entries are populated with the same p-value.
        """
        names = sorted(all_returns_dict.keys())
        result: Dict[str, Dict[str, Dict[str, float | str]]] = {
            n: {} for n in names
        }

        for name_a, name_b in itertools.combinations(names, 2):
            p = self.test(
                all_returns_dict[name_a],
                all_returns_dict[name_b],
                metric=metric,
            )
            entry = {"p_value": p, "significance": significance_label(p)}
            result[name_a][name_b] = entry
            result[name_b][name_a] = entry

        return result

    def summary_table(
        self,
        all_returns_dict: Dict[str, np.ndarray],
        metric: str = "sharpe",
    ) -> str:
        """Return a formatted text table of pairwise significance results.

        Useful for logging or inclusion in reports.

        Args:
            all_returns_dict: Strategy name -> returns mapping.
            metric:           Metric to compare.

        Returns:
            Multi-line string with a pairwise comparison table.
        """
        matrix = self.pairwise_significance_matrix(all_returns_dict, metric)
        names = sorted(all_returns_dict.keys())

        # Header
        col_w = max(len(n) for n in names) + 2
        header = " " * col_w + "".join(n.rjust(col_w) for n in names)
        lines = [f"Pairwise bootstrap test (metric={metric})", header]

        for row_name in names:
            row = row_name.ljust(col_w)
            for col_name in names:
                if col_name == row_name:
                    cell = "---"
                elif col_name in matrix[row_name]:
                    entry = matrix[row_name][col_name]
                    cell = f"{entry['p_value']:.3f}{entry['significance']}"
                else:
                    cell = ""
                row += cell.rjust(col_w)
            lines.append(row)

        return "\n".join(lines)
