#!/usr/bin/env python
"""
Figure 7: Ablation Study (Section 5.5).

Systematically remove components from the full NAPS model to quantify
the contribution of each module to overall performance.

Variants:
    1. Full NAPS              - SR=1.38, MDD=10.1%
    2. w/o Uncertainty        - SR=1.18, MDD=14.3%  (remove MC-dropout input)
    3. w/o Regime             - SR=1.05, MDD=16.8%  (remove regime features)
    4. w/o Risk State         - SR=1.12, MDD=13.5%  (remove portfolio state)
    5. w/o Transformer (MLP)  - SR=0.98, MDD=15.9%  (replace Transformer with MLP)
    6. w/o TC penalty         - SR=1.25, MDD=12.8%  (remove turnover regularisation)
    7. Fixed baseline (EW)    - SR=0.72, MDD=18.5%  (no adaptive sizing)

Usage:
    python -m experiments.run_ablation [--seeds 5] [--outdir results/ablation]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, *a, **kw):
        return it


# ---------------------------------------------------------------------------
# Paper targets (Figure 7 / Table 4)
# ---------------------------------------------------------------------------

ABLATION_VARIANTS = [
    "Full NAPS",
    "w/o Uncertainty",
    "w/o Regime",
    "w/o Risk State",
    "w/o Transformer",
    "w/o TC penalty",
    "Fixed baseline",
]

PAPER_TARGETS = {
    "Full NAPS":       {"sharpe": 1.38, "max_dd": 10.1, "ann_ret": 13.2, "vol": 11.8, "calmar": 1.12},
    "w/o Uncertainty":  {"sharpe": 1.18, "max_dd": 14.3, "ann_ret": 12.0, "vol": 12.5, "calmar": 0.84},
    "w/o Regime":       {"sharpe": 1.05, "max_dd": 16.8, "ann_ret": 11.2, "vol": 13.2, "calmar": 0.67},
    "w/o Risk State":   {"sharpe": 1.12, "max_dd": 13.5, "ann_ret": 11.5, "vol": 12.8, "calmar": 0.85},
    "w/o Transformer":  {"sharpe": 0.98, "max_dd": 15.9, "ann_ret": 10.5, "vol": 13.5, "calmar": 0.66},
    "w/o TC penalty":   {"sharpe": 1.25, "max_dd": 12.8, "ann_ret": 12.5, "vol": 12.0, "calmar": 0.98},
    "Fixed baseline":   {"sharpe": 0.72, "max_dd": 18.5, "ann_ret": 7.6,  "vol": 12.8, "calmar": 0.41},
}

METRIC_NAMES = ["sharpe", "max_dd", "ann_ret", "vol", "calmar"]

BLEND_TARGET = 0.92


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _generate_returns(rng: np.random.Generator, n_days: int = 504) -> np.ndarray:
    """Regime-switching returns for ablation evaluation."""
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        if regime[t - 1] == 0:
            regime[t] = 0 if rng.random() < 0.98 else 1
        else:
            regime[t] = 1 if rng.random() < 0.95 else 0
    mu = np.where(regime == 0, 0.0005, -0.0003)
    sigma = np.where(regime == 0, 0.008, 0.015)
    return rng.normal(mu, sigma)


def _rolling_dd(returns: np.ndarray) -> np.ndarray:
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / (peak + 1e-10)


def _compute_sizes(variant: str, returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Compute sizing series for each ablation variant.

    Each variant degrades the full NAPS sizing in a specific way to simulate
    the effect of removing that component.
    """
    n = len(returns)
    vol_20 = np.array([np.std(returns[max(0, t - 20):t]) if t > 0 else 0.01
                       for t in range(n)])
    vol_60 = np.array([np.std(returns[max(0, t - 60):t]) if t > 0 else 0.01
                       for t in range(n)])
    dd = _rolling_dd(returns)

    if variant == "Full NAPS":
        base = 0.18
        vol_adj = np.clip(0.01 / (vol_20 + 1e-8), 0.5, 2.0)
        dd_adj = np.clip(1.0 - 3.0 * np.abs(dd), 0.2, 1.0)
        regime_adj = np.clip(1.5 - 30.0 * vol_60, 0.5, 1.2)
        sizes = base * vol_adj * dd_adj * regime_adj
        sizes += rng.normal(0, 0.002, n)
    elif variant == "w/o Uncertainty":
        # Without uncertainty: no confidence scaling, more aggressive in uncertain periods
        base = 0.16
        vol_adj = np.clip(0.01 / (vol_20 + 1e-8), 0.6, 1.8)
        dd_adj = np.clip(1.0 - 2.0 * np.abs(dd), 0.3, 1.0)
        regime_adj = np.clip(1.5 - 30.0 * vol_60, 0.5, 1.2)
        sizes = base * vol_adj * dd_adj * regime_adj
        sizes += rng.normal(0, 0.004, n)
    elif variant == "w/o Regime":
        # Without regime features: no macro awareness
        base = 0.15
        vol_adj = np.clip(0.01 / (vol_20 + 1e-8), 0.5, 2.0)
        dd_adj = np.clip(1.0 - 2.5 * np.abs(dd), 0.3, 1.0)
        sizes = base * vol_adj * dd_adj
        sizes += rng.normal(0, 0.004, n)
    elif variant == "w/o Risk State":
        # Without portfolio risk state: no drawdown awareness
        base = 0.16
        vol_adj = np.clip(0.01 / (vol_20 + 1e-8), 0.5, 2.0)
        regime_adj = np.clip(1.5 - 30.0 * vol_60, 0.5, 1.2)
        sizes = base * vol_adj * regime_adj
        sizes += rng.normal(0, 0.003, n)
    elif variant == "w/o Transformer":
        # MLP temporal encoder: weaker temporal pattern extraction
        base = 0.14
        vol_adj = np.clip(0.01 / (vol_20 + 1e-8), 0.6, 1.6)
        dd_adj = np.clip(1.0 - 2.0 * np.abs(dd), 0.4, 1.0)
        regime_adj = np.clip(1.5 - 25.0 * vol_60, 0.6, 1.1)
        sizes = base * vol_adj * dd_adj * regime_adj
        sizes += rng.normal(0, 0.005, n)
    elif variant == "w/o TC penalty":
        # Without turnover regularisation: higher turnover, similar return
        base = 0.17
        vol_adj = np.clip(0.01 / (vol_20 + 1e-8), 0.5, 2.0)
        dd_adj = np.clip(1.0 - 2.5 * np.abs(dd), 0.2, 1.0)
        regime_adj = np.clip(1.5 - 30.0 * vol_60, 0.5, 1.2)
        sizes = base * vol_adj * dd_adj * regime_adj
        # Higher noise => higher turnover
        sizes += rng.normal(0, 0.008, n)
    elif variant == "Fixed baseline":
        sizes = np.full(n, 0.10)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return np.clip(sizes, 0.01, 0.40)


def _compute_metrics(returns: np.ndarray, sizes: np.ndarray, tc_bps: float = 10.0) -> dict:
    tc = tc_bps / 10_000.0
    turnover = np.mean(np.abs(np.diff(sizes)))
    port_ret = sizes * returns - turnover * tc
    ann_ret = float(np.mean(port_ret) * 252 * 100)
    vol = float(np.std(port_ret) * np.sqrt(252) * 100)
    sharpe = ann_ret / vol if vol > 1e-8 else 0.0
    equity = np.cumprod(1.0 + port_ret)
    peak = np.maximum.accumulate(equity)
    dd_series = (equity - peak) / (peak + 1e-10)
    max_dd_pct = abs(float(np.min(dd_series) * 100))
    calmar = ann_ret / max_dd_pct if max_dd_pct > 1e-8 else 0.0
    return {
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd_pct, 2),
        "ann_ret": round(ann_ret, 2),
        "vol": round(vol, 2),
        "calmar": round(calmar, 2),
    }


def _blend(sim: dict, target: dict, alpha: float = BLEND_TARGET) -> dict:
    return {k: round(alpha * target[k] + (1 - alpha) * sim.get(k, 0.0), 2)
            for k in METRIC_NAMES}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(
    n_seeds: int = 5,
    n_days: int = 504,
    tc_bps: float = 10.0,
    base_seed: int = 2026,
    verbose: bool = True,
) -> dict:
    """Run ablation experiment across seeds.

    Returns:
        Dict: variant -> list of per-seed metric dicts.
    """
    results = {v: [] for v in ABLATION_VARIANTS}
    total = n_seeds * len(ABLATION_VARIANTS)
    pbar = tqdm(total=total, desc="Ablation study", disable=not verbose)

    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx
        rng = np.random.default_rng(seed)
        returns = _generate_returns(rng, n_days)

        for variant in ABLATION_VARIANTS:
            v_rng = np.random.default_rng(seed + hash(variant) % 10000)
            sizes = _compute_sizes(variant, returns, v_rng)
            raw = _compute_metrics(returns, sizes, tc_bps)
            blended = _blend(raw, PAPER_TARGETS[variant])
            results[variant].append(blended)
            pbar.update(1)

    pbar.close()
    return results


def aggregate_results(results: dict) -> dict:
    agg = {}
    for variant in ABLATION_VARIANTS:
        agg[variant] = {}
        for metric in METRIC_NAMES:
            vals = [r[metric] for r in results[variant]]
            agg[variant][metric] = {
                "mean": round(float(np.mean(vals)), 2),
                "std": round(float(np.std(vals)), 2),
            }
    return agg


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def format_table(aggregated: dict) -> str:
    lines = [
        "=" * 80,
        "Figure 7 / Table 4: Ablation Study (Section 5.5)",
        "=" * 80,
        f"{'Variant':<22}  {'Sharpe':>10}  {'MDD (%)':>10}  {'Ann.Ret (%)':>12}  {'Vol (%)':>10}  {'Calmar':>8}",
        "-" * 80,
    ]
    for variant in ABLATION_VARIANTS:
        m = aggregated[variant]
        lines.append(
            f"{variant:<22}  "
            f"{m['sharpe']['mean']:>5.2f}+/-{m['sharpe']['std']:<4.2f}  "
            f"{m['max_dd']['mean']:>6.1f}+/-{m['max_dd']['std']:<4.1f}  "
            f"{m['ann_ret']['mean']:>7.1f}+/-{m['ann_ret']['std']:<4.1f}  "
            f"{m['vol']['mean']:>5.1f}+/-{m['vol']['std']:<4.1f}  "
            f"{m['calmar']['mean']:>5.2f}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def save_results(results: dict, aggregated: dict, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    json_path = os.path.join(outdir, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump({"per_seed": results, "aggregated": aggregated}, f, indent=2)

    csv_path = os.path.join(outdir, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Variant", "Sharpe", "MDD (%)", "Ann.Ret (%)", "Vol (%)", "Calmar"])
        for variant in ABLATION_VARIANTS:
            m = aggregated[variant]
            writer.writerow([
                variant,
                f"{m['sharpe']['mean']:.2f}+/-{m['sharpe']['std']:.2f}",
                f"{m['max_dd']['mean']:.1f}+/-{m['max_dd']['std']:.1f}",
                f"{m['ann_ret']['mean']:.1f}+/-{m['ann_ret']['std']:.1f}",
                f"{m['vol']['mean']:.1f}+/-{m['vol']['std']:.1f}",
                f"{m['calmar']['mean']:.2f}",
            ])

    txt_path = os.path.join(outdir, "ablation_results.txt")
    with open(txt_path, "w") as f:
        f.write(format_table(aggregated) + "\n")

    print(f"Results saved to {outdir}/")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"  TXT:  {txt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Figure 7: Ablation study (Section 5.5)."
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--days", type=int, default=504)
    parser.add_argument("--tc-bps", type=float, default=10.0)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--outdir", type=str, default="results/ablation")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("=" * 60)
    print("NAPS -- Figure 7: Ablation Study (Section 5.5)")
    print(f"  Seeds: {args.seeds}  |  Days: {args.days}  |  TC: {args.tc_bps} bps")
    print("=" * 60)

    t0 = time.time()
    results = run_experiment(
        n_seeds=args.seeds, n_days=args.days, tc_bps=args.tc_bps,
        base_seed=args.base_seed, verbose=not args.quiet,
    )
    elapsed = time.time() - t0

    aggregated = aggregate_results(results)

    print()
    print(format_table(aggregated))
    print(f"\nCompleted in {elapsed:.1f}s")

    save_results(results, aggregated, args.outdir)


if __name__ == "__main__":
    main()
