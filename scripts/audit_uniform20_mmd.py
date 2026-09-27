#!/usr/bin/env python3
"""Audit the Table-1 kFFM-vs-FFM MMD-RBF comparison at 20 seeds.

The statistical unit is one training seed. For Economy, the MMD values from
the available component series are averaged within seed before paired tests;
component-series observations are not counted as independent seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


TARGET_SEEDS = [2**index for index in range(20)]

COMPARISONS: dict[str, dict[str, Any]] = {
    "Gene Expr.": {
        "dataset": "expr_genes",
        "ffm": "none",
        "kffm": "signature",
        "config_label": "Sig/gp",
    },
    "Economy": {
        "dataset": "economy",
        "ffm": "none",
        "kffm": "signature",
        "config_label": "Sig/gp",
        "components": ["econ1_population", "econ2_gdp"],
    },
    "Stoch. NS": {
        "dataset": "stochastic_ns",
        "ffm": "none",
        "kffm": "sns_iid_rbf_s025_r005",
        "config_label": "RBF/wn",
    },
    "Navier-Stokes": {
        "dataset": "navier_stokes",
        "ffm": "none",
        "kffm": "ns_iid_rbf_s1_r002",
        "config_label": "RBF/wn",
    },
    "AEMET": {
        "dataset": "aemet",
        "ffm": "none",
        "kffm": "signature",
        "config_label": "Sig/gp",
    },
    "Heston": {
        "dataset": "heston",
        "ffm": "none",
        "kffm": "cfm_rbf_ot_s2",
        "config_label": "RBF/wn",
    },
    "KdV": {
        "dataset": "kdv",
        "ffm": "none",
        "kffm": "sobolev_rbf_s2",
        "config_label": "Sob-RBF/gp",
    },
    "Stoch. KdV": {
        "dataset": "stochastic_kdv",
        "ffm": "none",
        "kffm": "sobolev_rbf_s2",
        "config_label": "Sob-RBF/gp",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/seeded_runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/seeded_runs/uniform20_mmd_audit.json"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def read_mmd(path: Path) -> float:
    metrics = json.loads(path.read_text())
    return float(metrics["mmd_rbf"])


def load_seed_values(
    root: Path,
    dataset: str,
    method: str,
    components: list[str] | None,
) -> dict[int, float]:
    method_root = root / dataset / method
    if components is None:
        return {
            int(path.parent.name.removeprefix("seed_")): read_mmd(path)
            for path in method_root.glob("seed_*/quality_metrics.json")
        }

    component_values: dict[int, list[float]] = {}
    for component in components:
        paths = method_root.glob(f"{component}/seed_*/quality_metrics.json")
        for path in paths:
            seed = int(path.parent.name.removeprefix("seed_"))
            component_values.setdefault(seed, []).append(read_mmd(path))

    return {
        seed: float(np.mean(values))
        for seed, values in component_values.items()
        if len(values) == len(components)
    }


def paired_statistics(
    ffm: np.ndarray,
    kffm: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    difference = ffm - kffm  # Positive means lower/better kFFM MMD-RBF.
    t_p = float(ttest_rel(ffm, kffm).pvalue)
    wilcoxon_p = float(
        wilcoxon(ffm, kffm, alternative="two-sided", method="auto").pvalue
    )
    return {
        "n_seeds": int(len(ffm)),
        "ffm_mean": float(np.mean(ffm)),
        "ffm_sample_std": float(np.std(ffm, ddof=1)),
        "kffm_mean": float(np.mean(kffm)),
        "kffm_sample_std": float(np.std(kffm, ddof=1)),
        "mean_improvement": float(np.mean(difference)),
        "kffm_wins": int(np.sum(difference > 0)),
        "paired_t_p_two_sided": t_p,
        "wilcoxon_p_two_sided": wilcoxon_p,
        "significant_both_tests": bool(
            np.mean(difference) > 0 and t_p < alpha and wilcoxon_p < alpha
        ),
    }


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "metric": "mmd_rbf",
        "direction": "lower_is_better",
        "target_seed_count": 20,
        "target_seeds": TARGET_SEEDS,
        "alpha": args.alpha,
        "tests": ["two-sided paired t-test", "two-sided Wilcoxon signed-rank"],
        "selection_policy": (
            "Seed identities are fixed before reading outcomes. No seed subset is "
            "selected to optimize p-values."
        ),
        "economy_unit": (
            "One seed after averaging econ1_population and econ2_gdp within seed."
        ),
        "datasets": {},
    }

    for display_name, comparison in COMPARISONS.items():
        components = comparison.get("components")
        ffm_by_seed = load_seed_values(
            args.root,
            comparison["dataset"],
            comparison["ffm"],
            components,
        )
        kffm_by_seed = load_seed_values(
            args.root,
            comparison["dataset"],
            comparison["kffm"],
            components,
        )
        shared_seeds = [
            seed
            for seed in TARGET_SEEDS
            if seed in ffm_by_seed and seed in kffm_by_seed
        ]
        missing_ffm = [seed for seed in TARGET_SEEDS if seed not in ffm_by_seed]
        missing_kffm = [seed for seed in TARGET_SEEDS if seed not in kffm_by_seed]
        ffm = np.array([ffm_by_seed[seed] for seed in shared_seeds])
        kffm = np.array([kffm_by_seed[seed] for seed in shared_seeds])

        dataset_report = {
            **comparison,
            "shared_seeds": shared_seeds,
            "missing_ffm_seeds": missing_ffm,
            "missing_kffm_seeds": missing_kffm,
            "complete_20_seed_comparison": len(shared_seeds) == 20,
            "statistics": paired_statistics(ffm, kffm, args.alpha),
        }
        report["datasets"][display_name] = dataset_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(
        "| Dataset | n | FFM MMD-RBF x1e3 | kFFM MMD-RBF x1e3 | "
        "paired-t p | Wilcoxon p | Status |"
    )
    print("|---|---:|---:|---:|---:|---:|---|")
    for name, result in report["datasets"].items():
        stats = result["statistics"]
        status = (
            "significant"
            if result["complete_20_seed_comparison"]
            and stats["significant_both_tests"]
            else "incomplete"
            if not result["complete_20_seed_comparison"]
            else "not significant"
        )
        print(
            f"| {name} | {stats['n_seeds']} | "
            f"{stats['ffm_mean'] * 1e3:.3f} ± "
            f"{stats['ffm_sample_std'] * 1e3:.3f} | "
            f"{stats['kffm_mean'] * 1e3:.3f} ± "
            f"{stats['kffm_sample_std'] * 1e3:.3f} | "
            f"{stats['paired_t_p_two_sided']:.4g} | "
            f"{stats['wilcoxon_p_two_sided']:.4g} | {status} |"
        )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
