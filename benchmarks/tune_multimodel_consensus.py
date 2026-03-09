"""Tune experimental multi-model consensus variants on public benchmark cases."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from pharmacoml.covselect.public_benchmarks import (
    compare_hybrid_vs_multimodel,
    compare_multimodel_variants,
    load_fixed_public_cases,
    load_release_benchmark_cases,
    summarize_hybrid_vs_multimodel,
)


def main():
    parser = argparse.ArgumentParser(description="Tune experimental multi-model consensus settings.")
    parser.add_argument(
        "--suite",
        choices=["fixed", "release"],
        default="fixed",
        help="Benchmark case suite to use.",
    )
    args = parser.parse_args()

    if args.suite == "release":
        cases = load_release_benchmark_cases(include_optional_kekic=False)
    else:
        cases = load_fixed_public_cases()

    mm_details, mm_summary = compare_multimodel_variants(cases=cases)
    best_variant = mm_summary.iloc[0]["variant"] if not mm_summary.empty else None

    print("Benchmark cases:")
    for case in cases:
        print(f"  - {case.name} ({case.source})")

    print("\nMulti-model variant summary:")
    print(mm_summary.to_string(index=False))

    print("\nPer-case multi-model details:")
    print(mm_details.to_string(index=False))

    if best_variant is None:
        return

    best_kwargs = {
        "scikit_core": {
            "models": ["random_forest", "extra_trees", "gradient_boosting", "lasso", "aalasso"],
            "top_k": 3,
            "n_bootstrap": 3,
            "use_significance_filter": False,
            "min_model_frequency": 0.50,
            "min_family_support": 2,
            "run_permutation": False,
        },
        "scikit_core_relaxed": {
            "models": ["random_forest", "extra_trees", "gradient_boosting", "lasso", "aalasso"],
            "top_k": 3,
            "n_bootstrap": 3,
            "use_significance_filter": False,
            "min_model_frequency": 0.40,
            "min_family_support": 2,
            "run_permutation": False,
        },
        "linear_tree_balanced": {
            "models": ["random_forest", "extra_trees", "lasso", "elastic_net", "aalasso", "ridge"],
            "top_k": 2,
            "n_bootstrap": 3,
            "use_significance_filter": False,
            "min_model_frequency": 0.50,
            "min_family_support": 2,
            "run_permutation": False,
        },
    }[best_variant]

    hv_details, hv_summary = compare_hybrid_vs_multimodel(
        cases=cases,
        hybrid_kwargs={"n_bootstrap": 8, "run_permutation": False},
        multimodel_kwargs=best_kwargs,
    )
    comparison = summarize_hybrid_vs_multimodel(hv_details)

    print(f"\nBest multi-model variant: {best_variant}")
    print("\nHybrid vs best multi-model summary:")
    print(hv_summary.to_string(index=False))

    print("\nHybrid vs best multi-model details:")
    print(hv_details.to_string(index=False))

    print("\nPer-case verdict:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    main()
