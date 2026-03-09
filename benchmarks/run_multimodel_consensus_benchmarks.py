"""Compare the experimental multi-model consensus workflow to the default hybrid screener."""
from __future__ import annotations

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
    load_release_benchmark_cases,
    summarize_hybrid_vs_multimodel,
)


def main():
    cases = load_release_benchmark_cases()
    details, summary = compare_hybrid_vs_multimodel(cases=cases)
    comparison = summarize_hybrid_vs_multimodel(details)

    if details.empty:
        print("No benchmark cases available.")
        return

    print("Fixed public cases:")
    for case in cases:
        print(f"  - {case.name} ({case.source})")

    print("\nWorkflow summary:")
    print(summary.to_string(index=False))

    print("\nPer-case workflow details:")
    display_cols = ["workflow", "scenario", "tier", "precision", "recall", "F1", "FDR"]
    print(details[display_cols].to_string(index=False))

    print("\nWhere the experimental workflow helps or hurts:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)
    main()
