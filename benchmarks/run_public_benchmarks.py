"""Run the fixed public hybrid benchmark suite.

Usage:
    python benchmarks/run_public_benchmarks.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from pharmacoml.covselect.public_benchmarks import (
    compare_hybrid_variants,
    compare_summary_to_baseline,
    evaluate_release_thresholds,
    load_fixed_public_cases,
    load_release_benchmark_cases,
    load_public_benchmark_baseline,
    recommend_default_feature_flags,
)


DEFAULT_REPORT_DIR = ROOT / "benchmarks" / "reports" / "fixed_public"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return f"```\n{df.to_string(index=False)}\n```"


def _json_default(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _report_payload(cases, summary, details, comparison, decision, threshold_eval) -> dict:
    best = summary.iloc[0].to_dict() if len(summary) else {}
    return {
        "suite": "fixed_public_hybrid",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cases": [
            {
                "name": case.name,
                "source": case.source,
                "primary_tier": case.primary_tier,
            }
            for case in cases
        ],
        "decision": decision,
        "threshold_evaluation": threshold_eval,
        "best_variant": best,
        "summary": summary.to_dict(orient="records"),
        "details": details.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
    }


def _build_markdown_report(cases, summary, details, comparison, decision, threshold_eval) -> str:
    best = summary.iloc[0] if len(summary) else None
    lines = [
        "# Fixed Public Benchmark Report",
        "",
        f"Generated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        "",
        "## Cases",
        "",
    ]
    for case in cases:
        lines.append(f"- `{case.name}`: {case.source} (primary tier: `{case.primary_tier}`)")

    lines.extend(
        [
            "",
            "## Primary Summary",
            "",
            _markdown_table(summary),
            "",
            "## Per-Case Details",
            "",
            _markdown_table(
                details[
                    [
                        "variant",
                        "scenario",
                        "tier",
                        "primary",
                        "rfe_enabled",
                        "shrinkage_awareness",
                        "precision",
                        "recall",
                        "F1",
                        "FDR",
                    ]
                ]
            ),
            "",
            "## Baseline Comparison",
            "",
            _markdown_table(
                comparison[
                    [
                        "variant",
                        "primary_score_baseline",
                        "primary_score_current",
                        "primary_score_delta",
                        "meets_gate",
                    ]
                ]
            ) if len(comparison) else "_No baseline comparison available_",
            "",
            "## Recommended Defaults",
            "",
            f"- `rfe_enabled={decision['rfe_enabled']}`: {decision['rfe']['reason']}",
            f"- `shrinkage_awareness={decision['shrinkage_awareness']}`: {decision['shrinkage']['reason']}",
            "",
            "## Acceptance Thresholds",
            "",
            f"- Pass: `{threshold_eval['meets_thresholds']}`",
            f"- Checks: `{threshold_eval['checks']}`",
        ]
    )

    if best is not None:
        lines.extend(
            [
                "",
                "## Best Variant",
                "",
                f"- `{best['variant']}` with `primary_score={best['primary_score']:.4f}`",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def write_benchmark_report(cases, summary, details, comparison, decision, threshold_eval, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / "public_benchmark_report.md"
    summary_csv_path = output_dir / "public_benchmark_summary.csv"
    details_csv_path = output_dir / "public_benchmark_details.csv"
    json_path = output_dir / "public_benchmark_report.json"

    markdown_path.write_text(
        _build_markdown_report(cases, summary, details, comparison, decision, threshold_eval),
        encoding="utf-8",
    )
    summary.to_csv(summary_csv_path, index=False)
    details.to_csv(details_csv_path, index=False)
    json_path.write_text(
        json.dumps(
            _report_payload(cases, summary, details, comparison, decision, threshold_eval),
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return {
        "markdown": markdown_path,
        "summary_csv": summary_csv_path,
        "details_csv": details_csv_path,
        "json": json_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the fixed public hybrid benchmark suite.")
    parser.add_argument("--check", action="store_true", help="Exit non-zero if the current summary falls below the pinned baseline.")
    parser.add_argument("--baseline", default=None, help="Optional path to a baseline JSON file.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Directory where the benchmark report bundle will be written.")
    parser.add_argument("--no-report", action="store_true", help="Skip writing Markdown/CSV/JSON benchmark artifacts.")
    args = parser.parse_args()

    cases = load_release_benchmark_cases()
    details, summary = compare_hybrid_variants(cases=cases)

    if len(details) == 0:
        print("No benchmark cases available.")
        return

    print("Fixed public cases:")
    for case in cases:
        print(f"  - {case.name} ({case.source})")

    print("\nPrimary-tier summary:")
    print(summary.to_string(index=False))

    print("\nPer-case details:")
    display_cols = [
        "variant",
        "scenario",
        "tier",
        "primary",
        "rfe_enabled",
        "shrinkage_awareness",
        "precision",
        "recall",
        "F1",
        "FDR",
    ]
    print(details[display_cols].to_string(index=False))

    baseline = load_public_benchmark_baseline(args.baseline)
    comparison = compare_summary_to_baseline(summary, baseline=baseline)
    decision = recommend_default_feature_flags(summary)
    recommended_variant = "baseline"
    if decision["rfe_enabled"] and decision["shrinkage_awareness"]:
        recommended_variant = "rfe+shrinkage"
    elif decision["rfe_enabled"]:
        recommended_variant = "rfe"
    elif decision["shrinkage_awareness"]:
        recommended_variant = "shrinkage"
    threshold_eval = evaluate_release_thresholds(summary, details, variant=recommended_variant)

    if len(comparison) > 0:
        print("\nPinned baseline comparison:")
        display_cols = [
            "variant",
            "primary_score_baseline",
            "primary_score_current",
            "primary_score_delta",
            "meets_gate",
        ]
        print(comparison[display_cols].to_string(index=False))

    print("\nDefault flag recommendation:")
    print(
        f"  - rfe_enabled={decision['rfe_enabled']} "
        f"({decision['rfe']['reason']})"
    )
    print(
        f"  - shrinkage_awareness={decision['shrinkage_awareness']} "
        f"({decision['shrinkage']['reason']})"
    )
    print(
        f"\nAcceptance thresholds ({recommended_variant}): "
        f"pass={threshold_eval['meets_thresholds']} "
        f"checks={threshold_eval['checks']}"
    )

    best = summary.iloc[0]
    print(f"\nBest variant: {best['variant']} (primary_score={best['primary_score']:.4f})")

    if not args.no_report:
        outputs = write_benchmark_report(
            cases=cases,
            summary=summary,
            details=details,
            comparison=comparison,
            decision=decision,
            threshold_eval=threshold_eval,
            output_dir=args.report_dir,
        )
        print("\nReport artifacts:")
        for label, path in outputs.items():
            print(f"  - {label}: {path}")

    if args.check and len(comparison) > 0 and (not comparison["meets_gate"].all() or not threshold_eval["meets_thresholds"]):
        raise SystemExit(1)


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    main()
