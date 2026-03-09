import pandas as pd

from benchmarks.run_public_benchmarks import write_benchmark_report
from pharmacoml.covselect.public_benchmarks import PublicBenchmarkCase


def test_write_benchmark_report_creates_markdown_csv_and_json(tmp_path):
    cases = [
        PublicBenchmarkCase(
            name="toy_case",
            ebes=pd.DataFrame({"CL": [1.0, 1.2]}),
            covariates=pd.DataFrame({"WT": [70.0, 80.0]}),
            truth={("CL", "WT")},
            primary_tier="candidate",
            source="toy source",
        )
    ]
    summary = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "cases": 1,
                "mean_precision": 1.0,
                "mean_recall": 1.0,
                "mean_F1": 1.0,
                "mean_FDR": 0.0,
                "primary_score": 1.0,
                "rfe_enabled": False,
                "shrinkage_awareness": False,
            }
        ]
    )
    details = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "scenario": "toy_case",
                "tier": "candidate",
                "primary": True,
                "rfe_enabled": False,
                "shrinkage_awareness": False,
                "precision": 1.0,
                "recall": 1.0,
                "F1": 1.0,
                "FDR": 0.0,
            }
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "primary_score_baseline": 1.0,
                "primary_score_current": 1.0,
                "primary_score_delta": 0.0,
                "meets_gate": True,
            }
        ]
    )
    decision = {
        "rfe_enabled": False,
        "shrinkage_awareness": False,
        "rfe": {"reason": "toy rfe"},
        "shrinkage": {"reason": "toy shrinkage"},
    }
    threshold_eval = {
        "meets_thresholds": True,
        "checks": {"mean_precision_ok": True},
    }

    outputs = write_benchmark_report(
        cases=cases,
        summary=summary,
        details=details,
        comparison=comparison,
        decision=decision,
        threshold_eval=threshold_eval,
        output_dir=tmp_path,
    )

    assert outputs["markdown"].exists()
    assert outputs["summary_csv"].exists()
    assert outputs["details_csv"].exists()
    assert outputs["json"].exists()

    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert "# Fixed Public Benchmark Report" in markdown
    assert "toy_case" in markdown
    assert "baseline" in markdown
