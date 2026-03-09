import numpy as np
import pandas as pd
import pytest

import pharmacoml.covselect.experimental.consensus as consensus_module
from pharmacoml.covselect.experimental import MultiModelConsensusScreener
from pharmacoml.covselect.experimental.consensus import MultiModelConsensusResults
from pharmacoml.covselect.hybrid import HybridScreener
from pharmacoml.covselect.public_benchmarks import (
    PublicBenchmarkCase,
    compare_hybrid_vs_multimodel,
    compare_multimodel_variants,
    summarize_hybrid_vs_multimodel,
)


def make_reference_data(n=140, seed=42):
    rng = np.random.RandomState(seed)
    wt = rng.normal(70, 12, n)
    age = rng.normal(55, 10, n)
    alb = rng.normal(4.0, 0.4, n)
    smk = rng.binomial(1, 0.35, n).astype(float)
    cl = 5.0 * (wt / np.median(wt)) ** 0.8 * (1 + 0.18 * smk) * np.exp(rng.normal(0, 0.12, n))
    ebes = pd.DataFrame({"CL": cl})
    covs = pd.DataFrame({"WT": wt, "AGE": age, "ALB": alb, "SMK": smk})
    return ebes, covs


def _available_models():
    return {
        "random_forest": True,
        "extra_trees": True,
        "gradient_boosting": True,
        "lasso": True,
        "elastic_net": True,
        "aalasso": True,
        "ridge": True,
        "stg": False,
        "mlp": True,
        "tabnet": False,
        "xgboost": False,
        "lightgbm": False,
        "catboost": False,
        "adaptive_lasso": True,
    }


def test_default_models_resolve_to_curated_scikit_core(monkeypatch):
    monkeypatch.setattr(consensus_module, "check_engine_availability", _available_models)

    screener = MultiModelConsensusScreener()
    models = screener._resolve_models(n_rows=120)

    assert models == [
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "lasso",
        "aalasso",
    ]


def test_consensus_fit_recovers_weight_signal():
    ebes, covs = make_reference_data()
    report = MultiModelConsensusScreener(
        models=["random_forest", "extra_trees", "lasso", "ridge"],
        top_k=2,
        n_bootstrap=4,
        use_significance_filter=False,
        min_model_frequency=0.50,
        min_family_support=2,
        random_state=42,
    ).fit(ebes, covs)

    summary = report.consensus_summary().set_index("covariate")

    assert summary.loc["WT", "consensus"]
    assert summary.loc["WT", "n_models_selected"] >= 3
    assert summary.loc["WT", "family_support"] == 2
    assert summary.loc["WT", "selection_frequency"] >= 0.75


def test_family_summary_and_compare_with_hybrid():
    ebes, covs = make_reference_data()
    consensus = MultiModelConsensusScreener(
        models=["random_forest", "extra_trees", "lasso", "ridge"],
        top_k=2,
        n_bootstrap=4,
        use_significance_filter=False,
        min_model_frequency=0.50,
        min_family_support=2,
        random_state=42,
    ).fit(ebes, covs)
    family_summary = consensus.family_summary()

    assert {"parameter", "covariate", "family", "family_selection_frequency"}.issubset(
        family_summary.columns
    )
    assert "WT" in set(family_summary["covariate"])

    hybrid_report = HybridScreener(
        boosting_method="random_forest",
        penalized_method="aalasso",
        include_traditional=False,
        include_scm=False,
        include_stg=False,
        n_bootstrap=4,
        run_permutation=False,
        random_state=42,
    ).fit(ebes, covs)
    comparison = consensus.compare_with_hybrid(hybrid_report, hybrid_tier="candidate")

    wt_row = comparison[comparison["covariate"] == "WT"].iloc[0]
    assert wt_row["consensus_selected"]
    assert wt_row["hybrid_selected"]
    assert wt_row["agreement"]


def test_consensus_applies_mean_relative_importance_floor():
    summaries = {
        "tree_a": pd.DataFrame(
            [
                {
                    "parameter": "CL",
                    "covariate": "WT",
                    "family": "bagging",
                    "importance_rank": 1,
                    "relative_importance": 1.0,
                    "standardized_importance": 1.0,
                    "stability_frequency": 1.0,
                    "selected": True,
                },
                {
                    "parameter": "CL",
                    "covariate": "ASPHYXIA",
                    "family": "bagging",
                    "importance_rank": 2,
                    "relative_importance": 0.03,
                    "standardized_importance": 0.0,
                    "stability_frequency": 0.2,
                    "selected": True,
                },
            ]
        ),
        "linear_a": pd.DataFrame(
            [
                {
                    "parameter": "CL",
                    "covariate": "WT",
                    "family": "linear",
                    "importance_rank": 1,
                    "relative_importance": 1.0,
                    "standardized_importance": 1.0,
                    "stability_frequency": 1.0,
                    "selected": True,
                },
                {
                    "parameter": "CL",
                    "covariate": "ASPHYXIA",
                    "family": "linear",
                    "importance_rank": 2,
                    "relative_importance": 0.04,
                    "standardized_importance": 0.0,
                    "stability_frequency": 0.2,
                    "selected": True,
                },
            ]
        ),
    }

    report = MultiModelConsensusResults(
        method_results={},
        method_summaries=summaries,
        models=["tree_a", "linear_a"],
        top_k=2,
        min_model_frequency=0.5,
        min_family_support=2,
        min_mean_relative_importance=0.05,
        max_consensus_per_parameter=3,
    )
    summary = report.consensus_summary().set_index("covariate")

    assert summary.loc["WT", "consensus"]
    assert not summary.loc["ASPHYXIA", "consensus"]
    assert not summary.loc["ASPHYXIA", "importance_threshold_pass"]


def test_public_benchmark_comparison_smoke():
    ebes, covs = make_reference_data()
    case = PublicBenchmarkCase(
        name="synthetic_reference",
        ebes=ebes,
        covariates=covs,
        truth={("CL", "WT"), ("CL", "SMK")},
        truth_mode="parameter",
        primary_tier="candidate",
        source="synthetic test",
    )

    details, summary = compare_hybrid_vs_multimodel(
        cases=[case],
        hybrid_kwargs={
            "boosting_method": "random_forest",
            "penalized_method": "aalasso",
            "include_traditional": False,
            "include_scm": False,
            "include_stg": False,
            "n_bootstrap": 4,
            "run_permutation": False,
        },
        multimodel_kwargs={
            "models": ["random_forest", "extra_trees", "lasso", "ridge"],
            "top_k": 2,
            "n_bootstrap": 4,
            "use_significance_filter": False,
            "min_model_frequency": 0.50,
            "min_family_support": 2,
            "run_permutation": False,
        },
    )

    assert set(summary["workflow"]) == {"hybrid", "multimodel_consensus"}
    comparison = summarize_hybrid_vs_multimodel(details)
    assert len(comparison) == 1
    assert comparison.iloc[0]["scenario"] == "synthetic_reference"


def test_multimodel_variant_comparison_smoke():
    ebes, covs = make_reference_data()
    case = PublicBenchmarkCase(
        name="synthetic_reference",
        ebes=ebes,
        covariates=covs,
        truth={("CL", "WT"), ("CL", "SMK")},
        truth_mode="parameter",
        primary_tier="candidate",
        source="synthetic test",
    )
    details, summary = compare_multimodel_variants(
        cases=[case],
        variants={
            "strict": {
                "models": ["random_forest", "extra_trees", "lasso", "ridge"],
                "top_k": 2,
                "n_bootstrap": 4,
                "use_significance_filter": False,
                "min_model_frequency": 0.50,
                "min_family_support": 2,
                "run_permutation": False,
            },
            "relaxed": {
                "models": ["random_forest", "extra_trees", "lasso", "ridge"],
                "top_k": 3,
                "n_bootstrap": 4,
                "use_significance_filter": False,
                "min_model_frequency": 0.25,
                "min_family_support": 1,
                "run_permutation": False,
            },
        },
    )

    assert set(summary["variant"]) == {"strict", "relaxed"}
    assert set(details["variant"]) == {"strict", "relaxed"}
