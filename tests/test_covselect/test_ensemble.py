import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from pharmacoml.covselect.ensemble import EnsembleResults, EnsembleScreener
from pharmacoml.covselect.engines import DEFAULT_ENSEMBLE_METHODS, FULL_ENSEMBLE_METHODS
import pharmacoml.covselect.ensemble as ensemble_module


def _all_available():
    return {method: True for method in FULL_ENSEMBLE_METHODS}


class TestEnsembleMethodResolution:
    def test_default_methods_exclude_deep_learning(self, monkeypatch):
        monkeypatch.setattr(ensemble_module, "check_engine_availability", _all_available)
        screener = EnsembleScreener()
        assert screener._resolve_methods(n_rows=50) == DEFAULT_ENSEMBLE_METHODS

    def test_warns_when_deep_learning_used_on_small_dataset(self, monkeypatch):
        monkeypatch.setattr(ensemble_module, "check_engine_availability", _all_available)
        screener = EnsembleScreener(methods=["lasso", "tabnet"], deep_learning_min_rows=250)
        with pytest.warns(UserWarning, match="small popPK datasets"):
            screener._resolve_methods(n_rows=50)


class TestUtilityGate:
    def test_utility_gate_rejects_weak_incremental_covariate(self):
        rng = np.random.RandomState(42)
        wt = rng.normal(70, 12, 150)
        age = rng.normal(55, 10, 150)
        cl = 5 + 0.08 * wt + rng.normal(0, 0.35, 150)

        fake_results = SimpleNamespace(
            parameter_names=["CL"],
            _cov_data=pd.DataFrame({"WT": wt, "AGE": age}),
            _ebe_data=pd.DataFrame({"CL": cl}),
            _encoding_map={"WT": ["WT"], "AGE": ["AGE"]},
        )
        summary = pd.DataFrame([
            {
                "parameter": "CL",
                "covariate": "WT",
                "mean_importance": 0.80,
                "significant": True,
                "final_significant": True,
                "functional_form": "linear",
            },
            {
                "parameter": "CL",
                "covariate": "AGE",
                "mean_importance": 0.20,
                "significant": True,
                "final_significant": True,
                "functional_form": "linear",
            },
        ])

        screener = EnsembleScreener(
            methods=["lasso"],
            use_significance_filter=False,
            min_delta_r2=0.05,
            min_relative_importance=0.10,
        )
        annotated = screener._annotate_method_summary(summary, fake_results, "lasso")
        annotated = annotated.set_index("covariate")

        assert annotated.loc["WT", "utility_pass"]
        assert annotated.loc["WT", "ensemble_vote"]
        assert annotated.loc["AGE", "relative_importance_pass"]
        assert not annotated.loc["AGE", "utility_pass"]
        assert not annotated.loc["AGE", "ensemble_vote"]


class TestConsensusSummary:
    def test_consensus_uses_filtered_votes(self):
        method_a = pd.DataFrame([
            {
                "parameter": "CL",
                "covariate": "WT",
                "mean_importance": 0.80,
                "functional_form": "linear",
                "importance_rank": 1,
                "relative_importance": 1.0,
                "utility_delta_r2": 0.20,
                "ensemble_vote": True,
            },
            {
                "parameter": "CL",
                "covariate": "AGE",
                "mean_importance": 0.20,
                "functional_form": "linear",
                "importance_rank": 2,
                "relative_importance": 0.25,
                "utility_delta_r2": 0.00,
                "ensemble_vote": False,
            },
        ])
        method_b = pd.DataFrame([
            {
                "parameter": "CL",
                "covariate": "WT",
                "mean_importance": 0.75,
                "functional_form": "linear",
                "importance_rank": 1,
                "relative_importance": 1.0,
                "utility_delta_r2": 0.18,
                "ensemble_vote": True,
            },
            {
                "parameter": "CL",
                "covariate": "AGE",
                "mean_importance": 0.25,
                "functional_form": "linear",
                "importance_rank": 2,
                "relative_importance": 0.33,
                "utility_delta_r2": 0.00,
                "ensemble_vote": False,
            },
        ])

        results = EnsembleResults(
            method_results={},
            method_summaries={"a": method_a, "b": method_b},
            min_agreement=2,
            min_relative_importance=0.10,
        )
        consensus = results.consensus_summary().set_index("covariate")

        assert consensus.loc["WT", "consensus"]
        assert consensus.loc["WT", "n_methods_significant"] == 2
        assert not consensus.loc["AGE", "consensus"]
        assert consensus.loc["AGE", "n_methods_significant"] == 0
