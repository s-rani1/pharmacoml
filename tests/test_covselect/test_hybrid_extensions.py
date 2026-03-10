import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from tests.test_covselect._helpers import load_pheno_case_or_skip, pheno_case_available

from pharmacoml.covselect import HybridResults, HybridScreener, SCMBridge, STGScreener, SymbolicStructureScreener
from pharmacoml.covselect.benchmark import BenchmarkScenario, BenchmarkSuite
from pharmacoml.covselect.public_benchmarks import (
    load_age_pma_distinct_case,
    compare_hybrid_variants,
    compare_summary_to_baseline,
    evaluate_release_thresholds,
    load_asiimwe_style_cases,
    load_high_shrinkage_case,
    load_interaction_screening_case,
    load_pheno_case,
    load_public_benchmark_baseline,
    load_release_benchmark_cases,
    load_shapcov_style_cases,
    load_fixed_public_cases,
    _resolve_pharmpy_example_base,
    recommend_default_feature_flags,
)
from pharmacoml.covselect.screener import CovariateScreener
from pharmacoml.covselect.shapcov import ShapCovScreener


def make_reference_data(n=160, seed=42):
    rng = np.random.RandomState(seed)
    wt = rng.normal(70, 12, n)
    age = rng.normal(55, 10, n)
    smk = rng.binomial(1, 0.35, n).astype(float)
    bsa = wt / 40.0 + rng.normal(0, 0.02, n)

    cl = 5.0 * (wt / np.median(wt)) ** 0.75 * (1 + 0.2 * smk) * np.exp(rng.normal(0, 0.20, n))
    ebes = pd.DataFrame({"CL": cl})
    covs = pd.DataFrame({"WT": wt, "AGE": age, "SMK": smk, "BSA": bsa})
    return ebes, covs


def make_small_sample_secondary_effect(seed=7):
    rng = np.random.RandomState(seed)
    n = 72
    wt = rng.normal(70, 10, n)
    rare = rng.binomial(1, 0.15, n).astype(float)
    cl = 5.0 * (wt / np.median(wt)) ** 0.75 * np.exp(rng.normal(0, 0.18, n))
    vc = 50.0 * (wt / np.median(wt)) ** 1.0 * (1 + 0.22 * rare) * np.exp(rng.normal(0, 0.15, n))
    ebes = pd.DataFrame({"CL": cl, "VC": vc})
    covs = pd.DataFrame({"WT": wt, "RARE": rare})
    return ebes, covs


def test_aalasso_finds_weight_signal():
    ebes, covs = make_reference_data()
    results = CovariateScreener(method="aalasso", n_bootstrap=6, random_state=42).fit(ebes, covs)
    summary = results.summary()
    cl_df = summary[summary["parameter"] == "CL"].reset_index(drop=True)

    assert cl_df.iloc[0]["covariate"] == "WT"


def test_scm_bridge_prunes_noise_covariates():
    ebes, covs = make_reference_data()
    candidate_table = pd.DataFrame(
        [
            {"parameter": "CL", "covariate": "WT", "functional_form": "power", "tier": "candidate"},
            {"parameter": "CL", "covariate": "AGE", "functional_form": "linear", "tier": "candidate"},
            {"parameter": "CL", "covariate": "BSA", "functional_form": "linear", "tier": "candidate"},
        ]
    )

    bridge = SCMBridge(max_terms=2)
    results = bridge.fit(ebes, covs, candidate_table=candidate_table)
    selected = set(results.selected_covariates()["covariate"])

    assert "WT" in selected
    assert "AGE" not in selected


def test_hybrid_screener_exposes_scm_outputs():
    ebes, covs = make_reference_data()
    report = HybridScreener(
        boosting_method="random_forest",
        penalized_method="aalasso",
        include_scm=True,
        include_traditional=True,
        include_stg=False,
        n_bootstrap=8,
        run_permutation=False,
    ).fit(ebes, covs)

    scm_covs = report.scm_covariates()
    assert len(scm_covs) >= 1
    assert "WT" in set(scm_covs["covariate"])
    assert set(report.confirmed_covariates()["covariate"]) == set(scm_covs["covariate"])


def test_rescue_candidates_recovers_parameter_specific_secondary_effect():
    ebes, covs = make_small_sample_secondary_effect()
    summary = pd.DataFrame(
        [
            {
                "parameter": "CL",
                "covariate": "WT",
                "functional_form": "power",
                "shapcov_importance": 0.9,
                "shapcov_score": 1.0,
                "shapcov_selected": True,
                "penalized_importance": 0.8,
                "penalized_score": 0.9,
                "penalized_selected": True,
                "stg_importance": 0.0,
                "stg_score": 0.0,
                "stg_selected": False,
                "traditional_selected": True,
                "support_count": 3,
                "combined_score": 1.0,
                "tier": "core",
                "tier_rank": 0,
                "scm_selected": True,
                "support_label": "boosting+penalized+traditional+scm",
            },
            {
                "parameter": "CL",
                "covariate": "RARE",
                "functional_form": "categorical",
                "shapcov_importance": 0.1,
                "shapcov_score": 0.06,
                "shapcov_selected": True,
                "penalized_importance": 0.0,
                "penalized_score": 0.0,
                "penalized_selected": False,
                "stg_importance": 0.0,
                "stg_score": 0.0,
                "stg_selected": False,
                "traditional_selected": False,
                "support_count": 1,
                "combined_score": 0.06,
                "tier": "rejected",
                "tier_rank": 3,
                "scm_selected": False,
                "support_label": "boosting",
            },
            {
                "parameter": "VC",
                "covariate": "WT",
                "functional_form": "power",
                "shapcov_importance": 0.9,
                "shapcov_score": 1.0,
                "shapcov_selected": True,
                "penalized_importance": 0.8,
                "penalized_score": 0.9,
                "penalized_selected": True,
                "stg_importance": 0.0,
                "stg_score": 0.0,
                "stg_selected": False,
                "traditional_selected": True,
                "support_count": 3,
                "combined_score": 1.0,
                "tier": "core",
                "tier_rank": 0,
                "scm_selected": True,
                "support_label": "boosting+penalized+traditional+scm",
            },
            {
                "parameter": "VC",
                "covariate": "RARE",
                "functional_form": "categorical",
                "shapcov_importance": 0.1,
                "shapcov_score": 0.05,
                "shapcov_selected": True,
                "penalized_importance": 0.0,
                "penalized_score": 0.0,
                "penalized_selected": False,
                "stg_importance": 0.0,
                "stg_score": 0.0,
                "stg_selected": False,
                "traditional_selected": False,
                "support_count": 1,
                "combined_score": 0.05,
                "tier": "rejected",
                "tier_rank": 3,
                "scm_selected": False,
                "support_label": "boosting",
            },
        ]
    )

    screener = HybridScreener(
        boosting_method="random_forest",
        penalized_method="aalasso",
        include_scm=True,
        include_traditional=True,
        enable_rescue=True,
        rescue_score_floor=0.02,
        small_sample_n=100,
        small_sample_rescue_alpha=0.2,
        small_sample_rescue_min_delta_aic=0.1,
    )
    rescued = screener._apply_rescue_candidates(summary, ebes=ebes, covariates=covs)
    rescued = rescued.set_index(["parameter", "covariate"])

    assert not rescued.loc[("CL", "RARE"), "rescued_confirmed"]
    assert rescued.loc[("VC", "RARE"), "rescued_confirmed"]


def test_rescue_candidates_require_parameter_anchor_for_rescue_only_signal():
    rng = np.random.RandomState(17)
    n = 80
    wt = rng.normal(70, 10, n)
    age = rng.normal(55, 8, n)
    ebes = pd.DataFrame({
        "CL": 5.0 * (wt / np.median(wt)) ** 0.75 * np.exp(rng.normal(0, 0.1, n)),
        "KA": 1.0 + 0.25 * (wt - np.median(wt)) / np.std(wt) + rng.normal(0, 0.05, n),
    })
    covs = pd.DataFrame({"WT": wt, "AGE": age})
    summary = pd.DataFrame(
        [
            {
                "parameter": "CL",
                "covariate": "WT",
                "functional_form": "power",
                "shapcov_selected": True,
                "shapcov_importance": 0.45,
                "stg_selected": False,
                "traditional_selected": False,
                "support_count": 1,
                "combined_score": 0.45,
                "tier": "candidate",
                "tier_rank": 1,
                "scm_selected": True,
                "support_label": "scm",
                "screening_suppressed": False,
            },
            {
                "parameter": "KA",
                "covariate": "WT",
                "functional_form": "linear",
                "shapcov_selected": True,
                "shapcov_importance": 0.35,
                "stg_selected": False,
                "traditional_selected": False,
                "support_count": 0,
                "combined_score": 0.35,
                "tier": "rejected",
                "tier_rank": 3,
                "scm_selected": False,
                "support_label": "none",
                "screening_suppressed": False,
            },
        ]
    )

    screener = HybridScreener(enable_rescue=True, rescue_score_floor=0.02)
    rescued = screener._apply_rescue_candidates(summary, ebes=ebes, covariates=covs)
    ka_row = rescued.set_index(["parameter", "covariate"]).loc[("KA", "WT")]

    assert ka_row["tier"] == "rejected"
    assert ka_row["confirmation_status"] == "unconfirmed"


def test_prune_redundant_candidates_demotes_rescue_only_proxy():
    covs = pd.DataFrame(
        {
            "WT": [60, 65, 70, 75, 80, 85],
            "NOISE": [61, 66, 71, 76, 81, 86],
            "AGE": [40, 45, 50, 55, 60, 65],
        }
    )
    summary = pd.DataFrame(
        [
            {
                "parameter": "CL",
                "covariate": "WT",
                "tier": "candidate",
                "tier_rank": 1,
                "support_count": 2,
                "support_requirement": 2,
                "scm_selected": True,
                "traditional_selected": False,
                "penalized_selected": True,
                "score_rescue_candidate": True,
                "combined_score_adjusted": 0.55,
                "shapcov_score": 0.60,
            },
            {
                "parameter": "CL",
                "covariate": "NOISE",
                "tier": "candidate",
                "tier_rank": 1,
                "support_count": 0,
                "support_requirement": 2,
                "scm_selected": False,
                "traditional_selected": False,
                "penalized_selected": False,
                "score_rescue_candidate": True,
                "combined_score_adjusted": 0.75,
                "shapcov_score": 0.90,
            },
            {
                "parameter": "CL",
                "covariate": "AGE",
                "tier": "rejected",
                "tier_rank": 3,
                "support_count": 0,
                "support_requirement": 2,
                "scm_selected": False,
                "traditional_selected": False,
                "penalized_selected": False,
                "score_rescue_candidate": False,
                "combined_score_adjusted": 0.10,
                "shapcov_score": 0.10,
            },
        ]
    )

    screener = HybridScreener(rescue_redundancy_threshold=0.65)
    pruned = screener._prune_redundant_candidates(summary, covariates=covs)
    noise_row = pruned.set_index(["parameter", "covariate"]).loc[("CL", "NOISE")]

    assert noise_row["tier"] == "proxy"
    assert noise_row["proxy_for"] == "WT"


def test_prune_redundant_candidates_drops_unanchored_rescue_only_parameter():
    covs = pd.DataFrame(
        {
            "WT": [60, 65, 70, 75, 80, 85],
            "AGE": [40, 45, 50, 55, 60, 65],
        }
    )
    summary = pd.DataFrame(
        [
            {
                "parameter": "KA",
                "covariate": "WT",
                "tier": "candidate",
                "tier_rank": 1,
                "support_count": 0,
                "support_requirement": 2,
                "scm_selected": False,
                "traditional_selected": False,
                "penalized_selected": False,
                "score_rescue_candidate": True,
                "combined_score_adjusted": 0.70,
                "shapcov_score": 0.90,
            },
            {
                "parameter": "KA",
                "covariate": "AGE",
                "tier": "candidate",
                "tier_rank": 1,
                "support_count": 0,
                "support_requirement": 2,
                "scm_selected": False,
                "traditional_selected": False,
                "penalized_selected": False,
                "score_rescue_candidate": True,
                "combined_score_adjusted": 0.65,
                "shapcov_score": 0.85,
            },
        ]
    )

    screener = HybridScreener()
    pruned = screener._prune_redundant_candidates(summary, covariates=covs)

    assert set(pruned["tier"]) == {"rejected"}


def test_stg_screener_smoke():
    torch = pytest.importorskip("torch")
    assert torch is not None

    ebes, covs = make_reference_data(n=120)
    summary = STGScreener(n_bootstrap=3, random_state=42).fit(ebes, covs)
    assert {"parameter", "covariate", "final_significant", "mean_importance"}.issubset(summary.columns)


def test_symbolic_structure_screener_returns_formula():
    ebes, covs = make_reference_data(n=120)
    summary = SymbolicStructureScreener().fit(ebes, covs)
    wt_row = summary[(summary["parameter"] == "CL") & (summary["covariate"] == "WT")].iloc[0]
    assert wt_row["symbolic_form"] in {"linear", "power", "exponential", "quadratic"}
    assert isinstance(wt_row["symbolic_expression"], str)


def test_hybrid_screener_records_symbolic_backend():
    ebes, covs = make_reference_data(n=120)
    report = HybridScreener(
        boosting_method="random_forest",
        penalized_method="aalasso",
        include_symbolic=True,
        symbolic_backend="basis",
        include_scm=False,
        include_traditional=False,
        n_bootstrap=4,
        run_permutation=False,
    ).fit(ebes, covs)

    summary = report.summary()
    assert "symbolic_backend" in summary.columns
    assert (summary["symbolic_backend"] == "basis").any()


def test_hybrid_screener_omits_symbolic_backend_when_symbolic_disabled():
    ebes, covs = make_reference_data(n=120)
    report = HybridScreener(
        boosting_method="random_forest",
        penalized_method="aalasso",
        include_symbolic=False,
        include_scm=False,
        include_traditional=False,
        n_bootstrap=4,
        run_permutation=False,
    ).fit(ebes, covs)

    assert "symbolic_backend" not in report.summary().columns


@pytest.mark.parametrize("backend", ["gplearn", "pysr"])
def test_symbolic_structure_optional_backends_raise_clear_importerror(monkeypatch, backend):
    monkeypatch.setattr("pharmacoml.covselect.symbolic.find_spec", lambda name: None)

    ebes, covs = make_reference_data(n=120)
    screener = SymbolicStructureScreener(symbolic_backend=backend)

    with pytest.raises(ImportError, match=backend):
        screener.fit(ebes, covs)


def test_hybrid_results_interaction_accessor_filters_interaction_rows():
    summary = pd.DataFrame(
        [
            {
                "parameter": "CL",
                "covariate": "WT",
                "functional_form": "power",
                "combined_score": 0.8,
                "tier": "core",
                "tier_rank": 0,
            },
            {
                "parameter": "CL",
                "covariate": "WT__x__AGE",
                "functional_form": "interaction",
                "combined_score": 0.3,
                "tier": "candidate",
                "tier_rank": 1,
            },
        ]
    )
    report = HybridResults(summary, pd.DataFrame(), artifacts=None)

    interactions = report.interaction_covariates()
    assert len(interactions) == 1
    assert interactions.iloc[0]["covariate"] == "WT__x__AGE"


def test_benchmark_suite_supports_hybrid_and_scm():
    scenario = BenchmarkScenario(
        name="Tiny hybrid benchmark",
        n_subjects=120,
        true_covariates={
            ("CL", "WT"): {"form": "power", "effect": 0.75},
            ("CL", "SEX"): {"form": "categorical", "effect": -0.20},
        },
        n_noise_covariates=2,
        eta_sd={"CL": 0.20},
        seed=7,
    )

    suite = BenchmarkSuite(
        methods=["hybrid", "scm_bridge", "aalasso"],
        include_traditional=False,
        include_significance_filter=False,
        n_bootstrap=5,
        random_state=42,
        hybrid_kwargs={
            "boosting_method": "random_forest",
            "penalized_method": "aalasso",
            "include_scm": True,
            "include_traditional": False,
        },
    )
    results = suite.run(scenarios=[scenario])

    assert "Hybrid" in set(results["method"])
    assert "SCMBridge" in set(results["method"])


def test_shapcov_rfe_marks_retained_covariates():
    ebes, covs = make_reference_data(n=140)
    covs["NOISE_1"] = np.linspace(-1, 1, len(covs))
    covs["NOISE_2"] = np.random.RandomState(3).normal(size=len(covs))

    summary = ShapCovScreener(
        method="random_forest",
        n_bootstrap=4,
        random_state=42,
        rfe_enabled=True,
        rfe_drop_fraction=0.40,
        rfe_max_rounds=2,
    ).fit(ebes, covs)

    assert {"rfe_retained", "rfe_round"}.issubset(summary.columns)
    assert not summary["rfe_retained"].all()


def test_hybrid_shrinkage_awareness_relaxes_support_requirement():
    screener = HybridScreener(shrinkage_awareness=True, shrinkage_candidate_floor=0.35)
    shap_summary = pd.DataFrame(
        [
            {
                "parameter": "ETA_CL",
                "covariate": "WT",
                "functional_form": "power",
                "mean_importance": 0.5,
                "standardized_importance": 0.6,
                "stability_frequency": 0.5,
                "cv_r2": 0.05,
                "final_significant": True,
                "corr_filtered": False,
                "perm_p_value": np.nan,
                "perm_q_value": np.nan,
            }
        ]
    )
    penalized_summary = pd.DataFrame(
        [
            {
                "parameter": "ETA_CL",
                "covariate": "WT",
                "mean_importance": 0.0,
                "standardized_importance": 0.0,
                "stability_frequency": 0.0,
                "cv_r2": 0.05,
                "final_significant": False,
                "corr_filtered": False,
                "perm_p_value": np.nan,
                "perm_q_value": np.nan,
            }
        ]
    )

    merged = screener._merge_stage_outputs(
        shap_summary=shap_summary,
        penalized_summary=penalized_summary,
        stg_summary=None,
        traditional_summary=None,
        parameter_profiles={"ETA_CL": {"shrinkage_proxy": 0.8, "low_information": True}},
    )

    row = merged.iloc[0]
    assert row["support_requirement"] == 1
    assert row["combined_score_adjusted"] > row["combined_score"]


def test_user_supplied_shrinkage_suppresses_unreliable_parameter_screening():
    case = load_high_shrinkage_case()
    baseline = HybridScreener(
        boosting_method="random_forest",
        penalized_method="aalasso",
        include_scm=False,
        include_traditional=False,
        rfe_enabled=False,
        n_bootstrap=4,
        run_permutation=False,
        shrinkage_awareness=False,
    ).fit(
        case.ebes,
        case.covariates,
        parameter_shrinkage=case.parameter_shrinkage,
    )
    aware = HybridScreener(
        boosting_method="random_forest",
        penalized_method="aalasso",
        include_scm=False,
        include_traditional=False,
        rfe_enabled=False,
        n_bootstrap=4,
        run_permutation=False,
        shrinkage_awareness=True,
    ).fit(
        case.ebes,
        case.covariates,
        parameter_shrinkage=case.parameter_shrinkage,
    )

    baseline_pairs = set(map(tuple, baseline.candidate_covariates()[["parameter", "covariate"]].itertuples(index=False, name=None)))
    aware_pairs = set(map(tuple, aware.candidate_covariates()[["parameter", "covariate"]].itertuples(index=False, name=None)))

    assert ("CL", "WT") in aware_pairs
    assert any(param == "V" for param, _ in baseline_pairs)
    assert not any(param == "V" for param, _ in aware_pairs)


def test_public_benchmark_suite_runs_on_pheno_case():
    details, summary = compare_hybrid_variants(
        variants={"baseline": {"rfe_enabled": False}, "rfe": {"rfe_enabled": True}},
        cases=[load_pheno_case_or_skip()],
    )

    assert not details.empty
    assert not summary.empty
    assert set(summary["variant"]) == {"baseline", "rfe"}


def test_resolve_pharmpy_example_base_uses_installed_package_path():
    if not pheno_case_available():
        pytest.skip("pharmpy example models are unavailable in this environment.")
    base = _resolve_pharmpy_example_base()
    assert isinstance(base, Path)
    assert base.exists()
    assert (base / "pheno.dta").exists()


def test_load_fixed_public_cases_skips_missing_pheno(monkeypatch):
    stub_case = load_high_shrinkage_case()

    monkeypatch.setitem(
        load_fixed_public_cases.__globals__,
        "load_pheno_case",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing pheno")),
    )
    monkeypatch.setitem(load_fixed_public_cases.__globals__, "load_eleveld_case", lambda: stub_case)
    monkeypatch.setitem(load_fixed_public_cases.__globals__, "load_ggpmx_theophylline_case", lambda: stub_case)

    cases = load_fixed_public_cases()
    names = {case.name for case in cases}

    assert "pheno" not in names
    assert names == {"high_shrinkage_user_input"}


def test_release_benchmark_suite_includes_paper_cases():
    cases = load_release_benchmark_cases(include_optional_kekic=False)
    names = {case.name for case in cases}

    assert "high_shrinkage_user_input" in names
    assert "age_pma_distinct" in names
    assert "interaction_xor_screening" in names
    assert "asiimwe_correlated_small_n" in names
    assert "shapcov_collinear" in names
    if pheno_case_available():
        assert "pheno" in names


def test_paper_style_case_loaders_return_cases():
    assert len(load_asiimwe_style_cases()) >= 1
    assert len(load_shapcov_style_cases()) >= 1


def test_recommend_default_feature_flags_requires_actual_gain():
    summary = pd.DataFrame(
        [
            {"variant": "baseline", "primary_score": 0.80, "mean_precision": 0.90, "mean_F1": 0.70, "mean_FDR": 0.10, "rfe_enabled": False, "shrinkage_awareness": False},
            {"variant": "rfe", "primary_score": 0.80, "mean_precision": 0.90, "mean_F1": 0.70, "mean_FDR": 0.10, "rfe_enabled": True, "shrinkage_awareness": False},
            {"variant": "shrinkage", "primary_score": 0.79, "mean_precision": 0.90, "mean_F1": 0.69, "mean_FDR": 0.10, "rfe_enabled": False, "shrinkage_awareness": True},
            {"variant": "rfe+shrinkage", "primary_score": 0.80, "mean_precision": 0.89, "mean_F1": 0.70, "mean_FDR": 0.10, "rfe_enabled": True, "shrinkage_awareness": True},
        ]
    )

    decision = recommend_default_feature_flags(summary, min_primary_gain=0.001)
    assert not decision["rfe_enabled"]
    assert not decision["shrinkage_awareness"]


def test_high_shrinkage_benchmark_case_rewards_explicit_shrinkage_awareness():
    details, summary = compare_hybrid_variants(
        variants={
            "baseline": {
                "boosting_method": "random_forest",
                "penalized_method": "aalasso",
                "include_traditional": False,
                "include_scm": False,
                "rfe_enabled": False,
                "shrinkage_awareness": False,
                "n_bootstrap": 4,
            },
            "shrinkage": {
                "boosting_method": "random_forest",
                "penalized_method": "aalasso",
                "include_traditional": False,
                "include_scm": False,
                "rfe_enabled": False,
                "shrinkage_awareness": True,
                "n_bootstrap": 4,
            },
        },
        cases=[load_high_shrinkage_case()],
    )

    assert not details.empty
    score_map = dict(zip(summary["variant"], summary["primary_score"]))
    assert score_map["shrinkage"] > score_map["baseline"]


def test_age_pma_benchmark_case_keeps_both_covariates():
    case = load_age_pma_distinct_case()
    hybrid_kwargs = {
        "include_traditional": True,
        "include_scm": False,
        "shrinkage_awareness": True,
    }
    hybrid_kwargs.update(case.hybrid_kwargs or {})
    report = HybridScreener(**hybrid_kwargs).fit(case.ebes, case.covariates)

    selected = set(map(tuple, report.candidate_covariates()[["parameter", "covariate"]].itertuples(index=False, name=None)))
    assert ("CL", "AGE") in selected
    assert ("CL", "PMA") in selected


def test_interaction_benchmark_case_recovers_xor_term():
    case = load_interaction_screening_case()
    hybrid_kwargs = {
        "include_traditional": False,
        "include_scm": False,
        "shrinkage_awareness": True,
    }
    hybrid_kwargs.update(case.hybrid_kwargs or {})
    report = HybridScreener(**hybrid_kwargs).fit(case.ebes, case.covariates)

    selected = set(map(tuple, report.candidate_covariates()[["parameter", "covariate"]].itertuples(index=False, name=None)))
    assert ("CL", "COPD__xor__SMK") in selected


def test_compare_summary_to_baseline_smoke():
    baseline = load_public_benchmark_baseline()
    summary = pd.DataFrame(baseline["summary"])
    comparison = compare_summary_to_baseline(summary, baseline=baseline)

    assert not comparison.empty
    assert comparison["meets_gate"].all()


def test_evaluate_release_thresholds_smoke():
    details, summary = compare_hybrid_variants(
        variants={"baseline": {"rfe_enabled": False}},
        cases=[load_pheno_case_or_skip()],
    )
    evaluation = evaluate_release_thresholds(summary, details, variant="baseline")

    assert isinstance(evaluation["meets_thresholds"], bool)
