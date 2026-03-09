import numpy as np
import pandas as pd

from pharmacoml.covselect.hybrid import HybridResults, HybridScreener


def test_assign_proxy_groups_marks_correlated_alternatives():
    rng = np.random.RandomState(42)
    wt = rng.normal(70, 10, 100)
    covs = pd.DataFrame(
        {
            "WT": wt,
            "BBSA": wt / 40.0 + rng.normal(0, 0.01, 100),
            "SMK": rng.binomial(1, 0.4, 100).astype(float),
        }
    )
    summary = pd.DataFrame(
        [
            {
                "parameter": "ETA2",
                "covariate": "WT",
                "functional_form": "linear",
                "shapcov_importance": 0.9,
                "shapcov_score": 0.95,
                "shapcov_stability": 0.8,
                "shapcov_cv_r2": 0.5,
                "shapcov_selected": True,
                "shapcov_corr_filtered": False,
                "shapcov_perm_p": 0.01,
                "shapcov_perm_q": 0.02,
                "penalized_importance": 0.8,
                "penalized_score": 0.85,
                "penalized_stability": 0.8,
                "penalized_cv_r2": 0.4,
                "penalized_selected": True,
                "penalized_corr_filtered": False,
                "penalized_perm_p": 0.01,
                "penalized_perm_q": 0.02,
                "traditional_selected": True,
                "traditional_p_value": 0.001,
                "traditional_effect_size": 0.7,
                "combined_score": 0.91,
                "support_label": "boosting+penalized+traditional",
                "tier": "core",
                "tier_rank": 0,
            },
            {
                "parameter": "ETA2",
                "covariate": "BBSA",
                "functional_form": "linear",
                "shapcov_importance": 0.7,
                "shapcov_score": 0.75,
                "shapcov_stability": 0.7,
                "shapcov_cv_r2": 0.5,
                "shapcov_selected": True,
                "shapcov_corr_filtered": False,
                "shapcov_perm_p": 0.02,
                "shapcov_perm_q": 0.03,
                "penalized_importance": 0.5,
                "penalized_score": 0.55,
                "penalized_stability": 0.7,
                "penalized_cv_r2": 0.4,
                "penalized_selected": True,
                "penalized_corr_filtered": False,
                "penalized_perm_p": 0.02,
                "penalized_perm_q": 0.03,
                "traditional_selected": False,
                "traditional_p_value": 0.2,
                "traditional_effect_size": 0.2,
                "combined_score": 0.61,
                "support_label": "boosting+penalized",
                "tier": "candidate",
                "tier_rank": 1,
            },
        ]
    )

    screener = HybridScreener(boosting_method="random_forest", include_traditional=False)
    grouped = screener._assign_proxy_groups(summary, covs)
    grouped = grouped.set_index("covariate")

    assert grouped.loc["WT", "group_representative"]
    assert grouped.loc["BBSA", "tier"] == "proxy"
    assert grouped.loc["BBSA", "proxy_for"] == "WT"


def test_assign_proxy_groups_preserves_biologically_distinct_pairs():
    rng = np.random.RandomState(7)
    age = rng.normal(45, 12, 120)
    pma = age + 18.0 + rng.normal(0, 0.1, 120)
    covs = pd.DataFrame(
        {
            "AGE": age,
            "PMA": pma,
            "WT": rng.normal(70, 10, 120),
        }
    )
    summary = pd.DataFrame(
        [
            {
                "parameter": "CL",
                "covariate": "AGE",
                "functional_form": "linear",
                "shapcov_importance": 0.9,
                "shapcov_score": 0.9,
                "shapcov_stability": 0.8,
                "shapcov_cv_r2": 0.4,
                "shapcov_selected": True,
                "shapcov_corr_filtered": False,
                "shapcov_perm_p": 0.01,
                "shapcov_perm_q": 0.02,
                "penalized_importance": 0.8,
                "penalized_score": 0.8,
                "penalized_stability": 0.8,
                "penalized_cv_r2": 0.4,
                "penalized_selected": True,
                "penalized_corr_filtered": False,
                "penalized_perm_p": 0.01,
                "penalized_perm_q": 0.02,
                "traditional_selected": True,
                "traditional_p_value": 0.001,
                "traditional_effect_size": 0.6,
                "combined_score": 0.92,
                "support_label": "boosting+penalized+traditional",
                "tier": "core",
                "tier_rank": 0,
            },
            {
                "parameter": "CL",
                "covariate": "PMA",
                "functional_form": "linear",
                "shapcov_importance": 0.7,
                "shapcov_score": 0.75,
                "shapcov_stability": 0.8,
                "shapcov_cv_r2": 0.4,
                "shapcov_selected": True,
                "shapcov_corr_filtered": False,
                "shapcov_perm_p": 0.01,
                "shapcov_perm_q": 0.02,
                "penalized_importance": 0.7,
                "penalized_score": 0.7,
                "penalized_stability": 0.8,
                "penalized_cv_r2": 0.4,
                "penalized_selected": True,
                "penalized_corr_filtered": False,
                "penalized_perm_p": 0.01,
                "penalized_perm_q": 0.02,
                "traditional_selected": True,
                "traditional_p_value": 0.002,
                "traditional_effect_size": 0.5,
                "combined_score": 0.84,
                "support_label": "boosting+penalized+traditional",
                "tier": "candidate",
                "tier_rank": 1,
            },
        ]
    )

    screener = HybridScreener(
        boosting_method="random_forest",
        include_traditional=False,
        preserve_biological_distinctness=True,
    )
    grouped = screener._assign_proxy_groups(summary, covs).set_index("covariate")

    assert grouped.loc["AGE", "tier"] == "core"
    assert grouped.loc["PMA", "tier"] == "candidate"
    assert grouped.loc["PMA", "proxy_for"] == ""


def test_hybrid_results_exports_candidate_blocks():
    summary = pd.DataFrame(
        [
            {
                "parameter": "CL",
                "covariate": "WT",
                "functional_form": "power",
                "combined_score": 0.9,
                "tier": "core",
                "tier_rank": 0,
            },
            {
                "parameter": "CL",
                "covariate": "AGE",
                "functional_form": "linear",
                "combined_score": 0.4,
                "tier": "candidate",
                "tier_rank": 1,
            },
            {
                "parameter": "CL",
                "covariate": "BSA",
                "functional_form": "linear",
                "combined_score": 0.3,
                "tier": "proxy",
                "tier_rank": 2,
            },
        ]
    )
    report = HybridResults(summary, pd.DataFrame(), artifacts=None)

    assert list(report.core_covariates()["covariate"]) == ["WT"]
    assert set(report.candidate_covariates()["covariate"]) == {"WT", "AGE"}
    assert list(report.proxy_covariates()["covariate"]) == ["BSA"]
    assert "WT -> CL" in report.to_nonmem_candidates()
    assert "AGE -> CL" in report.to_nlmixr2_candidates()
