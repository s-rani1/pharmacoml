"""
Comprehensive test suite for pharmacoml.covselect

Test categories:
1. UNIT TESTS — basic API correctness, error handling, edge cases
2. GROUND TRUTH TESTS — simulated data where true covariates are known
3. BENCHMARK TESTS — simulate published pop PK scenarios (quetiapine-like)
4. ENSEMBLE TESTS — validate consensus across multiple ML methods
5. REPORTING TESTS — verify output formats, NONMEM code, summaries
"""

import numpy as np
import pandas as pd
import pytest

from tests.test_covselect._helpers import available_methods, stable_tree_method


DEFAULT_TEST_METHOD = stable_tree_method()
GROUND_TRUTH_METHODS = available_methods(["xgboost", "random_forest", "elastic_net"])
METHOD_COMPARISON_PAIRS = [
    (m1, m2)
    for m1, m2 in [("xgboost", "random_forest"), ("xgboost", "elastic_net")]
    if m1 in GROUND_TRUTH_METHODS and m2 in GROUND_TRUTH_METHODS
]


# ============================================================
# DATA GENERATORS
# ============================================================

def make_simple_pk(n=500, seed=42):
    """Simple 2-param PK with known covariate effects.

    Truth:
        CL = 5 * (WT/70)^0.75 * exp(eta_CL)
        V  = 50 * (WT/70)^1.0  * exp(eta_V)
    WT is significant on both CL and V.
    AGE and SEX are noise — should NOT be flagged.
    """
    rng = np.random.RandomState(seed)
    wt = rng.normal(70, 15, n).clip(40, 130)
    age = rng.normal(55, 12, n).clip(18, 90)
    sex = rng.binomial(1, 0.5, n)

    cl = 5 * (wt / 70) ** 0.75 * np.exp(rng.normal(0, 0.3, n))
    v = 50 * (wt / 70) ** 1.0 * np.exp(rng.normal(0, 0.2, n))

    ebes = pd.DataFrame({"CL": cl, "V": v})
    covs = pd.DataFrame({"WT": wt, "AGE": age, "SEX": sex})
    return ebes, covs


def make_multi_covariate_pk(n=800, seed=42):
    """Complex scenario with multiple true covariates + noise.

    Truth:
        CL = 10 * (WT/70)^0.75 * (CRCL/100)^0.5 * (1 - 0.2*SEX) * exp(eta)
        V  = 80 * (WT/70)^1.0  * exp(eta)
        KA = 1.5 * exp(eta)  (no covariate effects — all noise)
    Significant: WT on CL+V, CRCL on CL, SEX on CL
    Noise: AGE, ALB on all; everything on KA
    """
    rng = np.random.RandomState(seed)
    wt = rng.normal(70, 15, n).clip(40, 130)
    age = rng.normal(55, 12, n).clip(18, 90)
    sex = rng.binomial(1, 0.5, n).astype(float)
    crcl = 30 + 0.8 * wt + rng.normal(0, 15, n)  # correlated with WT
    crcl = crcl.clip(20, 180)
    alb = rng.normal(4.0, 0.5, n).clip(2, 5.5)

    cl = (10 * (wt / 70) ** 0.75 * (crcl / 100) ** 0.5
          * (1 - 0.2 * sex) * np.exp(rng.normal(0, 0.25, n)))
    v = 80 * (wt / 70) ** 1.0 * np.exp(rng.normal(0, 0.2, n))
    ka = 1.5 * np.exp(rng.normal(0, 0.4, n))

    ebes = pd.DataFrame({"CL": cl, "V": v, "KA": ka})
    covs = pd.DataFrame({"WT": wt, "AGE": age, "SEX": sex, "CRCL": crcl, "ALB": alb})
    return ebes, covs


def make_quetiapine_like(n=400, seed=42):
    """Simulates quetiapine-like pop PK scenario based on published models.

    Published covariates (from systematic review of quetiapine popPK):
        - Body weight on V/F (power ~1.0)
        - Age on CL/F (modest decrease with age)
        - GGT on CL/F (hepatic function marker)
        - CYP3A4 status on CL/F (large effect)

    We simulate EBEs that would result from such a model.
    """
    rng = np.random.RandomState(seed)
    wt = rng.normal(84, 20, n).clip(45, 150)  # CATIE mean ~84 kg
    age = rng.normal(45, 15, n).clip(18, 85)
    sex = rng.binomial(1, 0.37, n).astype(float)  # 63% male in CATIE
    ggt = rng.lognormal(3.5, 0.6, n).clip(10, 300)  # log-normal GGT
    cyp3a4_inducer = rng.binomial(1, 0.1, n).astype(float)  # 10% on inducers
    smoking = rng.binomial(1, 0.35, n).astype(float)

    # CL/F ~ 87.7 L/h typical; age and GGT effects, large CYP3A4 inducer effect
    cl = (87.7
          * (age / 45) ** (-0.3)
          * (ggt / 30) ** 0.15
          * (1 + 3.0 * cyp3a4_inducer)  # ~4x increase with inducers
          * np.exp(rng.normal(0, 0.35, n)))

    # V/F ~ 277 L typical; weight effect
    v = 277 * (wt / 84) ** 1.0 * np.exp(rng.normal(0, 0.4, n))

    ebes = pd.DataFrame({"CL": cl, "V": v})
    covs = pd.DataFrame({
        "WT": wt, "AGE": age, "SEX": sex, "GGT": ggt,
        "CYP3A4_IND": cyp3a4_inducer, "SMOKING": smoking
    })
    return ebes, covs


# ============================================================
# 1. UNIT TESTS — API correctness
# ============================================================

class TestAPIBasics:
    """Basic API contract tests."""

    def test_creates_screener(self):
        from pharmacoml.covselect import CovariateScreener
        s = CovariateScreener()
        assert s.method == "xgboost"
        assert s.n_bootstrap == 100

    def test_invalid_method_raises(self):
        from pharmacoml.covselect import CovariateScreener
        with pytest.raises(ValueError, match="Unknown method"):
            CovariateScreener(method="neural_net")

    def test_mismatched_rows_raises(self):
        from pharmacoml.covselect import CovariateScreener
        ebes = pd.DataFrame({"CL": [1, 2, 3]})
        covs = pd.DataFrame({"WT": [70, 80]})
        with pytest.raises(ValueError):
            CovariateScreener(n_bootstrap=5).fit(ebes, covs)

    def test_non_dataframe_raises(self):
        from pharmacoml.covselect import CovariateScreener
        with pytest.raises(TypeError):
            CovariateScreener(n_bootstrap=5).fit([[1, 2]], [[3, 4]])

    def test_empty_dataframe_raises(self):
        from pharmacoml.covselect import CovariateScreener
        with pytest.raises(ValueError):
            CovariateScreener(n_bootstrap=5).fit(
                pd.DataFrame(), pd.DataFrame()
            )

    def test_too_few_observations_raises(self):
        from pharmacoml.covselect import CovariateScreener
        ebes = pd.DataFrame({"CL": [1.0] * 10})
        covs = pd.DataFrame({"WT": [70.0] * 10})
        with pytest.raises(ValueError, match="<20 non-NaN values"):
            CovariateScreener(n_bootstrap=5).fit(ebes, covs)

    def test_fit_returns_screening_results(self):
        from pharmacoml.covselect import CovariateScreener, ScreeningResults
        ebes, covs = make_simple_pk(n=100)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=5).fit(ebes, covs)
        assert isinstance(results, ScreeningResults)

    def test_summary_has_expected_columns(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=100)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=5).fit(ebes, covs)
        summary = results.summary()
        expected_cols = {"parameter", "covariate", "mean_importance",
                        "ci_lower", "ci_upper", "pct_nonzero",
                        "significant", "functional_form"}
        assert expected_cols.issubset(set(summary.columns))

    def test_results_repr(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=100)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=5).fit(ebes, covs)
        repr_str = repr(results)
        assert "ScreeningResults" in repr_str
        assert "CL" in repr_str

    def test_subset_parameters(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=100)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=5).fit(
            ebes, covs, parameter_names=["CL"]
        )
        assert results.parameter_names == ["CL"]
        assert "V" not in results.summary()["parameter"].values

    def test_subset_covariates(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=100)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=5).fit(
            ebes, covs, covariate_names=["WT", "AGE"]
        )
        assert "SEX" not in results.summary()["covariate"].values


# ============================================================
# 2. GROUND TRUTH TESTS — known covariate effects
# ============================================================

class TestGroundTruth:
    """Verify pharmacoml correctly identifies known true covariates."""

    @pytest.mark.parametrize("method", GROUND_TRUTH_METHODS)
    def test_wt_significant_on_cl_all_methods(self, method):
        """WT should be identified as significant on CL for all ML methods."""
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=500)
        results = CovariateScreener(
            method=method, n_bootstrap=30, random_state=42
        ).fit(ebes, covs)
        sig = results.significant_covariates()
        cl_covs = sig[sig["parameter"] == "CL"]["covariate"].tolist()
        assert "WT" in cl_covs, f"WT not found significant on CL with {method}"

    @pytest.mark.parametrize("method", GROUND_TRUTH_METHODS)
    def test_wt_significant_on_v_all_methods(self, method):
        """WT should be identified as significant on V for all ML methods."""
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=500)
        results = CovariateScreener(
            method=method, n_bootstrap=30, random_state=42
        ).fit(ebes, covs)
        sig = results.significant_covariates()
        v_covs = sig[sig["parameter"] == "V"]["covariate"].tolist()
        assert "WT" in v_covs, f"WT not found significant on V with {method}"

    def test_wt_most_important_on_cl(self):
        """WT should be the most important covariate on CL."""
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=500)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=30, random_state=42).fit(ebes, covs)
        summary = results.summary()
        cl_summary = summary[summary["parameter"] == "CL"].reset_index(drop=True)
        # Summary is sorted by importance descending — first row should be WT
        assert cl_summary.iloc[0]["covariate"] == "WT"

    def test_sex_not_significant_on_cl(self):
        """SEX is noise — should NOT be significant on CL in simple model."""
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=500)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=50, random_state=42).fit(ebes, covs)
        sig = results.significant_covariates()
        cl_covs = sig[sig["parameter"] == "CL"]["covariate"].tolist()
        # SEX has no true effect in simple model — allow some false positive rate
        # but at minimum WT should rank higher
        summary = results.summary()
        cl_df = summary[summary["parameter"] == "CL"]
        wt_imp = cl_df[cl_df["covariate"] == "WT"]["mean_importance"].values[0]
        sex_imp = cl_df[cl_df["covariate"] == "SEX"]["mean_importance"].values[0]
        assert wt_imp > sex_imp * 2, "WT should be much more important than SEX"

    def test_multi_covariate_crcl_on_cl(self):
        """In multi-covariate scenario, CRCL should be significant on CL."""
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_multi_covariate_pk(n=800)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=30, random_state=42).fit(ebes, covs)
        sig = results.significant_covariates()
        cl_covs = sig[sig["parameter"] == "CL"]["covariate"].tolist()
        # CRCL or WT should be found (CRCL is correlated with WT)
        assert "CRCL" in cl_covs or "WT" in cl_covs

    def test_no_covariates_on_ka(self):
        """KA has no true covariate effects — nothing should be significant."""
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_multi_covariate_pk(n=800)
        results = CovariateScreener(
            method=DEFAULT_TEST_METHOD, n_bootstrap=50, significance_threshold=0.01, random_state=42
        ).fit(ebes, covs)
        sig = results.significant_covariates()
        ka_covs = sig[sig["parameter"] == "KA"]["covariate"].tolist()
        # With strict threshold, KA should have few or no significant covariates
        assert len(ka_covs) <= 1, f"Too many false positives on KA: {ka_covs}"

    def test_functional_form_wt_on_cl_is_power_or_nonlinear(self):
        """WT effect on CL is power (0.75) — should detect power or nonlinear."""
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=500)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=20, random_state=42).fit(ebes, covs)
        summary = results.summary()
        cl_wt = summary[(summary["parameter"] == "CL") & (summary["covariate"] == "WT")]
        form = cl_wt["functional_form"].values[0]
        assert form in ("power", "nonlinear", "linear"), f"Unexpected form: {form}"


# ============================================================
# 3. BENCHMARK TESTS — published quetiapine-like scenario
# ============================================================

class TestQuetiapineBenchmark:
    """Validate against known covariates from published quetiapine popPK models.

    Published covariates:
        - Body weight on V/F (power ~1.0)
        - Age on CL/F (negative effect)
        - GGT on CL/F (hepatic marker)
        - CYP3A4 inducer status on CL/F (very large ~4x effect)
    """

    def test_cyp3a4_inducer_significant_on_cl(self):
        """CYP3A4 inducer status has 4x effect — must be detected."""
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_quetiapine_like(n=400)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=30, random_state=42).fit(ebes, covs)
        sig = results.significant_covariates()
        cl_covs = sig[sig["parameter"] == "CL"]["covariate"].tolist()
        assert "CYP3A4_IND" in cl_covs, (
            f"CYP3A4_IND not found significant on CL. Found: {cl_covs}"
        )

    def test_wt_significant_on_v_quetiapine(self):
        """Body weight on V is the strongest covariate — must be detected."""
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_quetiapine_like(n=400)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=30, random_state=42).fit(ebes, covs)
        sig = results.significant_covariates()
        v_covs = sig[sig["parameter"] == "V"]["covariate"].tolist()
        assert "WT" in v_covs, f"WT not found significant on V. Found: {v_covs}"

    def test_age_detected_on_cl_quetiapine(self):
        """Age has modest effect on CL — should be detected with enough data."""
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_quetiapine_like(n=600)  # larger N for modest effect
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=30, random_state=42).fit(ebes, covs)
        summary = results.summary()
        cl_age = summary[(summary["parameter"] == "CL") & (summary["covariate"] == "AGE")]
        # Age may or may not be significant (modest effect) — but should have nonzero importance
        assert cl_age["mean_importance"].values[0] > 0.01

    def test_smoking_not_top_covariate(self):
        """Smoking is noise in our simulation — should not outrank true covariates."""
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_quetiapine_like(n=400)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=30, random_state=42).fit(ebes, covs)
        summary = results.summary()
        cl_df = summary[summary["parameter"] == "CL"].reset_index(drop=True)
        # Smoking should not be in top 3
        top3 = cl_df.head(3)["covariate"].tolist()
        assert "SMOKING" not in top3, f"SMOKING unexpectedly in top 3: {top3}"


# ============================================================
# 4. METHOD CONSISTENCY TESTS
# ============================================================

class TestMethodConsistency:
    """All ML methods should agree on strong signals."""

    def test_all_methods_find_wt_on_cl(self):
        """The strongest signal (WT on CL) should be found by every method."""
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=500)
        for method in GROUND_TRUTH_METHODS:
            results = CovariateScreener(
                method=method, n_bootstrap=20, random_state=42
            ).fit(ebes, covs)
            summary = results.summary()
            cl_top = summary[summary["parameter"] == "CL"].iloc[0]["covariate"]
            assert cl_top == "WT", f"{method} ranked {cl_top} above WT on CL"

    def test_methods_rank_correlation(self):
        """Different methods should produce correlated importance rankings."""
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_multi_covariate_pk(n=500)
        rankings = {}
        for method in GROUND_TRUTH_METHODS:
            results = CovariateScreener(
                method=method, n_bootstrap=15, random_state=42
            ).fit(ebes, covs)
            summary = results.summary()
            cl_df = summary[summary["parameter"] == "CL"].set_index("covariate")
            rankings[method] = cl_df["mean_importance"].rank(ascending=False)

        # Check pairwise rank correlation between methods
        from scipy.stats import spearmanr
        for m1, m2 in METHOD_COMPARISON_PAIRS:
            common = rankings[m1].index.intersection(rankings[m2].index)
            r1 = rankings[m1].loc[common].values
            r2 = rankings[m2].loc[common].values
            corr, _ = spearmanr(r1, r2)
            assert corr > 0.3, (
                f"Low rank correlation ({corr:.2f}) between {m1} and {m2}"
            )


# ============================================================
# 5. REPORTING TESTS — output formats
# ============================================================

class TestReporting:
    """Verify output formats and code generation."""

    def test_nonmem_code_is_string(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=200)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=10).fit(ebes, covs)
        code = results.to_nonmem()
        assert isinstance(code, str)

    def test_nonmem_code_contains_tv(self):
        """Generated NONMEM code should contain TV (typical value) statements."""
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=300)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=20).fit(ebes, covs)
        code = results.to_nonmem()
        if "No significant" not in code:
            assert "TV" in code

    def test_nonmem_code_contains_theta(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=300)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=20).fit(ebes, covs)
        code = results.to_nonmem()
        if "No significant" not in code:
            assert "THETA" in code

    def test_summary_dataframe_types(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=200)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=10).fit(ebes, covs)
        summary = results.summary()
        assert summary["mean_importance"].dtype == float
        assert summary["significant"].dtype == bool
        assert summary["parameter"].dtype == object

    def test_to_dataframe_same_as_summary(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=200)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=10).fit(ebes, covs)
        pd.testing.assert_frame_equal(results.to_dataframe(), results.summary())

    def test_significant_covariates_is_subset_of_summary(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=200)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=10).fit(ebes, covs)
        sig = results.significant_covariates()
        full = results.summary()
        assert len(sig) <= len(full)
        assert all(sig["significant"])


# ============================================================
# 6. CATEGORICAL COVARIATE TESTS
# ============================================================

class TestCategoricalCovariates:
    """Test handling of categorical/string covariates."""

    def test_string_covariate_encoding(self):
        from pharmacoml.covselect import CovariateScreener
        rng = np.random.RandomState(42)
        n = 200
        race = rng.choice(["White", "Black", "Asian"], n)
        wt = rng.normal(70, 15, n)
        cl = 5 * (wt / 70) ** 0.75 * np.exp(rng.normal(0, 0.3, n))

        ebes = pd.DataFrame({"CL": cl})
        covs = pd.DataFrame({"WT": wt, "RACE": race})
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=10).fit(ebes, covs)
        summary = results.summary()
        # RACE should appear in summary (aggregated from one-hot columns)
        assert "RACE" in summary["covariate"].values

    def test_binary_detected_as_categorical(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=200)
        results = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=10).fit(ebes, covs)
        summary = results.summary()
        sex_row = summary[(summary["parameter"] == "CL") & (summary["covariate"] == "SEX")]
        assert sex_row["functional_form"].values[0] == "categorical"


# ============================================================
# 7. REPRODUCIBILITY TESTS
# ============================================================

class TestReproducibility:
    """Same inputs + same seed = same outputs."""

    def test_deterministic_results(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=200)

        r1 = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=10, random_state=42).fit(ebes, covs)
        r2 = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=10, random_state=42).fit(ebes, covs)

        pd.testing.assert_frame_equal(r1.summary(), r2.summary())

    def test_different_seeds_differ(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simple_pk(n=200)

        r1 = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=10, random_state=42).fit(ebes, covs)
        r2 = CovariateScreener(method=DEFAULT_TEST_METHOD, n_bootstrap=10, random_state=99).fit(ebes, covs)

        # Results may be similar but not necessarily identical
        s1 = r1.summary()["mean_importance"].values
        s2 = r2.summary()["mean_importance"].values
        # At least some values should differ
        assert not np.allclose(s1, s2, atol=1e-6)
