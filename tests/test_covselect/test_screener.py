"""Test CovariateScreener with simulated population PK data."""
import numpy as np
import pandas as pd
import pytest

from tests.test_covselect._helpers import stable_tree_method


TREE_METHOD = stable_tree_method()


def make_simulated_pk_data(n=300, seed=42):
    """Simulate a simple pop PK dataset with known covariate effects.

    True model:
        CL = 5 * (WT/70)^0.75 * exp(eta_CL)
        V  = 50 * (WT/70)^1.0 * exp(eta_V)
    So WT should be identified as significant on both CL and V (power form).
    AGE and SEX are noise covariates — should NOT be significant.
    """
    rng = np.random.RandomState(seed)
    wt = rng.normal(70, 15, n).clip(40, 130)
    age = rng.normal(55, 12, n).clip(18, 90)
    sex = rng.binomial(1, 0.5, n)
    crcl = 50 + 0.5 * wt + rng.normal(0, 10, n)  # correlated with WT

    eta_cl = rng.normal(0, 0.3, n)
    eta_v = rng.normal(0, 0.2, n)

    cl = 5 * (wt / 70) ** 0.75 * np.exp(eta_cl)
    v = 50 * (wt / 70) ** 1.0 * np.exp(eta_v)

    ebes = pd.DataFrame({"CL": cl, "V": v})
    covs = pd.DataFrame({"WT": wt, "AGE": age, "SEX": sex, "CRCL": crcl})
    return ebes, covs


class TestCovariateScreener:
    def test_basic_fit(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simulated_pk_data()
        screener = CovariateScreener(method=TREE_METHOD, n_bootstrap=10, random_state=42)
        results = screener.fit(ebes, covs)
        summary = results.summary()
        assert len(summary) > 0
        assert "parameter" in summary.columns
        assert "covariate" in summary.columns
        assert "significant" in summary.columns

    def test_wt_is_significant_on_cl(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simulated_pk_data(n=500)
        screener = CovariateScreener(method=TREE_METHOD, n_bootstrap=30, random_state=42)
        results = screener.fit(ebes, covs)
        sig = results.significant_covariates()
        cl_covs = sig[sig["parameter"] == "CL"]["covariate"].tolist()
        # WT or CRCL (correlated with WT) should be picked up
        assert "WT" in cl_covs or "CRCL" in cl_covs

    def test_nonmem_code_generation(self):
        from pharmacoml.covselect import CovariateScreener
        ebes, covs = make_simulated_pk_data()
        screener = CovariateScreener(method=TREE_METHOD, n_bootstrap=10, random_state=42)
        results = screener.fit(ebes, covs)
        code = results.to_nonmem()
        assert isinstance(code, str)

    def test_invalid_method_raises(self):
        from pharmacoml.covselect import CovariateScreener
        with pytest.raises(ValueError):
            CovariateScreener(method="invalid_method")

    def test_mismatched_rows_raises(self):
        from pharmacoml.covselect import CovariateScreener
        ebes = pd.DataFrame({"CL": [1, 2, 3]})
        covs = pd.DataFrame({"WT": [70, 80]})
        screener = CovariateScreener()
        with pytest.raises(ValueError):
            screener.fit(ebes, covs)
