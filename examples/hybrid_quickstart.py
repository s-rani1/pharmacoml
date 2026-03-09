"""Quickstart for the default hybrid covariate screener."""

import numpy as np
import pandas as pd

from pharmacoml.covselect import HybridScreener


def make_demo_data(n=200, seed=42):
    rng = np.random.RandomState(seed)
    wt = rng.normal(70, 12, n)
    age = rng.normal(55, 10, n)
    smk = rng.binomial(1, 0.35, n).astype(float)
    bsa = wt / 40 + rng.normal(0, 0.03, n)

    eta2 = (
        0.6 * ((wt - wt.mean()) / wt.std()) +
        0.3 * smk -
        0.2 * ((age - age.mean()) / age.std()) +
        rng.normal(0, 0.35, n)
    )

    ebes = pd.DataFrame({"ETA2": eta2})
    covariates = pd.DataFrame({"WT": wt, "AGE": age, "SMK": smk, "BSA": bsa})
    return ebes, covariates


if __name__ == "__main__":
    ebes, covariates = make_demo_data()

    report = HybridScreener(
        boosting_method="random_forest",
        penalized_method="aalasso",
        include_scm=True,
        n_bootstrap=20,
        run_permutation=False,
    ).fit(ebes, covariates)

    print("\n=== Core covariates ===")
    print(report.core_covariates()[["parameter", "covariate", "combined_score"]].to_string(index=False))

    print("\n=== Candidate covariates ===")
    print(report.candidate_covariates()[["parameter", "covariate", "tier", "combined_score"]].to_string(index=False))

    print("\n=== Confirmed covariates ===")
    print(report.confirmed_covariates().to_string(index=False))

    print("\n=== SCM-confirmed covariates ===")
    print(report.scm_covariates().to_string(index=False))

    print("\n=== Proxy groups ===")
    print(report.proxy_groups().to_string(index=False))

    print("\n=== NONMEM candidates ===")
    print(report.to_nonmem_candidates())
