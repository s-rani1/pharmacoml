"""Experimental multi-model consensus example."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pharmacoml.covselect.experimental import MultiModelConsensusScreener


def make_demo_data(n: int = 180, seed: int = 42):
    rng = np.random.RandomState(seed)
    wt = rng.normal(70, 12, n)
    age = rng.normal(55, 10, n)
    alb = rng.normal(4.0, 0.4, n)
    sex = rng.binomial(1, 0.45, n).astype(float)

    cl = 5.0 * (wt / np.median(wt)) ** 0.75 * (1 - 0.12 * (alb < 3.6)) * np.exp(rng.normal(0, 0.15, n))
    ebes = pd.DataFrame({"CL": cl})
    covs = pd.DataFrame({"WT": wt, "AGE": age, "ALB": alb, "SEX": sex})
    return ebes, covs


if __name__ == "__main__":
    ebes, covs = make_demo_data()

    report = MultiModelConsensusScreener(
        top_k=3,
        n_bootstrap=8,
        include_optional_boosting=False,
        include_neural=False,
        run_permutation=False,
    ).fit(ebes, covs)

    print("\nConsensus covariates")
    print(report.consensus_covariates().to_string(index=False))

    print("\nSelection frequency table")
    print(report.selection_frequency_table().to_string(index=False))
