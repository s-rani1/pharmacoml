"""Penalized-regression confirmation workflow for hybrid covariate screening."""
from __future__ import annotations

import pandas as pd

from pharmacoml.covselect.screener import CovariateScreener
from pharmacoml.covselect.significance import SignificanceFilter


class PenalizedScreener:
    """Lasso / Elastic Net confirmation stage for hybrid workflows."""

    def __init__(
        self,
        method: str = "lasso",
        n_bootstrap: int = 50,
        significance_threshold: float = 0.05,
        random_state: int = 42,
        cv_splits: int = 5,
        min_r2: float = 0.10,
        corr_threshold: float = 0.80,
        perm_n: int = 20,
        perm_alpha: float = 0.05,
        run_permutation: bool = False,
    ):
        if method not in {"lasso", "elastic_net", "adaptive_lasso", "aalasso"}:
            raise ValueError(
                "PenalizedScreener supports only 'lasso', 'elastic_net', 'adaptive_lasso', and 'aalasso'"
            )
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.significance_threshold = significance_threshold
        self.random_state = random_state
        self.cv_splits = cv_splits
        self.run_permutation = run_permutation
        self._filter = SignificanceFilter(
            min_r2=min_r2,
            corr_threshold=corr_threshold,
            perm_n=perm_n,
            perm_alpha=perm_alpha,
            random_state=random_state,
        )
        self._results = None
        self._summary = None

    def fit(
        self,
        ebes: pd.DataFrame,
        covariates: pd.DataFrame,
        parameter_names: list[str] | None = None,
        covariate_names: list[str] | None = None,
    ) -> pd.DataFrame:
        screener = CovariateScreener(
            method=self.method,
            n_bootstrap=self.n_bootstrap,
            significance_threshold=self.significance_threshold,
            random_state=self.random_state,
            cv_splits=self.cv_splits,
        )
        results = screener.fit(
            ebes=ebes,
            covariates=covariates,
            parameter_names=parameter_names,
            covariate_names=covariate_names,
        )
        cov_input = covariates[covariate_names].copy() if covariate_names else covariates.copy()
        summary = self._filter.apply(
            results,
            cov_input,
            method_name=self.method,
            run_permutation=self.run_permutation,
        )
        summary = summary.copy()
        summary["screening_method"] = self.method
        summary["workflow"] = "penalized"
        summary["selected"] = summary["final_significant"].astype(bool)
        self._results = results
        self._summary = summary
        return summary

    @property
    def results(self):
        return self._results

    @property
    def summary_(self) -> pd.DataFrame | None:
        return None if self._summary is None else self._summary.copy()
