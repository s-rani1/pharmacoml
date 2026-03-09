"""SHAP-based boosting workflow for ML-assisted covariate preselection."""
from __future__ import annotations

import pandas as pd

from pharmacoml.covselect.screener import CovariateScreener
from pharmacoml.covselect.significance import SignificanceFilter


class ShapCovScreener:
    """Gradient-boosting covariate screener inspired by shap-cov style workflows."""

    def __init__(
        self,
        method: str = "catboost",
        n_bootstrap: int = 50,
        significance_threshold: float = 0.05,
        random_state: int = 42,
        cv_splits: int = 5,
        min_r2: float = 0.10,
        corr_threshold: float = 0.80,
        perm_n: int = 20,
        perm_alpha: float = 0.05,
        run_permutation: bool = False,
        rfe_enabled: bool = False,
        rfe_drop_fraction: float = 0.25,
        rfe_min_features: int = 3,
        rfe_max_rounds: int = 3,
        rfe_repeats: int = 3,
        rfe_retain_threshold: float = 0.60,
    ):
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.significance_threshold = significance_threshold
        self.random_state = random_state
        self.cv_splits = cv_splits
        self.run_permutation = run_permutation
        self.rfe_enabled = rfe_enabled
        self.rfe_drop_fraction = rfe_drop_fraction
        self.rfe_min_features = rfe_min_features
        self.rfe_max_rounds = rfe_max_rounds
        self.rfe_repeats = rfe_repeats
        self.rfe_retain_threshold = rfe_retain_threshold
        self._filter = SignificanceFilter(
            min_r2=min_r2,
            corr_threshold=corr_threshold,
            perm_n=perm_n,
            perm_alpha=perm_alpha,
            random_state=random_state,
        )
        self._results = None
        self._summary = None

    def _fit_once(
        self,
        ebes: pd.DataFrame,
        covariates: pd.DataFrame,
        parameter_names: list[str] | None,
        covariate_names: list[str] | None,
        random_state: int | None = None,
    ) -> tuple[CovariateScreener, pd.DataFrame]:
        screener = CovariateScreener(
            method=self.method,
            n_bootstrap=self.n_bootstrap,
            significance_threshold=self.significance_threshold,
            random_state=self.random_state if random_state is None else random_state,
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
        return results, summary

    def _run_rfe(
        self,
        ebes: pd.DataFrame,
        covariates: pd.DataFrame,
        parameter_names: list[str] | None,
        covariate_names: list[str] | None,
    ) -> tuple[CovariateScreener, pd.DataFrame]:
        full_covariates = covariate_names or list(covariates.columns)
        reference_results, reference_summary = self._fit_once(
            ebes=ebes,
            covariates=covariates,
            parameter_names=parameter_names,
            covariate_names=full_covariates,
            random_state=self.random_state,
        )

        if not self.rfe_enabled or len(full_covariates) <= self.rfe_min_features:
            reference_summary = reference_summary.copy()
            reference_summary["rfe_round"] = 0
            reference_summary["rfe_retained"] = True
            reference_summary["rfe_retain_frequency"] = 1.0
            return reference_results, reference_summary

        retained_sets: list[set[str]] = []
        repeat_summaries: list[pd.DataFrame] = []

        for repeat_idx in range(max(1, self.rfe_repeats)):
            active_covariates = list(full_covariates)
            current_results, current_summary = self._fit_once(
                ebes=ebes,
                covariates=covariates,
                parameter_names=parameter_names,
                covariate_names=active_covariates,
                random_state=self.random_state + repeat_idx,
            )

            for _round_idx in range(1, self.rfe_max_rounds + 1):
                if len(active_covariates) <= self.rfe_min_features:
                    break

                cov_scores = (
                    current_summary.groupby("covariate")
                    .agg(
                        max_importance=("standardized_importance", "max"),
                        selected=("final_significant", "max"),
                        stability=("stability_frequency", "max"),
                    )
                    .reset_index()
                )
                cov_scores["retain_score"] = (
                    0.55 * cov_scores["max_importance"].astype(float) +
                    0.25 * cov_scores["stability"].astype(float) +
                    0.20 * cov_scores["selected"].astype(float)
                )
                cov_scores = cov_scores.sort_values(
                    ["selected", "retain_score", "max_importance"],
                    ascending=[False, False, False],
                )
                keep_count = max(
                    self.rfe_min_features,
                    int(round(len(active_covariates) * (1.0 - self.rfe_drop_fraction))),
                )
                keepers = cov_scores.head(keep_count)["covariate"].tolist()
                if set(keepers) == set(active_covariates):
                    break

                active_covariates = keepers
                current_results, current_summary = self._fit_once(
                    ebes=ebes,
                    covariates=covariates,
                    parameter_names=parameter_names,
                    covariate_names=active_covariates,
                    random_state=self.random_state + repeat_idx + _round_idx,
                )

            retained_sets.append(set(active_covariates))
            repeat_summaries.append(current_summary.copy())

        retain_frequency = {
            cov: sum(cov in retained for retained in retained_sets) / max(len(retained_sets), 1)
            for cov in full_covariates
        }
        retained_covariates = [
            cov for cov, freq in retain_frequency.items()
            if freq >= self.rfe_retain_threshold
        ]
        if len(retained_covariates) < self.rfe_min_features:
            ranking = (
                reference_summary.groupby("covariate")
                .agg(
                    score=("standardized_importance", "max"),
                    stability=("stability_frequency", "max"),
                )
                .assign(
                    retain_frequency=lambda df: df.index.map(lambda cov: retain_frequency.get(cov, 0.0)),
                    rank_score=lambda df: 0.50 * df["retain_frequency"] + 0.35 * df["score"] + 0.15 * df["stability"],
                )
                .sort_values(["rank_score", "score"], ascending=False)
            )
            retained_covariates = ranking.head(self.rfe_min_features).index.tolist()

        final_results, final_summary = self._fit_once(
            ebes=ebes,
            covariates=covariates,
            parameter_names=parameter_names,
            covariate_names=retained_covariates,
            random_state=self.random_state + 997,
        )

        merged = reference_summary.copy()
        merged["rfe_retained"] = merged["covariate"].isin(retained_covariates)
        merged["rfe_retain_frequency"] = merged["covariate"].map(retain_frequency).fillna(0.0)
        merged["rfe_round"] = self.rfe_max_rounds
        replacement = final_summary.copy()
        replacement["rfe_retained"] = True
        replacement["rfe_round"] = self.rfe_max_rounds
        replacement["rfe_retain_frequency"] = replacement["covariate"].map(retain_frequency).fillna(1.0)
        merged = merged.merge(
            replacement,
            on=["parameter", "covariate"],
            how="left",
            suffixes=("", "__rfe"),
        )
        for col in replacement.columns:
            if col in {"parameter", "covariate"}:
                continue
            rfe_col = f"{col}__rfe"
            if rfe_col in merged.columns:
                merged[col] = merged[rfe_col].where(merged[rfe_col].notna(), merged[col])
                merged = merged.drop(columns=[rfe_col])

        merged["rfe_retained"] = merged["rfe_retained"].where(
            merged["rfe_retained"].notna(),
            False,
        ).astype(bool)
        merged["rfe_retain_frequency"] = merged["rfe_retain_frequency"].fillna(0.0).astype(float)
        merged["rfe_round"] = merged["rfe_round"].fillna(0).astype(int)
        for col in ["mean_importance", "standardized_importance", "stability_frequency", "cv_r2"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)
        for col in ["significant", "final_significant", "perm_significant", "corr_filtered", "selected"]:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), False).astype(bool)
        merged["pre_rfe_significant"] = merged["final_significant"].astype(bool)
        merged["final_significant"] = merged["final_significant"].astype(bool) & merged["rfe_retained"].astype(bool)
        merged["selected"] = merged["final_significant"]
        return final_results, merged

    def fit(
        self,
        ebes: pd.DataFrame,
        covariates: pd.DataFrame,
        parameter_names: list[str] | None = None,
        covariate_names: list[str] | None = None,
    ) -> pd.DataFrame:
        results, summary = self._run_rfe(
            ebes=ebes,
            covariates=covariates,
            parameter_names=parameter_names,
            covariate_names=covariate_names,
        )
        summary = summary.copy()
        summary["screening_method"] = self.method
        summary["workflow"] = "shapcov"
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
