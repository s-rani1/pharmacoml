"""
Ensemble covariate screener — filtered, rank-aware consensus across ML methods.

Raw majority voting is too liberal for pharmacometric covariate screening:
small but stable nonzero effects can get counted as "significant" across
methods and leak through as false positives. This module tightens the ensemble
workflow in six ways:

1. Per-method votes pass through ``SignificanceFilter`` by default.
2. Reliability and utility are scored out-of-fold rather than in-sample.
3. Secondary covariates must add incremental predictive value (delta R² gate).
4. Consensus is standardized, family-weighted, and stability-aware.
5. Deep learning engines are opt-in rather than default ensemble members.
6. Pairwise interactions can be screened in a second pass.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pharmacoml.covselect.engines import (
    DEEP_LEARNING_ENSEMBLE_METHODS,
    DEFAULT_ENSEMBLE_METHODS,
    FULL_ENSEMBLE_METHODS,
    ENGINE_FAMILY,
    check_engine_availability,
)
from pharmacoml.covselect.selection_utils import build_interaction_terms, cross_validated_r2
from pharmacoml.covselect.screener import CovariateScreener
from pharmacoml.covselect.significance import SignificanceFilter


class EnsembleScreener:
    """Consensus-based covariate screening across multiple ML methods.

    By default the ensemble uses the six classical tabular methods
    (XGBoost, LightGBM, CatBoost, Random Forest, Elastic Net, Lasso).
    Deep learning engines remain available but are opt-in because most
    population PK datasets are modest in size and overfitting risk is high.

    Parameters
    ----------
    methods : list[str] | None
        Methods to run explicitly. If omitted, the classical ensemble is used.
    min_agreement : int, default 4
        Minimum number of methods that must cast a valid vote.
    n_bootstrap : int, default 100
    significance_threshold : float, default 0.05
    use_significance_filter : bool, default True
        Apply R² reliability, correlation deduplication, and permutation testing
        before any per-method vote can count.
    run_permutation : bool, default True
        Whether the per-method significance filter runs the expensive
        permutation null test.
    min_delta_r2 : float, default 0.02
        Minimum incremental in-sample R² a covariate must add after stronger
        already-accepted covariates for the same parameter.
    min_relative_importance : float, default 0.10
        Minimum mean-importance ratio vs the top covariate for that parameter.
    include_deep_learning : bool, default False
        If ``methods`` is omitted, append TabNet and MLP to the default set.
    deep_learning_min_rows : int, default 250
        Warn when deep learning is used below this sample size.
    """

    def __init__(
        self,
        methods: list[str] | None = None,
        min_agreement: int = 4,
        n_bootstrap: int = 100,
        significance_threshold: float = 0.05,
        random_state: int = 42,
        use_significance_filter: bool = True,
        run_permutation: bool = True,
        min_r2: float = 0.10,
        corr_threshold: float = 0.80,
        perm_n: int = 20,
        perm_alpha: float = 0.05,
        min_delta_r2: float = 0.02,
        min_relative_importance: float = 0.10,
        min_stability_frequency: float = 0.40,
        include_deep_learning: bool = False,
        deep_learning_min_rows: int = 250,
        cv_splits: int = 5,
        enable_interactions: bool = False,
        interaction_seed_top_n: int = 6,
        max_interaction_pairs: int = 12,
        candidate_min_agreement: int | None = None,
        candidate_family_agreement: int = 2,
        candidate_score_threshold: float = 0.02,
    ):
        self._requested_methods = methods
        self.methods = list(methods) if methods is not None else []
        self.min_agreement = min_agreement
        self.n_bootstrap = n_bootstrap
        self.significance_threshold = significance_threshold
        self.random_state = random_state

        self.use_significance_filter = use_significance_filter
        self.run_permutation = run_permutation
        self.min_r2 = min_r2
        self.corr_threshold = corr_threshold
        self.perm_n = perm_n
        self.perm_alpha = perm_alpha

        self.min_delta_r2 = min_delta_r2
        self.min_relative_importance = min_relative_importance
        self.min_stability_frequency = min_stability_frequency
        self.include_deep_learning = include_deep_learning
        self.deep_learning_min_rows = deep_learning_min_rows
        self.cv_splits = cv_splits
        self.enable_interactions = enable_interactions
        self.interaction_seed_top_n = interaction_seed_top_n
        self.max_interaction_pairs = max_interaction_pairs
        self.candidate_min_agreement = candidate_min_agreement
        self.candidate_family_agreement = candidate_family_agreement
        self.candidate_score_threshold = candidate_score_threshold

        self._method_results: dict[str, object] = {}
        self._method_summaries: dict[str, pd.DataFrame] = {}
        self._interaction_metadata: dict[str, dict[str, str]] = {}

    def _resolve_methods(self, n_rows: int) -> list[str]:
        if self._requested_methods is not None:
            candidate_methods = list(self._requested_methods)
        else:
            candidate_methods = list(DEFAULT_ENSEMBLE_METHODS)
            if self.include_deep_learning:
                candidate_methods = list(FULL_ENSEMBLE_METHODS)

        available = check_engine_availability()
        methods = [m for m in candidate_methods if available.get(m, False)]
        unavailable = [m for m in candidate_methods if not available.get(m, False)]

        if unavailable:
            warnings.warn(
                f"Engines not available (install their packages): {unavailable}. "
                f"Running with: {methods}",
                stacklevel=3,
            )

        if not methods:
            raise ValueError("No ensemble engines are available in the current environment.")

        deep_learning_methods = [m for m in methods if m in DEEP_LEARNING_ENSEMBLE_METHODS]
        if deep_learning_methods and n_rows < self.deep_learning_min_rows:
            warnings.warn(
                f"Deep learning methods {deep_learning_methods} requested with n={n_rows}. "
                f"These are optional and may overfit small popPK datasets; "
                f"the default ensemble excludes them below ~{self.deep_learning_min_rows} rows.",
                stacklevel=3,
            )

        return methods

    def _build_significance_filter(self) -> SignificanceFilter:
        return SignificanceFilter(
            min_r2=self.min_r2,
            corr_threshold=self.corr_threshold,
            perm_n=self.perm_n,
            perm_alpha=self.perm_alpha,
            random_state=self.random_state,
        )

    def _empty_filter_columns(self, summary: pd.DataFrame) -> pd.DataFrame:
        summary = summary.copy()
        summary["r2"] = np.nan
        summary["r2_reliable"] = True
        summary["corr_filtered"] = False
        summary["perm_p_value"] = np.nan
        summary["perm_q_value"] = np.nan
        summary["perm_significant"] = summary["significant"]
        summary["final_significant"] = summary["significant"]
        return summary

    def _subset_r2(self, method: str, results, param: str, encoded_columns: tuple[str, ...]) -> float:
        if not encoded_columns:
            return 0.0

        y = results._ebe_data[param].values
        mask = ~np.isnan(y)
        X = results._cov_data.loc[mask, list(encoded_columns)].values
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = y[mask]

        try:
            return max(
                0.0,
                float(
                    cross_validated_r2(
                        method_name=method,
                        X=X,
                        y=y,
                        random_state=self.random_state,
                        n_splits=self.cv_splits,
                    )
                ),
            )
        except Exception:
            return 0.0

    def _apply_utility_gate(self, summary: pd.DataFrame, results, method: str) -> pd.DataFrame:
        summary = summary.copy()
        summary["utility_r2_base"] = np.nan
        summary["utility_r2_with_candidate"] = np.nan
        summary["utility_delta_r2"] = np.nan
        summary["utility_reference_covariates"] = ""
        summary["utility_pass"] = False

        vote_col = "final_significant" if "final_significant" in summary.columns else "significant"

        for param in results.parameter_names:
            pdf = summary[summary["parameter"] == param].sort_values(
                "mean_importance", ascending=False
            )
            if len(pdf) == 0:
                continue

            accepted_covariates: list[str] = []
            accepted_columns: list[str] = []
            r2_cache: dict[tuple[str, ...], float] = {}

            def cached_subset_r2(columns: list[str]) -> float:
                key = tuple(columns)
                if key not in r2_cache:
                    r2_cache[key] = self._subset_r2(method, results, param, key)
                return r2_cache[key]

            for idx, row in pdf.iterrows():
                covariate = row["covariate"]
                encoded_cols = [
                    col for col in results._encoding_map.get(covariate, [covariate])
                    if col in results._cov_data.columns
                ]
                if not encoded_cols:
                    continue

                summary.at[idx, "utility_reference_covariates"] = (
                    ", ".join(accepted_covariates) if accepted_covariates else "none"
                )

                if not bool(row[vote_col]):
                    continue

                if not accepted_columns:
                    with_candidate = cached_subset_r2(encoded_cols)
                    summary.at[idx, "utility_r2_base"] = 0.0
                    summary.at[idx, "utility_r2_with_candidate"] = round(with_candidate, 4)
                    summary.at[idx, "utility_delta_r2"] = round(with_candidate, 4)
                    summary.at[idx, "utility_pass"] = True
                    accepted_covariates.append(covariate)
                    accepted_columns.extend(encoded_cols)
                    continue

                base_r2 = cached_subset_r2(accepted_columns)
                trial_columns = accepted_columns + encoded_cols
                with_candidate = cached_subset_r2(trial_columns)
                delta_r2 = with_candidate - base_r2
                utility_pass = delta_r2 >= self.min_delta_r2

                summary.at[idx, "utility_r2_base"] = round(base_r2, 4)
                summary.at[idx, "utility_r2_with_candidate"] = round(with_candidate, 4)
                summary.at[idx, "utility_delta_r2"] = round(delta_r2, 4)
                summary.at[idx, "utility_pass"] = utility_pass

                if utility_pass:
                    accepted_covariates.append(covariate)
                    accepted_columns.extend(encoded_cols)

        return summary

    def _annotate_method_summary(self, summary: pd.DataFrame, results, method: str) -> pd.DataFrame:
        summary = summary.copy()
        summary["importance_rank"] = (
            summary.groupby("parameter")["mean_importance"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )
        max_rank = summary.groupby("parameter")["importance_rank"].transform("max").replace(0, 1)
        rank_weight = np.where(max_rank > 1, 1.0 - ((summary["importance_rank"] - 1) / (max_rank - 1)), 1.0)
        summary["relative_importance"] = summary.groupby("parameter")["mean_importance"].transform(
            lambda s: s / max(float(s.max()), 1e-12)
        )
        summary["relative_importance"] = summary["relative_importance"].fillna(0.0).round(4)
        summary["standardized_importance"] = (
            summary["relative_importance"] * rank_weight
        ).round(4)
        summary["relative_importance_pass"] = (
            (summary["importance_rank"] == 1) |
            (summary["relative_importance"] >= self.min_relative_importance)
        )
        if "stability_frequency" not in summary.columns:
            summary["stability_frequency"] = 0.0
        summary["stability_frequency"] = summary["stability_frequency"].fillna(0.0)
        summary["stability_pass"] = (
            (summary["importance_rank"] == 1) |
            (summary["stability_frequency"] >= self.min_stability_frequency)
        )

        summary = self._apply_utility_gate(summary, results, method)
        vote_col = "final_significant" if "final_significant" in summary.columns else "significant"
        summary["ensemble_vote"] = (
            summary[vote_col].astype(bool) &
            summary["relative_importance_pass"].astype(bool) &
            summary["utility_pass"].astype(bool) &
            summary["stability_pass"].astype(bool)
        )
        return summary

    def _run_methods(self, methods, ebes, covariates, parameter_names=None, covariate_names=None):
        cov_input = covariates[covariate_names].copy() if covariate_names else covariates.copy()
        method_results = {}
        method_summaries = {}

        for method in methods:
            screener = CovariateScreener(
                method=method,
                n_bootstrap=self.n_bootstrap,
                significance_threshold=self.significance_threshold,
                random_state=self.random_state,
                cv_splits=self.cv_splits,
            )
            results = screener.fit(ebes, covariates, parameter_names, covariate_names)
            method_results[method] = results

            if self.use_significance_filter:
                filter_ = self._build_significance_filter()
                summary = filter_.apply(
                    results,
                    cov_input,
                    method_name=method,
                    run_permutation=self.run_permutation,
                )
            else:
                summary = self._empty_filter_columns(results.summary())

            method_summaries[method] = self._annotate_method_summary(summary, results, method)

        return method_results, method_summaries

    def _select_interaction_seeds(self, method_summaries: dict[str, pd.DataFrame]) -> list[str]:
        if not method_summaries:
            return []

        seed_scores = {}
        for summary in method_summaries.values():
            df = summary.copy()
            df = df[~df["covariate"].str.contains("__x__|__xor__", regex=True)]
            if len(df) == 0:
                continue
            grouped = (
                df.groupby("covariate")[["standardized_importance", "stability_frequency"]]
                .mean()
            )
            for cov, row in grouped.iterrows():
                score = float(row["standardized_importance"]) * max(float(row["stability_frequency"]), 0.1)
                seed_scores[cov] = max(seed_scores.get(cov, 0.0), score)

        ordered = sorted(seed_scores.items(), key=lambda item: item[1], reverse=True)
        return [cov for cov, _ in ordered[: self.interaction_seed_top_n]]

    def fit(self, ebes: pd.DataFrame, covariates: pd.DataFrame,
            parameter_names=None, covariate_names=None) -> "EnsembleResults":
        """Run all methods and compute filtered consensus."""
        methods = self._resolve_methods(len(ebes))
        self.methods = methods
        cov_input = covariates[covariate_names].copy() if covariate_names else covariates.copy()

        method_results, method_summaries = self._run_methods(
            methods,
            ebes,
            covariates,
            parameter_names=parameter_names,
            covariate_names=covariate_names,
        )

        self._interaction_metadata = {}
        if self.enable_interactions:
            seeds = self._select_interaction_seeds(method_summaries)
            interaction_df, interaction_metadata = build_interaction_terms(
                cov_input,
                candidate_covariates=seeds,
                max_pairs=self.max_interaction_pairs,
            )
            if len(interaction_df.columns):
                cov_augmented = pd.concat([cov_input, interaction_df], axis=1)
                method_results, method_summaries = self._run_methods(
                    methods,
                    ebes,
                    cov_augmented,
                    parameter_names=parameter_names,
                    covariate_names=list(cov_augmented.columns),
                )
                self._interaction_metadata = interaction_metadata

        self._method_results = method_results
        self._method_summaries = method_summaries

        return EnsembleResults(
            method_results=self._method_results,
            method_summaries=self._method_summaries,
            min_agreement=min(self.min_agreement, len(methods)),
            min_relative_importance=self.min_relative_importance,
            min_delta_r2=self.min_delta_r2,
            candidate_min_agreement=self.candidate_min_agreement,
            candidate_family_agreement=self.candidate_family_agreement,
            candidate_score_threshold=self.candidate_score_threshold,
            interaction_metadata=self._interaction_metadata,
        )


class EnsembleResults:
    """Results from ``EnsembleScreener`` — filtered consensus across methods."""

    def __init__(
        self,
        method_results: dict[str, object],
        method_summaries: dict[str, pd.DataFrame],
        min_agreement: int,
        min_relative_importance: float,
        min_delta_r2: float = 0.02,
        candidate_min_agreement: int | None = None,
        candidate_family_agreement: int = 2,
        candidate_score_threshold: float = 0.02,
        interaction_metadata: dict[str, dict[str, str]] | None = None,
    ):
        self._method_results = method_results
        self._method_summaries = method_summaries
        self._min_agreement = min_agreement
        self._min_relative_importance = min_relative_importance
        self._min_delta_r2 = min_delta_r2
        self._candidate_min_agreement = candidate_min_agreement
        self._candidate_family_agreement = candidate_family_agreement
        self._candidate_score_threshold = candidate_score_threshold
        self._interaction_metadata = interaction_metadata or {}

    def per_method_summary(self) -> dict[str, pd.DataFrame]:
        """Return the filtered, ensemble-ready summary table for each method."""
        return {m: df.copy() for m, df in self._method_summaries.items()}

    def interaction_metadata(self) -> dict[str, dict[str, str]]:
        """Return metadata for engineered interaction terms."""
        return dict(self._interaction_metadata)

    def consensus_summary(self) -> pd.DataFrame:
        """Return consensus table showing agreement across methods.

        Consensus is based on each method's filtered ``ensemble_vote``. The
        summary also exposes method-level ranking and relative-strength metrics
        so users can see whether support comes from strong top-ranked covariates
        or weak secondary signals.
        """
        if not self._method_summaries:
            return pd.DataFrame()

        all_pairs = set()
        for df in self._method_summaries.values():
            all_pairs.update(zip(df["parameter"], df["covariate"]))

        n_methods = len(self._method_summaries)
        available_families = {ENGINE_FAMILY.get(method, method) for method in self._method_summaries}
        n_families = max(len(available_families), 1)
        core_family_agreement = max(2, int(np.ceil(n_families * 0.67)))
        candidate_min_agreement = self._candidate_min_agreement or max(2, self._min_agreement - 1)
        rows = []
        for param, cov in sorted(all_pairs):
            methods_voting = []
            families_voting = []
            method_ranks = []
            method_relative_importance = []
            standardized_importances = []
            stability_scores = []
            mean_importances = []
            functional_forms = []
            utility_deltas = []

            for method, df in self._method_summaries.items():
                match = df[(df["parameter"] == param) & (df["covariate"] == cov)]
                if len(match) == 0:
                    continue
                row = match.iloc[0]
                mean_importances.append(float(row["mean_importance"]))
                functional_forms.append(row.get("functional_form", "unknown"))
                method_ranks.append(float(row.get("importance_rank", np.nan)))
                method_relative_importance.append(float(row.get("relative_importance", 0.0)))
                standardized_importances.append(float(row.get("standardized_importance", 0.0)))
                stability_scores.append(float(row.get("stability_frequency", 0.0)))
                if bool(row.get("ensemble_vote", False)):
                    methods_voting.append(method)
                    families_voting.append(ENGINE_FAMILY.get(method, method))
                    utility_deltas.append(float(row.get("utility_delta_r2", 0.0)))

            n_votes = len(methods_voting)
            n_families_significant = len(set(families_voting))
            vote_ratio = n_votes / n_methods if n_methods else 0.0
            family_vote_ratio = n_families_significant / n_families if n_families else 0.0
            family_diversity = n_families_significant / n_votes if n_votes else 0.0
            mean_rank = float(np.nanmean(method_ranks)) if method_ranks else np.nan
            mean_relative_importance = (
                float(np.nanmean(method_relative_importance))
                if method_relative_importance else 0.0
            )
            mean_standardized_importance = (
                float(np.nanmean(standardized_importances))
                if standardized_importances else 0.0
            )
            mean_stability = (
                float(np.nanmean(stability_scores))
                if stability_scores else 0.0
            )
            rank_score = (
                float(np.nanmean([1.0 / max(rank, 1.0) for rank in method_ranks]))
                if method_ranks else 0.0
            )
            mean_delta_r2 = float(np.mean(utility_deltas)) if utility_deltas else 0.0
            delta_score = min(1.0, mean_delta_r2 / max(self._min_delta_r2, 1e-6))
            consensus_score = (
                vote_ratio *
                family_vote_ratio *
                max(family_diversity, 0.25) *
                max(mean_standardized_importance, 0.0) *
                max(mean_stability, 0.0) *
                max(rank_score, 0.1) *
                max(delta_score, 0.1)
            )
            core_consensus = (
                n_votes >= self._min_agreement and
                n_families_significant >= core_family_agreement and
                mean_relative_importance >= self._min_relative_importance
            )
            candidate_consensus = (
                not core_consensus and
                n_votes >= candidate_min_agreement and
                n_families_significant >= min(self._candidate_family_agreement, n_families) and
                mean_relative_importance >= self._min_relative_importance and
                consensus_score >= self._candidate_score_threshold
            )
            if core_consensus:
                consensus_tier = "core"
            elif candidate_consensus:
                consensus_tier = "candidate"
            else:
                consensus_tier = "none"

            rows.append({
                "parameter": param,
                "covariate": cov,
                "n_methods_significant": n_votes,
                "n_families_significant": n_families_significant,
                "methods_significant": ", ".join(methods_voting) or "none",
                "families_significant": ", ".join(sorted(set(families_voting))) or "none",
                "mean_importance_avg": round(float(np.mean(mean_importances)), 4) if mean_importances else 0.0,
                "mean_rank": round(mean_rank, 2) if not np.isnan(mean_rank) else np.nan,
                "mean_relative_importance": round(mean_relative_importance, 4),
                "mean_standardized_importance": round(mean_standardized_importance, 4),
                "mean_stability_frequency": round(mean_stability, 4),
                "mean_delta_r2": round(mean_delta_r2, 4),
                "method_vote_ratio": round(vote_ratio, 4),
                "family_vote_ratio": round(family_vote_ratio, 4),
                "consensus_score": round(consensus_score, 4),
                "consensus_form": (
                    max(set(functional_forms), key=functional_forms.count)
                    if functional_forms else "unknown"
                ),
                "consensus_tier": consensus_tier,
                "consensus": consensus_tier != "none",
                "is_interaction": "__x__" in cov or "__xor__" in cov,
            })

        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values(
                ["parameter", "n_methods_significant", "consensus_score", "mean_importance_avg"],
                ascending=[True, False, False, False],
            ).reset_index(drop=True)
        return df

    def significant_consensus(self) -> pd.DataFrame:
        """Return only covariate-parameter pairs with consensus agreement."""
        return self.consensus_summary().query("consensus").reset_index(drop=True)

    def comparison_table(self) -> pd.DataFrame:
        """Side-by-side comparison of filtered votes across methods."""
        methods = list(self._method_summaries.keys())
        all_pairs = set()
        vote_sets = {}

        for method, summary in self._method_summaries.items():
            for _, row in summary.iterrows():
                all_pairs.add((row["parameter"], row["covariate"]))
            vote_sets[method] = {
                (r["parameter"], r["covariate"])
                for _, r in summary[summary["ensemble_vote"]].iterrows()
            }

        rows = []
        for param, cov in sorted(all_pairs):
            row = {"parameter": param, "covariate": cov}
            for method in methods:
                row[f"{method}_significant"] = (param, cov) in vote_sets[method]
            rows.append(row)

        return pd.DataFrame(rows)

    def __repr__(self):
        n_consensus = len(self.significant_consensus())
        methods = list(self._method_summaries.keys())
        return (
            f"EnsembleResults(methods={methods}, "
            f"min_agreement={self._min_agreement}, "
            f"consensus_significant={n_consensus})"
        )
