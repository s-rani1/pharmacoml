"""Experimental multi-model consensus screening for broad model comparison."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pharmacoml.covselect.engines import (
    check_engine_availability,
    get_engine_family,
)
from pharmacoml.covselect.hybrid import HybridResults
from pharmacoml.covselect.screener import CovariateScreener
from pharmacoml.covselect.significance import SignificanceFilter


DEFAULT_MULTIMODEL_MODELS = [
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "lasso",
    "aalasso",
]
EXTENDED_MULTIMODEL_MODELS = ["elastic_net", "ridge"]
OPTIONAL_MULTIMODEL_MODELS = ["xgboost", "lightgbm", "catboost"]
NEURAL_MULTIMODEL_MODELS = ["stg", "mlp", "tabnet"]


class MultiModelConsensusResults:
    """Aggregated results for the experimental multi-model consensus workflow."""

    def __init__(
        self,
        method_results: dict[str, object],
        method_summaries: dict[str, pd.DataFrame],
        models: list[str],
        top_k: int,
        min_model_frequency: float,
        min_family_support: int,
        min_mean_relative_importance: float,
        max_consensus_per_parameter: int | None,
    ):
        self.method_results = method_results
        self.method_summaries = method_summaries
        self.models = models
        self.top_k = top_k
        self.min_model_frequency = min_model_frequency
        self.min_family_support = min_family_support
        self.min_mean_relative_importance = min_mean_relative_importance
        self.max_consensus_per_parameter = max_consensus_per_parameter

    def per_model_summary(self, model: str | None = None) -> pd.DataFrame:
        if model is None:
            frames = []
            for method_name, summary in self.method_summaries.items():
                frame = summary.copy()
                frame["model"] = method_name
                frames.append(frame)
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)
        if model not in self.method_summaries:
            raise ValueError(f"Unknown model '{model}'. Available: {sorted(self.method_summaries)}")
        return self.method_summaries[model].copy()

    def consensus_summary(self) -> pd.DataFrame:
        combined = self.per_model_summary()
        if combined.empty:
            return combined

        family_counts = (
            combined[["model", "family"]]
            .drop_duplicates()
            .groupby("family")
            .size()
            .to_dict()
        )
        n_models = max(len(self.models), 1)
        n_families = max(len(family_counts), 1)

        rows = []
        for (parameter, covariate), group in combined.groupby(["parameter", "covariate"], sort=False):
            selected = group[group["selected"]]
            selected_models = selected["model"].tolist()
            selected_families = sorted(selected["family"].unique().tolist())

            family_frequency_terms = []
            for family in selected_families:
                denom = max(family_counts.get(family, 1), 1)
                family_hits = int((selected["family"] == family).sum())
                family_frequency_terms.append(family_hits / denom)

            selection_frequency = float(len(selected_models) / n_models)
            family_support = len(selected_families)
            family_frequency = float(np.mean(family_frequency_terms)) if family_frequency_terms else 0.0
            mean_rank = float(group["importance_rank"].mean())
            mean_selected_rank = float(selected["importance_rank"].mean()) if len(selected) else np.nan
            mean_relative_importance = float(group["relative_importance"].mean())
            mean_standardized_importance = float(group["standardized_importance"].mean())
            mean_stability = float(group["stability_frequency"].fillna(0.0).mean())
            consensus_score = (
                0.45 * selection_frequency
                + 0.25 * (family_support / n_families)
                + 0.20 * mean_standardized_importance
                + 0.10 * mean_stability
            )
            rows.append(
                {
                    "parameter": parameter,
                    "covariate": covariate,
                    "n_models_selected": len(selected_models),
                    "selection_frequency": round(selection_frequency, 4),
                    "family_support": family_support,
                    "family_frequency": round(family_frequency, 4),
                    "mean_rank": round(mean_rank, 3),
                    "mean_selected_rank": round(mean_selected_rank, 3) if not np.isnan(mean_selected_rank) else np.nan,
                    "mean_relative_importance": round(mean_relative_importance, 4),
                    "mean_standardized_importance": round(mean_standardized_importance, 4),
                    "mean_stability_frequency": round(mean_stability, 4),
                    "models_selected": ", ".join(selected_models) if selected_models else "none",
                    "families_selected": ", ".join(selected_families) if selected_families else "none",
                    "consensus_score": round(consensus_score, 4),
                    "consensus": False,
                }
            )

        summary = pd.DataFrame(rows)
        if len(summary):
            summary["importance_threshold_pass"] = (
                summary["mean_relative_importance"] >= self.min_mean_relative_importance
            )
            summary["base_consensus"] = (
                (summary["selection_frequency"] >= self.min_model_frequency)
                & (summary["family_support"] >= self.min_family_support)
                & summary["importance_threshold_pass"]
            )
            summary["consensus"] = summary["base_consensus"]
            if self.max_consensus_per_parameter is not None:
                summary["consensus"] = False
                for parameter, pdf in summary.groupby("parameter", sort=False):
                    eligible = pdf[pdf["base_consensus"]].sort_values(
                        ["consensus_score", "selection_frequency", "mean_selected_rank"],
                        ascending=[False, False, True],
                    )
                    keep_index = eligible.head(self.max_consensus_per_parameter).index
                    summary.loc[keep_index, "consensus"] = True
            summary = summary.sort_values(
                ["parameter", "consensus", "consensus_score", "selection_frequency", "mean_rank"],
                ascending=[True, False, False, False, True],
            ).reset_index(drop=True)
        return summary

    def consensus_covariates(self) -> pd.DataFrame:
        summary = self.consensus_summary()
        if summary.empty:
            return summary
        return summary[summary["consensus"]].reset_index(drop=True)

    def selection_frequency_table(self) -> pd.DataFrame:
        summary = self.consensus_summary()
        if summary.empty:
            return summary
        columns = [
            "parameter",
            "covariate",
            "n_models_selected",
            "selection_frequency",
            "family_support",
            "family_frequency",
            "mean_rank",
            "mean_selected_rank",
            "consensus_score",
            "consensus",
        ]
        return summary[columns].copy()

    def family_summary(self) -> pd.DataFrame:
        combined = self.per_model_summary()
        if combined.empty:
            return combined

        family_size = (
            combined[["model", "family"]]
            .drop_duplicates()
            .groupby("family")
            .size()
            .rename("n_models_family")
        )
        selected = combined[combined["selected"]].copy()
        if selected.empty:
            return pd.DataFrame(
                columns=[
                    "parameter",
                    "covariate",
                    "family",
                    "n_models_selected",
                    "family_selection_frequency",
                ]
            )
        summary = (
            selected.groupby(["parameter", "covariate", "family"])
            .size()
            .rename("n_models_selected")
            .reset_index()
            .merge(family_size.reset_index(), on="family", how="left")
        )
        summary["family_selection_frequency"] = (
            summary["n_models_selected"] / summary["n_models_family"].clip(lower=1)
        ).round(4)
        return summary.sort_values(
            ["parameter", "covariate", "family_selection_frequency"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    def compare_with_hybrid(
        self,
        hybrid_report: HybridResults,
        hybrid_tier: str = "confirmed",
    ) -> pd.DataFrame:
        hybrid_getters = {
            "core": hybrid_report.core_covariates,
            "candidate": hybrid_report.candidate_covariates,
            "confirmed": hybrid_report.confirmed_covariates,
            "proxy": hybrid_report.proxy_covariates,
        }
        if hybrid_tier not in hybrid_getters:
            raise ValueError(f"Unknown hybrid_tier '{hybrid_tier}'. Choose from {sorted(hybrid_getters)}")

        hybrid_df = hybrid_getters[hybrid_tier]()[["parameter", "covariate"]].copy()
        hybrid_df["hybrid_selected"] = True
        consensus_df = self.consensus_summary()[["parameter", "covariate", "consensus"]].copy()
        consensus_df = consensus_df.rename(columns={"consensus": "consensus_selected"})

        comparison = consensus_df.merge(hybrid_df, on=["parameter", "covariate"], how="outer")
        comparison["consensus_selected"] = comparison["consensus_selected"].astype("boolean").fillna(False).astype(bool)
        comparison["hybrid_selected"] = comparison["hybrid_selected"].astype("boolean").fillna(False).astype(bool)
        comparison["agreement"] = (
            comparison["consensus_selected"] == comparison["hybrid_selected"]
        )
        return comparison.sort_values(["parameter", "covariate"]).reset_index(drop=True)

    def model_comparison_table(self) -> pd.DataFrame:
        combined = self.per_model_summary()
        if combined.empty:
            return combined
        columns = [
            "model",
            "family",
            "parameter",
            "covariate",
            "mean_importance",
            "importance_rank",
            "relative_importance",
            "standardized_importance",
            "selected",
        ]
        available = [column for column in columns if column in combined.columns]
        return combined[available].sort_values(
            ["parameter", "model", "importance_rank"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    def __repr__(self) -> str:
        return (
            "MultiModelConsensusResults("
            f"models={len(self.models)}, "
            f"consensus={len(self.consensus_covariates())})"
        )


class MultiModelConsensusScreener:
    """Experimental multi-model consensus across curated model families.

    This workflow is intentionally broader than :class:`HybridScreener`.
    It is meant for exploratory comparison and benchmark work rather than
    the default pharmacometric production path.
    """

    def __init__(
        self,
        models: list[str] | None = None,
        families: list[str] | None = None,
        top_k: int = 3,
        n_bootstrap: int = 20,
        significance_threshold: float = 0.05,
        random_state: int = 42,
        use_significance_filter: bool = True,
        run_permutation: bool = False,
        min_r2: float = 0.10,
        corr_threshold: float = 0.80,
        perm_n: int = 20,
        perm_alpha: float = 0.05,
        min_model_frequency: float = 0.40,
        min_family_support: int = 2,
        min_mean_relative_importance: float = 0.05,
        max_consensus_per_parameter: int | None = 3,
        include_extended_linear: bool = False,
        include_neural: bool = False,
        include_optional_boosting: bool = False,
        deep_learning_min_rows: int = 250,
        cv_splits: int = 5,
    ):
        self._requested_models = list(models) if models is not None else None
        self._requested_families = list(families) if families is not None else None
        self.top_k = top_k
        self.n_bootstrap = n_bootstrap
        self.significance_threshold = significance_threshold
        self.random_state = random_state
        self.use_significance_filter = use_significance_filter
        self.run_permutation = run_permutation
        self.min_r2 = min_r2
        self.corr_threshold = corr_threshold
        self.perm_n = perm_n
        self.perm_alpha = perm_alpha
        self.min_model_frequency = min_model_frequency
        self.min_family_support = min_family_support
        self.min_mean_relative_importance = min_mean_relative_importance
        self.max_consensus_per_parameter = max_consensus_per_parameter
        self.include_extended_linear = include_extended_linear
        self.include_neural = include_neural
        self.include_optional_boosting = include_optional_boosting
        self.deep_learning_min_rows = deep_learning_min_rows
        self.cv_splits = cv_splits
        self._results: MultiModelConsensusResults | None = None

    def _resolve_models(self, n_rows: int) -> list[str]:
        if self._requested_models is not None:
            candidates = list(self._requested_models)
        elif self._requested_families is not None:
            availability = check_engine_availability()
            candidates = [
                model
                for model, available in availability.items()
                if available and get_engine_family(model) in self._requested_families
            ]
        else:
            candidates = list(DEFAULT_MULTIMODEL_MODELS)
            if self.include_extended_linear:
                candidates.extend(EXTENDED_MULTIMODEL_MODELS)
            if self.include_optional_boosting:
                candidates.extend(OPTIONAL_MULTIMODEL_MODELS)
            if self.include_neural:
                candidates.extend(NEURAL_MULTIMODEL_MODELS)

        availability = check_engine_availability()
        models = [model for model in candidates if availability.get(model, False)]
        unavailable = [model for model in candidates if not availability.get(model, False)]

        if unavailable:
            warnings.warn(
                f"Consensus models not available (install their packages): {unavailable}. "
                f"Running with: {models}",
                stacklevel=3,
            )
        if not models:
            raise ValueError("No consensus models are available in the current environment.")

        deep_models = [model for model in models if get_engine_family(model) in {"sparse_neural", "deep_learning"}]
        if deep_models and n_rows < self.deep_learning_min_rows:
            warnings.warn(
                f"Neural models {deep_models} requested with n={n_rows}. "
                f"They are experimental and may overfit small popPK datasets.",
                stacklevel=3,
            )
        return models

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

    def _annotate_summary(self, summary: pd.DataFrame, model: str) -> pd.DataFrame:
        summary = summary.copy()
        summary["model"] = model
        summary["family"] = get_engine_family(model)
        summary["importance_rank"] = (
            summary.groupby("parameter")["mean_importance"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )
        max_rank = summary.groupby("parameter")["importance_rank"].transform("max").replace(0, 1)
        rank_weight = np.where(
            max_rank > 1,
            1.0 - ((summary["importance_rank"] - 1) / (max_rank - 1)),
            1.0,
        )
        summary["relative_importance"] = summary.groupby("parameter")["mean_importance"].transform(
            lambda series: series / max(float(series.max()), 1e-12)
        )
        summary["relative_importance"] = summary["relative_importance"].fillna(0.0).round(4)
        summary["standardized_importance"] = (
            summary["relative_importance"] * rank_weight
        ).round(4)
        summary["top_k_selected"] = summary["importance_rank"] <= self.top_k
        vote_col = "final_significant" if "final_significant" in summary.columns else "significant"
        summary["selected"] = summary["top_k_selected"] & summary[vote_col].astype(bool)
        return summary

    def fit(
        self,
        ebes: pd.DataFrame,
        covariates: pd.DataFrame,
        parameter_names: list[str] | None = None,
        covariate_names: list[str] | None = None,
    ) -> MultiModelConsensusResults:
        models = self._resolve_models(len(ebes))
        cov_input = covariates[covariate_names].copy() if covariate_names else covariates.copy()
        method_results: dict[str, object] = {}
        method_summaries: dict[str, pd.DataFrame] = {}

        for model in models:
            screener = CovariateScreener(
                method=model,
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
            method_results[model] = results

            if self.use_significance_filter:
                filter_ = self._build_significance_filter()
                summary = filter_.apply(
                    results,
                    cov_input,
                    method_name=model,
                    run_permutation=self.run_permutation,
                )
            else:
                summary = self._empty_filter_columns(results.summary())

            method_summaries[model] = self._annotate_summary(summary, model)

        self._results = MultiModelConsensusResults(
            method_results=method_results,
            method_summaries=method_summaries,
            models=models,
            top_k=self.top_k,
            min_model_frequency=self.min_model_frequency,
            min_family_support=self.min_family_support,
            min_mean_relative_importance=self.min_mean_relative_importance,
            max_consensus_per_parameter=self.max_consensus_per_parameter,
        )
        return self._results

    @property
    def results(self) -> MultiModelConsensusResults | None:
        return self._results
