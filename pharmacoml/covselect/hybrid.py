"""Primary hybrid ML-assisted covariate preselection workflow."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pharmacoml.covselect.scm import SCMBridge, SCMResults
from pharmacoml.covselect.penalized import PenalizedScreener
from pharmacoml.covselect.shapcov import ShapCovScreener
from pharmacoml.covselect.significance import CorrelationFilter
from pharmacoml.covselect.stg import STGScreener
from pharmacoml.covselect.symbolic import SymbolicStructureScreener
from pharmacoml.covselect.selection_utils import association_matrix, build_interaction_terms, estimate_parameter_information
from pharmacoml.covselect.traditional import TraditionalScreener


@dataclass
class HybridArtifacts:
    """Internal handles to the per-stage fitted screeners."""

    shapcov: ShapCovScreener
    penalized: PenalizedScreener
    stg: STGScreener | None
    traditional_summary: pd.DataFrame | None
    scm_results: SCMResults | None
    symbolic_summary: pd.DataFrame | None
    parameter_profiles: dict[str, dict] | None = None


class HybridScreener:
    """Hybrid PMx-ML preselection workflow for daily pharmacometric use.

    The default workflow mirrors the literature more closely than a broad
    ensemble: one explainable tree model for discovery, one penalized model
    for confirmation, and an optional traditional baseline for comparison.
    """

    def __init__(
        self,
        boosting_method: str = "catboost",
        penalized_method: str = "aalasso",
        n_bootstrap: int = 50,
        significance_threshold: float = 0.05,
        random_state: int = 42,
        cv_splits: int = 5,
        min_r2: float = 0.10,
        corr_threshold: float = 0.80,
        perm_n: int = 20,
        perm_alpha: float = 0.05,
        run_permutation: bool = False,
        include_traditional: bool = True,
        include_stg: bool = False,
        include_scm: bool = True,
        include_symbolic: bool = False,
        symbolic_backend: str = "basis",
        symbolic_backend_kwargs: dict | None = None,
        rfe_enabled: bool = True,
        rfe_drop_fraction: float = 0.25,
        rfe_min_features: int = 3,
        rfe_max_rounds: int = 3,
        rfe_repeats: int = 3,
        rfe_retain_threshold: float = 0.60,
        shrinkage_awareness: bool = True,
        shrinkage_score_boost: float = 0.25,
        shrinkage_support_relaxation: float = 0.45,
        shrinkage_candidate_floor: float = 0.45,
        shrinkage_warn_threshold: float = 0.20,
        shrinkage_tighten_threshold: float = 0.40,
        shrinkage_suppress_threshold: float = 0.70,
        candidate_score_threshold: float = 0.10,
        enable_rescue: bool = True,
        rescue_score_floor: float = 0.02,
        rescue_max_per_parameter: int = 2,
        rescue_alpha: float = 0.05,
        rescue_min_delta_aic: float = 1.0,
        rescue_requires_parameter_anchor: bool = True,
        rescue_redundancy_threshold: float = 0.65,
        small_sample_n: int = 100,
        small_sample_rescue_alpha: float = 0.15,
        small_sample_rescue_min_delta_aic: float = 0.25,
        low_prevalence_threshold: float = 0.25,
        preserve_proxy_pairs: list[tuple[str, str]] | None = None,
        preserve_biological_distinctness: bool = False,
        proxy_preserve_corr_threshold: float = 0.995,
        proxy_preserve_r2_threshold: float = 0.99,
        proxy_preserve_intercept_std_ratio: float = 0.10,
        include_interactions: bool = False,
        interaction_top_n: int = 5,
        interaction_max_pairs: int = 10,
    ):
        self.boosting_method = boosting_method
        self.penalized_method = penalized_method
        self.n_bootstrap = n_bootstrap
        self.significance_threshold = significance_threshold
        self.random_state = random_state
        self.cv_splits = cv_splits
        self.min_r2 = min_r2
        self.corr_threshold = corr_threshold
        self.perm_n = perm_n
        self.perm_alpha = perm_alpha
        self.run_permutation = run_permutation
        self.include_traditional = include_traditional
        self.include_stg = include_stg
        self.include_scm = include_scm
        self.include_symbolic = include_symbolic
        self.symbolic_backend = symbolic_backend
        self.symbolic_backend_kwargs = dict(symbolic_backend_kwargs or {})
        self.rfe_enabled = rfe_enabled
        self.rfe_drop_fraction = rfe_drop_fraction
        self.rfe_min_features = rfe_min_features
        self.rfe_max_rounds = rfe_max_rounds
        self.rfe_repeats = rfe_repeats
        self.rfe_retain_threshold = rfe_retain_threshold
        self.shrinkage_awareness = shrinkage_awareness
        self.shrinkage_score_boost = shrinkage_score_boost
        self.shrinkage_support_relaxation = shrinkage_support_relaxation
        self.shrinkage_candidate_floor = shrinkage_candidate_floor
        self.shrinkage_warn_threshold = shrinkage_warn_threshold
        self.shrinkage_tighten_threshold = shrinkage_tighten_threshold
        self.shrinkage_suppress_threshold = shrinkage_suppress_threshold
        self.candidate_score_threshold = candidate_score_threshold
        self.enable_rescue = enable_rescue
        self.rescue_score_floor = rescue_score_floor
        self.rescue_max_per_parameter = rescue_max_per_parameter
        self.rescue_alpha = rescue_alpha
        self.rescue_min_delta_aic = rescue_min_delta_aic
        self.rescue_requires_parameter_anchor = rescue_requires_parameter_anchor
        self.rescue_redundancy_threshold = rescue_redundancy_threshold
        self.small_sample_n = small_sample_n
        self.small_sample_rescue_alpha = small_sample_rescue_alpha
        self.small_sample_rescue_min_delta_aic = small_sample_rescue_min_delta_aic
        self.low_prevalence_threshold = low_prevalence_threshold
        self.preserve_proxy_pairs = {
            frozenset((left, right))
            for left, right in (preserve_proxy_pairs or [])
            if left and right
        }
        self.preserve_biological_distinctness = preserve_biological_distinctness
        self.proxy_preserve_corr_threshold = proxy_preserve_corr_threshold
        self.proxy_preserve_r2_threshold = proxy_preserve_r2_threshold
        self.proxy_preserve_intercept_std_ratio = proxy_preserve_intercept_std_ratio
        self.include_interactions = include_interactions
        self.interaction_top_n = interaction_top_n
        self.interaction_max_pairs = interaction_max_pairs

        self._artifacts: HybridArtifacts | None = None
        self._report: HybridResults | None = None

    def _build_shapcov(self) -> ShapCovScreener:
        return ShapCovScreener(
            method=self.boosting_method,
            n_bootstrap=self.n_bootstrap,
            significance_threshold=self.significance_threshold,
            random_state=self.random_state,
            cv_splits=self.cv_splits,
            min_r2=self.min_r2,
            corr_threshold=self.corr_threshold,
            perm_n=self.perm_n,
            perm_alpha=self.perm_alpha,
            run_permutation=self.run_permutation,
            rfe_enabled=self.rfe_enabled,
            rfe_drop_fraction=self.rfe_drop_fraction,
            rfe_min_features=self.rfe_min_features,
            rfe_max_rounds=self.rfe_max_rounds,
            rfe_repeats=self.rfe_repeats,
            rfe_retain_threshold=self.rfe_retain_threshold,
        )

    def _build_penalized(self) -> PenalizedScreener:
        return PenalizedScreener(
            method=self.penalized_method,
            n_bootstrap=self.n_bootstrap,
            significance_threshold=self.significance_threshold,
            random_state=self.random_state,
            cv_splits=self.cv_splits,
            min_r2=self.min_r2,
            corr_threshold=self.corr_threshold,
            perm_n=self.perm_n,
            perm_alpha=self.perm_alpha,
            run_permutation=self.run_permutation,
        )

    def _build_stg(self) -> STGScreener:
        return STGScreener(
            n_bootstrap=max(20, self.n_bootstrap // 2),
            significance_threshold=self.significance_threshold,
            random_state=self.random_state,
            cv_splits=self.cv_splits,
            min_r2=self.min_r2,
            corr_threshold=self.corr_threshold,
            perm_n=self.perm_n,
            perm_alpha=self.perm_alpha,
            run_permutation=self.run_permutation,
        )

    def fit(
        self,
        ebes: pd.DataFrame,
        covariates: pd.DataFrame,
        parameter_names: list[str] | None = None,
        covariate_names: list[str] | None = None,
        parameter_shrinkage: dict[str, float] | None = None,
    ) -> "HybridResults":
        cov_input = covariates[covariate_names].copy() if covariate_names else covariates.copy()
        ebe_input = ebes[parameter_names] if parameter_names else ebes
        parameter_profiles = self._resolve_parameter_profiles(
            ebe_input=ebe_input,
            parameter_shrinkage=parameter_shrinkage,
        )

        shapcov = self._build_shapcov()
        shap_summary = shapcov.fit(
            ebes=ebe_input,
            covariates=cov_input,
            parameter_names=parameter_names,
            covariate_names=list(cov_input.columns),
        )

        penalized = self._build_penalized()
        penalized_summary = penalized.fit(
            ebes=ebe_input,
            covariates=cov_input,
            parameter_names=parameter_names,
            covariate_names=list(cov_input.columns),
        )

        stg = None
        stg_summary = None
        if self.include_stg:
            stg = self._build_stg()
            if stg.is_available():
                stg_summary = stg.fit(
                    ebes=ebe_input,
                    covariates=cov_input,
                    parameter_names=parameter_names,
                    covariate_names=list(cov_input.columns),
                )

        traditional_summary = None
        if self.include_traditional:
            trad = TraditionalScreener(alpha=0.01)
            traditional_summary = trad.fit(ebe_input, cov_input).summary()

        merged = self._merge_stage_outputs(
            shap_summary=shap_summary,
            penalized_summary=penalized_summary,
            stg_summary=stg_summary,
            traditional_summary=traditional_summary,
            parameter_profiles=parameter_profiles,
        )
        if self.include_interactions:
            merged = self._augment_with_interactions(
                summary=merged,
                ebes=ebe_input,
                covariates=cov_input,
                parameter_names=list(ebe_input.columns),
            )
        scm_results = None
        if self.include_scm:
            scm_bridge = SCMBridge()
            scm_results = scm_bridge.fit(
                ebes=ebe_input,
                covariates=cov_input,
                candidate_table=merged[merged["tier"].isin(["core", "candidate"])],
                parameter_names=parameter_names,
            )
            merged = self._merge_scm_outputs(merged, scm_results.summary())
        else:
            merged["scm_selected"] = False
            merged["scm_p_value"] = np.nan
            merged["final_aic"] = np.nan
            merged["final_bic"] = np.nan
            merged["final_r2"] = np.nan
            merged["target_scale"] = ""
        merged = self._apply_rescue_candidates(
            merged,
            ebes=ebe_input,
            covariates=cov_input,
        )
        merged = self._prune_redundant_candidates(merged, cov_input)
        symbolic_summary = None
        if self.include_symbolic:
            symbolic = SymbolicStructureScreener(
                symbolic_backend=self.symbolic_backend,
                symbolic_backend_kwargs=self.symbolic_backend_kwargs,
                random_state=self.random_state,
            )
            symbolic_summary = symbolic.fit(
                ebes=ebe_input,
                covariates=cov_input,
                candidate_table=merged[merged["tier"].isin(["core", "candidate"])],
                parameter_names=parameter_names,
            )
            merged = self._merge_symbolic_outputs(merged, symbolic_summary)
        else:
            merged["symbolic_selected"] = False
            merged["symbolic_form"] = ""
            merged["symbolic_expression"] = ""
            merged["symbolic_score"] = 0.0
            merged["symbolic_p_value"] = np.nan
        merged = self._assign_proxy_groups(merged, cov_input)
        merged = merged.sort_values(
            ["parameter", "tier_rank", "combined_score", "shapcov_importance"],
            ascending=[True, True, False, False],
        ).reset_index(drop=True)

        proxy_groups = self._proxy_group_table(merged)
        self._artifacts = HybridArtifacts(
            shapcov=shapcov,
            penalized=penalized,
            stg=stg,
            traditional_summary=traditional_summary,
            scm_results=scm_results,
            symbolic_summary=symbolic_summary,
            parameter_profiles=parameter_profiles,
        )
        self._report = HybridResults(
            summary_df=merged,
            proxy_groups_df=proxy_groups,
            artifacts=self._artifacts,
        )
        return self._report

    def _resolve_parameter_profiles(
        self,
        ebe_input: pd.DataFrame,
        parameter_shrinkage: dict[str, float] | None = None,
    ) -> dict[str, dict]:
        profiles = {
            param: dict(estimate_parameter_information(ebe_input[param]))
            for param in ebe_input.columns
        }
        supplied = parameter_shrinkage or {}

        for param, profile in profiles.items():
            proxy = float(profile.get("shrinkage_proxy", 0.0))
            if param in supplied and supplied[param] is not None:
                value = float(np.clip(supplied[param], 0.0, 1.0))
                source = "user"
            else:
                value = proxy
                source = "proxy"

            if value >= self.shrinkage_suppress_threshold:
                status = "suppressed"
            elif value >= self.shrinkage_tighten_threshold:
                status = "tightened"
            elif value >= self.shrinkage_warn_threshold:
                status = "warn"
            else:
                status = "normal"

            profile["shrinkage_proxy"] = proxy
            profile["shrinkage_value"] = round(value, 4)
            profile["shrinkage_source"] = source
            profile["shrinkage_status"] = status
            if source == "user":
                profile["low_information"] = bool(value >= self.shrinkage_tighten_threshold)

        return profiles

    def _merge_stage_outputs(
        self,
        shap_summary: pd.DataFrame,
        penalized_summary: pd.DataFrame,
        stg_summary: pd.DataFrame | None,
        traditional_summary: pd.DataFrame | None,
        parameter_profiles: dict[str, dict] | None = None,
    ) -> pd.DataFrame:
        shap_summary = shap_summary.copy()
        for col, default in {
            "rfe_retained": False,
            "rfe_retain_frequency": 0.0,
        }.items():
            if col not in shap_summary.columns:
                shap_summary[col] = default

        base = shap_summary[
            [
                "parameter",
                "covariate",
                "functional_form",
                "mean_importance",
                "standardized_importance",
                "stability_frequency",
                "cv_r2",
                "final_significant",
                "corr_filtered",
                "perm_p_value",
                "perm_q_value",
                "rfe_retained",
                "rfe_retain_frequency",
            ]
        ].rename(
            columns={
                "mean_importance": "shapcov_importance",
                "standardized_importance": "shapcov_score",
                "stability_frequency": "shapcov_stability",
                "cv_r2": "shapcov_cv_r2",
                "final_significant": "shapcov_selected",
                "corr_filtered": "shapcov_corr_filtered",
                "perm_p_value": "shapcov_perm_p",
                "perm_q_value": "shapcov_perm_q",
                "rfe_retained": "shapcov_rfe_retained",
                "rfe_retain_frequency": "shapcov_rfe_retain_frequency",
            }
        )

        penalized = penalized_summary[
            [
                "parameter",
                "covariate",
                "mean_importance",
                "standardized_importance",
                "stability_frequency",
                "cv_r2",
                "final_significant",
                "corr_filtered",
                "perm_p_value",
                "perm_q_value",
            ]
        ].rename(
            columns={
                "mean_importance": "penalized_importance",
                "standardized_importance": "penalized_score",
                "stability_frequency": "penalized_stability",
                "cv_r2": "penalized_cv_r2",
                "final_significant": "penalized_selected",
                "corr_filtered": "penalized_corr_filtered",
                "perm_p_value": "penalized_perm_p",
                "perm_q_value": "penalized_perm_q",
            }
        )

        merged = base.merge(penalized, on=["parameter", "covariate"], how="outer")

        if stg_summary is not None and len(stg_summary):
            stg = stg_summary[
                [
                    "parameter",
                    "covariate",
                    "mean_importance",
                    "standardized_importance",
                    "stability_frequency",
                    "cv_r2",
                    "final_significant",
                    "corr_filtered",
                    "perm_p_value",
                    "perm_q_value",
                ]
            ].rename(
                columns={
                    "mean_importance": "stg_importance",
                    "standardized_importance": "stg_score",
                    "stability_frequency": "stg_stability",
                    "cv_r2": "stg_cv_r2",
                    "final_significant": "stg_selected",
                    "corr_filtered": "stg_corr_filtered",
                    "perm_p_value": "stg_perm_p",
                    "perm_q_value": "stg_perm_q",
                }
            )
            merged = merged.merge(stg, on=["parameter", "covariate"], how="left")
        else:
            merged["stg_importance"] = 0.0
            merged["stg_score"] = 0.0
            merged["stg_stability"] = 0.0
            merged["stg_cv_r2"] = 0.0
            merged["stg_selected"] = False
            merged["stg_corr_filtered"] = False
            merged["stg_perm_p"] = np.nan
            merged["stg_perm_q"] = np.nan

        if traditional_summary is not None and len(traditional_summary):
            traditional = traditional_summary[
                ["parameter", "covariate", "significant", "p_value", "effect_size"]
            ].rename(
                columns={
                    "significant": "traditional_selected",
                    "p_value": "traditional_p_value",
                    "effect_size": "traditional_effect_size",
                }
            )
            merged = merged.merge(traditional, on=["parameter", "covariate"], how="left")
        else:
            merged["traditional_selected"] = False
            merged["traditional_p_value"] = np.nan
            merged["traditional_effect_size"] = np.nan

        for col in [
            "shapcov_importance",
            "shapcov_score",
            "shapcov_stability",
            "shapcov_cv_r2",
            "penalized_importance",
            "penalized_score",
            "penalized_stability",
            "penalized_cv_r2",
            "stg_importance",
            "stg_score",
            "stg_stability",
            "stg_cv_r2",
            "traditional_effect_size",
            "shapcov_rfe_retain_frequency",
        ]:
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = merged[col].fillna(0.0)

        for col in [
            "shapcov_selected",
            "penalized_selected",
            "stg_selected",
            "traditional_selected",
            "shapcov_corr_filtered",
            "penalized_corr_filtered",
            "stg_corr_filtered",
            "shapcov_rfe_retained",
        ]:
            if col not in merged.columns:
                merged[col] = False
            merged[col] = merged[col].fillna(False).astype(bool)

        stage_weights = {
            "shapcov_score": 0.50,
            "penalized_score": 0.25,
            "traditional_selected": 0.10,
        }
        if self.include_stg:
            stage_weights["stg_score"] = 0.15

        weighted = np.zeros(len(merged), dtype=float)
        total_weight = 0.0
        for col, weight in stage_weights.items():
            values = merged[col].astype(float) if col != "traditional_selected" else merged[col].astype(float)
            weighted += weight * values
            total_weight += weight
        merged["combined_score"] = np.round(weighted / max(total_weight, 1e-12), 4)
        parameter_profiles = parameter_profiles or {}
        merged["shrinkage_proxy"] = merged["parameter"].map(
            lambda p: float(parameter_profiles.get(p, {}).get("shrinkage_proxy", 0.0))
        ).fillna(0.0)
        merged["shrinkage_value"] = merged["parameter"].map(
            lambda p: float(parameter_profiles.get(p, {}).get("shrinkage_value", parameter_profiles.get(p, {}).get("shrinkage_proxy", 0.0)))
        ).fillna(0.0)
        merged["shrinkage_source"] = merged["parameter"].map(
            lambda p: str(parameter_profiles.get(p, {}).get("shrinkage_source", "proxy"))
        ).fillna("proxy")
        merged["shrinkage_status"] = merged["parameter"].map(
            lambda p: str(parameter_profiles.get(p, {}).get("shrinkage_status", "normal"))
        ).fillna("normal")
        merged["low_information_parameter"] = merged["parameter"].map(
            lambda p: bool(parameter_profiles.get(p, {}).get("low_information", False))
        ).fillna(False)
        merged["stability_consensus"] = merged[
            ["shapcov_stability", "penalized_stability", "stg_stability"]
        ].max(axis=1).fillna(0.0)
        merged["r2_consensus"] = merged[
            ["shapcov_cv_r2", "penalized_cv_r2", "stg_cv_r2"]
        ].max(axis=1).fillna(0.0)
        merged["screening_suppressed"] = False
        merged["support_requirement"] = 2
        merged["combined_score_adjusted"] = merged["combined_score"]
        merged["candidate_threshold_effective"] = self.candidate_score_threshold
        if self.shrinkage_awareness:
            proxy_mask = merged["shrinkage_source"] != "user"
            if proxy_mask.any():
                stability_bonus = (
                    merged.loc[proxy_mask, "shrinkage_proxy"] *
                    merged.loc[proxy_mask, "stability_consensus"] *
                    self.shrinkage_score_boost
                )
                instability_penalty = (
                    merged.loc[proxy_mask, "shrinkage_proxy"] *
                    (1.0 - merged.loc[proxy_mask, "stability_consensus"]) *
                    0.10
                )
                merged.loc[proxy_mask, "combined_score_adjusted"] = np.round(
                    merged.loc[proxy_mask, "combined_score"] * (1.0 + stability_bonus - instability_penalty),
                    4,
                )
                merged.loc[proxy_mask, "candidate_threshold_effective"] = np.round(
                    self.candidate_score_threshold * (
                        1.0 - self.shrinkage_support_relaxation *
                        merged.loc[proxy_mask, "shrinkage_proxy"] *
                        merged.loc[proxy_mask, "stability_consensus"]
                    ),
                    4,
                ).clip(lower=self.candidate_score_threshold * 0.55)

            user_mask = merged["shrinkage_source"] == "user"
            if user_mask.any():
                user_values = merged.loc[user_mask, "shrinkage_value"].astype(float)
                moderate_mask = user_values >= self.shrinkage_warn_threshold
                high_mask = user_values >= self.shrinkage_tighten_threshold
                suppress_mask = user_values >= self.shrinkage_suppress_threshold

                penalty = np.where(high_mask, 0.35, np.where(moderate_mask, 0.15, 0.0))
                threshold_multiplier = np.where(high_mask, 1.35, np.where(moderate_mask, 1.10, 1.0))

                merged.loc[user_mask, "combined_score_adjusted"] = np.round(
                    merged.loc[user_mask, "combined_score"].astype(float) * (1.0 - penalty),
                    4,
                )
                merged.loc[user_mask, "candidate_threshold_effective"] = np.round(
                    self.candidate_score_threshold * threshold_multiplier,
                    4,
                )
                high_index = user_values.index[high_mask]
                if len(high_index):
                    merged.loc[high_index, "support_requirement"] = 3
                suppressed_index = user_values.index[suppress_mask]
                if len(suppressed_index):
                    merged.loc[suppressed_index, "screening_suppressed"] = True
                    merged.loc[suppressed_index, "combined_score_adjusted"] = 0.0
                    merged.loc[suppressed_index, "candidate_threshold_effective"] = np.inf
                    merged.loc[suppressed_index, "support_requirement"] = 99
        merged["support_count"] = (
            merged["shapcov_selected"].astype(int) +
            merged["penalized_selected"].astype(int) +
            merged["stg_selected"].astype(int) +
            merged["traditional_selected"].astype(int)
        )
        merged["score_rescue_candidate"] = (
            merged["shapcov_rfe_retained"] &
            (merged["shapcov_score"] >= 0.70) &
            (merged["shapcov_stability"] >= 0.80)
        ) | (
            (merged["penalized_score"] >= 0.90) &
            (merged["penalized_stability"] >= 0.85)
        )
        if self.shrinkage_awareness:
            relax_mask = (
                (merged["shrinkage_source"] != "user") &
                merged["low_information_parameter"] &
                (merged["combined_score_adjusted"] >= self.shrinkage_candidate_floor) &
                (merged["stability_consensus"] >= 0.40) &
                (merged["shapcov_selected"] | merged["penalized_selected"] | merged["stg_selected"])
            )
            merged.loc[relax_mask, "support_requirement"] = 1
        merged["support_label"] = merged.apply(self._support_label, axis=1)

        merged["tier"] = "rejected"
        core_mask = (
            merged["shapcov_selected"] &
            merged["penalized_selected"] &
            (merged["combined_score_adjusted"] >= merged["candidate_threshold_effective"]) &
            (~merged["screening_suppressed"]) &
            ~(
                (merged["shrinkage_source"] == "user") &
                (merged["shrinkage_value"] >= self.shrinkage_tighten_threshold)
            )
        )
        candidate_mask = (
            (~merged["screening_suppressed"]) &
            (merged["combined_score_adjusted"] >= merged["candidate_threshold_effective"]) &
            (
                core_mask |
                (merged["support_count"] >= merged["support_requirement"]) |
                merged["score_rescue_candidate"]
            )
        )
        merged.loc[candidate_mask, "tier"] = "candidate"
        merged.loc[core_mask, "tier"] = "core"
        merged["tier_rank"] = merged["tier"].map({"core": 0, "candidate": 1, "proxy": 2, "rejected": 3}).fillna(4).astype(int)
        return merged

    @staticmethod
    def _support_label(row) -> str:
        labels = []
        if bool(row.get("shapcov_selected", False)):
            labels.append("boosting")
        if bool(row.get("penalized_selected", False)):
            labels.append("penalized")
        if bool(row.get("stg_selected", False)):
            labels.append("stg")
        if bool(row.get("traditional_selected", False)):
            labels.append("traditional")
        if bool(row.get("score_rescue_candidate", False)):
            labels.append("score-rescue")
        return "+".join(labels) if labels else "none"

    def _augment_with_interactions(
        self,
        summary: pd.DataFrame,
        ebes: pd.DataFrame,
        covariates: pd.DataFrame,
        parameter_names: list[str],
    ) -> pd.DataFrame:
        summary = summary.copy()
        interaction_rows: list[dict] = []

        for param in parameter_names:
            param_summary = summary[
                (summary["parameter"] == param) &
                (~summary["covariate"].astype(str).str.contains("__", regex=False))
            ].copy()
            if len(param_summary) == 0:
                continue

            preferred = param_summary[param_summary["tier"].isin(["core", "candidate"])].copy()
            if preferred.empty:
                preferred = param_summary.copy()
            preferred = preferred.sort_values(
                ["tier_rank", "combined_score_adjusted", "combined_score", "shapcov_score", "penalized_score"],
                ascending=[True, False, False, False, False],
            )
            candidate_covs = preferred["covariate"].drop_duplicates().head(self.interaction_top_n).tolist()
            if len(candidate_covs) < 2:
                continue

            interaction_df, metadata = build_interaction_terms(
                covariates,
                candidate_covariates=candidate_covs,
                max_pairs=self.interaction_max_pairs,
            )
            if interaction_df.empty:
                continue

            interaction_summary = PenalizedScreener(
                method="aalasso",
                n_bootstrap=max(3, min(self.n_bootstrap, 8)),
                significance_threshold=self.significance_threshold,
                random_state=self.random_state,
                cv_splits=self.cv_splits,
                min_r2=self.min_r2,
                corr_threshold=self.corr_threshold,
                perm_n=self.perm_n,
                perm_alpha=self.perm_alpha,
                run_permutation=self.run_permutation,
            ).fit(
                ebes=ebes[[param]],
                covariates=interaction_df,
                parameter_names=[param],
                covariate_names=list(interaction_df.columns),
            )

            for _, row in interaction_summary.iterrows():
                cov = str(row["covariate"])
                meta = metadata.get(cov)
                if meta is None:
                    continue
                interaction_selected = bool(row.get("final_significant", False))
                interaction_score = float(row.get("standardized_importance", 0.0))
                candidate_floor = max(self.candidate_score_threshold, 0.05)
                tier = "candidate" if interaction_selected and interaction_score >= candidate_floor else "rejected"
                support_label = "interaction-penalized" if tier == "candidate" else "none"
                interaction_rows.append(
                    {
                        "parameter": param,
                        "covariate": cov,
                        "functional_form": "interaction",
                        "shapcov_importance": 0.0,
                        "shapcov_score": 0.0,
                        "shapcov_stability": 0.0,
                        "shapcov_cv_r2": 0.0,
                        "shapcov_selected": False,
                        "shapcov_corr_filtered": False,
                        "shapcov_perm_p": np.nan,
                        "shapcov_perm_q": np.nan,
                        "shapcov_rfe_retained": False,
                        "shapcov_rfe_retain_frequency": 0.0,
                        "penalized_importance": float(row.get("mean_importance", 0.0)),
                        "penalized_score": interaction_score,
                        "penalized_stability": float(row.get("stability_frequency", 0.0)),
                        "penalized_cv_r2": float(row.get("cv_r2", 0.0)),
                        "penalized_selected": interaction_selected,
                        "penalized_corr_filtered": bool(row.get("corr_filtered", False)),
                        "penalized_perm_p": row.get("perm_p_value", np.nan),
                        "penalized_perm_q": row.get("perm_q_value", np.nan),
                        "stg_importance": 0.0,
                        "stg_score": 0.0,
                        "stg_stability": 0.0,
                        "stg_cv_r2": 0.0,
                        "stg_selected": False,
                        "stg_corr_filtered": False,
                        "stg_perm_p": np.nan,
                        "stg_perm_q": np.nan,
                        "traditional_selected": False,
                        "traditional_p_value": np.nan,
                        "traditional_effect_size": np.nan,
                        "combined_score": round(interaction_score, 4),
                        "shrinkage_proxy": float(summary.loc[summary["parameter"] == param, "shrinkage_proxy"].max()),
                        "shrinkage_value": float(summary.loc[summary["parameter"] == param, "shrinkage_value"].max()),
                        "shrinkage_source": str(summary.loc[summary["parameter"] == param, "shrinkage_source"].iloc[0]),
                        "shrinkage_status": str(summary.loc[summary["parameter"] == param, "shrinkage_status"].iloc[0]),
                        "low_information_parameter": bool(summary.loc[summary["parameter"] == param, "low_information_parameter"].max()),
                        "stability_consensus": float(row.get("stability_frequency", 0.0)),
                        "r2_consensus": float(row.get("cv_r2", 0.0)),
                        "screening_suppressed": False,
                        "combined_score_adjusted": round(interaction_score, 4),
                        "candidate_threshold_effective": candidate_floor,
                        "support_count": int(interaction_selected),
                        "score_rescue_candidate": interaction_selected,
                        "support_requirement": 1,
                        "support_label": support_label,
                        "tier": tier,
                        "tier_rank": 1 if tier == "candidate" else 3,
                        "interaction_selected": interaction_selected,
                        "interaction_left": meta["left"],
                        "interaction_right": meta["right"],
                        "interaction_operator": meta["operator"],
                    }
                )

        if not interaction_rows:
            return summary

        interaction_df = pd.DataFrame(interaction_rows)
        for column in interaction_df.columns:
            if column not in summary.columns:
                summary[column] = np.nan
        for column in summary.columns:
            if column not in interaction_df.columns:
                interaction_df[column] = np.nan

        combined = pd.concat([summary, interaction_df[summary.columns]], ignore_index=True, sort=False)
        combined["tier_rank"] = combined["tier"].map({"core": 0, "candidate": 1, "proxy": 2, "rejected": 3}).fillna(4).astype(int)
        return combined

    def _merge_scm_outputs(self, summary: pd.DataFrame, scm_summary: pd.DataFrame) -> pd.DataFrame:
        merged = summary.merge(
            scm_summary[
                [
                    "parameter",
                    "covariate",
                    "selected",
                    "scm_p_value",
                    "final_aic",
                    "final_bic",
                    "final_r2",
                    "target_scale",
                ]
            ].rename(columns={"selected": "scm_selected"}),
            on=["parameter", "covariate"],
            how="left",
        )
        merged["scm_selected"] = merged["scm_selected"].where(merged["scm_selected"].notna(), False).astype(bool)
        merged["scm_p_value"] = merged["scm_p_value"].fillna(np.nan)
        merged["target_scale"] = merged["target_scale"].fillna("")
        merged["pre_scm_candidate"] = merged["tier"].isin(["core", "candidate"])
        merged.loc[merged["scm_selected"], "support_label"] = (
            merged.loc[merged["scm_selected"], "support_label"].replace("none", "") + "+scm"
        ).str.strip("+")

        core_mask = (
            (merged["tier"] == "core") &
            (
                merged["scm_selected"] |
                (merged["support_count"] >= 3) |
                merged["traditional_selected"]
            ) &
            (~merged["screening_suppressed"]) &
            ~(
                (merged["shrinkage_source"] == "user") &
                (merged["shrinkage_value"] >= self.shrinkage_tighten_threshold)
            )
        )
        candidate_mask = (
            merged["pre_scm_candidate"] &
            (~merged["screening_suppressed"]) &
            (
                core_mask |
                merged["scm_selected"] |
                (merged["support_count"] >= merged.get("support_requirement", 2)) |
                merged["score_rescue_candidate"]
            )
        )

        merged["tier"] = "rejected"
        merged.loc[candidate_mask, "tier"] = "candidate"
        merged.loc[core_mask, "tier"] = "core"
        merged["tier_rank"] = merged["tier"].map({"core": 0, "candidate": 1, "proxy": 2, "rejected": 3}).fillna(4).astype(int)
        return merged

    @staticmethod
    def _is_binary_series(series: pd.Series) -> bool:
        vals = pd.Series(series).dropna().unique().tolist()
        return len(vals) <= 2 and set(vals).issubset({0, 1, 0.0, 1.0, False, True})

    def _apply_rescue_candidates(
        self,
        summary: pd.DataFrame,
        ebes: pd.DataFrame,
        covariates: pd.DataFrame,
    ) -> pd.DataFrame:
        summary = summary.copy()
        if "screening_suppressed" not in summary.columns:
            summary["screening_suppressed"] = False
        summary["rescued_confirmed"] = False
        summary["rescue_p_value"] = np.nan
        summary["rescue_robust_p_value"] = np.nan
        summary["rescue_delta_aic"] = np.nan
        summary["rescue_reason"] = ""
        summary["confirmation_status"] = np.where(summary.get("scm_selected", False), "scm", "unconfirmed")

        if not self.enable_rescue:
            return summary

        bridge = SCMBridge(
            enter_alpha=self.rescue_alpha,
            stay_alpha=max(self.rescue_alpha, self.small_sample_rescue_alpha),
            min_delta_aic=self.rescue_min_delta_aic,
            max_terms=max(2, self.rescue_max_per_parameter),
        )
        rescue_pool = summary[
            (summary["tier"] == "rejected") &
            (~summary["screening_suppressed"]) &
            ((summary["shapcov_selected"]) | (summary["stg_selected"]) | (summary.get("score_rescue_candidate", False))) &
            (summary["combined_score"] >= self.rescue_score_floor)
        ].copy()
        top_covariate_indices = {}
        if len(rescue_pool):
            ranked_pool = rescue_pool.sort_values(
                ["combined_score", "shapcov_importance"],
                ascending=[False, False],
            )
            top_covariate_indices = {
                cov: ranked_pool[ranked_pool["covariate"] == cov].index[0]
                for cov in ranked_pool["covariate"].unique()
            }
        rescue_groups = rescue_pool.groupby("covariate").groups if len(rescue_pool) else {}
        multi_parameter_covariates = {
            cov
            for cov, pdf in rescue_pool.groupby("covariate")
            if pdf["parameter"].nunique() > 1
        }
        eval_cache: dict[int, dict] = {}

        for param in summary["parameter"].unique():
            param_mask = summary["parameter"] == param
            param_df = summary.loc[param_mask].copy()
            if len(param_df) == 0:
                continue

            y = ebes[param].astype(float)
            n_samples = int(y.notna().sum())
            is_small_sample = n_samples <= self.small_sample_n

            base_table = param_df[param_df["scm_selected"] | param_df["tier"].isin(["core", "candidate"])].copy()
            rescue_candidates = param_df[
                (param_df["tier"] == "rejected") &
                (~param_df["screening_suppressed"]) &
                ((param_df["shapcov_selected"]) | (param_df["stg_selected"])) &
                (param_df["combined_score"] >= self.rescue_score_floor)
            ].copy()
            if len(rescue_candidates) == 0:
                continue

            for idx, row in rescue_candidates.iterrows():
                cov = row["covariate"]
                if cov not in covariates.columns:
                    continue
                series = covariates[cov]
                is_binary = self._is_binary_series(series)
                prevalence = float(pd.Series(series).dropna().astype(float).mean()) if is_binary else np.nan
                effective_alpha = self.small_sample_rescue_alpha if is_small_sample else self.rescue_alpha
                effective_delta_aic = (
                    self.small_sample_rescue_min_delta_aic if is_small_sample else self.rescue_min_delta_aic
                )
                reason_bits = []
                if is_small_sample:
                    reason_bits.append("small-sample")
                if is_binary and prevalence <= self.low_prevalence_threshold:
                    effective_alpha = max(effective_alpha, self.small_sample_rescue_alpha)
                    effective_delta_aic = min(effective_delta_aic, self.small_sample_rescue_min_delta_aic)
                    reason_bits.append("low-prevalence-binary")

                eval_result = bridge.evaluate_candidate(
                    ebes=ebes,
                    covariates=covariates,
                    parameter=param,
                    candidate_row=row,
                    base_table=base_table,
                    use_robust_se=is_binary,
                )
                effective_p = (
                    eval_result["robust_p_value"]
                    if is_binary and not np.isnan(eval_result["robust_p_value"])
                    else eval_result["p_value"]
                )
                base_selected = (
                    not np.isnan(effective_p) and
                    effective_p <= effective_alpha and
                    eval_result["delta_aic"] >= effective_delta_aic
                )
                low_support_multi_param = (
                    is_small_sample and
                    is_binary and
                    prevalence <= self.low_prevalence_threshold and
                    row["support_count"] <= 1 and
                    cov in multi_parameter_covariates
                )
                strong_selected = False
                if base_selected:
                    if not low_support_multi_param:
                        strong_selected = True
                    else:
                        strong_alpha = min(effective_alpha * 0.5, 0.05)
                        strong_delta_aic = max(effective_delta_aic * 2.0, 2.0)
                        strong_selected = (
                            effective_p <= strong_alpha and
                            eval_result["delta_aic"] >= strong_delta_aic
                        )

                eval_cache[idx] = {
                    "parameter": param,
                    "covariate": cov,
                    "is_small_sample": is_small_sample,
                    "is_binary": is_binary,
                    "prevalence": prevalence,
                    "effective_alpha": effective_alpha,
                    "effective_delta_aic": effective_delta_aic,
                    "effective_p": effective_p,
                    "reason_bits": reason_bits,
                    "eval_result": eval_result,
                    "base_selected": base_selected,
                    "low_support_multi_param": low_support_multi_param,
                    "strong_selected": strong_selected,
                }

        for param in summary["parameter"].unique():
            param_mask = summary["parameter"] == param
            param_df = summary.loc[param_mask].copy()
            if len(param_df) == 0:
                continue

            param_has_anchor = bool(
                (
                    param_df.get("scm_selected", False).astype(bool) |
                    param_df.get("traditional_selected", False).astype(bool) |
                    (param_df.get("support_count", 0).astype(int) >= 1) |
                    param_df["tier"].isin(["core"])
                ).any()
            )

            base_table = param_df[param_df["scm_selected"] | param_df["tier"].isin(["core", "candidate"])].copy()
            rescue_candidates = param_df[
                (param_df["tier"] == "rejected") &
                (~param_df["screening_suppressed"]) &
                ((param_df["shapcov_selected"]) | (param_df["stg_selected"])) &
                (param_df["combined_score"] >= self.rescue_score_floor)
            ].copy()
            if len(rescue_candidates) == 0:
                continue

            rescue_candidates = rescue_candidates.sort_values(
                ["support_count", "combined_score", "shapcov_importance"],
                ascending=[False, False, False],
            )
            accepted = 0
            for idx, row in rescue_candidates.iterrows():
                if idx not in eval_cache:
                    continue
                cov = row["covariate"]
                meta = eval_cache[idx]
                eval_result = meta["eval_result"]
                effective_p = meta["effective_p"]
                is_small_sample = meta["is_small_sample"]
                is_binary = meta["is_binary"]
                prevalence = meta["prevalence"]
                is_top_parameter_signal = top_covariate_indices.get(cov) == idx
                has_strong_rescue = any(
                    eval_cache[group_idx]["strong_selected"]
                    for group_idx in rescue_groups.get(cov, [])
                    if group_idx in eval_cache
                )
                selected = meta["strong_selected"] or (
                    meta["base_selected"] and
                    not meta["low_support_multi_param"]
                )
                if self.rescue_requires_parameter_anchor and not param_has_anchor:
                    selected = False

                summary.loc[idx, "rescue_p_value"] = eval_result["p_value"]
                summary.loc[idx, "rescue_robust_p_value"] = eval_result["robust_p_value"]
                summary.loc[idx, "rescue_delta_aic"] = eval_result["delta_aic"]
                summary.loc[idx, "rescue_reason"] = (
                    ",".join(meta["reason_bits"]) if meta["reason_bits"] else "incremental-utility"
                )

                if selected and accepted < self.rescue_max_per_parameter:
                    summary.loc[idx, "rescued_confirmed"] = True
                    summary.loc[idx, "confirmation_status"] = "rescued"
                    summary.loc[idx, "tier"] = "candidate"
                    summary.loc[idx, "support_label"] = (
                        str(summary.loc[idx, "support_label"]).replace("none", "").strip("+") + "+rescue"
                    ).strip("+")
                    if meta["low_support_multi_param"]:
                        summary.loc[idx, "rescue_reason"] = ",".join(meta["reason_bits"] + ["strong-incremental-evidence"]).strip(",")
                    accepted += 1
                    base_table = pd.concat([base_table, summary.loc[[idx]]], ignore_index=True)
                elif (
                    (not self.rescue_requires_parameter_anchor or param_has_anchor) and
                    is_small_sample and
                    is_binary and
                    prevalence <= self.low_prevalence_threshold and
                    is_top_parameter_signal and
                    row["support_count"] <= 1 and
                    row["combined_score"] >= max(self.rescue_score_floor, 0.04) and
                    not has_strong_rescue
                ):
                    summary.loc[idx, "confirmation_status"] = "rescued_candidate"
                    summary.loc[idx, "tier"] = "candidate"
                    summary.loc[idx, "support_label"] = (
                        str(summary.loc[idx, "support_label"]).replace("none", "").strip("+") + "+candidate-rescue"
                    ).strip("+")
                    summary.loc[idx, "rescue_reason"] = ",".join(meta["reason_bits"] + ["top-parameter-ml-signal"]).strip(",")

        summary.loc[summary["scm_selected"], "confirmation_status"] = "scm"
        summary["tier_rank"] = summary["tier"].map({"core": 0, "candidate": 1, "proxy": 2, "rejected": 3}).fillna(4).astype(int)
        return summary

    def _prune_redundant_candidates(
        self,
        summary: pd.DataFrame,
        covariates: pd.DataFrame,
    ) -> pd.DataFrame:
        summary = summary.copy()
        if len(summary) == 0 or self.rescue_redundancy_threshold <= 0:
            return summary

        for col, default in {
            "support_count": 0,
            "support_requirement": 2,
            "scm_selected": False,
            "traditional_selected": False,
            "penalized_selected": False,
            "score_rescue_candidate": False,
            "combined_score_adjusted": 0.0,
            "shapcov_score": 0.0,
        }.items():
            if col not in summary.columns:
                summary[col] = default

        assoc = association_matrix(covariates)
        corr_filter = CorrelationFilter(threshold=self.corr_threshold)
        corr_filter.fit(covariates)
        strict_collinearity = any(len(members) >= 4 for members in corr_filter.groups.values() if len(members) > 1)

        for param in summary["parameter"].unique():
            param_mask = summary["parameter"] == param
            param_df = summary.loc[param_mask].copy()
            selected = param_df[param_df["tier"].isin(["core", "candidate"])].copy()
            if len(selected) < 2:
                if len(selected) == 1:
                    selected["rescued_only"] = (
                        selected.get("score_rescue_candidate", False).astype(bool) &
                        (~selected.get("scm_selected", False).astype(bool)) &
                        (~selected.get("traditional_selected", False).astype(bool)) &
                        (selected.get("support_count", 0).astype(int) < selected.get("support_requirement", 2).astype(int))
                    )
                    param_has_anchor = bool(
                        (
                            selected.get("scm_selected", False).astype(bool) |
                            selected.get("traditional_selected", False).astype(bool) |
                            (selected.get("support_count", 0).astype(int) >= selected.get("support_requirement", 2).astype(int)) |
                            (~selected["rescued_only"])
                        ).any()
                    )
                    if not param_has_anchor and bool(selected["rescued_only"].iloc[0]):
                        idx_mask = (summary["parameter"] == param) & (summary["covariate"] == selected["covariate"].iloc[0])
                        summary.loc[idx_mask, "tier"] = "rejected"
                continue

            selected["rescued_only"] = (
                selected.get("score_rescue_candidate", False).astype(bool) &
                (~selected.get("scm_selected", False).astype(bool)) &
                (~selected.get("traditional_selected", False).astype(bool)) &
                (selected.get("support_count", 0).astype(int) < selected.get("support_requirement", 2).astype(int))
            )
            param_has_anchor = bool(
                (
                    selected.get("scm_selected", False).astype(bool) |
                    selected.get("traditional_selected", False).astype(bool) |
                    (selected.get("support_count", 0).astype(int) >= selected.get("support_requirement", 2).astype(int)) |
                    (~selected["rescued_only"])
                ).any()
            )
            if not param_has_anchor:
                rescue_mask = (
                    (summary["parameter"] == param) &
                    summary["tier"].isin(["core", "candidate"]) &
                    summary.get("score_rescue_candidate", False).astype(bool)
                )
                summary.loc[rescue_mask, "tier"] = "rejected"
                continue

            if strict_collinearity:
                weak_mask = (
                    (summary["parameter"] == param) &
                    (summary["tier"] == "candidate") &
                    (summary.get("support_count", 0).astype(int) <= 1) &
                    (~summary.get("penalized_selected", False).astype(bool))
                )
                summary.loc[weak_mask, "tier"] = "rejected"
                param_df = summary.loc[param_mask].copy()
                selected = param_df[param_df["tier"].isin(["core", "candidate"])].copy()
                if len(selected) < 2:
                    continue
                selected["rescued_only"] = (
                    selected.get("score_rescue_candidate", False).astype(bool) &
                    (~selected.get("scm_selected", False).astype(bool)) &
                    (~selected.get("traditional_selected", False).astype(bool)) &
                    (selected.get("support_count", 0).astype(int) < selected.get("support_requirement", 2).astype(int))
                )

            candidate_cols = [c for c in selected["covariate"].tolist() if c in assoc.index]
            if not candidate_cols:
                continue
            centrality = assoc.loc[candidate_cols, candidate_cols].copy()
            np.fill_diagonal(centrality.values, np.nan)
            centrality = centrality.mean(axis=1, skipna=True).fillna(0.0).to_dict()
            selected["group_centrality"] = selected["covariate"].map(lambda c: float(centrality.get(c, 0.0)))

            ranked = selected.sort_values(
                [
                    "tier_rank",
                    "support_count",
                    "scm_selected",
                    "traditional_selected",
                    "penalized_selected",
                    "group_centrality",
                    "combined_score_adjusted",
                    "shapcov_score",
                ],
                ascending=[True, False, False, False, False, False, False, False],
            )

            survivors: list[str] = []
            for _, row in ranked.iterrows():
                cov = row["covariate"]
                if cov not in assoc.index:
                    survivors.append(cov)
                    continue
                if row["tier"] == "core":
                    survivors.append(cov)
                    continue

                demote_to = None
                for survivor in survivors:
                    if survivor not in assoc.index:
                        continue
                    if self._should_preserve_proxy_pair(covariates, survivor, cov):
                        continue
                    if float(assoc.loc[cov, survivor]) >= self.rescue_redundancy_threshold:
                        demote_to = survivor
                        break

                if demote_to is None:
                    survivors.append(cov)
                    continue

                idx_mask = (summary["parameter"] == param) & (summary["covariate"] == cov)
                summary.loc[idx_mask, "tier"] = "proxy"
                summary.loc[idx_mask, "proxy_for"] = demote_to

        summary["tier_rank"] = summary["tier"].map({"core": 0, "candidate": 1, "proxy": 2, "rejected": 3}).fillna(4).astype(int)
        return summary

    @staticmethod
    def _merge_symbolic_outputs(summary: pd.DataFrame, symbolic_summary: pd.DataFrame) -> pd.DataFrame:
        merged = summary.merge(
            symbolic_summary[
                [
                    "parameter",
                    "covariate",
                    "symbolic_form",
                    "symbolic_expression",
                    "symbolic_score",
                    "symbolic_p_value",
                    "symbolic_selected",
                    "symbolic_backend",
                ]
            ],
            on=["parameter", "covariate"],
            how="left",
        )
        merged["symbolic_selected"] = merged["symbolic_selected"].where(merged["symbolic_selected"].notna(), False).astype(bool)
        merged["symbolic_form"] = merged["symbolic_form"].fillna("")
        merged["symbolic_expression"] = merged["symbolic_expression"].fillna("")
        merged["symbolic_score"] = merged["symbolic_score"].fillna(0.0)
        merged["symbolic_p_value"] = merged["symbolic_p_value"].fillna(np.nan)
        merged["symbolic_backend"] = merged["symbolic_backend"].fillna("")
        merged.loc[merged["symbolic_selected"], "support_label"] = (
            merged.loc[merged["symbolic_selected"], "support_label"].replace("none", "") + "+symbolic"
        ).str.strip("+")
        return merged

    def _assign_proxy_groups(self, summary: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
        summary = summary.copy()
        summary["proxy_group_id"] = ""
        summary["proxy_for"] = ""
        summary["group_representative"] = False
        for col, default in {
            "support_count": 0,
            "scm_selected": False,
            "traditional_selected": False,
            "penalized_selected": False,
            "combined_score": 0.0,
            "shapcov_importance": 0.0,
            "penalized_importance": 0.0,
        }.items():
            if col not in summary.columns:
                summary[col] = default

        corr_filter = CorrelationFilter(threshold=self.corr_threshold)
        corr_filter.fit(covariates)
        assoc = association_matrix(covariates)
        groups = [members for members in corr_filter.groups.values() if len(members) > 1]
        if not groups:
            return summary

        for group_idx, members in enumerate(groups, start=1):
            group_id = f"G{group_idx}"
            group_centrality = {}
            present_members = [m for m in members if m in assoc.index]
            if present_members:
                sub = assoc.loc[present_members, present_members].copy()
                np.fill_diagonal(sub.values, np.nan)
                group_centrality = sub.mean(axis=1, skipna=True).fillna(0.0).to_dict()
            for param in summary["parameter"].unique():
                mask = (summary["parameter"] == param) & (summary["covariate"].isin(members))
                pdf = summary.loc[mask].copy()
                if len(pdf) == 0:
                    continue
                pdf["group_centrality"] = pdf["covariate"].map(lambda c: float(group_centrality.get(c, 0.0)))
                ranked = pdf.sort_values(
                    [
                        "tier_rank",
                        "support_count",
                        "scm_selected",
                        "traditional_selected",
                        "penalized_selected",
                        "group_centrality",
                        "combined_score",
                        "shapcov_importance",
                        "penalized_importance",
                    ],
                    ascending=[True, False, False, False, False, False, False, False, False],
                )
                representative = ranked.iloc[0]["covariate"]
                summary.loc[mask, "proxy_group_id"] = group_id
                summary.loc[(summary["parameter"] == param) & (summary["covariate"] == representative), "group_representative"] = True

                for _, row in ranked.iloc[1:].iterrows():
                    idx_mask = (summary["parameter"] == param) & (summary["covariate"] == row["covariate"])
                    if self._should_preserve_proxy_pair(covariates, representative, row["covariate"]):
                        continue
                    if summary.loc[idx_mask, "tier"].iloc[0] in {"core", "candidate"}:
                        summary.loc[idx_mask, "tier"] = "proxy"
                        summary.loc[idx_mask, "proxy_for"] = representative

        summary["tier_rank"] = summary["tier"].map({"core": 0, "candidate": 1, "proxy": 2, "rejected": 3}).fillna(4).astype(int)
        return summary

    def _should_preserve_proxy_pair(
        self,
        covariates: pd.DataFrame,
        representative: str,
        candidate: str,
    ) -> bool:
        pair_key = frozenset((representative, candidate))
        if pair_key in self.preserve_proxy_pairs:
            return True
        if not self.preserve_biological_distinctness:
            return False
        if representative not in covariates.columns or candidate not in covariates.columns:
            return False

        left = pd.Series(covariates[representative])
        right = pd.Series(covariates[candidate])
        if (
            not pd.api.types.is_numeric_dtype(left) or
            not pd.api.types.is_numeric_dtype(right) or
            self._is_binary_series(left) or
            self._is_binary_series(right)
        ):
            return False

        pair = pd.concat([left.astype(float), right.astype(float)], axis=1).dropna()
        if len(pair) < 10:
            return False

        x = pair.iloc[:, 0].to_numpy(dtype=float)
        y = pair.iloc[:, 1].to_numpy(dtype=float)
        if np.nanstd(x) <= 1e-8 or np.nanstd(y) <= 1e-8:
            return False

        corr = abs(float(np.corrcoef(x, y)[0, 1]))
        if not np.isfinite(corr) or corr < self.proxy_preserve_corr_threshold:
            return False

        slope, intercept = np.polyfit(x, y, 1)
        predicted = intercept + slope * x
        ss_res = float(np.sum((y - predicted) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
        if r2 < self.proxy_preserve_r2_threshold:
            return False

        intercept_ratio = abs(float(intercept)) / max(float(np.nanstd(y)), 1e-8)
        return intercept_ratio >= self.proxy_preserve_intercept_std_ratio

    @staticmethod
    def _proxy_group_table(summary: pd.DataFrame) -> pd.DataFrame:
        rows = []
        grouped = summary[summary["proxy_group_id"] != ""].groupby(["parameter", "proxy_group_id"])
        for (param, group_id), pdf in grouped:
            representative = pdf.loc[pdf["group_representative"], "covariate"]
            representative = representative.iloc[0] if len(representative) else pdf.iloc[0]["covariate"]
            rows.append(
                {
                    "parameter": param,
                    "proxy_group_id": group_id,
                    "representative": representative,
                    "members": ", ".join(sorted(pdf["covariate"].tolist())),
                    "selected_members": ", ".join(sorted(pdf[pdf["tier"].isin(["core", "candidate", "proxy"])]["covariate"].tolist())) or "none",
                }
            )
        return pd.DataFrame(rows)

    @property
    def report(self) -> "HybridResults | None":
        return self._report


class HybridResults:
    """Tiered results from the default hybrid screening workflow."""

    def __init__(
        self,
        summary_df: pd.DataFrame,
        proxy_groups_df: pd.DataFrame,
        artifacts: HybridArtifacts | None,
    ):
        self._summary = summary_df
        self._proxy_groups = proxy_groups_df
        self._artifacts = artifacts

    def summary(self) -> pd.DataFrame:
        return self._summary.copy()

    def core_covariates(self) -> pd.DataFrame:
        return self._summary[self._summary["tier"] == "core"].reset_index(drop=True)

    def candidate_covariates(self) -> pd.DataFrame:
        return self._summary[self._summary["tier"].isin(["core", "candidate"])].reset_index(drop=True)

    def proxy_covariates(self) -> pd.DataFrame:
        return self._summary[self._summary["tier"] == "proxy"].reset_index(drop=True)

    def proxy_groups(self) -> pd.DataFrame:
        return self._proxy_groups.copy()

    def functional_forms(self) -> pd.DataFrame:
        cols = ["parameter", "covariate", "functional_form", "tier"]
        return self._summary[self._summary["tier"].isin(["core", "candidate"])][cols].reset_index(drop=True)

    def shortlist(self) -> pd.DataFrame:
        return self._summary[self._summary["tier"].isin(["core", "candidate", "proxy"])].reset_index(drop=True)

    def to_nonmem_candidates(self) -> str:
        return self._render_candidates(target="nonmem")

    def to_nlmixr2_candidates(self) -> str:
        return self._render_candidates(target="nlmixr2")

    def compare_with_traditional(self) -> pd.DataFrame:
        if self._artifacts is None or self._artifacts.traditional_summary is None:
            return pd.DataFrame()
        trad = self._artifacts.traditional_summary[["parameter", "covariate", "significant"]].rename(
            columns={"significant": "traditional_selected"}
        )
        shortlist = self.shortlist()[["parameter", "covariate", "tier"]]
        merged = trad.merge(shortlist, on=["parameter", "covariate"], how="outer")
        merged["traditional_selected"] = merged["traditional_selected"].fillna(False)
        merged["tier"] = merged["tier"].fillna("rejected")
        return merged.sort_values(["parameter", "covariate"]).reset_index(drop=True)

    def scm_covariates(self) -> pd.DataFrame:
        if self._artifacts is None or self._artifacts.scm_results is None:
            return pd.DataFrame()
        return self._artifacts.scm_results.selected_covariates()

    def rescued_covariates(self) -> pd.DataFrame:
        if "rescued_confirmed" not in self._summary.columns:
            return pd.DataFrame()
        return self._summary[self._summary["rescued_confirmed"]].reset_index(drop=True)

    def confirmed_covariates(self) -> pd.DataFrame:
        """Return the recommended final tier for daily use.

        If SCM confirmation is available, this is the preferred answer for
        users who want a compact, pharmacometric-style confirmed set.
        Otherwise, fall back to the core tier.
        """
        scm = self.scm_covariates()
        rescued = self.rescued_covariates()
        if len(scm) or len(rescued):
            frames = []
            if len(scm):
                scm = scm.copy()
                scm["confirmation_status"] = "scm"
                frames.append(scm)
            if len(rescued):
                frames.append(rescued)
            confirmed = pd.concat(frames, ignore_index=True, sort=False)
            return confirmed.drop_duplicates(subset=["parameter", "covariate"]).reset_index(drop=True)
        return self.core_covariates()

    def compare_with_scm(self) -> pd.DataFrame:
        scm = self.scm_covariates()
        if len(scm) == 0:
            return pd.DataFrame()
        scm = scm[["parameter", "covariate", "selected"]].rename(columns={"selected": "scm_selected"})
        shortlist = self.shortlist()[["parameter", "covariate", "tier"]]
        merged = scm.merge(shortlist, on=["parameter", "covariate"], how="outer")
        merged["scm_selected"] = merged["scm_selected"].fillna(False)
        merged["tier"] = merged["tier"].fillna("rejected")
        return merged.sort_values(["parameter", "covariate"]).reset_index(drop=True)

    def symbolic_covariates(self) -> pd.DataFrame:
        if "symbolic_selected" not in self._summary.columns:
            return pd.DataFrame()
        symbolic = self._summary[self._summary["symbolic_selected"]]
        return symbolic.reset_index(drop=True)

    def interaction_covariates(self) -> pd.DataFrame:
        if "functional_form" not in self._summary.columns:
            return pd.DataFrame()
        interactions = self._summary[self._summary["functional_form"] == "interaction"]
        return interactions.reset_index(drop=True)

    def _render_candidates(self, target: str) -> str:
        shortlist = self.candidate_covariates()
        if len(shortlist) == 0:
            return "# No candidate covariates identified"

        lines = [
            f"# pharmacoml hybrid candidate covariates ({target})",
            "# core = strongest evidence, candidate = carry forward to SCM/backward elimination",
        ]
        for _, row in shortlist.sort_values(["parameter", "tier_rank", "combined_score"], ascending=[True, True, False]).iterrows():
            tier = row["tier"].upper()
            param = row["parameter"]
            cov = row["covariate"]
            form = row.get("functional_form", "unknown")
            score = row["combined_score"]
            confirmed = " scm=yes" if bool(row.get("scm_selected", False)) else ""
            if target == "nonmem":
                lines.append(f"; [{tier}] {cov} -> {param} | form={form} | score={score:.3f}{confirmed}")
            else:
                lines.append(f"# [{tier}] {cov} -> {param} | form={form} | score={score:.3f}{confirmed}")
        return "\n".join(lines)

    def __repr__(self):
        core = len(self.core_covariates())
        candidate = len(self.candidate_covariates())
        proxy = len(self.proxy_covariates())
        return f"HybridResults(core={core}, candidate={candidate}, proxy={proxy})"
