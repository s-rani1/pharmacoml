"""SCM-style stepwise bridge for EBE/individual-parameter workflows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class _SCMTerm:
    name: str
    covariate: str
    functional_form: str
    columns: list[str]
    design: pd.DataFrame


class SCMBridge:
    """Stepwise covariate confirmation over EBE-style targets.

    This is an SCM-inspired bridge for users who already have subject-level
    ETAs or individual parameters. It does not replace a final NONMEM/PsN SCM
    run, but it gives users a rigorous, backend-agnostic confirmation step.
    """

    def __init__(
        self,
        enter_alpha: float = 0.05,
        stay_alpha: float = 0.10,
        min_delta_aic: float = 2.0,
        max_terms: int = 6,
        use_log_target: bool = True,
    ):
        self.enter_alpha = enter_alpha
        self.stay_alpha = stay_alpha
        self.min_delta_aic = min_delta_aic
        self.max_terms = max_terms
        self.use_log_target = use_log_target
        self._summary = None
        self._steps = None

    def fit(
        self,
        ebes: pd.DataFrame,
        covariates: pd.DataFrame,
        candidate_table: pd.DataFrame | None = None,
        parameter_names: list[str] | None = None,
        candidate_tiers: tuple[str, ...] = ("core", "candidate"),
    ) -> "SCMResults":
        params = parameter_names or list(ebes.columns)
        summary_rows = []
        step_rows = []

        for param in params:
            y = ebes[param].astype(float)
            terms = self._build_terms(
                covariates=covariates,
                candidate_table=candidate_table,
                parameter=param,
                candidate_tiers=candidate_tiers,
            )
            if not terms:
                continue

            y_model, target_scale = self._prepare_target(y)
            model_df = pd.DataFrame({"__target__": y_model})
            for term in terms:
                model_df = pd.concat([model_df, term.design], axis=1)
            model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
            if len(model_df) < 20:
                continue

            y_clean = model_df["__target__"]
            current_terms: list[_SCMTerm] = []
            remaining = {term.name: term for term in terms}
            current_model = self._fit_model(y_clean, current_terms, model_df)
            step_number = 0

            while remaining and len(current_terms) < self.max_terms:
                best_name = None
                best_candidate = None
                best_p = 1.0
                best_delta_aic = -np.inf

                for name, term in remaining.items():
                    trial_terms = current_terms + [term]
                    trial_model = self._fit_model(y_clean, trial_terms, model_df)
                    if trial_model is None:
                        continue
                    _, p_value, _ = trial_model.compare_f_test(current_model)
                    delta_aic = float(current_model.aic - trial_model.aic)
                    if p_value < best_p or (np.isclose(p_value, best_p) and delta_aic > best_delta_aic):
                        best_name = name
                        best_candidate = trial_model
                        best_p = float(p_value)
                        best_delta_aic = delta_aic

                if best_name is None:
                    break
                if best_p > self.enter_alpha or best_delta_aic < self.min_delta_aic:
                    break

                added_term = remaining.pop(best_name)
                current_terms.append(added_term)
                current_model = best_candidate
                step_number += 1
                step_rows.append(
                    {
                        "parameter": param,
                        "step": step_number,
                        "action": "add",
                        "covariate": added_term.covariate,
                        "functional_form": added_term.functional_form,
                        "p_value": round(best_p, 4),
                        "delta_aic": round(best_delta_aic, 4),
                    }
                )

                changed = True
                while changed and len(current_terms) > 1:
                    changed = False
                    worst_name = None
                    worst_p = -np.inf
                    worst_reduced = None
                    worst_delta_aic = 0.0

                    for term in list(current_terms):
                        reduced_terms = [t for t in current_terms if t.name != term.name]
                        reduced_model = self._fit_model(y_clean, reduced_terms, model_df)
                        if reduced_model is None:
                            continue
                        _, p_value, _ = current_model.compare_f_test(reduced_model)
                        delta_aic = float(reduced_model.aic - current_model.aic)
                        if p_value > worst_p:
                            worst_name = term.name
                            worst_p = float(p_value)
                            worst_reduced = reduced_model
                            worst_delta_aic = delta_aic

                    if worst_name is not None and worst_p > self.stay_alpha and worst_delta_aic <= self.min_delta_aic:
                        removed = next(t for t in current_terms if t.name == worst_name)
                        current_terms = [t for t in current_terms if t.name != worst_name]
                        current_model = worst_reduced
                        step_number += 1
                        step_rows.append(
                            {
                                "parameter": param,
                                "step": step_number,
                                "action": "remove",
                                "covariate": removed.covariate,
                                "functional_form": removed.functional_form,
                                "p_value": round(worst_p, 4),
                                "delta_aic": round(worst_delta_aic, 4),
                            }
                        )
                        changed = True

            selected_names = {term.name for term in current_terms}
            final_metrics = {
                "target_scale": target_scale,
                "final_aic": round(float(current_model.aic), 4),
                "final_bic": round(float(current_model.bic), 4),
                "final_r2": round(float(current_model.rsquared), 4),
            }

            for term in terms:
                if term.name in selected_names:
                    reduced_terms = [t for t in current_terms if t.name != term.name]
                    reduced_model = self._fit_model(y_clean, reduced_terms, model_df)
                    if reduced_model is not None:
                        _, p_value, _ = current_model.compare_f_test(reduced_model)
                    else:
                        p_value = np.nan
                else:
                    p_value = np.nan

                summary_rows.append(
                    {
                        "parameter": param,
                        "covariate": term.covariate,
                        "functional_form": term.functional_form,
                        "selected": term.name in selected_names,
                        "scm_p_value": round(float(p_value), 4) if not np.isnan(p_value) else np.nan,
                        **final_metrics,
                    }
                )

        self._summary = pd.DataFrame(summary_rows)
        self._steps = pd.DataFrame(step_rows)
        return SCMResults(self._summary, self._steps)

    def evaluate_candidate(
        self,
        ebes: pd.DataFrame,
        covariates: pd.DataFrame,
        parameter: str,
        candidate_row: pd.Series | dict,
        base_table: pd.DataFrame | None = None,
        use_robust_se: bool = True,
    ) -> dict:
        """Evaluate one candidate term against a parameter-specific base model."""
        if isinstance(candidate_row, dict):
            candidate_row = pd.Series(candidate_row)

        y = ebes[parameter].astype(float)
        y_model, target_scale = self._prepare_target(y)
        base_terms = self._build_terms(
            covariates=covariates,
            candidate_table=base_table,
            parameter=parameter,
            candidate_tiers=("core", "candidate", "rescue"),
        )
        candidate_df = pd.DataFrame(
            [
                {
                    "parameter": parameter,
                    "covariate": candidate_row["covariate"],
                    "functional_form": candidate_row.get("functional_form", "linear"),
                    "tier": "rescue",
                }
            ]
        )
        candidate_terms = self._build_terms(
            covariates=covariates,
            candidate_table=candidate_df,
            parameter=parameter,
            candidate_tiers=("rescue",),
        )
        if len(candidate_terms) == 0:
            return {
                "selected": False,
                "p_value": np.nan,
                "robust_p_value": np.nan,
                "delta_aic": 0.0,
                "target_scale": target_scale,
            }

        candidate_term = candidate_terms[0]
        base_model_df = self._assemble_model_df(y_model, base_terms, candidate_term, covariates.index)
        base_model_df = base_model_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        if len(base_model_df) < 20:
            return {
                "selected": False,
                "p_value": np.nan,
                "robust_p_value": np.nan,
                "delta_aic": 0.0,
                "target_scale": target_scale,
            }

        y_clean = base_model_df["__target__"]
        base_model = self._fit_model(y_clean, base_terms, base_model_df)
        full_model = self._fit_model(y_clean, base_terms + [candidate_term], base_model_df)
        if base_model is None or full_model is None:
            return {
                "selected": False,
                "p_value": np.nan,
                "robust_p_value": np.nan,
                "delta_aic": 0.0,
                "target_scale": target_scale,
            }

        _, p_value, _ = full_model.compare_f_test(base_model)
        delta_aic = float(base_model.aic - full_model.aic)
        robust_p = np.nan
        if use_robust_se and len(candidate_term.columns) == 1:
            try:
                robust = full_model.get_robustcov_results(cov_type="HC3")
                idx = full_model.model.exog_names.index(candidate_term.columns[0])
                robust_p = float(robust.pvalues[idx])
            except Exception:
                robust_p = np.nan

        effective_p = robust_p if use_robust_se and not np.isnan(robust_p) else float(p_value)
        return {
            "selected": bool((effective_p <= self.enter_alpha) and (delta_aic >= self.min_delta_aic)),
            "p_value": round(float(p_value), 4),
            "robust_p_value": round(float(robust_p), 4) if not np.isnan(robust_p) else np.nan,
            "delta_aic": round(delta_aic, 4),
            "target_scale": target_scale,
        }

    def _prepare_target(self, y: pd.Series) -> tuple[pd.Series, str]:
        y = y.astype(float)
        if self.use_log_target and bool((y > 0).all()):
            return np.log(y), "log"
        return y, "identity"

    def _assemble_model_df(
        self,
        y_model: pd.Series,
        base_terms: list[_SCMTerm],
        candidate_term: _SCMTerm | None,
        index,
    ) -> pd.DataFrame:
        model_df = pd.DataFrame({"__target__": y_model}, index=index)
        for term in base_terms:
            model_df = pd.concat([model_df, term.design], axis=1)
        if candidate_term is not None:
            model_df = pd.concat([model_df, candidate_term.design], axis=1)
        return model_df

    def _build_terms(
        self,
        covariates: pd.DataFrame,
        candidate_table: pd.DataFrame | None,
        parameter: str,
        candidate_tiers: tuple[str, ...],
    ) -> list[_SCMTerm]:
        if candidate_table is None or len(candidate_table) == 0:
            rows = pd.DataFrame(
                {
                    "parameter": parameter,
                    "covariate": list(covariates.columns),
                    "functional_form": [
                        "categorical" if self._is_categorical(covariates[cov]) else "linear"
                        for cov in covariates.columns
                    ],
                    "tier": "candidate",
                }
            )
        else:
            rows = candidate_table.copy()
            if "tier" in rows.columns:
                rows = rows[rows["tier"].isin(candidate_tiers)]
            rows = rows[rows["parameter"] == parameter]

        terms = []
        for _, row in rows.drop_duplicates(subset=["covariate"]).iterrows():
            cov = row["covariate"]
            form = row.get("functional_form", "linear")
            if cov not in covariates.columns and "__" not in cov:
                continue
            design = self._transform_covariate(covariates, cov, form)
            if design is None or design.shape[1] == 0:
                continue
            terms.append(
                _SCMTerm(
                    name=cov,
                    covariate=cov,
                    functional_form=form,
                    columns=list(design.columns),
                    design=design,
                )
            )
        return terms

    def _transform_covariate(self, covariates: pd.DataFrame, covariate: str, form: str) -> pd.DataFrame | None:
        if covariate in covariates.columns:
            series = covariates[covariate]
        elif "__xor__" in covariate:
            left, right = covariate.split("__xor__")
            if left not in covariates.columns or right not in covariates.columns:
                return None
            series = covariates[left].fillna(0).astype(int).astype(bool) ^ covariates[right].fillna(0).astype(int).astype(bool)
            series = pd.Series(series.astype(float), index=covariates.index, name=covariate)
        elif "__x__" in covariate:
            left, right = covariate.split("__x__")
            if left not in covariates.columns or right not in covariates.columns:
                return None
            series = covariates[left].astype(float) * covariates[right].astype(float)
            series = pd.Series(series, index=covariates.index, name=covariate)
        else:
            return None

        if form == "categorical" or self._is_categorical(series):
            if self._is_binary(series):
                return pd.DataFrame({covariate: series.astype(float)})
            dummies = pd.get_dummies(series.astype("category"), prefix=covariate, drop_first=True).astype(float)
            return dummies

        x = series.astype(float)
        median = float(np.nanmedian(x)) if np.isfinite(np.nanmedian(x)) else 0.0
        scale = float(np.nanstd(x)) if np.isfinite(np.nanstd(x)) and np.nanstd(x) > 1e-8 else 1.0

        if form == "power" and bool((x > 0).all()):
            transformed = np.log(x / max(median, 1e-6))
        elif form == "exponential":
            transformed = (x - median) / max(abs(median), 1.0)
        else:
            transformed = (x - median) / scale

        return pd.DataFrame({covariate: transformed.astype(float)})

    @staticmethod
    def _fit_model(y: pd.Series, terms: list[_SCMTerm], model_df: pd.DataFrame):
        cols = []
        for term in terms:
            cols.extend(term.columns)
        X = model_df[cols] if cols else pd.DataFrame(index=model_df.index)
        X = sm.add_constant(X, has_constant="add")
        try:
            return sm.OLS(y, X).fit()
        except Exception:
            return None

    @staticmethod
    def _is_binary(series: pd.Series) -> bool:
        vals = pd.Series(series).dropna().unique().tolist()
        return len(vals) <= 2 and set(vals).issubset({0, 1, 0.0, 1.0, False, True})

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        return series.dtype == "object" or str(series.dtype).startswith("category") or SCMBridge._is_binary(series)


class SCMResults:
    """Results from the SCM-style stepwise bridge."""

    def __init__(self, summary_df: pd.DataFrame, steps_df: pd.DataFrame):
        self._summary = summary_df if summary_df is not None else pd.DataFrame()
        self._steps = steps_df if steps_df is not None else pd.DataFrame()

    def summary(self) -> pd.DataFrame:
        if len(self._summary) == 0:
            return self._summary.copy()
        return self._summary.sort_values(["parameter", "selected", "covariate"], ascending=[True, False, True]).reset_index(drop=True)

    def selected_covariates(self) -> pd.DataFrame:
        if len(self._summary) == 0:
            return self._summary.copy()
        return self.summary().query("selected").reset_index(drop=True)

    def steps(self) -> pd.DataFrame:
        return self._steps.copy()

    def __repr__(self):
        return f"SCMResults(selected={len(self.selected_covariates())})"
