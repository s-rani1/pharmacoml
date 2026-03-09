"""Symbolic covariate-structure search."""
from __future__ import annotations

from importlib.util import find_spec
import re

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


class SymbolicStructureScreener:
    """Symbolic covariate-structure layer.

    The default ``basis`` backend keeps the original pharmacometric basis-form
    search. Optional ``gplearn`` and ``pysr`` backends perform true symbolic
    regression over the candidate covariate set for each parameter.
    """

    SUPPORTED_BACKENDS = {"basis", "gplearn", "pysr"}

    def __init__(
        self,
        alpha: float = 0.05,
        min_delta_aic: float = 2.0,
        use_log_target: bool = True,
        symbolic_backend: str = "basis",
        symbolic_backend_kwargs: dict | None = None,
        random_state: int = 42,
    ):
        if symbolic_backend not in self.SUPPORTED_BACKENDS:
            available = ", ".join(sorted(self.SUPPORTED_BACKENDS))
            raise ValueError(f"Unknown symbolic backend '{symbolic_backend}'. Available: {available}")
        self.alpha = alpha
        self.min_delta_aic = min_delta_aic
        self.use_log_target = use_log_target
        self.symbolic_backend = symbolic_backend
        self.symbolic_backend_kwargs = symbolic_backend_kwargs or {}
        self.random_state = random_state
        self._summary = None

    def fit(
        self,
        ebes: pd.DataFrame,
        covariates: pd.DataFrame,
        candidate_table: pd.DataFrame | None = None,
        parameter_names: list[str] | None = None,
        candidate_tiers: tuple[str, ...] = ("core", "candidate"),
    ) -> pd.DataFrame:
        params = parameter_names or list(ebes.columns)
        rows = []

        for param in params:
            y = ebes[param].astype(float)
            y_model = np.log(y) if self.use_log_target and bool((y > 0).all()) else y

            if candidate_table is None or len(candidate_table) == 0:
                cov_rows = pd.DataFrame({"covariate": list(covariates.columns), "tier": "candidate"})
            else:
                cov_rows = candidate_table[candidate_table["parameter"] == param].copy()
                if "tier" in cov_rows.columns:
                    cov_rows = cov_rows[cov_rows["tier"].isin(candidate_tiers)]

            candidate_covariates = [cov for cov in cov_rows["covariate"].drop_duplicates() if cov in covariates.columns]
            if not candidate_covariates:
                continue

            if self.symbolic_backend == "basis":
                intercept = sm.OLS(y_model, sm.add_constant(pd.DataFrame(index=y.index), has_constant="add")).fit()
                for cov in candidate_covariates:
                    best = self._best_structure(y_model, intercept, covariates[cov], cov)
                    best["parameter"] = param
                    best["covariate"] = cov
                    best["symbolic_backend"] = "basis"
                    rows.append(best)
            else:
                rows.extend(
                    self._fit_symbolic_backend(
                        y=y_model,
                        covariates=covariates,
                        candidate_covariates=candidate_covariates,
                        parameter=param,
                    )
                )

        self._summary = pd.DataFrame(rows)
        return self.summary()

    def _fit_symbolic_backend(
        self,
        y: pd.Series,
        covariates: pd.DataFrame,
        candidate_covariates: list[str],
        parameter: str,
    ) -> list[dict]:
        X, feature_to_cov = self._build_symbolic_matrix(covariates, candidate_covariates)
        if X.empty:
            return self._empty_symbolic_rows(parameter, candidate_covariates, self.symbolic_backend)

        model_df = pd.concat([pd.Series(y, name="__target__"), X], axis=1)
        model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(model_df) < 20:
            return self._empty_symbolic_rows(parameter, candidate_covariates, self.symbolic_backend)

        y_clean = model_df["__target__"].astype(float)
        X_clean = model_df.drop(columns="__target__").astype(float)

        if self.symbolic_backend == "gplearn":
            selected_features, expression, preds = self._fit_gplearn_backend(X_clean, y_clean)
        elif self.symbolic_backend == "pysr":
            selected_features, expression, preds = self._fit_pysr_backend(X_clean, y_clean)
        else:
            raise ValueError(f"Unsupported symbolic backend '{self.symbolic_backend}'")

        selected_covariates = {
            feature_to_cov[feature]
            for feature in selected_features
            if feature in feature_to_cov
        }
        metrics = self._symbolic_regression_metrics(
            y=y_clean.to_numpy(dtype=float),
            preds=np.asarray(preds, dtype=float).ravel(),
            complexity=max(len(selected_features), 1),
        )
        selected_flag = bool(
            len(selected_covariates) > 0 and
            metrics["symbolic_delta_aic"] >= self.min_delta_aic and
            (
                np.isnan(metrics["symbolic_p_value"]) or
                metrics["symbolic_p_value"] < self.alpha
            )
        )

        rows = []
        for cov in candidate_covariates:
            is_selected = selected_flag and cov in selected_covariates
            rows.append(
                {
                    "parameter": parameter,
                    "covariate": cov,
                    "symbolic_form": self.symbolic_backend if is_selected else "none",
                    "symbolic_expression": expression if is_selected else "",
                    "symbolic_score": metrics["symbolic_score"] if is_selected else 0.0,
                    "symbolic_p_value": metrics["symbolic_p_value"] if is_selected else np.nan,
                    "symbolic_delta_aic": metrics["symbolic_delta_aic"] if is_selected else 0.0,
                    "symbolic_selected": is_selected,
                    "symbolic_backend": self.symbolic_backend,
                }
            )
        return rows

    def _fit_gplearn_backend(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[set[str], str, np.ndarray]:
        if find_spec("gplearn") is None:
            raise ImportError(
                "SymbolicStructureScreener backend='gplearn' requires gplearn. "
                "Install the optional symbolic dependency first."
            )

        from gplearn.genetic import SymbolicRegressor

        kwargs = {
            "population_size": 1000,
            "generations": 20,
            "stopping_criteria": 0.001,
            "function_set": ("add", "sub", "mul", "div"),
            "parsimony_coefficient": 0.001,
            "max_samples": 0.9,
            "p_crossover": 0.7,
            "p_subtree_mutation": 0.1,
            "p_hoist_mutation": 0.05,
            "p_point_mutation": 0.1,
            "random_state": self.random_state,
            "verbose": 0,
        }
        kwargs.update(self.symbolic_backend_kwargs)

        model = SymbolicRegressor(**kwargs)
        model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        preds = np.asarray(model.predict(X.to_numpy(dtype=float)), dtype=float).ravel()
        expression = str(model._program)
        selected_features = set()
        for match in re.findall(r"X(\d+)", expression):
            idx = int(match)
            if 0 <= idx < len(X.columns):
                selected_features.add(X.columns[idx])
        return selected_features, expression, preds

    def _fit_pysr_backend(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[set[str], str, np.ndarray]:
        if find_spec("pysr") is None:
            raise ImportError(
                "SymbolicStructureScreener backend='pysr' requires pysr. "
                "Install pysr and its Julia runtime before using this backend."
            )

        from pysr import PySRRegressor

        kwargs = {
            "niterations": 40,
            "populations": 8,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["exp", "log"],
            "model_selection": "best",
            "random_state": self.random_state,
            "procs": 0,
            "verbosity": 0,
        }
        kwargs.update(self.symbolic_backend_kwargs)

        model = PySRRegressor(
            variable_names=list(X.columns),
            **kwargs,
        )
        model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        preds = np.asarray(model.predict(X.to_numpy(dtype=float)), dtype=float).ravel()

        try:
            expr_obj = model.sympy()
            expression = str(expr_obj)
            selected_features = {str(symbol) for symbol in getattr(expr_obj, "free_symbols", set())}
        except Exception:
            expression = str(model)
            selected_features = {
                column
                for column in X.columns
                if re.search(rf"\\b{re.escape(column)}\\b", expression)
            }

        selected_features = {feature for feature in selected_features if feature in X.columns}
        return selected_features, expression, preds

    def _symbolic_regression_metrics(
        self,
        y: np.ndarray,
        preds: np.ndarray,
        complexity: int,
    ) -> dict:
        y = np.asarray(y, dtype=float).ravel()
        preds = np.asarray(preds, dtype=float).ravel()
        rss0 = float(np.sum((y - y.mean()) ** 2))
        rss1 = float(np.sum((y - preds) ** 2))
        rss1 = max(rss1, 1e-12)
        n = max(len(y), 1)

        aic0 = n * np.log(max(rss0 / max(n, 1), 1e-12)) + 2.0
        aic1 = n * np.log(max(rss1 / max(n, 1), 1e-12)) + 2.0 * (complexity + 1)
        delta_aic = float(aic0 - aic1)

        if rss0 <= rss1 or n <= (complexity + 1):
            p_value = np.nan
        else:
            df_num = max(int(complexity), 1)
            df_den = max(int(n - complexity - 1), 1)
            f_stat = ((rss0 - rss1) / df_num) / (rss1 / df_den)
            p_value = float(stats.f.sf(f_stat, df_num, df_den)) if np.isfinite(f_stat) else np.nan

        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - (rss1 / ss_tot) if ss_tot > 1e-12 else 0.0
        score = max(delta_aic, 0.0) + max(r2, 0.0)
        return {
            "symbolic_score": round(float(score), 4),
            "symbolic_p_value": round(float(p_value), 4) if np.isfinite(p_value) else np.nan,
            "symbolic_delta_aic": round(float(delta_aic), 4),
        }

    def _best_structure(self, y: pd.Series, intercept, x: pd.Series, covariate: str) -> dict:
        candidates = self._candidate_designs(x, covariate)
        best = {
            "symbolic_form": "none",
            "symbolic_expression": "",
            "symbolic_score": 0.0,
            "symbolic_p_value": np.nan,
            "symbolic_delta_aic": 0.0,
            "symbolic_selected": False,
        }

        for form, expression, design in candidates:
            design = design.replace([np.inf, -np.inf], np.nan).dropna()
            aligned_y = y.loc[design.index]
            if len(design) < 20:
                continue
            model = sm.OLS(aligned_y, sm.add_constant(design, has_constant="add")).fit()
            _, p_value, _ = model.compare_f_test(
                sm.OLS(aligned_y, sm.add_constant(pd.DataFrame(index=design.index), has_constant="add")).fit()
            )
            delta_aic = float(
                sm.OLS(aligned_y, sm.add_constant(pd.DataFrame(index=design.index), has_constant="add")).fit().aic - model.aic
            )
            score = max(delta_aic, 0.0) + max(float(model.rsquared), 0.0)
            if score > best["symbolic_score"]:
                best = {
                    "symbolic_form": form,
                    "symbolic_expression": expression,
                    "symbolic_score": round(score, 4),
                    "symbolic_p_value": round(float(p_value), 4),
                    "symbolic_delta_aic": round(delta_aic, 4),
                    "symbolic_selected": bool(p_value < self.alpha and delta_aic >= self.min_delta_aic),
                }
        return best

    def _candidate_designs(self, x: pd.Series, covariate: str):
        if self._is_categorical(x):
            if self._is_binary(x):
                design = pd.DataFrame({covariate: x.astype(float)})
                return [("categorical", covariate, design)]
            dummies = pd.get_dummies(x.astype("category"), prefix=covariate, drop_first=True).astype(float)
            return [("categorical", f"C({covariate})", dummies)]

        x = x.astype(float)
        median = float(np.nanmedian(x)) if np.isfinite(np.nanmedian(x)) else 0.0
        scale = float(np.nanstd(x)) if np.isfinite(np.nanstd(x)) and np.nanstd(x) > 1e-8 else 1.0
        centered = (x - median) / scale

        candidates = [
            ("linear", f"({covariate} - median_{covariate})/{scale:.4g}", pd.DataFrame({covariate: centered})),
            ("quadratic", f"(({covariate} - median_{covariate})/{scale:.4g})^2", pd.DataFrame({covariate: centered ** 2})),
            ("exponential", f"exp(({covariate} - median_{covariate})/{max(abs(median), 1.0):.4g})", pd.DataFrame({covariate: (x - median) / max(abs(median), 1.0)})),
        ]
        if bool((x > 0).all()):
            candidates.append(
                ("power", f"log({covariate}/median_{covariate})", pd.DataFrame({covariate: np.log(x / max(median, 1e-6))}))
            )
        return candidates

    def _build_symbolic_matrix(
        self,
        covariates: pd.DataFrame,
        candidate_covariates: list[str],
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        frames = []
        feature_to_cov: dict[str, str] = {}

        for cov in candidate_covariates:
            if cov not in covariates.columns:
                continue
            series = covariates[cov]
            if self._is_categorical(series) and not self._is_binary(series):
                dummies = pd.get_dummies(series.astype("category"), prefix=cov, drop_first=True).astype(float)
                if dummies.empty:
                    continue
                for column in dummies.columns:
                    feature_to_cov[column] = cov
                frames.append(dummies)
            else:
                frames.append(pd.DataFrame({cov: series.astype(float)}))
                feature_to_cov[cov] = cov

        if not frames:
            return pd.DataFrame(index=covariates.index), {}
        return pd.concat(frames, axis=1), feature_to_cov

    @staticmethod
    def _empty_symbolic_rows(parameter: str, candidate_covariates: list[str], backend: str) -> list[dict]:
        return [
            {
                "parameter": parameter,
                "covariate": cov,
                "symbolic_form": "none",
                "symbolic_expression": "",
                "symbolic_score": 0.0,
                "symbolic_p_value": np.nan,
                "symbolic_delta_aic": 0.0,
                "symbolic_selected": False,
                "symbolic_backend": backend,
            }
            for cov in candidate_covariates
        ]

    @staticmethod
    def _is_binary(series: pd.Series) -> bool:
        vals = pd.Series(series).dropna().unique().tolist()
        return len(vals) <= 2 and set(vals).issubset({0, 1, 0.0, 1.0, False, True})

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        return series.dtype == "object" or str(series.dtype).startswith("category") or SymbolicStructureScreener._is_binary(series)

    def summary(self) -> pd.DataFrame:
        if self._summary is None or len(self._summary) == 0:
            return pd.DataFrame()
        return self._summary.sort_values(["parameter", "symbolic_score"], ascending=[True, False]).reset_index(drop=True)
