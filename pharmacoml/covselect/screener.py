"""Core CovariateScreener class — NONMEM-agnostic ML covariate identification."""
from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd
from pharmacoml.covselect.engines import get_engine, ENGINE_REGISTRY
from pharmacoml.covselect.results import ScreeningResults
from pharmacoml.covselect.functional_form import detect_functional_forms
from pharmacoml.covselect.selection_utils import cross_validated_r2


class CovariateScreener:
    """ML-based covariate screening for population PK/PD models.

    Works with EBEs from any estimation tool: NONMEM, nlmixr2, Monolix, Pumas.

    Parameters
    ----------
    method : {"xgboost", "lightgbm", "catboost", "random_forest", "elastic_net", "lasso"}
    n_bootstrap : int — bootstrap iterations for confidence intervals
    significance_threshold : float — covariate significant if nonzero in (1-threshold)*100% of bootstraps
    """

    _VALID_METHODS = tuple(ENGINE_REGISTRY.keys())

    def __init__(self, method: str = "xgboost", n_bootstrap: int = 100,
                 significance_threshold: float = 0.05, random_state: int | None = 42,
                 cv_splits: int = 5):
        if method not in self._VALID_METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {self._VALID_METHODS}")
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.significance_threshold = significance_threshold
        self.random_state = random_state
        self.cv_splits = cv_splits
        self._results = None

    def fit(self, ebes: pd.DataFrame, covariates: pd.DataFrame,
            parameter_names: list[str] | None = None,
            covariate_names: list[str] | None = None) -> ScreeningResults:
        """Screen covariates for all PK/PD parameters.

        Parameters
        ----------
        ebes : DataFrame — columns are PK/PD parameters (CL, V, KA), rows are subjects
        covariates : DataFrame — columns are covariates (WT, AGE, SEX), rows aligned with ebes
        """
        if not isinstance(ebes, pd.DataFrame) or not isinstance(covariates, pd.DataFrame):
            raise TypeError("ebes and covariates must be pandas DataFrames")
        if len(ebes) != len(covariates):
            raise ValueError("ebes and covariates must have same number of rows")

        params = parameter_names or list(ebes.columns)
        covs = covariate_names or list(covariates.columns)
        ebe_sub = ebes[params].copy()
        cov_sub = covariates[covs].copy()

        cov_encoded, encoding_map = self._encode_covariates(cov_sub)

        all_importances, all_shap_values, all_models = {}, {}, {}
        cv_diagnostics = {}

        for param in params:
            y = ebe_sub[param].values
            X = cov_encoded.values
            mask = ~np.isnan(y)
            y_clean, X_clean = y[mask], X[mask]

            if len(y_clean) < 20:
                raise ValueError(f"'{param}' has <20 non-NaN values")

            engine = get_engine(self.method, random_state=self.random_state)

            # Bootstrap
            rng = np.random.RandomState(self.random_state)
            imp_rows = []
            for _ in range(self.n_bootstrap):
                idx = rng.choice(len(y_clean), size=len(y_clean), replace=True)
                eng = get_engine(self.method, random_state=self.random_state)
                eng.fit(X_clean[idx], y_clean[idx])
                imp_rows.append(eng.feature_importances(cov_encoded.columns.tolist()))
            all_importances[param] = pd.DataFrame(imp_rows)

            # Final model for SHAP
            engine.fit(X_clean, y_clean)
            all_shap_values[param] = engine.shap_values(X_clean)
            all_models[param] = engine
            cv_diagnostics[param] = {
                "r2": round(
                    cross_validated_r2(
                        method_name=self.method,
                        X=X_clean,
                        y=y_clean,
                        random_state=self.random_state or 42,
                        n_splits=self.cv_splits,
                    ),
                    4,
                )
            }

        functional_forms = detect_functional_forms(
            all_shap_values, cov_encoded, ebe_sub, encoding_map
        )

        self._results = ScreeningResults(
            importances=all_importances, shap_values=all_shap_values,
            models=all_models, parameter_names=params, covariate_names=covs,
            covariate_names_encoded=cov_encoded.columns.tolist(),
            encoding_map=encoding_map, functional_forms=functional_forms,
            significance_threshold=self.significance_threshold,
            n_bootstrap=self.n_bootstrap,
            covariate_data=cov_encoded, ebe_data=ebe_sub,
            cv_diagnostics=cv_diagnostics,
        )
        return self._results

    def _encode_covariates(self, covariates):
        encoding_map = {}
        parts = []
        for col in covariates.columns:
            if covariates[col].dtype == "object" or covariates[col].dtype.name == "category":
                dummies = pd.get_dummies(covariates[col], prefix=col, drop_first=True).astype(float)
                encoding_map[col] = dummies.columns.tolist()
                parts.append(dummies)
            else:
                encoding_map[col] = [col]
                parts.append(covariates[[col]])
        return pd.concat(parts, axis=1), encoding_map

    @property
    def results(self):
        return self._results
