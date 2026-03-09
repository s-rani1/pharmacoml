"""ScreeningResults — summary, visualization, and NONMEM code generation."""
from __future__ import annotations
import numpy as np
import pandas as pd


class ScreeningResults:
    def __init__(self, importances, shap_values, models, parameter_names, covariate_names,
                 covariate_names_encoded, encoding_map, functional_forms,
                 significance_threshold, n_bootstrap, covariate_data, ebe_data,
                 cv_diagnostics=None):
        self._importances = importances
        self._shap_values = shap_values
        self._models = models
        self.parameter_names = parameter_names
        self.covariate_names = covariate_names
        self._cov_encoded = covariate_names_encoded
        self._encoding_map = encoding_map
        self._func_forms = functional_forms
        self._sig_thresh = significance_threshold
        self._n_boot = n_bootstrap
        self._cov_data = covariate_data
        self._ebe_data = ebe_data
        self._cv_diagnostics = cv_diagnostics or {}

    def summary(self) -> pd.DataFrame:
        rows = []
        for param in self.parameter_names:
            imp_df = self._importances[param]
            cov_imp_df = pd.DataFrame({
                orig_cov: imp_df[[c for c in enc_cols if c in imp_df.columns]].sum(axis=1)
                for orig_cov, enc_cols in self._encoding_map.items()
                if any(c in imp_df.columns for c in enc_cols)
            })
            if len(cov_imp_df.columns):
                rank_df = cov_imp_df.rank(axis=1, method="dense", ascending=False)
                denom = cov_imp_df.max(axis=1).replace(0, np.nan)
                rel_df = cov_imp_df.div(denom, axis=0).fillna(0.0)
                top_k = max(1, int(np.ceil(len(cov_imp_df.columns) * 0.25)))
            else:
                rank_df = pd.DataFrame(index=imp_df.index)
                rel_df = pd.DataFrame(index=imp_df.index)
                top_k = 1

            for orig_cov, enc_cols in self._encoding_map.items():
                matching = [c for c in enc_cols if c in imp_df.columns]
                if not matching:
                    continue
                cov_imp = imp_df[matching].sum(axis=1)
                pct_nz = (cov_imp > 1e-10).mean() * 100
                stability_mask = pd.Series(False, index=cov_imp.index)
                if orig_cov in rank_df.columns:
                    stability_mask = (
                        (rank_df[orig_cov] <= top_k) |
                        (rel_df[orig_cov] >= 0.20)
                    )
                rows.append({
                    "parameter": param, "covariate": orig_cov,
                    "mean_importance": round(cov_imp.mean(), 4),
                    "ci_lower": round(np.percentile(cov_imp, 2.5), 4),
                    "ci_upper": round(np.percentile(cov_imp, 97.5), 4),
                    "pct_nonzero": round(pct_nz, 1),
                    "mean_bootstrap_rank": round(float(rank_df[orig_cov].mean()), 2) if orig_cov in rank_df.columns else np.nan,
                    "stability_frequency": round(float(stability_mask.mean()), 4),
                    "cv_r2": round(float(self._cv_diagnostics.get(param, {}).get("r2", np.nan)), 4)
                    if param in self._cv_diagnostics else np.nan,
                    "significant": pct_nz >= (1 - self._sig_thresh) * 100,
                    "functional_form": self._func_forms.get((param, orig_cov), "unknown"),
                })
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values(["parameter", "mean_importance"], ascending=[True, False]).reset_index(drop=True)
        return df

    def significant_covariates(self) -> pd.DataFrame:
        return self.summary().query("significant").reset_index(drop=True)

    def to_dataframe(self) -> pd.DataFrame:
        return self.summary()

    def plot_importance(self, parameter=None, top_n=10):
        import matplotlib.pyplot as plt
        params = [parameter] if parameter else self.parameter_names
        fig, axes = plt.subplots(1, len(params), figsize=(6 * len(params), 5))
        if len(params) == 1: axes = [axes]
        for ax, param in zip(axes, params):
            pdf = self.summary().query(f"parameter == '{param}'").head(top_n)
            colors = ["#2196F3" if s else "#BDBDBD" for s in pdf["significant"]]
            ax.barh(range(len(pdf)), pdf["mean_importance"], color=colors)
            ax.set_yticks(range(len(pdf)))
            ax.set_yticklabels(pdf["covariate"])
            ax.set_xlabel("Importance")
            ax.set_title(f"Covariates: {param}")
            ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_dependence(self, parameter, covariate):
        import matplotlib.pyplot as plt
        enc_cols = self._encoding_map.get(covariate)
        col_idx = self._cov_encoded.index(enc_cols[0])
        sv = self._shap_values[parameter]
        x = self._cov_data[enc_cols[0]].values
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, sv[:, col_idx], alpha=0.5, s=20, color="#2196F3")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(covariate)
        ax.set_ylabel(f"SHAP value ({parameter})")
        form = self._func_forms.get((parameter, covariate), "?")
        ax.set_title(f"{covariate} → {parameter} (form: {form})")
        plt.tight_layout()
        plt.show()

    def to_nonmem(self, parameter=None) -> str:
        sig = self.significant_covariates()
        if parameter:
            sig = sig[sig["parameter"] == parameter]
        if len(sig) == 0:
            return "; No significant covariates identified"
        lines = ["; --- Covariate relationships (pharmacoml) ---"]
        for _, row in sig.iterrows():
            p, c, f = row["parameter"], row["covariate"], row["functional_form"]
            if f == "power":
                lines.append(f"TV{p} = THETA(X) * ({c}/median_{c})**THETA(Y)")
            elif f == "linear":
                lines.append(f"TV{p} = THETA(X) * (1 + THETA(Y)*({c}-median_{c}))")
            elif f == "exponential":
                lines.append(f"TV{p} = THETA(X) * EXP(THETA(Y)*({c}-median_{c}))")
            elif f == "categorical":
                lines.append(f"IF ({c}.EQ.1) TV{p} = THETA(X) * THETA(Y)")
            else:
                lines.append(f"; {c} on {p}: form={f} — specify manually")
            lines.append(f"; importance={row['mean_importance']:.4f} CI=[{row['ci_lower']:.4f},{row['ci_upper']:.4f}]")
        code = "\n".join(lines)
        print(code)
        return code

    def __repr__(self):
        n_sig = len(self.significant_covariates())
        return f"ScreeningResults(params={self.parameter_names}, sig={n_sig})"
