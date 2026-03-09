"""
pharmacoml.covselect.traditional — Traditional covariate screening baseline.

Implements standard pre-ML methods for fair comparison:
  1. Univariate Pearson/Spearman correlation
  2. Linear regression R² + p-value
  3. Welch t-test / ANOVA for categoricals

References: Jonsson & Karlsson (1998), Karlsen et al. (2025) CPT:PSP
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


class TraditionalScreener:
    """Traditional pharmacometric covariate screening."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def fit(self, ebes: pd.DataFrame, covariates: pd.DataFrame) -> "TraditionalResults":
        rows = []
        for param in ebes.columns:
            y = ebes[param].values
            mask_y = ~np.isnan(y)
            y_c = y[mask_y]
            for cov in covariates.columns:
                x = covariates[cov].values[mask_y]
                mask_x = ~np.isnan(x)
                yv, xv = y_c[mask_x], x[mask_x]
                if len(xv) < 10: continue
                row = {"parameter": param, "covariate": cov}
                uniq = np.unique(xv)
                if len(uniq) <= 2 and len(uniq) >= 2:
                    g1 = yv[xv == uniq[0]]; g2 = yv[xv == uniq[1]]
                    if len(g1) >= 3 and len(g2) >= 3:
                        t, p = stats.ttest_ind(g1, g2, equal_var=False)
                        ps = np.sqrt((np.var(g1)+np.var(g2))/2)
                        row.update(test="welch_t", p_value=float(p),
                                   effect_size=round(abs(np.mean(g1)-np.mean(g2))/(ps+1e-10),4))
                    else:
                        row.update(test="welch_t", p_value=1.0, effect_size=0.0)
                else:
                    r_p, p_p = stats.pearsonr(xv, yv)
                    r_s, p_s = stats.spearmanr(xv, yv)
                    _, _, r_v, p_r, _ = stats.linregress(xv, yv)
                    row.update(test="correlation", pearson_r=round(float(r_p),4),
                               pearson_p=float(p_p), spearman_r=round(float(r_s),4),
                               spearman_p=float(p_s), regression_r2=round(float(r_v**2),4),
                               p_value=min(float(p_p), float(p_s), float(p_r)),
                               effect_size=round(float(r_v**2),4))
                row["significant"] = row["p_value"] < self.alpha
                rows.append(row)
        return TraditionalResults(pd.DataFrame(rows), self.alpha)


class TraditionalResults:
    def __init__(self, df, alpha):
        self._df = df; self.alpha = alpha
    def summary(self):
        if len(self._df)==0: return self._df
        return self._df.sort_values(["parameter","p_value"],ascending=[True,True]).reset_index(drop=True)
    def significant_covariates(self):
        return self.summary().query("significant").reset_index(drop=True)
    def compare_with(self, ml_summary_df, ml_label="ML"):
        trad_sig = self.significant_covariates()
        trad_pairs = set(zip(trad_sig["parameter"], trad_sig["covariate"]))
        sig_col = "final_significant" if "final_significant" in ml_summary_df.columns else "significant"
        ml_sig = ml_summary_df[ml_summary_df[sig_col]]
        ml_pairs = set(zip(ml_sig["parameter"], ml_sig["covariate"]))
        all_pairs = trad_pairs | ml_pairs
        rows = []
        for param, cov in sorted(all_pairs):
            in_t = (param,cov) in trad_pairs; in_m = (param,cov) in ml_pairs
            if in_t and in_m: agree = "✅ Both"
            elif in_t: agree = "📊 Trad only"
            elif in_m: agree = "🤖 ML only"
            else: agree = "❌ Neither"
            tp = self._df[(self._df["parameter"]==param)&(self._df["covariate"]==cov)]
            t_p = round(tp["p_value"].values[0],4) if len(tp)>0 else "-"
            mp = ml_summary_df[(ml_summary_df["parameter"]==param)&(ml_summary_df["covariate"]==cov)]
            m_i = round(mp["mean_importance"].values[0],4) if len(mp)>0 else "-"
            rows.append({"parameter":param,"covariate":cov,"trad_sig":in_t,"trad_p":t_p,
                          f"{ml_label}_sig":in_m,f"{ml_label}_imp":m_i,"agreement":agree})
        return pd.DataFrame(rows)
    def __repr__(self):
        return f"TraditionalResults(sig={len(self.significant_covariates())}, alpha={self.alpha})"
