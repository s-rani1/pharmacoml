"""Functional form detection from SHAP dependence patterns."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit


def detect_functional_forms(shap_values, covariate_data, ebe_data, encoding_map):
    results = {}
    col_list = covariate_data.columns.tolist()
    for param, sv in shap_values.items():
        for orig_cov, enc_cols in encoding_map.items():
            if "__xor__" in orig_cov:
                results[(param, orig_cov)] = "interaction_xor"
                continue
            if "__x__" in orig_cov:
                results[(param, orig_cov)] = "interaction_product"
                continue
            if len(enc_cols) > 1:
                results[(param, orig_cov)] = "categorical"
                continue
            col_name = enc_cols[0]
            if col_name not in col_list:
                continue
            col_idx = col_list.index(col_name)
            x = covariate_data[col_name].values
            y = sv[:, col_idx]
            if np.std(y) < 1e-8:
                results[(param, orig_cov)] = "none"
                continue
            uniq = np.unique(x[~np.isnan(x)])
            if len(uniq) <= 2:
                results[(param, orig_cov)] = "categorical"
                continue
            results[(param, orig_cov)] = _best_form(x, y)
    return results


def _best_form(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return "unknown"
    scores = {}
    try:
        _, _, r, _, _ = stats.linregress(x, y)
        scores["linear"] = r ** 2
    except Exception:
        scores["linear"] = 0
    xp = x[x > 0]
    yp = y[x > 0]
    if len(xp) > 10:
        try:
            s, i, r2, _, _ = stats.linregress(np.log(xp), yp)
            pred = i + s * np.log(xp)
            ss_r = np.sum((yp - pred) ** 2)
            ss_t = np.sum((yp - np.mean(yp)) ** 2)
            scores["power"] = max(0, 1 - ss_r / ss_t) if ss_t > 0 else 0
        except Exception:
            scores["power"] = 0
    try:
        xn = (x - np.mean(x)) / (np.std(x) + 1e-10)
        popt, _ = curve_fit(lambda x, a, b, c: a * np.exp(b * x) + c, xn, y, p0=[1, .1, 0], maxfev=2000)
        pred = popt[0] * np.exp(popt[1] * xn) + popt[2]
        ss_r = np.sum((y - pred) ** 2)
        ss_t = np.sum((y - np.mean(y)) ** 2)
        scores["exponential"] = max(0, 1 - ss_r / ss_t) if ss_t > 0 else 0
    except Exception:
        scores["exponential"] = 0
    if not scores:
        return "nonlinear"
    best = max(scores, key=scores.get)
    if scores[best] < 0.3:
        return "nonlinear"
    if best in ("power", "exponential") and scores[best] - scores.get("linear", 0) < 0.05:
        return "linear"
    return best
