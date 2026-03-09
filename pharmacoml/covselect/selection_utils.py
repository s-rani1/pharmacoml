"""Shared selection utilities for robust covariate screening."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def resolve_cv_splits(n_samples: int, preferred: int = 5) -> int:
    """Pick a defensible CV split count for the available sample size."""
    if n_samples < 40:
        return 2
    if n_samples < 80:
        return 3
    if n_samples < 160:
        return 4
    return min(preferred, 5)


def cross_validated_predictions(
    method_name: str,
    X,
    y,
    random_state: int = 42,
    n_splits: int = 5,
):
    """Generate out-of-fold predictions for a given engine."""
    from pharmacoml.covselect.engines import get_engine

    X_arr = np.asarray(X)
    y_arr = np.asarray(y).ravel()
    splits = resolve_cv_splits(len(y_arr), preferred=n_splits)
    if splits < 2:
        return np.full(len(y_arr), np.nan)

    preds = np.full(len(y_arr), np.nan, dtype=float)
    cv = KFold(n_splits=splits, shuffle=True, random_state=random_state)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_arr, y_arr)):
        engine = get_engine(method_name, random_state=(random_state or 42) + fold_idx)
        engine.fit(X_arr[train_idx], y_arr[train_idx])
        preds[test_idx] = np.asarray(engine.predict(X_arr[test_idx])).ravel()
    return preds


def cross_validated_r2(
    method_name: str,
    X,
    y,
    random_state: int = 42,
    n_splits: int = 5,
) -> float:
    """Compute out-of-fold R² for a given engine and feature subset."""
    y_arr = np.asarray(y).ravel()
    preds = cross_validated_predictions(
        method_name=method_name,
        X=X,
        y=y_arr,
        random_state=random_state,
        n_splits=n_splits,
    )
    mask = ~(np.isnan(y_arr) | np.isnan(preds))
    if mask.sum() < 3:
        return 0.0
    return float(r2_score(y_arr[mask], preds[mask]))


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return np.array([], dtype=float)

    order = np.argsort(p)
    ranked = p[order]
    m = float(len(ranked))
    adjusted = np.empty_like(ranked)

    prev = 1.0
    for idx in range(len(ranked) - 1, -1, -1):
        rank = idx + 1
        candidate = ranked[idx] * m / rank
        prev = min(prev, candidate)
        adjusted[idx] = min(prev, 1.0)

    q = np.empty_like(adjusted)
    q[order] = adjusted
    return q


def is_binary_series(series: pd.Series) -> bool:
    vals = pd.Series(series).dropna().unique()
    if len(vals) == 0 or len(vals) > 2:
        return False
    if pd.api.types.is_bool_dtype(series):
        return True
    return set(np.asarray(vals).tolist()).issubset({0, 1, 0.0, 1.0, False, True})


def _series_kind(series: pd.Series) -> str:
    if is_binary_series(series):
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        return "continuous"
    return "categorical"


def _correlation_ratio(categories, measurements) -> float:
    categories = pd.Series(categories)
    measurements = pd.Series(measurements)
    mask = ~(categories.isna() | measurements.isna())
    categories = categories[mask]
    measurements = measurements[mask].astype(float)
    if len(categories) < 3:
        return 0.0

    overall_mean = measurements.mean()
    numerator = 0.0
    denominator = float(((measurements - overall_mean) ** 2).sum())
    if denominator <= 0:
        return 0.0

    for level, group in measurements.groupby(categories):
        if len(group) == 0:
            continue
        numerator += len(group) * (group.mean() - overall_mean) ** 2
    return float(np.sqrt(max(numerator / denominator, 0.0)))


def _cramers_v(a, b) -> float:
    table = pd.crosstab(pd.Series(a), pd.Series(b))
    if table.empty:
        return 0.0
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    if n <= 0:
        return 0.0
    r, k = table.shape
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / denom))


def association_strength(series_a: pd.Series, series_b: pd.Series) -> float:
    """Mixed-type association score on [0, 1]."""
    a = pd.Series(series_a)
    b = pd.Series(series_b)
    mask = ~(a.isna() | b.isna())
    a = a[mask]
    b = b[mask]
    if len(a) < 3:
        return 0.0

    kind_a = _series_kind(a)
    kind_b = _series_kind(b)

    if kind_a == "continuous" and kind_b == "continuous":
        try:
            pearson = abs(float(stats.pearsonr(a.astype(float), b.astype(float)).statistic))
        except Exception:
            pearson = 0.0
        try:
            spearman = abs(float(stats.spearmanr(a.astype(float), b.astype(float)).statistic))
        except Exception:
            spearman = 0.0
        return max(pearson, spearman)

    if kind_a == "binary" and kind_b == "continuous":
        try:
            return abs(float(stats.pointbiserialr(a.astype(int), b.astype(float)).statistic))
        except Exception:
            return 0.0

    if kind_a == "continuous" and kind_b == "binary":
        try:
            return abs(float(stats.pointbiserialr(b.astype(int), a.astype(float)).statistic))
        except Exception:
            return 0.0

    if kind_a == "continuous" and kind_b == "categorical":
        return _correlation_ratio(b, a)

    if kind_a == "categorical" and kind_b == "continuous":
        return _correlation_ratio(a, b)

    return _cramers_v(a, b)


def association_matrix(covariates: pd.DataFrame) -> pd.DataFrame:
    """Pairwise mixed-type association matrix."""
    cols = list(covariates.columns)
    matrix = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols, dtype=float)
    for i, c1 in enumerate(cols):
        for j in range(i + 1, len(cols)):
            c2 = cols[j]
            assoc = association_strength(covariates[c1], covariates[c2])
            matrix.loc[c1, c2] = assoc
            matrix.loc[c2, c1] = assoc
    return matrix


def build_interaction_terms(
    covariates: pd.DataFrame,
    candidate_covariates: list[str] | None = None,
    max_pairs: int = 12,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """Create product and binary XOR interaction terms from candidate covariates."""
    if candidate_covariates:
        cols = [c for c in candidate_covariates if c in covariates.columns]
    else:
        cols = list(covariates.columns)

    numeric_cols = []
    binary_cols = set()
    for col in cols:
        if not pd.api.types.is_numeric_dtype(covariates[col]):
            continue
        numeric_cols.append(col)
        if is_binary_series(covariates[col]):
            binary_cols.add(col)

    pairs = list(combinations(numeric_cols, 2))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    interaction_data = {}
    metadata: dict[str, dict[str, str]] = {}
    for left, right in pairs:
        product_name = f"{left}__x__{right}"
        interaction_data[product_name] = covariates[left].astype(float) * covariates[right].astype(float)
        metadata[product_name] = {"left": left, "right": right, "operator": "product"}

        if left in binary_cols and right in binary_cols:
            xor_name = f"{left}__xor__{right}"
            left_bool = covariates[left].fillna(0).astype(int).astype(bool)
            right_bool = covariates[right].fillna(0).astype(int).astype(bool)
            interaction_data[xor_name] = left_bool ^ right_bool
            interaction_data[xor_name] = interaction_data[xor_name].astype(float)
            metadata[xor_name] = {"left": left, "right": right, "operator": "xor"}

    if not interaction_data:
        return pd.DataFrame(index=covariates.index), {}
    return pd.DataFrame(interaction_data, index=covariates.index), metadata


def estimate_parameter_information(
    series: pd.Series,
    use_log: bool = True,
) -> dict[str, float | bool]:
    """Empirical low-information profile for EBE/individual-parameter targets.

    This is not a formal NONMEM shrinkage estimate. It is a pragmatic proxy
    intended for benchmarked threshold calibration when only subject-level
    targets are available.
    """
    values = pd.Series(series).dropna().astype(float)
    if len(values) < 20:
        return {
            "n_samples": int(len(values)),
            "dispersion": 0.0,
            "unique_ratio": 0.0,
            "shrinkage_proxy": 1.0,
            "low_information": True,
        }

    if use_log and bool((values > 0).all()):
        values = np.log(values)

    dispersion = float(np.nanstd(values))
    rounded = np.round(values, 6)
    unique_ratio = float(min(len(np.unique(rounded)) / max(len(values), 1), 1.0))

    # Reference dispersion chosen to match typical eta/log-parameter spread in
    # the benchmark cases. Lower spread and fewer unique subject values imply
    # a more shrinkage-like, low-information target.
    spread_score = min(dispersion / 0.35, 1.0)
    information_score = 0.75 * spread_score + 0.25 * unique_ratio
    shrinkage_proxy = float(np.clip(1.0 - information_score, 0.0, 1.0))

    return {
        "n_samples": int(len(values)),
        "dispersion": round(dispersion, 4),
        "unique_ratio": round(unique_ratio, 4),
        "shrinkage_proxy": round(shrinkage_proxy, 4),
        "low_information": bool(shrinkage_proxy >= 0.45),
    }
