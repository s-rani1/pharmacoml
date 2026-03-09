"""
pharmacoml.covselect.significance — Rigorous false-positive control.

Three-layer filter based on methods from:
  - Asiimwe et al. (2024) AAPS J — correlation-aware screening
  - Brooks et al. (2025) CPT:PSP (shap-cov) — R² reliability gate
  - Sibieude et al. (2021) JPKPD — permutation-based null distributions

Layer 1: R² reliability gate — reject parameters where ML can't predict EBEs
Layer 2: Correlation filter — deduplicate correlated covariates
Layer 3: Permutation test — only keep covariates whose importance exceeds null
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pharmacoml.covselect.selection_utils import (
    association_matrix,
    benjamini_hochberg,
    cross_validated_r2,
)


class ReliabilityGate:
    """If ML R² < threshold for a parameter, flag results as low-confidence."""
    def __init__(self, min_r2: float = 0.10):
        self.min_r2 = min_r2

    def check(self, screening_results, method_name: str, param: str) -> dict:
        try:
            diagnostics = screening_results._cv_diagnostics.get(param, {})
            if "r2" in diagnostics:
                r2 = float(diagnostics["r2"])
            else:
                X = screening_results._cov_data.values
                y = screening_results._ebe_data[param].values
                mask = ~np.isnan(y)
                r2 = cross_validated_r2(method_name, X[mask], y[mask])
        except Exception:
            r2 = 0.0
        return {"r2": round(r2, 4), "reliable": r2 >= self.min_r2,
                "message": f"R²={r2:.3f}" + ("" if r2 >= self.min_r2 else " — LOW CONFIDENCE")}


class CorrelationFilter:
    """Group covariates by mixed-type association; keep one representative per group."""
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold
        self._groups = None
        self._assoc_matrix = None

    def fit(self, covariates: pd.DataFrame):
        self._assoc_matrix = association_matrix(covariates)
        cols = list(self._assoc_matrix.columns)
        parent = {c: c for c in cols}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb

        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i < j and self._assoc_matrix.loc[c1, c2] > self.threshold:
                    union(c1, c2)
        groups = {}
        for c in cols:
            r = find(c)
            groups.setdefault(r, []).append(c)
        self._groups = {k: v for k, v in groups.items() if len(v) > 1}
        return self

    def filter_results(self, summary_df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        if not self._groups:
            summary_df = summary_df.copy()
            summary_df["corr_filtered"] = False
            return summary_df, []
        notes = []
        remove_pairs = set()
        for root, members in self._groups.items():
            notes.append({"group": members, "threshold": self.threshold})
            for param in summary_df["parameter"].unique():
                pdf = summary_df[(summary_df["parameter"] == param) & (summary_df["covariate"].isin(members))]
                if len(pdf) > 1:
                    sort_cols = [c for c in ["standardized_importance", "mean_importance", "stability_frequency"] if c in pdf.columns]
                    ranked = pdf.sort_values(sort_cols, ascending=[False] * len(sort_cols)) if sort_cols else pdf
                    best_idx = ranked.index[0]
                    for idx in pdf.index:
                        if idx != best_idx:
                            remove_pairs.add((param, summary_df.loc[idx, "covariate"]))
        filtered = summary_df.copy()
        filtered["corr_filtered"] = False
        for param, cov in remove_pairs:
            mask = (filtered["parameter"] == param) & (filtered["covariate"] == cov)
            filtered.loc[mask, "corr_filtered"] = True
        return filtered, notes

    @property
    def groups(self): return self._groups or {}


class PermutationImportanceTest:
    """Shuffle each covariate n times, refit, compare true vs null importance."""
    def __init__(self, n_permutations: int = 30, alpha: float = 0.05, random_state: int = 42):
        self.n_permutations = n_permutations
        self.random_state = random_state

        # Auto-correct alpha: minimum achievable p-value is 1/(n_perm+1)
        # Alpha must be strictly greater than this, otherwise nothing can pass
        min_achievable_p = 1.0 / (n_permutations + 1)
        if alpha <= min_achievable_p:
            corrected = min_achievable_p + 0.01
            import warnings
            warnings.warn(
                f"perm_alpha={alpha} is unreachable with perm_n={n_permutations} "
                f"(minimum p-value = 1/{n_permutations+1} = {min_achievable_p:.4f}). "
                f"Auto-correcting to perm_alpha={corrected:.3f}. "
                f"To use alpha=0.05, set perm_n >= {int(1/alpha)}.",
                stacklevel=3,
            )
            self.alpha = corrected
        else:
            self.alpha = alpha

    def test(self, engine_name: str, X, y, feature_names, true_importances) -> dict:
        from pharmacoml.covselect.engines import get_engine
        rng = np.random.RandomState(self.random_state)
        results = {}
        for feat_idx, feat_name in enumerate(feature_names):
            null_imps = []
            for _ in range(self.n_permutations):
                X_perm = X.copy()
                X_perm[:, feat_idx] = rng.permutation(X_perm[:, feat_idx])
                try:
                    eng = get_engine(engine_name, random_state=self.random_state)
                    eng.fit(X_perm, y)
                    imp = eng.feature_importances(feature_names)
                    null_imps.append(imp.get(feat_name, 0.0))
                except Exception:
                    null_imps.append(0.0)
            null_arr = np.array(null_imps)
            true_imp = true_importances.get(feat_name, 0.0)
            p_value = (np.sum(null_arr >= true_imp) + 1) / (self.n_permutations + 1)
            results[feat_name] = {
                "true_importance": round(true_imp, 6),
                "null_mean": round(float(np.mean(null_arr)), 6),
                "null_p95": round(float(np.percentile(null_arr, 95)), 6),
                "p_value": round(float(p_value), 4),
                "perm_significant": p_value < self.alpha,
            }
        return results


class SignificanceFilter:
    """Combined three-layer filter. The recommended way to control false positives.

    Usage:
        from pharmacoml.covselect.significance import SignificanceFilter
        sf = SignificanceFilter()
        filtered = sf.apply(results, covariates, method_name="xgboost")
        sf.print_diagnostics()
    """
    def __init__(self, min_r2=0.10, corr_threshold=0.80,
                 perm_n=30, perm_alpha=0.05, random_state=42):
        # Validate and auto-correct perm_n / perm_alpha compatibility
        min_achievable_p = 1.0 / (perm_n + 1)
        if perm_alpha <= min_achievable_p:
            corrected_alpha = min_achievable_p + 0.01
            import warnings
            warnings.warn(
                f"perm_alpha={perm_alpha} is unreachable with perm_n={perm_n} "
                f"(min p-value = {min_achievable_p:.4f}). "
                f"Auto-correcting to perm_alpha={corrected_alpha:.3f}. "
                f"Recommendation: use perm_n>={int(1/perm_alpha)} for alpha={perm_alpha}, "
                f"or perm_n>=99 for alpha=0.05, or perm_n>=19 for alpha=0.10.",
                stacklevel=2,
            )
            perm_alpha = corrected_alpha

        self.r2_gate = ReliabilityGate(min_r2=min_r2)
        self.corr_filter = CorrelationFilter(threshold=corr_threshold)
        self.perm_test = PermutationImportanceTest(n_permutations=perm_n, alpha=perm_alpha, random_state=random_state)
        self.random_state = random_state
        self._diagnostics = {}

    def apply(self, screening_results, covariate_data: pd.DataFrame,
              method_name: str = "xgboost", run_permutation: bool = True) -> pd.DataFrame:
        """Apply all three layers. Returns filtered summary DataFrame.

        Adds columns: r2, r2_reliable, corr_filtered, perm_significant, final_significant
        """
        summary = screening_results.summary().copy()
        summary["standardized_importance"] = summary.groupby("parameter")["mean_importance"].transform(
            lambda s: s / max(float(s.max()), 1e-12)
        ).round(4)

        # ── Layer 1: R² gate ──
        r2_info = {}
        for param in screening_results.parameter_names:
            if param in screening_results._models:
                r2_info[param] = self.r2_gate.check(screening_results, method_name, param)

        summary["r2"] = summary["parameter"].map(lambda p: r2_info.get(p, {}).get("r2", 0))
        summary["r2_reliable"] = summary["parameter"].map(lambda p: r2_info.get(p, {}).get("reliable", False))

        # ── Layer 2: Correlation filter ──
        self.corr_filter.fit(covariate_data)
        summary, corr_notes = self.corr_filter.filter_results(summary)

        # ── Layer 3: Permutation test (optional — slow but rigorous) ──
        if run_permutation:
            perm_results = {}
            for param in screening_results.parameter_names:
                if param in screening_results._models:
                    engine = screening_results._models[param]
                    X = screening_results._cov_data.values
                    y = screening_results._ebe_data[param].values
                    mask = ~np.isnan(y)
                    true_imp = engine.feature_importances(screening_results._cov_data.columns.tolist())
                    perm = self.perm_test.test(method_name, X[mask], y[mask],
                                               screening_results._cov_data.columns.tolist(), true_imp)
                    perm_results[param] = perm

            # Map permutation significance back
            def get_perm_sig(row):
                param_perm = perm_results.get(row["parameter"], {})
                # Map original covariate name to encoded columns
                enc_map = screening_results._encoding_map
                enc_cols = enc_map.get(row["covariate"], [row["covariate"]])
                # Covariate is permutation-significant if any of its encoded columns are
                for ec in enc_cols:
                    if ec in param_perm and param_perm[ec]["perm_significant"]:
                        return True
                return False

            def get_perm_p(row):
                param_perm = perm_results.get(row["parameter"], {})
                enc_map = screening_results._encoding_map
                enc_cols = enc_map.get(row["covariate"], [row["covariate"]])
                p_values = []
                for ec in enc_cols:
                    if ec in param_perm:
                        p_values.append(param_perm[ec]["p_value"])
                return min(p_values) if p_values else 1.0

            summary["perm_p_value"] = summary.apply(get_perm_p, axis=1)
            summary["perm_significant"] = summary.apply(get_perm_sig, axis=1)
            summary["perm_q_value"] = summary.groupby("parameter")["perm_p_value"].transform(
                lambda s: benjamini_hochberg(s.values)
            )
            summary["perm_significant"] = summary["perm_q_value"] < self.perm_test.alpha
        else:
            summary["perm_p_value"] = np.nan
            summary["perm_q_value"] = np.nan
            summary["perm_significant"] = summary["significant"]  # fall back to bootstrap

        # ── Final significance: must pass ALL layers ──
        summary["final_significant"] = (
            summary["significant"] &         # bootstrap importance > 0 in 95% of samples
            summary["r2_reliable"] &          # ML model explains enough variance
            ~summary["corr_filtered"] &       # not a redundant correlated covariate
            summary["perm_significant"]       # beats null distribution
        )

        self._diagnostics = {
            "r2": r2_info,
            "correlation_groups": self.corr_filter.groups,
            "correlation_notes": corr_notes,
            "permutation": perm_results if run_permutation else None,
        }
        return summary

    def print_diagnostics(self):
        print("=" * 60)
        print("SIGNIFICANCE FILTER DIAGNOSTICS")
        print("=" * 60)
        print("\n--- R² Reliability ---")
        for p, info in self._diagnostics.get("r2", {}).items():
            s = "✅" if info["reliable"] else "⚠️"
            print(f"  {p}: R²={info['r2']:.3f} {s}")
        groups = self._diagnostics.get("correlation_groups", {})
        if groups:
            print("\n--- Correlated Groups (deduplicated) ---")
            for root, members in groups.items():
                print(f"  {members}")
        else:
            print("\n--- No correlated groups found ---")
        if self._diagnostics.get("permutation"):
            print("\n--- Permutation Test Results ---")
            for param, perm in self._diagnostics["permutation"].items():
                print(f"\n  {param}:")
                for feat, info in sorted(perm.items(), key=lambda x: x[1]["p_value"]):
                    s = "✅" if info["perm_significant"] else "❌"
                    print(f"    {feat:<15} p={info['p_value']:.3f} true={info['true_importance']:.4f} null_mean={info['null_mean']:.4f} {s}")
