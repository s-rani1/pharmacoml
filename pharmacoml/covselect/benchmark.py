"""
pharmacoml.covselect.benchmark — Simulation-based benchmarking framework.

Generates PK datasets with known true covariates at configurable effect sizes,
runs all screening methods, and computes standard metrics:
  - Sensitivity (true positive rate)
  - Specificity (true negative rate)
  - F1 score (harmonic mean of precision and recall)
  - False Discovery Rate (FDR)
  - Precision, Recall

Based on methodology from:
  - Asiimwe et al. (2024) — F1 comparison across ML and PMX methods
  - Sibieude et al. (2021) — ROC curves, scenario-based simulations
  - Karlsen et al. (2025) — systematic review of covariate selection methods
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkScenario:
    """Definition of a single simulation scenario."""
    name: str
    n_subjects: int = 300
    true_covariates: dict = field(default_factory=dict)
    # Format: {(param, cov): {"form": "power", "effect": 0.75}}
    n_noise_covariates: int = 5
    noise_correlation: float = 0.0  # correlation between noise covariates
    eta_sd: dict = field(default_factory=lambda: {"CL": 0.30, "V": 0.25})
    seed: int = 42


@dataclass
class ExternalBenchmarkCase:
    """External benchmark case loaded from a public dataset."""
    name: str
    ebes: pd.DataFrame
    covariates: pd.DataFrame
    ground_truth: dict
    source: str = "external"


def simulate_scenario(scenario: BenchmarkScenario) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Simulate a PK dataset for a given scenario.

    Returns (ebes, covariates, ground_truth) where ground_truth is
    {(param, cov): True/False} for all pairs.
    """
    rng = np.random.RandomState(scenario.seed)
    n = scenario.n_subjects

    # Generate standard covariates
    covs = {
        "WT": rng.normal(70, 15, n).clip(40, 130),
        "AGE": rng.normal(55, 12, n).clip(18, 90),
        "SEX": rng.binomial(1, 0.5, n).astype(float),
        "CRCL": rng.normal(90, 25, n).clip(20, 180),
        "ALB": rng.normal(4.0, 0.5, n).clip(2.5, 5.5),
        "HT": rng.normal(170, 10, n).clip(145, 200),
    }
    # Add noise covariates
    for i in range(scenario.n_noise_covariates):
        if scenario.noise_correlation > 0 and "WT" in covs:
            # Correlated with WT
            covs[f"NOISE_{i+1}"] = (covs["WT"] * scenario.noise_correlation +
                                     rng.normal(0, 1, n) * (1-scenario.noise_correlation))
        else:
            covs[f"NOISE_{i+1}"] = rng.normal(0, 1, n)

    cov_df = pd.DataFrame(covs)

    # Generate EBEs based on true model
    params = {}
    for param_name, eta_sd in scenario.eta_sd.items():
        eta = rng.normal(0, eta_sd, n)
        base = 5.0 if param_name == "CL" else 50.0  # typical values

        value = np.full(n, base)
        for (p, c), effect_info in scenario.true_covariates.items():
            if p != param_name:
                continue
            x = cov_df[c].values
            form = effect_info.get("form", "power")
            effect = effect_info.get("effect", 0.5)
            median_x = np.median(x)

            if form == "power":
                value = value * (x / median_x) ** effect
            elif form == "linear":
                value = value * (1 + effect * (x - median_x) / median_x)
            elif form == "categorical":
                value = value * (1 + effect * x)  # x is 0/1
            elif form == "exponential":
                value = value * np.exp(effect * (x - median_x) / median_x)

        value = value * np.exp(eta)
        params[param_name] = value

    ebe_df = pd.DataFrame(params)

    # Ground truth
    all_covs = list(cov_df.columns)
    all_params = list(ebe_df.columns)
    ground_truth = {}
    for param in all_params:
        for cov in all_covs:
            ground_truth[(param, cov)] = (param, cov) in scenario.true_covariates

    return ebe_df, cov_df, ground_truth


def compute_metrics(predicted: set, truth: dict) -> dict:
    """Compute classification metrics for covariate selection.

    Parameters
    ----------
    predicted : set of (param, cov) tuples that were flagged as significant
    truth : dict of {(param, cov): True/False}

    Returns dict with TP, FP, TN, FN, sensitivity, specificity, precision, recall, F1, FDR
    """
    tp = sum(1 for k, v in truth.items() if v and k in predicted)
    fp = sum(1 for k, v in truth.items() if not v and k in predicted)
    tn = sum(1 for k, v in truth.items() if not v and k not in predicted)
    fn = sum(1 for k, v in truth.items() if v and k not in predicted)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "F1": round(f1, 3),
        "FDR": round(fdr, 3),
    }


# ──────────────────────────────────────────────────────────────
# Pre-built scenarios matching published papers
# ──────────────────────────────────────────────────────────────

SCENARIO_SIMPLE = BenchmarkScenario(
    name="Simple: 2 true covariates, low noise",
    n_subjects=300,
    true_covariates={
        ("CL", "WT"): {"form": "power", "effect": 0.75},
        ("V", "WT"): {"form": "power", "effect": 1.0},
    },
    n_noise_covariates=3,
    eta_sd={"CL": 0.30, "V": 0.25},
)

SCENARIO_MODERATE = BenchmarkScenario(
    name="Moderate: 4 true covariates, some noise",
    n_subjects=400,
    true_covariates={
        ("CL", "WT"): {"form": "power", "effect": 0.75},
        ("CL", "CRCL"): {"form": "power", "effect": 0.50},
        ("CL", "SEX"): {"form": "categorical", "effect": -0.20},
        ("V", "WT"): {"form": "power", "effect": 1.0},
    },
    n_noise_covariates=5,
    eta_sd={"CL": 0.30, "V": 0.25},
)

SCENARIO_COMPLEX = BenchmarkScenario(
    name="Complex: 6 true covariates, high noise, correlations",
    n_subjects=500,
    true_covariates={
        ("CL", "WT"): {"form": "power", "effect": 0.75},
        ("CL", "AGE"): {"form": "power", "effect": -0.30},
        ("CL", "CRCL"): {"form": "power", "effect": 0.50},
        ("CL", "SEX"): {"form": "categorical", "effect": -0.15},
        ("V", "WT"): {"form": "power", "effect": 1.0},
        ("V", "AGE"): {"form": "linear", "effect": -0.10},
    },
    n_noise_covariates=8,
    noise_correlation=0.3,
    eta_sd={"CL": 0.35, "V": 0.30},
)

SCENARIO_WEAK_EFFECTS = BenchmarkScenario(
    name="Weak effects: small covariate effects, hard to detect",
    n_subjects=200,
    true_covariates={
        ("CL", "WT"): {"form": "power", "effect": 0.30},
        ("CL", "AGE"): {"form": "linear", "effect": -0.10},
        ("V", "WT"): {"form": "power", "effect": 0.50},
    },
    n_noise_covariates=5,
    eta_sd={"CL": 0.40, "V": 0.35},  # high variability relative to effects
)

ALL_SCENARIOS = [SCENARIO_SIMPLE, SCENARIO_MODERATE, SCENARIO_COMPLEX, SCENARIO_WEAK_EFFECTS]


KEKIC_TRUTH_DEFAULTS = {
    "reference": {"AGE", "SMK", "BWT", "COPD"},
    "linearly_dependent": {"AGE", "SMK", "BWT", "COPD"},
    "low_frequency": {"AGE", "SMK", "BWT", "COPD"},
    "high_iiv": {"AGE", "SMK", "BWT", "COPD"},
    "pop_100": {"AGE", "SMK", "BWT", "COPD"},
    # Main-effect approximation for the XOR scenario. The paper scenario is
    # interaction-driven, so users should interpret this case cautiously.
    "xor": {"SMK", "COPD"},
}


def load_kekic_case(base_path: str | Path, scenario_name: str) -> ExternalBenchmarkCase:
    """Load a public synthetic scenario from the Kekic et al. 2026 repo."""
    scenario_path = Path(base_path) / scenario_name / "train.csv"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Could not find Kekic scenario at {scenario_path}")

    df = pd.read_csv(scenario_path)
    feature_cols = [c for c in df.columns if c not in {"ID", "STUDYID", "ETA2", "train_flag"}]
    ebes = df[["ETA2"]].copy()
    covs = df[feature_cols].copy()

    truth_covariates = KEKIC_TRUTH_DEFAULTS.get(scenario_name)
    if truth_covariates is None:
        raise ValueError(f"No ground truth configured for Kekic scenario '{scenario_name}'")

    truth = {("ETA2", cov): cov in truth_covariates for cov in covs.columns}
    return ExternalBenchmarkCase(
        name=f"Kekic:{scenario_name}",
        ebes=ebes,
        covariates=covs,
        ground_truth=truth,
        source="Kekic et al. 2026 public synthetic scenario",
    )


class BenchmarkSuite:
    """Run standardized benchmarks across methods and scenarios.

    Usage:
        suite = BenchmarkSuite(methods=["xgboost", "random_forest", "lasso"],
                                include_traditional=True, include_significance_filter=True)
        results_df = suite.run(scenarios=ALL_SCENARIOS)
        suite.print_results()
    """

    def __init__(self, methods: list[str] | None = None,
                 include_traditional: bool = True,
                 include_significance_filter: bool = True,
                 n_bootstrap: int = 20,
                 random_state: int = 42,
                 hybrid_tier: str = "confirmed",
                 hybrid_kwargs: dict | None = None):
        self.methods = methods or ["hybrid", "random_forest", "aalasso"]
        self.include_traditional = include_traditional
        self.include_significance_filter = include_significance_filter
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.hybrid_tier = hybrid_tier
        self.hybrid_kwargs = hybrid_kwargs or {}
        self._results_df = None

    def _predict_pairs(self, method: str, ebes: pd.DataFrame, covs: pd.DataFrame):
        from pharmacoml.covselect import CovariateScreener, HybridScreener, SCMBridge, STGScreener

        if method == "hybrid":
            hybrid_kwargs = {
                "n_bootstrap": self.n_bootstrap,
                "random_state": self.random_state,
                "include_traditional": self.include_traditional,
            }
            hybrid_kwargs.update(self.hybrid_kwargs)
            screener = HybridScreener(**hybrid_kwargs)
            report = screener.fit(ebes, covs)
            if self.hybrid_tier == "core":
                selected = report.core_covariates()
            elif self.hybrid_tier == "scm":
                selected = report.scm_covariates()
            elif self.hybrid_tier == "confirmed":
                selected = report.confirmed_covariates()
            else:
                selected = report.candidate_covariates()
            predicted = set(zip(selected["parameter"], selected["covariate"])) if len(selected) else set()
            return predicted, report.summary(), "Hybrid"

        if method == "scm_bridge":
            results = SCMBridge().fit(ebes, covs)
            selected = results.selected_covariates()
            predicted = set(zip(selected["parameter"], selected["covariate"])) if len(selected) else set()
            return predicted, results.summary(), "SCMBridge"

        if method == "stg":
            screener = STGScreener(
                n_bootstrap=self.n_bootstrap,
                random_state=self.random_state,
                run_permutation=self.include_significance_filter,
            )
            summary = screener.fit(ebes, covs)
            selected = summary[summary["final_significant"]]
            predicted = set(zip(selected["parameter"], selected["covariate"])) if len(selected) else set()
            return predicted, summary, "STG"

        screener = CovariateScreener(
            method=method,
            n_bootstrap=self.n_bootstrap,
            random_state=self.random_state,
        )
        results = screener.fit(ebes, covs)
        summary = results.summary()
        if self.include_significance_filter:
            from pharmacoml.covselect.significance import SignificanceFilter

            sf = SignificanceFilter(perm_n=10, random_state=self.random_state)
            summary = sf.apply(results, covs, method_name=method, run_permutation=True)
            selected = summary[summary["final_significant"]]
            label = f"ML:{method}+filter"
        else:
            selected = results.significant_covariates()
            label = f"ML:{method}"
        predicted = set(zip(selected["parameter"], selected["covariate"])) if len(selected) else set()
        return predicted, summary, label

    def _run_single_case(self, name: str, ebes: pd.DataFrame, covs: pd.DataFrame, truth: dict) -> list[dict]:
        from pharmacoml.covselect.traditional import TraditionalScreener

        all_rows = []

        for method in self.methods:
            try:
                predicted, _, label = self._predict_pairs(method, ebes, covs)
                metrics = compute_metrics(predicted, truth)
                metrics["method"] = label
                metrics["scenario"] = name
                all_rows.append(metrics)
            except Exception as e:
                print(f"  {method}: ERROR — {str(e)[:50]}")

        if self.include_traditional:
            try:
                trad = TraditionalScreener(alpha=0.01)
                trad_results = trad.fit(ebes, covs)
                trad_sig = trad_results.significant_covariates()
                predicted_t = set(zip(trad_sig["parameter"], trad_sig["covariate"]))
                metrics_t = compute_metrics(predicted_t, truth)
                metrics_t["method"] = "Traditional:SCM"
                metrics_t["scenario"] = name
                all_rows.append(metrics_t)
            except Exception as e:
                print(f"  Traditional: ERROR — {str(e)[:50]}")

        return all_rows

    def run(self, scenarios: list[BenchmarkScenario] | None = None) -> pd.DataFrame:
        """Run all methods on all scenarios. Returns results DataFrame."""
        if scenarios is None:
            scenarios = ALL_SCENARIOS

        from pharmacoml.covselect.traditional import TraditionalScreener

        all_rows = []

        for scenario in scenarios:
            ebes, covs, truth = simulate_scenario(scenario)
            print(f"\n{'─'*60}")
            print(f"Scenario: {scenario.name}")
            print(f"  Subjects={scenario.n_subjects}, True covariates={sum(truth.values())}, "
                  f"Total pairs={len(truth)}")

            # ML methods
            for method in self.methods:
                try:
                    predicted, _, label = self._predict_pairs(method, ebes, covs)
                    metrics = compute_metrics(predicted, truth)
                    metrics["method"] = label
                    metrics["scenario"] = scenario.name
                    all_rows.append(metrics)

                    print(f"  {label}: F1={metrics['F1']:.3f} "
                          f"(TP={metrics['TP']},FP={metrics['FP']},FN={metrics['FN']})")

                except Exception as e:
                    print(f"  {method}: ERROR — {str(e)[:50]}")

            # Traditional method
            if self.include_traditional:
                try:
                    trad = TraditionalScreener(alpha=0.01)
                    trad_results = trad.fit(ebes, covs)
                    trad_sig = trad_results.significant_covariates()
                    predicted_t = set(zip(trad_sig["parameter"], trad_sig["covariate"]))
                    metrics_t = compute_metrics(predicted_t, truth)
                    metrics_t["method"] = "Traditional:SCM"
                    metrics_t["scenario"] = scenario.name
                    all_rows.append(metrics_t)
                    print(f"  Traditional SCM: F1={metrics_t['F1']:.3f} "
                          f"(TP={metrics_t['TP']},FP={metrics_t['FP']},FN={metrics_t['FN']})")
                except Exception as e:
                    print(f"  Traditional: ERROR — {str(e)[:50]}")

        self._results_df = pd.DataFrame(all_rows)
        return self._results_df

    def run_external_cases(self, cases: list[ExternalBenchmarkCase]) -> pd.DataFrame:
        """Run methods on externally loaded benchmark cases."""
        all_rows = []
        for case in cases:
            print(f"\n{'─' * 60}")
            print(f"External case: {case.name} ({case.source})")
            print(f"  Subjects={len(case.ebes)}, Total pairs={len(case.ground_truth)}")
            all_rows.extend(
                self._run_single_case(
                    name=case.name,
                    ebes=case.ebes,
                    covs=case.covariates,
                    truth=case.ground_truth,
                )
            )

        self._results_df = pd.DataFrame(all_rows)
        return self._results_df

    def print_results(self):
        """Print formatted benchmark results table."""
        if self._results_df is None:
            print("No results. Call run() first.")
            return

        print("\n" + "=" * 90)
        print("BENCHMARK RESULTS: F1 Score Comparison")
        print("=" * 90)

        pivot = self._results_df.pivot_table(
            index="method", columns="scenario", values="F1", aggfunc="first"
        )
        print(pivot.to_string(float_format="%.3f"))

        print("\n" + "=" * 90)
        print("BENCHMARK RESULTS: False Discovery Rate (lower is better)")
        print("=" * 90)
        pivot_fdr = self._results_df.pivot_table(
            index="method", columns="scenario", values="FDR", aggfunc="first"
        )
        print(pivot_fdr.to_string(float_format="%.3f"))

        print("\n" + "=" * 90)
        print("BENCHMARK RESULTS: Sensitivity (higher is better)")
        print("=" * 90)
        pivot_sens = self._results_df.pivot_table(
            index="method", columns="scenario", values="sensitivity", aggfunc="first"
        )
        print(pivot_sens.to_string(float_format="%.3f"))
