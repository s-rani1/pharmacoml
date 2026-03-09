"""Fixed public benchmark suite for benchmark-gated hybrid development."""
from __future__ import annotations

import importlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from pharmacoml.covselect.benchmark import (
    BenchmarkScenario,
    compute_metrics,
    load_kekic_case,
    simulate_scenario,
)
from pharmacoml.covselect.experimental import MultiModelConsensusScreener
from pharmacoml.covselect.hybrid import HybridScreener

BASELINE_BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "fixed_public_baseline.json"


@dataclass
class PublicBenchmarkCase:
    name: str
    ebes: pd.DataFrame
    covariates: pd.DataFrame
    truth: set
    truth_mode: str = "parameter"
    primary_tier: str = "confirmed"
    source: str = "public"
    parameter_shrinkage: dict[str, float] | None = None
    hybrid_kwargs: dict | None = None


@dataclass(frozen=True)
class BenchmarkAcceptanceThresholds:
    min_mean_precision: float = 0.65
    min_mean_recall: float = 0.75
    min_mean_f1: float = 0.70
    max_mean_fdr: float = 0.35
    min_case_f1: float = 0.30
    min_case_precision: float = 0.25


DEFAULT_ACCEPTANCE_THRESHOLDS = BenchmarkAcceptanceThresholds()


def _default_data_path(env_name: str, fallback: str) -> Path:
    value = os.environ.get(env_name, fallback)
    return Path(value)


def _resolve_pharmpy_example_base() -> Path:
    override = os.environ.get("PHARMACOML_PHENO_BASE")
    if override:
        base = Path(override)
    else:
        try:
            pharmpy = importlib.import_module("pharmpy")
        except ImportError as exc:
            raise FileNotFoundError(
                "Pharmpy is not installed, so the public pheno benchmark cannot be loaded."
            ) from exc
        base = Path(pharmpy.__file__).resolve().parent / "internals" / "example_models"

    if not base.exists():
        raise FileNotFoundError(
            f"Pharmpy example model directory not found at {base}. "
            "Set PHARMACOML_PHENO_BASE to the example_models directory if needed."
        )
    return base


def load_pheno_case() -> PublicBenchmarkCase:
    base = _resolve_pharmpy_example_base()
    obs = pd.read_csv(base / "pheno.dta", sep=r"\s+")
    subj = obs.groupby("ID", as_index=False).first()[["ID", "WGT", "APGR"]]
    subj["ASPHYXIA"] = (subj["APGR"] < 5).astype(float)

    phi = pd.read_csv(base / "pheno.phi", sep=r"\s+", skiprows=1)
    phi = phi.rename(columns={"ETA(1)": "ETA_CL", "ETA(2)": "ETA_VC"})[["ID", "ETA_CL", "ETA_VC"]]
    ext = pd.read_csv(base / "pheno.ext", sep=r"\s+", skiprows=1)
    final = ext.loc[ext["ITERATION"] == -1000000000].iloc[0]

    theta1 = float(final["THETA1"])
    theta2 = float(final["THETA2"])
    theta3 = float(final["THETA3"])

    df = subj.merge(phi, on="ID")
    df["TVCL"] = theta1 * df["WGT"]
    df["TVV"] = theta2 * df["WGT"]
    df.loc[df["APGR"] < 5, "TVV"] = df.loc[df["APGR"] < 5, "TVV"] * (1.0 + theta3)
    df["CL"] = df["TVCL"] * np.exp(df["ETA_CL"])
    df["VC"] = df["TVV"] * np.exp(df["ETA_VC"])

    return PublicBenchmarkCase(
        name="pheno",
        ebes=df[["CL", "VC"]],
        covariates=df[["WGT", "ASPHYXIA"]],
        truth={("CL", "WGT"), ("VC", "WGT"), ("VC", "ASPHYXIA")},
        truth_mode="parameter",
        primary_tier="candidate",
        source="Pharmpy phenobarbital example",
    )


def load_eleveld_case(
    dataframe_path: str | Path | None = None,
    modelparams_path: str | Path | None = None,
) -> PublicBenchmarkCase:
    dataframe_path = Path(dataframe_path) if dataframe_path else _default_data_path(
        "PHARMACOML_ELEVELD_DATAFRAME",
        "/tmp/propofol_wahlquist.csv",
    )
    modelparams_path = Path(modelparams_path) if modelparams_path else _default_data_path(
        "PHARMACOML_ELEVELD_MODELPARAMS",
        "/tmp/eleveld_modelparams.csv",
    )
    if not dataframe_path.exists() or not modelparams_path.exists():
        raise FileNotFoundError("Eleveld benchmark files not found. Set PHARMACOML_ELEVELD_* env vars or place files in /tmp.")

    raw = pd.read_csv(dataframe_path)
    subj = raw.groupby("ID", as_index=False).first()[["ID", "AGE", "WGT", "HT", "M1F2", "A1V2", "NOADD1ADD2", "PMA", "BMI"]]
    subj = subj.rename(columns={"HT": "HGT", "NOADD1ADD2": "TECH"})
    params = pd.read_csv(modelparams_path)
    df = params.merge(subj, on="ID", how="inner")

    truth = {"AGE", "WGT", "HGT", "M1F2", "A1V2", "TECH", "PMA"}
    return PublicBenchmarkCase(
        name="eleveld_union",
        ebes=df[["V1", "V2", "V3", "CL", "Q2", "Q3"]],
        covariates=df[["AGE", "WGT", "HGT", "M1F2", "A1V2", "TECH", "PMA", "BMI"]],
        truth=truth,
        truth_mode="union",
        primary_tier="confirmed",
        source="Eleveld/Wahlquist public propofol data",
    )


def load_ggpmx_theophylline_case(
    data_path: str | Path | None = None,
    eta_path: str | Path | None = None,
    estimates_path: str | Path | None = None,
) -> PublicBenchmarkCase:
    data_path = Path(data_path) if data_path else _default_data_path(
        "PHARMACOML_GGPMX_DATA",
        "/tmp/ggpmx_theophylline.csv",
    )
    eta_path = Path(eta_path) if eta_path else _default_data_path(
        "PHARMACOML_GGPMX_ETA",
        "/tmp/ggpmx_indiv_eta.txt",
    )
    estimates_path = Path(estimates_path) if estimates_path else _default_data_path(
        "PHARMACOML_GGPMX_ESTIMATES",
        "/tmp/ggpmx_estimates.txt",
    )
    if not data_path.exists() or not eta_path.exists() or not estimates_path.exists():
        raise FileNotFoundError("ggPMX theophylline benchmark files not found. Set PHARMACOML_GGPMX_* env vars or place files in /tmp.")

    raw = pd.read_csv(data_path)
    subj = raw.groupby("ID", as_index=False).first()[["ID", "WT0", "AGE0"]]
    eta = pd.read_csv(eta_path, sep=r"\s+")
    eta = eta[["ID", "eta_ka_mode", "eta_V_mode", "eta_Cl_mode", "tWT0", "tAGE0", "SEX_1", "STUD_2"]]
    est = pd.read_csv(estimates_path, sep=";", engine="python")
    coef = dict(zip(est.iloc[:, 0].astype(str).str.strip(), pd.to_numeric(est.iloc[:, 1], errors="coerce")))

    df = eta.merge(subj, on="ID", how="left")
    df["KA"] = coef["ka_pop"] * np.exp(df["eta_ka_mode"])
    df["V"] = coef["V_pop"] * np.exp(coef["beta_V_tWT0"] * df["tWT0"] + df["eta_V_mode"])
    df["CL"] = coef["Cl_pop"] * np.exp(
        coef["beta_Cl_tWT0"] * df["tWT0"] +
        coef["beta_Cl_tAGE0"] * df["tAGE0"] +
        coef["beta_Cl_SEX_1"] * df["SEX_1"] +
        coef["beta_Cl_STUD_2"] * df["STUD_2"] +
        df["eta_Cl_mode"]
    )

    return PublicBenchmarkCase(
        name="ggpmx_theophylline",
        ebes=df[["KA", "V", "CL"]],
        covariates=df[["WT0", "AGE0", "SEX_1", "STUD_2"]],
        truth={("V", "WT0"), ("CL", "WT0"), ("CL", "AGE0"), ("CL", "SEX_1"), ("CL", "STUD_2")},
        truth_mode="parameter",
        primary_tier="candidate",
        source="ggPMX Monolix theophylline example",
    )


def load_fixed_public_cases() -> list[PublicBenchmarkCase]:
    cases = []
    for loader in [load_pheno_case, load_eleveld_case, load_ggpmx_theophylline_case]:
        try:
            cases.append(loader())
        except (FileNotFoundError, ImportError):
            continue
    return cases


def load_high_shrinkage_case() -> PublicBenchmarkCase:
    rng = np.random.RandomState(808)
    n = 120
    wt = rng.normal(70, 12, n).clip(40, 130)
    age = rng.normal(56, 11, n).clip(18, 90)
    sex = rng.binomial(1, 0.5, n).astype(float)
    formulation = rng.binomial(1, 0.35, n).astype(float)

    cl = 5.2 * (wt / np.median(wt)) ** 0.75 * np.exp(rng.normal(0, 0.18, n))
    # This parameter intentionally carries covariate-driven artefact so the
    # benchmark can verify that explicit user-supplied shrinkage suppresses
    # unreliable screening results on the affected parameter.
    v_artifact = 48.0 + 0.45 * (age - np.median(age)) + 6.0 * formulation + rng.normal(0, 0.8, n)

    return PublicBenchmarkCase(
        name="high_shrinkage_user_input",
        ebes=pd.DataFrame({"CL": cl, "V": v_artifact}),
        covariates=pd.DataFrame(
            {
                "WT": wt,
                "AGE": age,
                "SEX": sex,
                "FORMULATION": formulation,
            }
        ),
        truth={("CL", "WT")},
        truth_mode="parameter",
        primary_tier="candidate",
        source="High-shrinkage synthetic benchmark with explicit user shrinkage",
        parameter_shrinkage={"CL": 0.12, "V": 0.82},
    )


def load_age_pma_distinct_case() -> PublicBenchmarkCase:
    rng = np.random.RandomState(911)
    n = 140
    age = rng.normal(45, 12, n).clip(18, 80)
    pma = age + 18.0 + rng.normal(0, 0.05, n)
    wt = rng.normal(70, 13, n).clip(40, 130)
    sex = rng.binomial(1, 0.5, n).astype(float)

    cl = (
        5.0 *
        (wt / np.median(wt)) ** 0.75 *
        (1 + 0.30 * (age - age.mean()) / max(age.std(), 1e-6)) *
        (1 + 0.40 * (pma - pma.mean()) / max(pma.std(), 1e-6)) *
        np.exp(rng.normal(0, 0.10, n))
    )

    return PublicBenchmarkCase(
        name="age_pma_distinct",
        ebes=pd.DataFrame({"CL": cl}),
        covariates=pd.DataFrame({"WT": wt, "AGE": age, "PMA": pma, "SEX": sex}),
        truth={("CL", "WT"), ("CL", "AGE"), ("CL", "PMA")},
        truth_mode="parameter",
        primary_tier="candidate",
        source="Biological AGE/PMA proxy-preservation benchmark",
        hybrid_kwargs={
            "boosting_method": "random_forest",
            "penalized_method": "elastic_net",
            "include_scm": False,
            "rfe_enabled": False,
            "candidate_score_threshold": 0.0,
            "preserve_proxy_pairs": [("AGE", "PMA")],
            "n_bootstrap": 1,
        },
    )


def load_interaction_screening_case() -> PublicBenchmarkCase:
    rng = np.random.RandomState(1204)
    n = 220
    smk = rng.binomial(1, 0.35, n).astype(float)
    copd = rng.binomial(1, 0.22, n).astype(float)
    age = rng.normal(57, 11, n).clip(18, 90)
    wt = rng.normal(72, 13, n).clip(40, 130)

    xor = (smk.astype(bool) ^ copd.astype(bool)).astype(float)
    cl = 5.1 * (1 + 0.50 * xor) * np.exp(rng.normal(0, 0.10, n))

    return PublicBenchmarkCase(
        name="interaction_xor_screening",
        ebes=pd.DataFrame({"CL": cl}),
        covariates=pd.DataFrame({"WT": wt, "SMK": smk, "COPD": copd, "AGE": age}),
        truth={("CL", "SMK"), ("CL", "COPD"), ("CL", "COPD__xor__SMK")},
        truth_mode="parameter",
        primary_tier="candidate",
        source="Interaction screening benchmark",
        hybrid_kwargs={
            "boosting_method": "random_forest",
            "include_traditional": False,
            "include_scm": False,
            "include_interactions": True,
            "interaction_top_n": 3,
            "interaction_max_pairs": 4,
            "candidate_score_threshold": 0.05,
            "n_bootstrap": 1,
        },
    )


def _scenario_to_case(
    name: str,
    scenario: BenchmarkScenario,
    source: str,
    primary_tier: str = "confirmed",
) -> PublicBenchmarkCase:
    ebes, covariates, truth = simulate_scenario(scenario)
    true_pairs = {pair for pair, selected in truth.items() if selected}
    return PublicBenchmarkCase(
        name=name,
        ebes=ebes,
        covariates=covariates,
        truth=true_pairs,
        truth_mode="parameter",
        primary_tier=primary_tier,
        source=source,
    )


def load_asiimwe_style_cases() -> list[PublicBenchmarkCase]:
    scenarios = [
        BenchmarkScenario(
            name="asiimwe_correlated_small_n",
            n_subjects=114,
            true_covariates={
                ("CL", "WT"): {"form": "power", "effect": 0.70},
                ("CL", "CRCL"): {"form": "power", "effect": 0.35},
                ("CL", "SEX"): {"form": "categorical", "effect": -0.12},
                ("V", "WT"): {"form": "power", "effect": 0.90},
            },
            n_noise_covariates=8,
            noise_correlation=0.70,
            eta_sd={"CL": 0.35, "V": 0.30},
            seed=424,
        ),
    ]
    return [
        _scenario_to_case(
            name=scenario.name,
            scenario=scenario,
            source="Asiimwe-style public correlated-covariate simulation",
            primary_tier="candidate",
        )
        for scenario in scenarios
    ]


def load_shapcov_style_cases() -> list[PublicBenchmarkCase]:
    scenarios = [
        BenchmarkScenario(
            name="shapcov_collinear",
            n_subjects=180,
            true_covariates={
                ("CL", "WT"): {"form": "power", "effect": 0.75},
                ("CL", "AGE"): {"form": "power", "effect": -0.25},
                ("CL", "CRCL"): {"form": "power", "effect": 0.45},
                ("V", "WT"): {"form": "power", "effect": 1.0},
            },
            n_noise_covariates=6,
            noise_correlation=0.65,
            eta_sd={"CL": 0.32, "V": 0.28},
            seed=725,
        ),
    ]
    return [
        _scenario_to_case(
            name=scenario.name,
            scenario=scenario,
            source="Shap-Cov-style public collinear simulation",
            primary_tier="candidate",
        )
        for scenario in scenarios
    ]


def load_optional_kekic_cases(
    base_path: str | Path | None = None,
    scenario_names: tuple[str, ...] = ("reference", "linearly_dependent"),
) -> list[PublicBenchmarkCase]:
    if base_path is None:
        env_path = os.environ.get("PHARMACOML_KEKIC_DIR")
        if not env_path:
            return []
        base_path = env_path

    cases: list[PublicBenchmarkCase] = []
    for scenario_name in scenario_names:
        try:
            loaded = load_kekic_case(base_path, scenario_name)
        except FileNotFoundError:
            continue
        true_pairs = {pair for pair, selected in loaded.ground_truth.items() if selected}
        cases.append(
            PublicBenchmarkCase(
                name=f"kekic_{scenario_name}",
                ebes=loaded.ebes,
                covariates=loaded.covariates,
                truth=true_pairs,
                truth_mode="parameter",
                primary_tier="candidate",
                source=loaded.source,
            )
        )
    return cases


def load_release_benchmark_cases(include_optional_kekic: bool = True) -> list[PublicBenchmarkCase]:
    cases = []
    cases.extend(load_fixed_public_cases())
    cases.append(load_high_shrinkage_case())
    cases.append(load_age_pma_distinct_case())
    cases.append(load_interaction_screening_case())
    cases.extend(load_asiimwe_style_cases())
    cases.extend(load_shapcov_style_cases())
    if include_optional_kekic:
        cases.extend(load_optional_kekic_cases())
    return cases


def _extract_tier_set(report, tier: str, truth_mode: str) -> set:
    if tier == "confirmed":
        df = report.confirmed_covariates()
    elif tier == "candidate":
        df = report.candidate_covariates()
    else:
        df = report.core_covariates()

    if len(df) == 0:
        return set()
    if truth_mode == "union":
        return set(df["covariate"].unique().tolist())
    return set(map(tuple, df[["parameter", "covariate"]].itertuples(index=False, name=None)))


def _extract_multimodel_set(report, truth_mode: str) -> set:
    df = report.consensus_covariates()
    if len(df) == 0:
        return set()
    if truth_mode == "union":
        return set(df["covariate"].unique().tolist())
    return set(map(tuple, df[["parameter", "covariate"]].itertuples(index=False, name=None)))


def _truth_dict_for_case(case: PublicBenchmarkCase) -> dict:
    if case.truth_mode == "union":
        universe = set(case.covariates.columns.tolist())
        return {cov: (cov in case.truth) for cov in universe}
    universe = {(param, cov) for param in case.ebes.columns for cov in case.covariates.columns}
    return {pair: (pair in case.truth) for pair in universe}


def compare_hybrid_variants(
    variants: dict[str, dict] | None = None,
    cases: list[PublicBenchmarkCase] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if variants is None:
        variants = {
            "baseline": {"rfe_enabled": False, "shrinkage_awareness": False},
            "rfe": {"rfe_enabled": True, "shrinkage_awareness": False},
            "shrinkage": {"rfe_enabled": False, "shrinkage_awareness": True},
            "rfe+shrinkage": {"rfe_enabled": True, "shrinkage_awareness": True},
        }
    if cases is None:
        cases = load_release_benchmark_cases()

    details = []
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for variant_name, kwargs in variants.items():
        rfe_enabled = bool(kwargs.get("rfe_enabled", False))
        shrinkage_awareness = bool(kwargs.get("shrinkage_awareness", False))
        for case in cases:
            hybrid_kwargs = {
                "n_bootstrap": 10,
                "run_permutation": False,
            }
            if case.hybrid_kwargs:
                hybrid_kwargs.update(case.hybrid_kwargs)
            hybrid_kwargs.update(kwargs)
            report = HybridScreener(**hybrid_kwargs).fit(
                case.ebes,
                case.covariates,
                parameter_shrinkage=case.parameter_shrinkage,
            )
            for tier in ["confirmed", "candidate"]:
                selected = _extract_tier_set(report, tier=tier, truth_mode=case.truth_mode)
                truth_dict = _truth_dict_for_case(case)
                metrics = compute_metrics(selected, truth_dict)
                details.append(
                    {
                        "variant": variant_name,
                        "scenario": case.name,
                        "tier": tier,
                        "primary": tier == case.primary_tier,
                        "truth_mode": case.truth_mode,
                        "source": case.source,
                        "rfe_enabled": rfe_enabled,
                        "shrinkage_awareness": shrinkage_awareness,
                        **metrics,
                    }
                )

    detail_df = pd.DataFrame(details)
    if len(detail_df) == 0:
        return detail_df, pd.DataFrame()

    summary_df = (
        detail_df[detail_df["primary"]]
        .groupby("variant")
        .agg(
            cases=("scenario", "count"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_F1=("F1", "mean"),
            mean_FDR=("FDR", "mean"),
        )
        .reset_index()
    )
    flags = (
        detail_df.groupby("variant")[["rfe_enabled", "shrinkage_awareness"]]
        .max()
        .reset_index()
    )
    summary_df = summary_df.merge(flags, on="variant", how="left")
    summary_df["primary_score"] = (
        0.60 * summary_df["mean_F1"] +
        0.25 * summary_df["mean_precision"] +
        0.15 * (1.0 - summary_df["mean_FDR"])
    ).round(4)
    summary_df = summary_df.sort_values("primary_score", ascending=False).reset_index(drop=True)
    return detail_df, summary_df


def compare_hybrid_vs_multimodel(
    cases: list[PublicBenchmarkCase] | None = None,
    hybrid_kwargs: dict | None = None,
    multimodel_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cases is None:
        cases = load_release_benchmark_cases()

    hybrid_defaults = {
        "n_bootstrap": 10,
        "run_permutation": False,
    }
    if hybrid_kwargs:
        hybrid_defaults.update(hybrid_kwargs)

    multimodel_defaults = {
        "n_bootstrap": 10,
        "run_permutation": False,
        "include_optional_boosting": False,
        "include_neural": False,
    }
    if multimodel_kwargs:
        multimodel_defaults.update(multimodel_kwargs)

    details = []
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for case in cases:
        truth_dict = _truth_dict_for_case(case)

        current_hybrid_kwargs = dict(hybrid_defaults)
        if case.hybrid_kwargs:
            current_hybrid_kwargs.update(case.hybrid_kwargs)
        hybrid_report = HybridScreener(**current_hybrid_kwargs).fit(
            case.ebes,
            case.covariates,
            parameter_shrinkage=case.parameter_shrinkage,
        )
        hybrid_selected = _extract_tier_set(hybrid_report, tier=case.primary_tier, truth_mode=case.truth_mode)
        hybrid_metrics = compute_metrics(hybrid_selected, truth_dict)
        details.append(
            {
                "workflow": "hybrid",
                "scenario": case.name,
                "tier": case.primary_tier,
                "truth_mode": case.truth_mode,
                "source": case.source,
                **hybrid_metrics,
            }
        )

        multimodel_report = MultiModelConsensusScreener(**multimodel_defaults).fit(case.ebes, case.covariates)
        multimodel_selected = _extract_multimodel_set(multimodel_report, truth_mode=case.truth_mode)
        multimodel_metrics = compute_metrics(multimodel_selected, truth_dict)
        details.append(
            {
                "workflow": "multimodel_consensus",
                "scenario": case.name,
                "tier": "consensus",
                "truth_mode": case.truth_mode,
                "source": case.source,
                **multimodel_metrics,
            }
        )

    detail_df = pd.DataFrame(details)
    if detail_df.empty:
        return detail_df, pd.DataFrame()

    summary_df = (
        detail_df.groupby("workflow")
        .agg(
            cases=("scenario", "count"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_F1=("F1", "mean"),
            mean_FDR=("FDR", "mean"),
        )
        .reset_index()
    )
    summary_df["primary_score"] = (
        0.60 * summary_df["mean_F1"] +
        0.25 * summary_df["mean_precision"] +
        0.15 * (1.0 - summary_df["mean_FDR"])
    ).round(4)
    summary_df = summary_df.sort_values("primary_score", ascending=False).reset_index(drop=True)
    return detail_df, summary_df


def compare_multimodel_variants(
    variants: dict[str, dict] | None = None,
    cases: list[PublicBenchmarkCase] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cases is None:
        cases = load_fixed_public_cases()

    if variants is None:
        variants = {
            "scikit_core": {
                "models": ["random_forest", "extra_trees", "gradient_boosting", "lasso", "aalasso"],
                "top_k": 3,
                "n_bootstrap": 3,
                "use_significance_filter": False,
                "min_model_frequency": 0.50,
                "min_family_support": 2,
                "run_permutation": False,
            },
            "scikit_core_relaxed": {
                "models": ["random_forest", "extra_trees", "gradient_boosting", "lasso", "aalasso"],
                "top_k": 3,
                "n_bootstrap": 3,
                "use_significance_filter": False,
                "min_model_frequency": 0.40,
                "min_family_support": 2,
                "run_permutation": False,
            },
            "linear_tree_balanced": {
                "models": ["random_forest", "extra_trees", "lasso", "elastic_net", "aalasso", "ridge"],
                "top_k": 2,
                "n_bootstrap": 3,
                "use_significance_filter": False,
                "min_model_frequency": 0.50,
                "min_family_support": 2,
                "run_permutation": False,
            },
        }

    details = []
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for variant_name, kwargs in variants.items():
        for case in cases:
            truth_dict = _truth_dict_for_case(case)
            report = MultiModelConsensusScreener(**kwargs).fit(case.ebes, case.covariates)
            selected = _extract_multimodel_set(report, truth_mode=case.truth_mode)
            metrics = compute_metrics(selected, truth_dict)
            details.append(
                {
                    "variant": variant_name,
                    "scenario": case.name,
                    "truth_mode": case.truth_mode,
                    "source": case.source,
                    **metrics,
                }
            )

    detail_df = pd.DataFrame(details)
    if detail_df.empty:
        return detail_df, pd.DataFrame()

    summary_df = (
        detail_df.groupby("variant")
        .agg(
            cases=("scenario", "count"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_F1=("F1", "mean"),
            mean_FDR=("FDR", "mean"),
        )
        .reset_index()
    )
    summary_df["primary_score"] = (
        0.60 * summary_df["mean_F1"] +
        0.25 * summary_df["mean_precision"] +
        0.15 * (1.0 - summary_df["mean_FDR"])
    ).round(4)
    return detail_df, summary_df.sort_values("primary_score", ascending=False).reset_index(drop=True)


def summarize_hybrid_vs_multimodel(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return detail_df

    hybrid = (
        detail_df[detail_df["workflow"] == "hybrid"]
        .set_index("scenario")[["precision", "recall", "F1", "FDR"]]
        .rename(
            columns={
                "precision": "hybrid_precision",
                "recall": "hybrid_recall",
                "F1": "hybrid_F1",
                "FDR": "hybrid_FDR",
            }
        )
    )
    multimodel = (
        detail_df[detail_df["workflow"] == "multimodel_consensus"]
        .set_index("scenario")[["precision", "recall", "F1", "FDR"]]
        .rename(
            columns={
                "precision": "multimodel_precision",
                "recall": "multimodel_recall",
                "F1": "multimodel_F1",
                "FDR": "multimodel_FDR",
            }
        )
    )
    comparison = hybrid.join(multimodel, how="outer").reset_index()
    comparison["delta_precision"] = (comparison["multimodel_precision"] - comparison["hybrid_precision"]).round(3)
    comparison["delta_recall"] = (comparison["multimodel_recall"] - comparison["hybrid_recall"]).round(3)
    comparison["delta_F1"] = (comparison["multimodel_F1"] - comparison["hybrid_F1"]).round(3)
    comparison["delta_FDR"] = (comparison["multimodel_FDR"] - comparison["hybrid_FDR"]).round(3)
    comparison["verdict"] = np.select(
        [
            comparison["delta_F1"] > 0.02,
            comparison["delta_F1"] < -0.02,
        ],
        [
            "multimodel helps",
            "hybrid stronger",
        ],
        default="rough tie",
    )
    return comparison.sort_values("scenario").reset_index(drop=True)


def load_public_benchmark_baseline(path: str | Path | None = None) -> dict:
    path = Path(path) if path else BASELINE_BENCHMARK_PATH
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_summary_to_baseline(
    summary_df: pd.DataFrame,
    baseline: dict | None = None,
    score_tolerance: float = 0.0025,
) -> pd.DataFrame:
    baseline = baseline or load_public_benchmark_baseline()
    expected = pd.DataFrame(baseline.get("summary", []))
    if expected.empty:
        return pd.DataFrame()

    compare_cols = [
        "variant",
        "cases",
        "mean_precision",
        "mean_recall",
        "mean_F1",
        "mean_FDR",
        "primary_score",
    ]
    current = summary_df[compare_cols].copy()
    merged = expected[compare_cols].merge(current, on="variant", how="outer", suffixes=("_baseline", "_current"), indicator=True)
    if "primary_score_current" in merged:
        merged["primary_score_delta"] = merged["primary_score_current"] - merged["primary_score_baseline"]
        merged["case_count_match"] = merged["cases_current"].fillna(-1).astype(int) == merged["cases_baseline"].fillna(-2).astype(int)
        merged["meets_gate"] = (
            merged["case_count_match"] &
            (merged["primary_score_delta"].fillna(-np.inf) >= (-score_tolerance))
        )
    else:
        merged["primary_score_delta"] = np.nan
        merged["case_count_match"] = False
        merged["meets_gate"] = False
    return merged


def evaluate_release_thresholds(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    variant: str = "baseline",
    thresholds: BenchmarkAcceptanceThresholds = DEFAULT_ACCEPTANCE_THRESHOLDS,
) -> dict:
    row = summary_df[summary_df["variant"] == variant]
    if row.empty:
        return {"variant": variant, "meets_thresholds": False, "reason": f"Variant '{variant}' not found."}

    row = row.iloc[0]
    primary_details = detail_df[(detail_df["variant"] == variant) & (detail_df["primary"])]
    min_case_f1 = float(primary_details["F1"].min()) if len(primary_details) else 0.0
    min_case_precision = float(primary_details["precision"].min()) if len(primary_details) else 0.0

    checks = {
        "mean_precision_ok": float(row["mean_precision"]) >= thresholds.min_mean_precision,
        "mean_recall_ok": float(row["mean_recall"]) >= thresholds.min_mean_recall,
        "mean_f1_ok": float(row["mean_F1"]) >= thresholds.min_mean_f1,
        "mean_fdr_ok": float(row["mean_FDR"]) <= thresholds.max_mean_fdr,
        "case_f1_ok": min_case_f1 >= thresholds.min_case_f1,
        "case_precision_ok": min_case_precision >= thresholds.min_case_precision,
    }
    return {
        "variant": variant,
        "meets_thresholds": bool(all(checks.values())),
        "checks": checks,
        "min_case_f1": round(min_case_f1, 3),
        "min_case_precision": round(min_case_precision, 3),
    }


def recommend_default_feature_flags(
    summary_df: pd.DataFrame,
    min_primary_gain: float = 0.0025,
    precision_tolerance: float = 0.0,
    fdr_tolerance: float = 0.01,
) -> dict:
    if summary_df.empty:
        return {
            "rfe_enabled": False,
            "shrinkage_awareness": False,
            "reason": "No benchmark summary available.",
        }

    def _best(df: pd.DataFrame) -> pd.Series | None:
        if df.empty:
            return None
        order = df.sort_values(["primary_score", "mean_precision", "mean_F1"], ascending=False)
        return order.iloc[0]

    def _feature_decision(flag_col: str) -> dict:
        best_with = _best(summary_df[summary_df[flag_col]])
        best_without = _best(summary_df[~summary_df[flag_col]])
        if best_with is None or best_without is None:
            return {
                "enabled": False,
                "best_variant": None,
                "score_delta": 0.0,
                "reason": f"No comparison variants available for {flag_col}.",
            }

        score_delta = float(best_with["primary_score"] - best_without["primary_score"])
        precision_ok = float(best_with["mean_precision"]) >= float(best_without["mean_precision"]) - precision_tolerance
        fdr_ok = float(best_with["mean_FDR"]) <= float(best_without["mean_FDR"]) + fdr_tolerance
        enabled = score_delta > min_primary_gain and precision_ok and fdr_ok
        if enabled:
            reason = (
                f"{flag_col} improves primary_score by {score_delta:.4f} "
                f"without hurting precision/FDR."
            )
        else:
            reason = (
                f"{flag_col} does not beat the fixed public benchmark baseline "
                f"(delta={score_delta:.4f})."
            )
        return {
            "enabled": bool(enabled),
            "best_variant": str(best_with["variant"]),
            "score_delta": round(score_delta, 4),
            "reason": reason,
        }

    rfe_decision = _feature_decision("rfe_enabled")
    shrinkage_decision = _feature_decision("shrinkage_awareness")
    return {
        "rfe_enabled": rfe_decision["enabled"],
        "shrinkage_awareness": shrinkage_decision["enabled"],
        "rfe": rfe_decision,
        "shrinkage": shrinkage_decision,
    }
