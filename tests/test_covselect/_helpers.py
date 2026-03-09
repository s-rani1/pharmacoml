"""Shared test helpers for optional backends and public benchmark assets."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _module_name_for_method(method: str) -> str | None:
    mapping = {
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "tabnet": "pytorch_tabnet",
        "mlp": "torch",
        "stg": "torch",
    }
    return mapping.get(method)


@lru_cache(maxsize=None)
def method_available(method: str) -> bool:
    module_name = _module_name_for_method(method)
    if module_name is None:
        return True
    if find_spec(module_name) is None:
        return False
    if method != "xgboost":
        return True

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    code = """
import numpy as np
from xgboost import XGBRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(32, 3))
y = rng.normal(size=32)
model = XGBRegressor(
    n_estimators=4,
    max_depth=2,
    learning_rate=0.3,
    subsample=1.0,
    colsample_bytree=1.0,
    verbosity=0,
    random_state=0,
)
model.fit(X, y)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        check=False,
    )
    return result.returncode == 0


def require_method(method: str) -> None:
    if not method_available(method):
        pytest.skip(f"Optional backend '{method}' is unavailable or unstable in this environment.")


def available_methods(methods: list[str] | tuple[str, ...]) -> list[str]:
    return [method for method in methods if method_available(method)]


def stable_tree_method() -> str:
    return "xgboost" if method_available("xgboost") else "random_forest"


def pheno_case_available() -> bool:
    try:
        from pharmacoml.covselect.public_benchmarks import _resolve_pharmpy_example_base

        base = _resolve_pharmpy_example_base()
    except (FileNotFoundError, ImportError):
        return False
    return isinstance(base, Path) and (base / "pheno.dta").exists()


def load_pheno_case_or_skip():
    from pharmacoml.covselect.public_benchmarks import load_pheno_case

    try:
        return load_pheno_case()
    except (FileNotFoundError, ImportError) as exc:
        pytest.skip(f"pheno benchmark unavailable: {exc}")
