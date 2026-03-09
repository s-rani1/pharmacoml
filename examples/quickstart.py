"""
pharmacoml Quick Start Example
================================
Demonstrates ML-based covariate screening on simulated pop PK data.

True model:
    CL = 5 * (WT/70)^0.75 * exp(eta_CL)
    V  = 50 * (WT/70)^1.0  * exp(eta_V)

Expected result: WT identified as significant on both CL and V.
AGE and SEX should be flagged as non-significant.
"""

import numpy as np
import pandas as pd
from pharmacoml.covselect import CovariateScreener

# --- Simulate data ---
np.random.seed(42)
n = 500

wt = np.random.normal(70, 15, n).clip(40, 130)
age = np.random.normal(55, 12, n).clip(18, 90)
sex = np.random.binomial(1, 0.5, n)
crcl = 50 + 0.5 * wt + np.random.normal(0, 10, n)

cl = 5 * (wt / 70) ** 0.75 * np.exp(np.random.normal(0, 0.3, n))
v = 50 * (wt / 70) ** 1.0 * np.exp(np.random.normal(0, 0.2, n))

ebes = pd.DataFrame({"CL": cl, "V": v})
covs = pd.DataFrame({"WT": wt, "AGE": age, "SEX": sex, "CRCL": crcl})

# --- Run screening ---
screener = CovariateScreener(method="xgboost", n_bootstrap=50, random_state=42)
results = screener.fit(ebes, covs)

# --- Inspect results ---
print("=" * 60)
print("FULL SUMMARY")
print("=" * 60)
print(results.summary().to_string(index=False))

print("\n" + "=" * 60)
print("SIGNIFICANT COVARIATES ONLY")
print("=" * 60)
print(results.significant_covariates().to_string(index=False))

print("\n" + "=" * 60)
print("NONMEM CODE SNIPPET")
print("=" * 60)
results.to_nonmem()

# --- Uncomment for plots (requires display) ---
# results.plot_importance()
# results.plot_dependence("CL", "WT")
