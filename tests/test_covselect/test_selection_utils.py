import numpy as np
import pandas as pd

from pharmacoml.covselect.ensemble import EnsembleResults
from pharmacoml.covselect.selection_utils import (
    association_matrix,
    benjamini_hochberg,
    build_interaction_terms,
)


def test_benjamini_hochberg_preserves_order_and_bounds():
    q = benjamini_hochberg([0.01, 0.04, 0.20, 0.03])
    assert len(q) == 4
    assert np.all(q >= 0)
    assert np.all(q <= 1)
    assert q[0] <= q[2]


def test_association_matrix_handles_mixed_types():
    df = pd.DataFrame(
        {
            "WT": [60, 65, 70, 80, 85, 90],
            "BBSA": [1.6, 1.7, 1.8, 2.0, 2.1, 2.2],
            "SMK": [0, 0, 1, 1, 0, 1],
            "COPD": [0, 0, 1, 1, 0, 1],
        }
    )
    assoc = association_matrix(df)
    assert assoc.loc["WT", "BBSA"] > 0.9
    assert assoc.loc["SMK", "COPD"] > 0.9
    assert assoc.loc["WT", "SMK"] >= 0.0


def test_build_interaction_terms_creates_product_and_xor():
    covs = pd.DataFrame(
        {
            "SMK": [0, 1, 0, 1],
            "COPD": [0, 0, 1, 1],
            "AGE": [40, 50, 60, 70],
        }
    )
    interactions, metadata = build_interaction_terms(
        covs,
        candidate_covariates=["SMK", "COPD", "AGE"],
        max_pairs=3,
    )

    assert "SMK__x__COPD" in interactions.columns
    assert "SMK__xor__COPD" in interactions.columns
    assert "AGE__x__SMK" in interactions.columns or "SMK__x__AGE" in interactions.columns
    xor_col = "SMK__xor__COPD"
    assert interactions[xor_col].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert metadata[xor_col]["operator"] == "xor"


def test_candidate_consensus_tier_survives_strict_core_threshold():
    method_a = pd.DataFrame(
        [
            {
                "parameter": "ETA2",
                "covariate": "BWT",
                "mean_importance": 0.7,
                "functional_form": "linear",
                "importance_rank": 1,
                "relative_importance": 1.0,
                "standardized_importance": 1.0,
                "stability_frequency": 0.8,
                "utility_delta_r2": 0.08,
                "ensemble_vote": True,
            }
        ]
    )
    method_b = pd.DataFrame(
        [
            {
                "parameter": "ETA2",
                "covariate": "BWT",
                "mean_importance": 0.6,
                "functional_form": "linear",
                "importance_rank": 1,
                "relative_importance": 1.0,
                "standardized_importance": 1.0,
                "stability_frequency": 0.8,
                "utility_delta_r2": 0.06,
                "ensemble_vote": True,
            }
        ]
    )
    method_c = pd.DataFrame(
        [
            {
                "parameter": "ETA2",
                "covariate": "BWT",
                "mean_importance": 0.5,
                "functional_form": "linear",
                "importance_rank": 1,
                "relative_importance": 1.0,
                "standardized_importance": 1.0,
                "stability_frequency": 0.8,
                "utility_delta_r2": 0.05,
                "ensemble_vote": True,
            }
        ]
    )

    results = EnsembleResults(
        method_results={},
        method_summaries={
            "xgboost": method_a,
            "random_forest": method_b,
            "lasso": method_c,
        },
        min_agreement=4,
        min_relative_importance=0.10,
        min_delta_r2=0.02,
        candidate_score_threshold=0.02,
    )

    consensus = results.consensus_summary().set_index("covariate")
    assert consensus.loc["BWT", "consensus"]
    assert consensus.loc["BWT", "consensus_tier"] == "candidate"
