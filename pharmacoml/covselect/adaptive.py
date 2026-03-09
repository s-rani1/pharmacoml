"""Adaptive penalized screeners for correlated covariates."""
from __future__ import annotations

from pharmacoml.covselect.penalized import PenalizedScreener


class AdaptiveLassoScreener(PenalizedScreener):
    """Adaptive LASSO confirmation stage."""

    def __init__(self, **kwargs):
        kwargs.setdefault("method", "adaptive_lasso")
        super().__init__(**kwargs)


class AALassoScreener(PenalizedScreener):
    """Augmented adaptive LASSO confirmation stage."""

    def __init__(self, **kwargs):
        kwargs.setdefault("method", "aalasso")
        super().__init__(**kwargs)


__all__ = ["AdaptiveLassoScreener", "AALassoScreener"]
