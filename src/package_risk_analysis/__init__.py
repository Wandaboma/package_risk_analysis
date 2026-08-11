"""Public API for the Rust package-risk analysis."""

from .api import get_crate_hazard_score, get_hazard_scores, get_hazard_scores_csv

__all__ = [
    "get_crate_hazard_score",
    "get_hazard_scores",
    "get_hazard_scores_csv",
]

__version__ = "0.3.1"
