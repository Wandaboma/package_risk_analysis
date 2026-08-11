"""Read-only access to the package hazard-score results."""

from __future__ import annotations

import csv
import gzip
import io
from importlib.resources import files
from typing import TypedDict


class HazardScoreRecord(TypedDict):
    """One package hazard-score result."""

    crate_name: str
    score: float


def _resource_bytes() -> bytes:
    resource = files("package_risk_analysis").joinpath("resources").joinpath(
        "crate_hazard_score.csv.gz"
    )
    return resource.read_bytes()


def get_hazard_scores_csv() -> str:
    """Return all hazard scores as UTF-8 CSV response content."""
    return gzip.decompress(_resource_bytes()).decode("utf-8")


def get_hazard_scores(limit: int | None = None) -> list[HazardScoreRecord]:
    """Return hazard scores ordered as stored in the experiment result.

    Args:
        limit: Optional maximum number of records. ``None`` returns every record.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    if limit == 0:
        return []

    records: list[HazardScoreRecord] = []
    reader = csv.DictReader(io.StringIO(get_hazard_scores_csv()))
    for row in reader:
        records.append(
            {
                "crate_name": row["crate_name"],
                "score": float(row["score"]),
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def get_crate_hazard_score(crate_name: str) -> HazardScoreRecord | None:
    """Return the case-insensitive hazard score for one crate, if present."""
    normalized = crate_name.strip().casefold()
    if not normalized:
        raise ValueError("crate_name must not be empty")
    for record in get_hazard_scores():
        if record["crate_name"].casefold() == normalized:
            return record
    return None
