"""Read-only access to the reference package-importance results."""

from __future__ import annotations

import csv
import gzip
import io
from importlib.resources import files
from typing import TypedDict


class ImportanceRecord(TypedDict):
    """One reference importance result."""

    crate_name: str
    importance: float


def _resource_bytes() -> bytes:
    resource = files("package_risk_analysis").joinpath("resources").joinpath(
        "crate_importance_reference.csv.gz"
    )
    return resource.read_bytes()


def get_reference_scores_csv() -> str:
    """Return all reference scores as UTF-8 CSV response content."""
    return gzip.decompress(_resource_bytes()).decode("utf-8")


def get_reference_scores(limit: int | None = None) -> list[ImportanceRecord]:
    """Return reference scores ordered as stored in the experiment result.

    Args:
        limit: Optional maximum number of records. ``None`` returns every record.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    if limit == 0:
        return []

    records: list[ImportanceRecord] = []
    reader = csv.DictReader(io.StringIO(get_reference_scores_csv()))
    for row in reader:
        records.append(
            {
                "crate_name": row["crate_name"],
                "importance": float(row["importance"]),
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def get_crate_importance(crate_name: str) -> ImportanceRecord | None:
    """Return the case-insensitive reference result for one crate, if present."""
    normalized = crate_name.strip().casefold()
    if not normalized:
        raise ValueError("crate_name must not be empty")
    for record in get_reference_scores():
        if record["crate_name"].casefold() == normalized:
            return record
    return None
