#!/usr/bin/env python3
"""Compatibility entry point for recent crates.io version-download retrieval.

Prefer the installed command documented in README.md:
    package-risk-data --data-dir data downloads --days 90
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from package_risk_analysis.data import ensure_version_downloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_version_downloads(args.data_dir.resolve(), args.days, args.end_date, args.force)


if __name__ == "__main__":
    main()
