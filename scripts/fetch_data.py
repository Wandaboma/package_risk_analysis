#!/usr/bin/env python3
"""Fetch all public data required by the package-risk analysis.

Existing non-empty files are retained by default. Functional summaries and embedding
vectors are derived model outputs and therefore are not downloaded by this script.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from package_risk_analysis.data import (  # noqa: E402
    build_project_list,
    csvs_to_sqlite,
    ensure_advisories,
    ensure_crates_dump,
    ensure_gharchive,
    ensure_version_downloads,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data")
    parser.add_argument("--days", type=int, default=90, help="download-history window")
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--advisories-since", default=None)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--sqlite", action="store_true", help="also create crates.sqlite3")
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    parser.add_argument("--gharchive-root", type=Path)
    parser.add_argument("--gharchive-start", type=date.fromisoformat)
    parser.add_argument("--gharchive-end", type=date.fromisoformat)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    if args.gharchive_root and not (args.gharchive_start and args.gharchive_end):
        parser.error("--gharchive-root requires --gharchive-start and --gharchive-end")
    if (args.gharchive_start or args.gharchive_end) and not args.gharchive_root:
        parser.error("GH Archive dates require --gharchive-root")

    ensure_crates_dump(data_dir, force=args.force)
    build_project_list(data_dir, force=args.force)
    ensure_version_downloads(
        data_dir,
        days=args.days,
        end=args.end_date,
        force=args.force,
    )
    ensure_advisories(
        data_dir,
        token_file=args.token_file,
        since=args.advisories_since,
        force=args.force,
    )

    if args.sqlite:
        csvs_to_sqlite(data_dir, data_dir / "crates.sqlite3", force=args.force)
    if args.gharchive_root:
        ensure_gharchive(
            args.gharchive_root.resolve(),
            args.gharchive_start,
            args.gharchive_end,
            force=args.force,
        )

    print("Public data preparation complete.")


if __name__ == "__main__":
    main()
