#!/usr/bin/env python3
"""Export the first package hazard scores and their packaged gzip resource."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path


def export_hazard_scores(
    source: Path,
    csv_output: Path,
    gzip_output: Path,
    max_lines: int = 10_000,
) -> int:
    """Export a header and hazard-score rows, returning the data-record count."""
    if max_lines < 2:
        raise ValueError("max_lines must allow one header and at least one data record")
    record_limit = max_lines - 1
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    gzip_output.parent.mkdir(parents=True, exist_ok=True)

    temporary_csv = csv_output.with_name(csv_output.name + ".part")
    count = 0
    with source.open("r", encoding="utf-8", newline="") as input_file, temporary_csv.open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError(f"source CSV has no header: {source}")
        name_column = "crate_name"
        score_column = (
            "score"
            if "score" in reader.fieldnames
            else "importance_with_download_portion"
        )
        missing = {name_column, score_column} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"source CSV is missing columns: {sorted(missing)}")

        writer = csv.DictWriter(output_file, fieldnames=["crate_name", "score"])
        writer.writeheader()
        for row in reader:
            name = row[name_column].strip()
            score_text = row[score_column].strip()
            if not name or not score_text:
                continue
            float(score_text)
            writer.writerow({"crate_name": name, "score": score_text})
            count += 1
            if count >= record_limit:
                break
    temporary_csv.replace(csv_output)

    temporary_gzip = gzip_output.with_name(gzip_output.name + ".part")
    content = csv_output.read_bytes()
    with temporary_gzip.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
            compressed.write(content)
    temporary_gzip.replace(gzip_output)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("csv_output", type=Path)
    parser.add_argument("gzip_output", type=Path)
    parser.add_argument("--max-lines", type=int, default=10_000)
    args = parser.parse_args()
    count = export_hazard_scores(
        args.source.resolve(),
        args.csv_output.resolve(),
        args.gzip_output.resolve(),
        args.max_lines,
    )
    print(f"Exported {count} hazard-score records across {count + 1} CSV lines")


if __name__ == "__main__":
    main()
