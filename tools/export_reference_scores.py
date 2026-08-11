#!/usr/bin/env python3
"""Export the two-column importance reference dataset and packaged gzip resource."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path


def export_reference_scores(source: Path, csv_output: Path, gzip_output: Path) -> int:
    """Export crate names and importance values, returning the record count."""
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
        importance_column = (
            "importance"
            if "importance" in reader.fieldnames
            else "importance_with_download_portion"
        )
        missing = {name_column, importance_column} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"source CSV is missing columns: {sorted(missing)}")

        writer = csv.DictWriter(output_file, fieldnames=["crate_name", "importance"])
        writer.writeheader()
        for row in reader:
            name = row[name_column].strip()
            importance_text = row[importance_column].strip()
            if not name or not importance_text:
                continue
            importance = float(importance_text)
            writer.writerow({"crate_name": name, "importance": format(importance, ".17g")})
            count += 1
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
    args = parser.parse_args()
    count = export_reference_scores(
        args.source.resolve(),
        args.csv_output.resolve(),
        args.gzip_output.resolve(),
    )
    print(f"Exported {count} reference records")


if __name__ == "__main__":
    main()
