#!/usr/bin/env python3
"""Create small schema fixtures from a full local experiment-data directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CSV_INPUTS = (
    "crates.csv",
    "versions.csv",
    "dependencies.csv",
    "version_downloads.csv",
    "crate_downloads.csv",
    "crates_with_stars.csv",
    "crate_function_conclude.csv",
    "deprecated_pairs.csv",
    "keywords.csv",
    "crates_keywords.csv",
)


def clean_cell(value: str) -> str:
    """Remove insignificant line-end whitespace from human-readable sample fields."""
    return "\n".join(line.rstrip() for line in value.splitlines())


def sample_csv(source: Path, destination: Path, count: int) -> None:
    with source.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.reader(input_file)
        header = [clean_cell(cell) for cell in next(reader)]
        rows = []
        for _ in range(count):
            try:
                rows.append([clean_cell(cell) for cell in next(reader)])
            except StopIteration:
                break
    with destination.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
        writer.writerows(rows)


def sample_jsonl(source: Path, destination: Path, count: int) -> None:
    with source.open("r", encoding="utf-8") as input_file, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        written = 0
        for line in input_file:
            if not line.strip():
                continue
            json.loads(line)
            output_file.write(line.rstrip("\r\n") + "\n")
            written += 1
            if written == count:
                break


def sample_json(source: Path, destination: Path, count: int) -> None:
    with source.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if isinstance(data, list):
        sample = data[:count]
    elif isinstance(data, dict):
        sample = dict(list(data.items())[:count])
    else:
        raise TypeError(f"expected a JSON array or object in {source}")
    destination.write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_samples(source_dir: Path, destination_dir: Path, count: int = 10) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for filename in CSV_INPUTS:
        source = source_dir / filename
        if source.is_file():
            sample_csv(source, destination_dir / filename, count)

    sample_json(source_dir / "project_list.json", destination_dir / "project_list.json", count)
    sample_jsonl(
        source_dir / "rust_advisories_stream.jsonl",
        destination_dir / "rust_advisories_stream.jsonl",
        count,
    )

    monthly_sources = sorted((source_dir / "monthly").glob("delta_*.json"))
    if not monthly_sources:
        raise FileNotFoundError(f"no monthly delta JSON found below {source_dir / 'monthly'}")
    monthly_destination = destination_dir / "monthly"
    monthly_destination.mkdir(parents=True, exist_ok=True)
    sample_json(monthly_sources[0], monthly_destination / monthly_sources[0].name, count)

    print(f"Wrote {count}-record fixtures to {destination_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="full experiment data directory")
    parser.add_argument("destination", type=Path, help="fixture output directory")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    make_samples(args.source.resolve(), args.destination.resolve(), args.count)


if __name__ == "__main__":
    main()
