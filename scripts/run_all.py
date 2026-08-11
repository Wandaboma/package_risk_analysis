#!/usr/bin/env python3
"""Run the complete package-risk analysis pipeline in dependency order."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ALL_STAGES = ("criticality", "maintenance", "replaceability", "combine", "validate")


def require_files(paths: list[Path], stage: str) -> None:
    missing = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Cannot run {stage}; required inputs are missing:\n{formatted}")


def require_monthly_data(monthly_dir: Path, minimum: int = 24) -> None:
    files = sorted(monthly_dir.glob("delta_*.json"))
    if len(files) < minimum:
        raise FileNotFoundError(
            f"Maintenance requires at least {minimum} monthly activity files in "
            f"{monthly_dir}; found {len(files)}"
        )


def display_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def execute(command: list[str], dry_run: bool) -> None:
    print(f"\n$ {display_command(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT_DIR, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter")
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=list(ALL_STAGES))
    parser.add_argument("--fetch", action="store_true", help="fetch public data first")
    parser.add_argument("--days", type=int, default=90, help="download window with --fetch")
    parser.add_argument("--token-file", type=Path, help="GitHub token file with --fetch")
    parser.add_argument("--gharchive-root", type=Path, help="raw GH Archive input/output root")
    parser.add_argument("--gharchive-start", type=date.fromisoformat)
    parser.add_argument("--gharchive-end", type=date.fromisoformat)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--advisory-start-date", default="2025-11-01")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if (args.gharchive_start or args.gharchive_end) and not args.gharchive_root:
        parser.error("GH Archive dates require --gharchive-root")
    if args.gharchive_root and not (args.gharchive_start and args.gharchive_end):
        parser.error("--gharchive-root requires --gharchive-start and --gharchive-end")
    python = str(Path(args.python))
    data = ROOT_DIR / "data"
    result = ROOT_DIR / "result"
    result.mkdir(parents=True, exist_ok=True)

    criticality = result / "crate_importance_metric.csv"
    maintenance_dir = result / "maintenance_model"
    maintenance = maintenance_dir / "mamba_activity_prediction.csv"
    replacement = result / "crate_replacement_metric.csv"
    combined = result / "crate_combined_criticality.csv"

    if args.fetch:
        fetch_command = [python, str(ROOT_DIR / "scripts" / "fetch_data.py"), "--days", str(args.days)]
        if args.token_file:
            fetch_command.extend(["--token-file", str(args.token_file)])
        if args.gharchive_root:
            fetch_command.extend(
                [
                    "--gharchive-root", str(args.gharchive_root),
                    "--gharchive-start", args.gharchive_start.isoformat(),
                    "--gharchive-end", args.gharchive_end.isoformat(),
                ]
            )
        execute(fetch_command, args.dry_run)

    if args.gharchive_root and "maintenance" in args.stages:
        execute(
            [
                python,
                str(ROOT_DIR / "code" / "helper" / "gharchive_info_collect.py"),
                "all",
                "--input-root", str(args.gharchive_root),
                "--repo-list", str(data / "project_list.json"),
                "--parquet-root", str(data / "gha_parquet"),
                "--monthly-dir", str(data / "monthly"),
                "--start-month", args.gharchive_start.strftime("%Y-%m"),
                "--end-month", args.gharchive_end.strftime("%Y-%m"),
            ],
            args.dry_run,
        )

    for stage in args.stages:
        if stage == "criticality":
            inputs = [
                data / "crates.csv",
                data / "versions.csv",
                data / "dependencies.csv",
                data / "version_downloads.csv",
            ]
            if not args.dry_run:
                require_files(inputs, stage)
            execute(
                [
                    python,
                    str(ROOT_DIR / "code" / "structual_importance.py"),
                    "--crates", str(inputs[0]),
                    "--versions", str(inputs[1]),
                    "--deps", str(inputs[2]),
                    "--version-downloads", str(inputs[3]),
                    "--out-dir", str(result),
                    "--crate-name", "",
                ],
                args.dry_run,
            )

        elif stage == "maintenance":
            if not args.dry_run:
                require_files([data / "crates.csv"], stage)
                require_monthly_data(data / "monthly")
            execute(
                [
                    python,
                    str(ROOT_DIR / "code" / "advanced_maintenance_prediction.py"),
                    "--models", "Mamba",
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--output-dir", str(maintenance_dir),
                ],
                args.dry_run,
            )

        elif stage == "replaceability":
            inputs = [
                data / "crates.csv",
                data / "crate_function_conclude.csv",
                data / "deprecated_pairs.csv",
                data / "embeddings_cache.npz",
            ]
            if not args.dry_run:
                require_files(inputs, stage)
            execute([python, str(ROOT_DIR / "code" / "similarity_eval.py")], args.dry_run)

        elif stage == "combine":
            inputs = [criticality, maintenance, replacement]
            if not args.dry_run:
                require_files(inputs, stage)
            execute(
                [
                    python,
                    str(ROOT_DIR / "code" / "combine_metrics.py"),
                    "--importance", str(criticality),
                    "--activity", str(maintenance),
                    "--replacement", str(replacement),
                    "--combine-method", "both",
                    "--output", str(combined),
                ],
                args.dry_run,
            )

        elif stage == "validate":
            inputs = [
                data / "rust_advisories_stream.jsonl",
                data / "crates.csv",
                criticality,
                maintenance,
                replacement,
            ]
            if not args.dry_run:
                require_files(inputs, stage)
            execute(
                [
                    python,
                    str(ROOT_DIR / "code" / "validate_combined_metric_correlation.py"),
                    "--advisories", str(inputs[0]),
                    "--crates", str(inputs[1]),
                    "--importance", str(inputs[2]),
                    "--activity", str(inputs[3]),
                    "--replacement", str(inputs[4]),
                    "--start-date", args.advisory_start_date,
                    "--output", str(result / "combined_metric_validation.csv"),
                ],
                args.dry_run,
            )

    print("\nPipeline complete." if not args.dry_run else "\nDry run complete.")


if __name__ == "__main__":
    main()
