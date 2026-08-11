"""Acquire and prepare the public datasets used by the analysis.

The functions in this module are deliberately idempotent: an existing non-empty
output is retained unless ``force=True``. Network access only occurs for missing
inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CRATES_DUMP_URL = "https://static.crates.io/db-dump.tar.gz"
DOWNLOAD_ARCHIVE_URL = "https://static.crates.io/archive/version-downloads"
GHARCHIVE_URL = "https://data.gharchive.org"
GITHUB_ADVISORIES_URL = "https://api.github.com/advisories"
CORE_TABLES = ("crates", "versions", "dependencies")
USER_AGENT = "package-risk-analysis/0.2"
GITHUB_REPOSITORY_PATTERN = re.compile(
    r"(?:https?://|git\+https?://|git@)github\.com[/:]([^/]+)/([^/#]+)",
    re.IGNORECASE,
)


def _present(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _download(url: str, destination: Path, headers: dict[str, str] | None = None) -> None:
    """Download *url* atomically to *destination*."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    request = Request(url, headers=request_headers)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_crates_dump(data_dir: Path, force: bool = False) -> list[Path]:
    """Download the nightly crates.io dump and extract missing core CSV tables."""
    data_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: data_dir / f"{name}.csv" for name in CORE_TABLES}
    missing = {name for name, path in targets.items() if force or not _present(path)}
    if not missing:
        print("[skip] crates.io core CSV files are already present")
        return list(targets.values())

    archive = data_dir / "db-dump.tar.gz"
    if force or not _present(archive):
        print(f"[download] {CRATES_DUMP_URL}")
        _download(CRATES_DUMP_URL, archive)

    extracted: set[str] = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            filename = Path(member.name).name
            table = Path(filename).stem
            if table not in missing or filename != f"{table}.csv" or not member.isfile():
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            destination = targets[table]
            temporary = destination.with_name(destination.name + ".part")
            with source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            temporary.replace(destination)
            extracted.add(table)
            print(f"[extract] {member.name} -> {destination}")

    unresolved = missing - extracted
    if unresolved:
        raise RuntimeError(f"crates.io dump did not contain expected tables: {sorted(unresolved)}")
    return list(targets.values())


def build_project_list(data_dir: Path, force: bool = False) -> Path:
    """Derive the GH Archive repository target list from crates.csv."""
    source = data_dir / "crates.csv"
    destination = data_dir / "project_list.json"
    if _present(destination) and not force:
        print(f"[skip] repository list is already present: {destination}")
        return destination
    if not _present(source):
        raise FileNotFoundError(f"cannot build repository list; missing: {source}")

    repositories: set[str] = set()
    with source.open("r", encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            repository = (row.get("repository") or "").strip()
            match = GITHUB_REPOSITORY_PATTERN.search(repository)
            if not match:
                continue
            owner = match.group(1).strip()
            name = match.group(2).removesuffix(".git").strip()
            if owner and name:
                repositories.add(f"github:{owner}/{name}")

    temporary = destination.with_name(destination.name + ".part")
    temporary.write_text(
        json.dumps(sorted(repositories, key=str.casefold), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(f"[write] {len(repositories)} repositories to {destination}")
    return destination


def _date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def ensure_version_downloads(
    data_dir: Path,
    days: int = 90,
    end: date | None = None,
    force: bool = False,
) -> Path:
    """Download and combine the requested daily version-download archives."""
    if days < 1:
        raise ValueError("days must be positive")
    combined = data_dir / "version_downloads.csv"
    if _present(combined) and not force:
        print(f"[skip] combined download file is already present: {combined}")
        return combined
    end = end or (datetime.now(timezone.utc).date() - timedelta(days=1))
    start = end - timedelta(days=days - 1)
    daily_dir = data_dir / "version_downloads"
    daily_dir.mkdir(parents=True, exist_ok=True)

    for day in _date_range(start, end):
        destination = daily_dir / f"{day.isoformat()}.csv"
        if not force and _present(destination):
            continue
        url = f"{DOWNLOAD_ARCHIVE_URL}/{day.isoformat()}.csv"
        try:
            print(f"[download] {url}")
            _download(url, destination)
        except HTTPError as exc:
            if exc.code == 404:
                print(f"[missing] no download archive for {day.isoformat()}")
                continue
            raise

    temporary = combined.with_name(combined.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["date", "downloads", "version_id"])
        writer.writeheader()
        for day in _date_range(start, end):
            source = daily_dir / f"{day.isoformat()}.csv"
            if not _present(source):
                continue
            with source.open("r", encoding="utf-8", newline="") as input_file:
                for row in csv.DictReader(input_file):
                    writer.writerow(
                        {
                            "date": day.isoformat(),
                            "downloads": row["downloads"],
                            "version_id": row["version_id"],
                        }
                    )
    temporary.replace(combined)
    print(f"[write] {combined}")
    return combined


def read_token(token_file: Path | None = None) -> str | None:
    """Read a GitHub token from a file or the environment, never from argv."""
    if token_file is not None:
        token = token_file.expanduser().read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError(f"token file is empty: {token_file}")
        return token
    return os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")


def ensure_advisories(
    data_dir: Path,
    token_file: Path | None = None,
    since: str | None = None,
    force: bool = False,
) -> Path:
    """Download Rust GitHub Security Advisories as analysis-compatible JSONL."""
    destination = data_dir / "rust_advisories_stream.jsonl"
    if _present(destination) and not force:
        print(f"[skip] advisory data is already present: {destination}")
        return destination

    token = read_token(token_file)
    params = {
        "ecosystem": "rust",
        "per_page": 100,
        "sort": "published",
        "direction": "desc",
        "state": "published",
    }
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    since_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc) if since else None
    page = 1
    rows: list[dict[str, object]] = []

    while True:
        params["page"] = page
        url = f"{GITHUB_ADVISORIES_URL}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": USER_AGENT, **headers})
        with urlopen(request, timeout=120) as response:
            advisories = json.loads(response.read().decode("utf-8"))
        if not advisories:
            break
        reached_cutoff = False
        for advisory in advisories:
            published_text = advisory.get("published_at")
            published = datetime.fromisoformat(published_text.replace("Z", "+00:00"))
            if since_dt and published < since_dt:
                reached_cutoff = True
                continue
            cvss = advisory.get("cvss") or {}
            for vulnerability in advisory.get("vulnerabilities") or []:
                package = (vulnerability.get("package") or {}).get("name")
                if package:
                    rows.append(
                        {
                            "package": package,
                            "ghsaId": advisory.get("ghsa_id"),
                            "severity": advisory.get("severity"),
                            "publishedAt": published_text,
                            "cvss_score": cvss.get("score"),
                            "cvss_vector": cvss.get("vector_string"),
                        }
                    )
        if reached_cutoff or len(advisories) < 100:
            break
        page += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    print(f"[write] {len(rows)} advisory-package rows to {destination}")
    return destination


def ensure_gharchive(
    output_dir: Path,
    start: date,
    end: date,
    force: bool = False,
) -> list[Path]:
    """Download missing hourly GH Archive files for an explicit inclusive date range."""
    if end < start:
        raise ValueError("end date must not be earlier than start date")
    outputs: list[Path] = []
    for day in _date_range(start, end):
        year_dir = output_dir / str(day.year)
        for hour in range(24):
            filename = f"{day.isoformat()}-{hour}.json.gz"
            destination = year_dir / filename
            outputs.append(destination)
            if not force and _present(destination):
                continue
            print(f"[download] {filename}")
            _download(f"{GHARCHIVE_URL}/{filename}", destination)
    return outputs


def csvs_to_sqlite(data_dir: Path, database: Path, force: bool = False) -> Path:
    """Import the raw core CSVs into SQLite for transparent SQL-based inspection."""
    if _present(database) and not force:
        print(f"[skip] SQLite database is already present: {database}")
        return database
    missing = [data_dir / f"{table}.csv" for table in CORE_TABLES if not _present(data_dir / f"{table}.csv")]
    if missing:
        raise FileNotFoundError(f"cannot build SQLite database; missing: {missing}")

    database.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=database.parent, suffix=".sqlite3", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with sqlite3.connect(temporary) as connection:
            for table in CORE_TABLES:
                source = data_dir / f"{table}.csv"
                with source.open("r", encoding="utf-8", newline="") as input_file:
                    reader = csv.reader(input_file)
                    columns = next(reader)
                    quoted = ", ".join(f'"{column}" TEXT' for column in columns)
                    connection.execute(f'CREATE TABLE "{table}" ({quoted})')
                    placeholders = ", ".join("?" for _ in columns)
                    connection.executemany(
                        f'INSERT INTO "{table}" VALUES ({placeholders})',
                        reader,
                    )
                print(f"[import] {source} -> {table}")
            connection.execute('CREATE INDEX crates_id_idx ON crates (id)')
            connection.execute('CREATE INDEX versions_id_idx ON versions (id)')
            connection.execute('CREATE INDEX versions_crate_idx ON versions (crate_id)')
            connection.execute('CREATE INDEX dependencies_version_idx ON dependencies (version_id)')
            connection.commit()
        temporary.replace(database)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[write] {database}")
    return database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true", help="replace existing generated inputs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("crates", help="obtain missing crates.io core dump CSVs")
    downloads = subparsers.add_parser("downloads", help="obtain recent version downloads")
    downloads.add_argument("--days", type=int, default=90)
    downloads.add_argument("--end-date", type=date.fromisoformat)
    advisories = subparsers.add_parser("advisories", help="obtain missing Rust advisories")
    advisories.add_argument("--since")
    advisories.add_argument("--token-file", type=Path)
    gharchive = subparsers.add_parser("gharchive", help="obtain missing hourly GH Archive files")
    gharchive.add_argument("--start-date", type=date.fromisoformat, required=True)
    gharchive.add_argument("--end-date", type=date.fromisoformat, required=True)
    gharchive.add_argument("--output-dir", type=Path, required=True)
    sqlite_parser = subparsers.add_parser("sqlite", help="convert core CSVs to SQLite")
    sqlite_parser.add_argument("--output", type=Path, default=Path("data/crates.sqlite3"))
    all_parser = subparsers.add_parser("all", help="obtain all public inputs used by core analysis")
    all_parser.add_argument("--days", type=int, default=90)
    all_parser.add_argument("--end-date", type=date.fromisoformat)
    all_parser.add_argument("--since")
    all_parser.add_argument("--token-file", type=Path)
    all_parser.add_argument("--sqlite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data_dir.resolve()
    if args.command in {"crates", "all"}:
        ensure_crates_dump(data_dir, args.force)
        build_project_list(data_dir, args.force)
    if args.command in {"downloads", "all"}:
        ensure_version_downloads(data_dir, args.days, args.end_date, args.force)
    if args.command in {"advisories", "all"}:
        ensure_advisories(data_dir, args.token_file, args.since, args.force)
    if args.command == "gharchive":
        ensure_gharchive(args.output_dir.resolve(), args.start_date, args.end_date, args.force)
    if args.command == "sqlite" or (args.command == "all" and args.sqlite):
        output = args.output if args.command == "sqlite" else data_dir / "crates.sqlite3"
        csvs_to_sqlite(data_dir, output.resolve(), args.force)


if __name__ == "__main__":
    main()
