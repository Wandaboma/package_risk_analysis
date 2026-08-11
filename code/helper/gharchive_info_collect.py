#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GH Archive -> filtered Parquet -> monthly maintenance JSON pipeline.

Designed for a large local GH Archive and a target list of ~100k repositories.
The expensive gzip/JSON scan is performed only once. Later feature revisions can
re-run the fast Parquet aggregation stage without reading the raw archive.

Input layout
------------
GHARCHIVE_ROOT/
  2019/2019-01-01-0.json.gz
  2019/2019-01-01-1.json.gz
  ...

Parquet layout
--------------
PARQUET_ROOT/
  year=2019/month=01/2019-01-01-0.parquet
  year=2019/month=01/2019-01-01-1.parquet
  ...

Monthly output
--------------
MONTHLY_ROOT/
  delta_2019_01.json
  delta_2019_02.json
  ...

Dependencies
------------
  python -m pip install pyarrow orjson tqdm

Examples
--------
Run both stages:
  python code/helper/gharchive_info_collect.py all ^
    --input-root "D:\\gharchive" ^
    --repo-list "D:\\project\\data\\project_list.json" ^
    --parquet-root "D:\\project\\data\\gha_parquet" ^
    --monthly-dir "D:\\project\\data\\monthly" ^
    --start-month 2024-01 --end-month 2025-12 --workers 6

Only filter raw GH Archive to Parquet:
  python code/helper/gharchive_info_collect.py extract ...

Rebuild monthly JSON after changing feature logic:
  python code/helper/gharchive_info_collect.py aggregate ^
    --repo-list project_list.json ^
    --parquet-root gha_parquet ^
    --monthly-dir monthly ^
    --start-month 2024-01 --end-month 2025-12

Notes
-----
* Modern GitHub Events API / GH Archive records are expected.
* Repository matching is case-insensitive and uses github:owner/repo keys.
* Each raw hourly file has a deterministic Parquet output and an SQLite
  checkpoint. Changed source files are automatically reprocessed.
* Parquet files are written atomically.
* A failed/corrupt gzip is recorded and other files continue.
* The aggregation output is sparse by default. Pass --zero-fill to include all
  target repositories in every month. Sparse output is much smaller; the model
  loader must treat a missing repo-month as zeros, not forward-fill it.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import gzip
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

try:
    import orjson
except ImportError as exc:
    raise SystemExit("Missing dependency 'orjson'. Run: python -m pip install orjson") from exc

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit("Missing dependency 'pyarrow'. Run: python -m pip install pyarrow") from exc

try:
    from tqdm.auto import tqdm
except ImportError as exc:
    raise SystemExit("Missing dependency 'tqdm'. Run: python -m pip install tqdm") from exc


SCRIPT_VERSION = "1.0.0"
CHECKPOINT_SCHEMA_VERSION = 1
MIN_SUPPORTED_YEAR = 2015
FILE_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-(?P<hour>\d{1,2})\.json\.gz$"
)
MONTH_RE = re.compile(r"^(?P<year>\d{4})[-_](?P<month>\d{2})$")
GITHUB_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com[/:](?P<owner>[^/#?\s]+)/(?P<repo>[^/#?\s]+)",
    re.IGNORECASE,
)

# Worker-global target set. Initialized once per process.
_TARGET_REPOS: set[str] = set()


PARQUET_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string()),
        pa.field("created_at", pa.timestamp("ms", tz="UTC")),
        pa.field("repo_key", pa.string()),
        pa.field("repo_name", pa.string()),
        pa.field("actor_login", pa.string()),
        pa.field("actor_type", pa.string()),
        pa.field("event_type", pa.string()),
        pa.field("action", pa.string()),
        pa.field("author_association", pa.string()),
        pa.field("push_size", pa.int32()),
        pa.field("push_distinct_size", pa.int32()),
        pa.field("issue_number", pa.int64()),
        pa.field("issue_is_pull_request", pa.bool_()),
        pa.field("issue_created_at", pa.timestamp("ms", tz="UTC")),
        pa.field("issue_closed_at", pa.timestamp("ms", tz="UTC")),
        pa.field("pull_number", pa.int64()),
        pa.field("pull_created_at", pa.timestamp("ms", tz="UTC")),
        pa.field("pull_closed_at", pa.timestamp("ms", tz="UTC")),
        pa.field("pull_merged_at", pa.timestamp("ms", tz="UTC")),
        pa.field("pull_merged", pa.bool_()),
        pa.field("review_state", pa.string()),
        pa.field("release_published_at", pa.timestamp("ms", tz="UTC")),
        pa.field("ref_type", pa.string()),
        pa.field("title_chars", pa.int32()),
        pa.field("body_chars", pa.int32()),
        pa.field("comment_chars", pa.int32()),
    ]
)

PARQUET_COLUMNS = [field.name for field in PARQUET_SCHEMA]


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def raise_csv_field_limit() -> int:
    limit = sys.maxsize
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit //= 10
    raise RuntimeError("Unable to increase csv.field_size_limit")


def normalize_repo(value: Any) -> str:
    """Normalize URL, owner/repo, or github:owner/repo to a lowercase key."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""

    text = text.strip("\"'")
    if text.lower().startswith("github:"):
        text = text.split(":", 1)[1]
    else:
        match = GITHUB_URL_RE.search(text)
        if match:
            text = f"{match.group('owner')}/{match.group('repo')}"

    text = text.replace("\\", "/").strip("/")
    if text.lower().endswith(".git"):
        text = text[:-4]
    text = text.split("?", 1)[0].split("#", 1)[0].strip("/")
    parts = text.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return ""
    owner, repo = parts[0], parts[1]
    return f"github:{owner.lower()}/{repo.lower()}"


def load_repo_list(path: Path) -> set[str]:
    """Load repositories from JSON, TXT, or crates-style CSV."""
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    repos: set[str] = set()

    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            obj = json.load(handle)
        if isinstance(obj, list):
            values = obj
        elif isinstance(obj, dict):
            # Accept {repo: ...}, {"repositories": [...]}, or mapping output.
            if isinstance(obj.get("repositories"), list):
                values = obj["repositories"]
            else:
                values = obj.keys()
        else:
            raise ValueError("JSON repo list must be an array or object")
        for value in values:
            key = normalize_repo(value)
            if key:
                repos.add(key)

    elif suffix == ".csv":
        raise_csv_field_limit()
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return repos
            fields = {name.lower(): name for name in reader.fieldnames}
            candidates = [
                fields.get("repo_key"),
                fields.get("repository"),
                fields.get("repo"),
                fields.get("repo_name"),
                fields.get("homepage"),
            ]
            candidates = [c for c in candidates if c]
            if not candidates:
                raise ValueError(
                    "CSV needs one of: repo_key, repository, repo, repo_name, homepage"
                )
            for row in reader:
                for column in candidates:
                    key = normalize_repo(row.get(column))
                    if key:
                        repos.add(key)
                        break
    else:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                key = normalize_repo(line)
                if key:
                    repos.add(key)

    if not repos:
        raise ValueError(f"No valid GitHub repositories found in {path}")
    return repos


def parse_month(text: str) -> tuple[int, int]:
    match = MONTH_RE.match(text.strip())
    if not match:
        raise argparse.ArgumentTypeError("month must be YYYY-MM")
    year, month = int(match.group("year")), int(match.group("month"))
    if year < MIN_SUPPORTED_YEAR or not 1 <= month <= 12:
        raise argparse.ArgumentTypeError(
            f"month must be between {MIN_SUPPORTED_YEAR}-01 and 9999-12"
        )
    return year, month


def month_key(year: int, month: int) -> str:
    return f"{year:04d}_{month:02d}"


def month_dash(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def iter_months(start: tuple[int, int], end: tuple[int, int]) -> Iterator[tuple[int, int]]:
    y, m = start
    while (y, m) <= end:
        yield y, m
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1


def infer_available_months(input_root: Path) -> list[tuple[int, int]]:
    months: set[tuple[int, int]] = set()
    year_dirs = sorted(p for p in input_root.iterdir() if p.is_dir() and p.name.isdigit())
    for year_dir in tqdm(year_dirs, desc="Scanning year folders", unit="year"):
        year = int(year_dir.name)
        if year < MIN_SUPPORTED_YEAR:
            continue
        for path in year_dir.glob("*.json.gz"):
            match = FILE_RE.match(path.name)
            if match:
                months.add((int(match.group("year")), int(match.group("month"))))
    return sorted(months)


def atomic_json_dump(path: Path, data: Any, *, indent: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=indent, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def chars(value: Any) -> int:
    return len(value) if isinstance(value, str) else 0


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Checkpoint database
# ---------------------------------------------------------------------------

class CheckpointDB:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS source_files (
                source_path TEXT PRIMARY KEY,
                source_size INTEGER NOT NULL,
                source_mtime_ns INTEGER NOT NULL,
                output_path TEXT,
                status TEXT NOT NULL,
                total_events INTEGER NOT NULL DEFAULT 0,
                matched_events INTEGER NOT NULL DEFAULT 0,
                malformed_lines INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
            (str(CHECKPOINT_SCHEMA_VERSION),),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def is_current(self, source: Path, output: Path) -> bool:
        stat = source.stat()
        row = self.conn.execute(
            """
            SELECT source_size, source_mtime_ns, output_path, status
            FROM source_files WHERE source_path = ?
            """,
            (str(source.resolve()),),
        ).fetchone()
        if not row:
            return False
        size, mtime_ns, output_path, status = row
        if status != "done" or size != stat.st_size or mtime_ns != stat.st_mtime_ns:
            return False
        # Empty files intentionally have output_path=NULL.
        return (output_path is None) or output.exists()

    def mark_result(self, result: "ExtractResult") -> None:
        source = Path(result.source_path)
        stat = source.stat()
        self.conn.execute(
            """
            INSERT INTO source_files(
                source_path, source_size, source_mtime_ns, output_path, status,
                total_events, matched_events, malformed_lines, error, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                source_size=excluded.source_size,
                source_mtime_ns=excluded.source_mtime_ns,
                output_path=excluded.output_path,
                status=excluded.status,
                total_events=excluded.total_events,
                matched_events=excluded.matched_events,
                malformed_lines=excluded.malformed_lines,
                error=excluded.error,
                updated_at=excluded.updated_at
            """,
            (
                str(source.resolve()),
                stat.st_size,
                stat.st_mtime_ns,
                result.output_path,
                "done" if result.error is None else "failed",
                result.total_events,
                result.matched_events,
                result.malformed_lines,
                result.error,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()


# ---------------------------------------------------------------------------
# Raw event -> flat Parquet row
# ---------------------------------------------------------------------------

def _worker_init(repo_values: Sequence[str]) -> None:
    global _TARGET_REPOS
    _TARGET_REPOS = set(repo_values)


def association_from(event: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    candidates = [
        payload.get("author_association"),
        (payload.get("issue") or {}).get("author_association") if isinstance(payload.get("issue"), dict) else None,
        (payload.get("pull_request") or {}).get("author_association") if isinstance(payload.get("pull_request"), dict) else None,
        (payload.get("comment") or {}).get("author_association") if isinstance(payload.get("comment"), dict) else None,
        (payload.get("review") or {}).get("author_association") if isinstance(payload.get("review"), dict) else None,
    ]
    for value in candidates:
        if value:
            return str(value).upper()
    return ""


def event_to_row(event: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    repo_obj = event.get("repo")
    if not isinstance(repo_obj, dict):
        return None
    repo_name = repo_obj.get("name")
    if not repo_name:
        return None
    repo_key = normalize_repo(repo_name)
    if not repo_key or repo_key not in _TARGET_REPOS:
        return None

    payload = event.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    actor = event.get("actor")
    if not isinstance(actor, dict):
        actor = {}

    issue = payload.get("issue") if isinstance(payload.get("issue"), dict) else {}
    pull = payload.get("pull_request") if isinstance(payload.get("pull_request"), dict) else {}
    review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
    release = payload.get("release") if isinstance(payload.get("release"), dict) else {}
    comment = payload.get("comment") if isinstance(payload.get("comment"), dict) else {}

    issue_is_pr = bool(issue.get("pull_request"))
    title_value = pull.get("title") or issue.get("title") or ""
    body_value = pull.get("body") or issue.get("body") or review.get("body") or ""
    comment_value = comment.get("body") or ""

    merged = pull.get("merged")
    if merged is None:
        merged = bool(pull.get("merged_at"))

    return {
        "event_id": str(event.get("id") or ""),
        "created_at": parse_dt(event.get("created_at")),
        "repo_key": repo_key,
        "repo_name": str(repo_name).lower(),
        "actor_login": str(actor.get("login") or "").lower(),
        "actor_type": str(actor.get("type") or ""),
        "event_type": str(event.get("type") or ""),
        "action": str(payload.get("action") or ""),
        "author_association": association_from(event, payload),
        "push_size": safe_int(payload.get("size") or len(payload.get("commits") or [])),
        "push_distinct_size": safe_int(payload.get("distinct_size")),
        "issue_number": safe_int(issue.get("number")) or None,
        "issue_is_pull_request": issue_is_pr,
        "issue_created_at": parse_dt(issue.get("created_at")),
        "issue_closed_at": parse_dt(issue.get("closed_at")),
        "pull_number": safe_int(payload.get("number") or pull.get("number")) or None,
        "pull_created_at": parse_dt(pull.get("created_at")),
        "pull_closed_at": parse_dt(pull.get("closed_at")),
        "pull_merged_at": parse_dt(pull.get("merged_at")),
        "pull_merged": bool(merged),
        "review_state": str(review.get("state") or "").lower(),
        "release_published_at": parse_dt(release.get("published_at") or release.get("created_at")),
        "ref_type": str(payload.get("ref_type") or "").lower(),
        "title_chars": chars(title_value),
        "body_chars": chars(body_value),
        "comment_chars": chars(comment_value),
    }


@dataclass
class ExtractResult:
    source_path: str
    output_path: Optional[str]
    total_events: int
    matched_events: int
    malformed_lines: int
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


def extract_one_file(task: tuple[str, str, int]) -> ExtractResult:
    """Worker function: scan one gzip and atomically write one Parquet file."""
    source_text, output_text, row_group_size = task
    source = Path(source_text)
    output = Path(output_text)
    started = time.perf_counter()
    total = matched = malformed = 0
    rows: list[dict[str, Any]] = []

    try:
        with gzip.open(source, "rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                try:
                    event = orjson.loads(line)
                except orjson.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(event, dict):
                    malformed += 1
                    continue
                row = event_to_row(event)
                if row is not None:
                    rows.append(row)
                    matched += 1

        # A zero-match source is represented only in SQLite; no empty Parquet.
        if not rows:
            if output.exists():
                output.unlink()
            return ExtractResult(
                source_path=str(source), output_path=None,
                total_events=total, matched_events=0,
                malformed_lines=malformed,
                elapsed_seconds=time.perf_counter() - started,
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
        tmp = output.with_name(output.name + f".{os.getpid()}.tmp")
        pq.write_table(
            table,
            tmp,
            compression="zstd",
            compression_level=3,
            use_dictionary=["repo_key", "repo_name", "actor_login", "event_type", "action", "author_association"],
            row_group_size=row_group_size,
            write_statistics=True,
        )
        os.replace(tmp, output)
        return ExtractResult(
            source_path=str(source), output_path=str(output),
            total_events=total, matched_events=matched,
            malformed_lines=malformed,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return ExtractResult(
            source_path=str(source), output_path=None,
            total_events=total, matched_events=matched,
            malformed_lines=malformed,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - started,
        )


def source_files_for_month(input_root: Path, year: int, month: int) -> list[Path]:
    folder = input_root / str(year)
    if not folder.exists():
        return []
    result: list[Path] = []
    for path in folder.glob(f"{year:04d}-{month:02d}-*.json.gz"):
        match = FILE_RE.match(path.name)
        if match and int(match.group("year")) == year and int(match.group("month")) == month:
            result.append(path)
    return sorted(result)


def parquet_output_for(parquet_root: Path, source: Path) -> Path:
    match = FILE_RE.match(source.name)
    if not match:
        raise ValueError(f"Unexpected GH Archive filename: {source.name}")
    year, month = match.group("year"), match.group("month")
    stem = source.name[:-len(".json.gz")]
    return parquet_root / f"year={year}" / f"month={month}" / f"{stem}.parquet"


def run_extract(args: argparse.Namespace, repos: set[str], months: list[tuple[int, int]]) -> None:
    args.parquet_root.mkdir(parents=True, exist_ok=True)
    state_path = args.state_db or (args.parquet_root / ".gharchive_parquet_state.sqlite")
    db = CheckpointDB(state_path)

    try:
        all_sources: list[Path] = []
        for year, month in months:
            all_sources.extend(source_files_for_month(args.input_root, year, month))

        if not all_sources:
            raise RuntimeError("No GH Archive .json.gz files found for the selected months")

        tasks: list[tuple[str, str, int]] = []
        skipped = 0
        for source in all_sources:
            output = parquet_output_for(args.parquet_root, source)
            if not args.force and db.is_current(source, output):
                skipped += 1
                continue
            tasks.append((str(source), str(output), args.row_group_size))

        logging.info(
            "Extraction selection: %d files total, %d current/skipped, %d to process",
            len(all_sources), skipped, len(tasks),
        )
        logging.info("Target repository set: %,d", len(repos))

        if not tasks:
            logging.info("All selected source files are already current.")
            return

        workers = max(1, args.workers)
        processed = failed = total_events = matched_events = malformed = 0
        start = time.perf_counter()

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(tuple(repos),),
        ) as executor:
            futures = {
                executor.submit(extract_one_file, task): task[0]
                for task in tasks
            }
            with tqdm(total=len(futures), desc="Filtering GH Archive", unit="file", dynamic_ncols=True) as bar:
                for future in as_completed(futures):
                    source_name = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # protects the parent process
                        result = ExtractResult(
                            source_path=source_name,
                            output_path=None,
                            total_events=0,
                            matched_events=0,
                            malformed_lines=0,
                            error=f"WorkerFailure: {exc}",
                        )
                    db.mark_result(result)
                    processed += 1
                    total_events += result.total_events
                    matched_events += result.matched_events
                    malformed += result.malformed_lines
                    if result.error:
                        failed += 1
                        logging.error("Failed %s: %s", result.source_path, result.error)
                    bar.set_postfix(
                        matched=f"{matched_events:,}",
                        scanned=f"{total_events:,}",
                        failed=failed,
                        refresh=False,
                    )
                    bar.update(1)

        elapsed = time.perf_counter() - start
        manifest = {
            "pipeline_version": SCRIPT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_root": str(args.input_root.resolve()),
            "parquet_root": str(args.parquet_root.resolve()),
            "repo_list": str(args.repo_list.resolve()),
            "repository_count": len(repos),
            "months": [month_dash(y, m) for y, m in months],
            "source_files_selected": len(all_sources),
            "source_files_processed_this_run": processed,
            "source_files_skipped": skipped,
            "source_files_failed_this_run": failed,
            "events_scanned_this_run": total_events,
            "events_matched_this_run": matched_events,
            "malformed_lines_this_run": malformed,
            "elapsed_seconds": elapsed,
            "workers": workers,
            "parquet_schema": str(PARQUET_SCHEMA),
        }
        atomic_json_dump(args.parquet_root / "_manifest.json", manifest, indent=2)
        logging.info(
            "Extraction complete: %,d matched / %,d scanned; %d failed; %.1f s",
            matched_events, total_events, failed, elapsed,
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Parquet -> monthly aggregate
# ---------------------------------------------------------------------------

MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
EXTERNAL_ASSOCIATIONS = {"CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER", "NONE"}


def role_of(association: str) -> str:
    value = (association or "").upper()
    if value in MAINTAINER_ASSOCIATIONS:
        return "maintainer"
    if value in EXTERNAL_ASSOCIATIONS:
        return "external"
    return "unknown"


@dataclass
class RepoMonthStats:
    counts: Counter[str] = field(default_factory=Counter)
    event_type_counts: Counter[str] = field(default_factory=Counter)
    event_action_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    actors: set[str] = field(default_factory=set)
    maintainer_actors: set[str] = field(default_factory=set)
    external_actors: set[str] = field(default_factory=set)
    unknown_actors: set[str] = field(default_factory=set)
    code_actors: set[str] = field(default_factory=set)
    active_dates: set[str] = field(default_factory=set)
    contributor_days: set[tuple[str, str]] = field(default_factory=set)
    first_event_at: Optional[datetime] = None
    last_event_at: Optional[datetime] = None
    last_release_in_month_at: Optional[datetime] = None
    issue_resolution_days_sum: float = 0.0
    issue_resolution_count: int = 0
    pr_close_days_sum: float = 0.0
    pr_close_count: int = 0
    pr_merge_days_sum: float = 0.0
    pr_merge_count: int = 0

    def add_actor(self, actor: str, role: str, event_type: str, created_at: Optional[datetime]) -> None:
        if not actor:
            return
        actor_key = f"github:{actor.lower()}"
        self.actors.add(actor_key)
        if role == "maintainer":
            self.maintainer_actors.add(actor_key)
        elif role == "external":
            self.external_actors.add(actor_key)
        else:
            self.unknown_actors.add(actor_key)
        if event_type in {"PushEvent", "PullRequestEvent", "PullRequestReviewEvent", "PullRequestReviewCommentEvent", "CommitCommentEvent", "ReleaseEvent", "CreateEvent", "DeleteEvent"}:
            self.code_actors.add(actor_key)
        if created_at:
            date = created_at.date().isoformat()
            self.active_dates.add(date)
            self.contributor_days.add((actor_key, date))

    def to_dict(self, month_end: datetime, prior_last_release: Optional[datetime]) -> tuple[dict[str, Any], Optional[datetime]]:
        last_release = self.last_release_in_month_at or prior_last_release
        days_since_release: Optional[int]
        if last_release is None:
            days_since_release = None
        else:
            days_since_release = max(0, (month_end.date() - last_release.date()).days)

        result: dict[str, Any] = {
            "push_events": self.counts["push_events"],
            "push_commits": self.counts["push_commits"],
            "push_distinct_commits": self.counts["push_distinct_commits"],
            "issues_opened": self.counts["issues_opened"],
            "issues_closed": self.counts["issues_closed"],
            "issues_reopened": self.counts["issues_reopened"],
            "issue_comments": self.counts["issue_comments_created"],
            "issue_comments_created": self.counts["issue_comments_created"],
            "issue_only_comments": self.counts["issue_only_comments"],
            "pr_conversation_comments": self.counts["pr_conversation_comments"],
            "maintainer_issue_comments": self.counts["maintainer_issue_comments"],
            "external_issue_comments": self.counts["external_issue_comments"],
            "unknown_role_issue_comments": self.counts["unknown_role_issue_comments"],
            "issue_resolution_count": self.issue_resolution_count,
            "issue_resolution_days_mean": (
                self.issue_resolution_days_sum / self.issue_resolution_count
                if self.issue_resolution_count else None
            ),
            "prs_opened": self.counts["prs_opened"],
            "prs_closed": self.counts["prs_closed"],
            "prs_merged": self.counts["prs_merged"],
            "prs_reopened": self.counts["prs_reopened"],
            "pr_reviews_submitted": self.counts["pr_reviews_submitted"],
            "pr_reviews_approved": self.counts["pr_reviews_approved"],
            "pr_reviews_changes_requested": self.counts["pr_reviews_changes_requested"],
            "pr_review_comments": self.counts["pr_review_comments_created"],
            "pr_review_comments_created": self.counts["pr_review_comments_created"],
            "maintainer_pr_reviews": self.counts["maintainer_pr_reviews"],
            "external_pr_reviews": self.counts["external_pr_reviews"],
            "maintainer_pr_review_comments": self.counts["maintainer_pr_review_comments"],
            "external_pr_review_comments": self.counts["external_pr_review_comments"],
            "pr_close_days_mean": self.pr_close_days_sum / self.pr_close_count if self.pr_close_count else None,
            "pr_merge_days_mean": self.pr_merge_days_sum / self.pr_merge_count if self.pr_merge_count else None,
            "releases_published": self.counts["releases_published"],
            "last_release_at": iso(last_release),
            "release_observed": last_release is not None,
            "days_since_last_release": days_since_release,
            "watch_started": self.counts["watch_started"],
            "forks": self.counts["forks"],
            "branches_created": self.counts["branches_created"],
            "branches_deleted": self.counts["branches_deleted"],
            "tags_created": self.counts["tags_created"],
            "tags_deleted": self.counts["tags_deleted"],
            "active_days": len(self.active_dates),
            "contributor_days": len(self.contributor_days),
            "active_contributors_count": len(self.actors),
            "code_contributors_count": len(self.code_actors),
            "maintainer_contributors_count": len(self.maintainer_actors),
            "external_contributors_count": len(self.external_actors),
            "unknown_role_contributors_count": len(self.unknown_actors),
            "active_contributors": sorted(self.actors),
            "maintainer_contributors": sorted(self.maintainer_actors),
            "external_contributors": sorted(self.external_actors),
            "first_event_at": iso(self.first_event_at),
            "last_event_at": iso(self.last_event_at),
            "total_events": sum(self.event_type_counts.values()),
            "event_type_counts": dict(sorted(self.event_type_counts.items())),
            "event_action_counts": {
                event_type: dict(sorted(actions.items()))
                for event_type, actions in sorted(self.event_action_counts.items())
            },
        }
        return result, last_release


def empty_month_dict(month_end: datetime, prior_last_release: Optional[datetime]) -> tuple[dict[str, Any], Optional[datetime]]:
    return RepoMonthStats().to_dict(month_end, prior_last_release)


def add_duration(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    if start is None or end is None or end < start:
        return None
    return (end - start).total_seconds() / 86400.0


def consume_row(stats: RepoMonthStats, row: Mapping[str, Any]) -> None:
    event_type = row.get("event_type") or ""
    action = row.get("action") or ""
    actor = row.get("actor_login") or ""
    association = row.get("author_association") or ""
    role = role_of(association)
    created_at = parse_dt(row.get("created_at"))

    stats.event_type_counts[event_type] += 1
    if action:
        stats.event_action_counts[event_type][action] += 1
    stats.add_actor(actor, role, event_type, created_at)
    if created_at:
        if stats.first_event_at is None or created_at < stats.first_event_at:
            stats.first_event_at = created_at
        if stats.last_event_at is None or created_at > stats.last_event_at:
            stats.last_event_at = created_at

    if event_type == "PushEvent":
        stats.counts["push_events"] += 1
        stats.counts["push_commits"] += safe_int(row.get("push_size"))
        stats.counts["push_distinct_commits"] += safe_int(row.get("push_distinct_size"))

    elif event_type == "IssuesEvent":
        if action == "opened":
            stats.counts["issues_opened"] += 1
        elif action == "closed":
            stats.counts["issues_closed"] += 1
            duration = add_duration(parse_dt(row.get("issue_created_at")), parse_dt(row.get("issue_closed_at")) or created_at)
            if duration is not None:
                stats.issue_resolution_days_sum += duration
                stats.issue_resolution_count += 1
        elif action == "reopened":
            stats.counts["issues_reopened"] += 1

    elif event_type == "IssueCommentEvent":
        if action in {"", "created"}:
            stats.counts["issue_comments_created"] += 1
            if bool(row.get("issue_is_pull_request")):
                stats.counts["pr_conversation_comments"] += 1
            else:
                stats.counts["issue_only_comments"] += 1
            stats.counts[f"{'unknown_role' if role == 'unknown' else role}_issue_comments"] += 1

    elif event_type == "PullRequestEvent":
        if action == "opened":
            stats.counts["prs_opened"] += 1
        elif action == "closed":
            stats.counts["prs_closed"] += 1
            closed_at = parse_dt(row.get("pull_closed_at")) or created_at
            created = parse_dt(row.get("pull_created_at"))
            duration = add_duration(created, closed_at)
            if duration is not None:
                stats.pr_close_days_sum += duration
                stats.pr_close_count += 1
            if bool(row.get("pull_merged")) or row.get("pull_merged_at"):
                stats.counts["prs_merged"] += 1
                merge_duration = add_duration(created, parse_dt(row.get("pull_merged_at")) or closed_at)
                if merge_duration is not None:
                    stats.pr_merge_days_sum += merge_duration
                    stats.pr_merge_count += 1
        elif action == "reopened":
            stats.counts["prs_reopened"] += 1

    elif event_type == "PullRequestReviewEvent":
        if action in {"", "created", "submitted"}:
            stats.counts["pr_reviews_submitted"] += 1
            stats.counts[f"{'unknown_role' if role == 'unknown' else role}_pr_reviews"] += 1
            state = (row.get("review_state") or "").lower()
            if state == "approved":
                stats.counts["pr_reviews_approved"] += 1
            elif state == "changes_requested":
                stats.counts["pr_reviews_changes_requested"] += 1

    elif event_type == "PullRequestReviewCommentEvent":
        if action in {"", "created"}:
            stats.counts["pr_review_comments_created"] += 1
            stats.counts[f"{'unknown_role' if role == 'unknown' else role}_pr_review_comments"] += 1

    elif event_type == "ReleaseEvent":
        if action == "published":
            stats.counts["releases_published"] += 1
            release_dt = parse_dt(row.get("release_published_at")) or created_at
            if release_dt and (stats.last_release_in_month_at is None or release_dt > stats.last_release_in_month_at):
                stats.last_release_in_month_at = release_dt

    elif event_type == "WatchEvent" and action == "started":
        stats.counts["watch_started"] += 1
    elif event_type == "ForkEvent":
        stats.counts["forks"] += 1
    elif event_type == "CreateEvent":
        ref_type = (row.get("ref_type") or "").lower()
        if ref_type == "branch":
            stats.counts["branches_created"] += 1
        elif ref_type == "tag":
            stats.counts["tags_created"] += 1
    elif event_type == "DeleteEvent":
        ref_type = (row.get("ref_type") or "").lower()
        if ref_type == "branch":
            stats.counts["branches_deleted"] += 1
        elif ref_type == "tag":
            stats.counts["tags_deleted"] += 1


def parquet_files_for_month(parquet_root: Path, year: int, month: int) -> list[Path]:
    folder = parquet_root / f"year={year:04d}" / f"month={month:02d}"
    if not folder.exists():
        return []
    return sorted(folder.glob("*.parquet"))


def aggregate_one_month(
    files: list[Path],
    batch_size: int,
) -> tuple[dict[str, RepoMonthStats], int]:
    stats: dict[str, RepoMonthStats] = {}
    rows_seen = 0
    if not files:
        return stats, rows_seen

    dataset = ds.dataset([str(path) for path in files], format="parquet")
    scanner = dataset.scanner(columns=PARQUET_COLUMNS, batch_size=batch_size, use_threads=True)
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)

    with tqdm(total=total_rows, desc="Reading Parquet rows", unit="event", leave=False, dynamic_ncols=True) as bar:
        for batch in scanner.to_batches():
            for row in batch.to_pylist():
                repo_key = row.get("repo_key")
                if not repo_key:
                    continue
                repo_stats = stats.get(repo_key)
                if repo_stats is None:
                    repo_stats = RepoMonthStats()
                    stats[repo_key] = repo_stats
                consume_row(repo_stats, row)
            rows_seen += batch.num_rows
            bar.update(batch.num_rows)
    return stats, rows_seen


def run_aggregate(args: argparse.Namespace, repos: set[str], months: list[tuple[int, int]]) -> None:
    args.monthly_dir.mkdir(parents=True, exist_ok=True)
    prior_release: dict[str, datetime] = {}
    total_rows = 0
    manifest_months: list[dict[str, Any]] = []

    # To calculate days_since_last_release correctly when starting in the middle,
    # optionally bootstrap release state from earlier selected Parquet partitions.
    if args.bootstrap_release_history:
        first_year, first_month = months[0]
        earlier_partitions: list[tuple[int, int]] = []
        for year_dir in args.parquet_root.glob("year=*"):
            try:
                year = int(year_dir.name.split("=", 1)[1])
            except (ValueError, IndexError):
                continue
            for month_dir in year_dir.glob("month=*"):
                try:
                    month = int(month_dir.name.split("=", 1)[1])
                except (ValueError, IndexError):
                    continue
                if (year, month) < (first_year, first_month):
                    earlier_partitions.append((year, month))
        for year, month in tqdm(sorted(earlier_partitions), desc="Bootstrapping releases", unit="month"):
            files = parquet_files_for_month(args.parquet_root, year, month)
            if not files:
                continue
            dataset = ds.dataset([str(p) for p in files], format="parquet")
            scanner = dataset.scanner(
                columns=["repo_key", "event_type", "action", "release_published_at", "created_at"],
                filter=(ds.field("event_type") == "ReleaseEvent") & (ds.field("action") == "published"),
                batch_size=args.batch_size,
                use_threads=True,
            )
            for batch in scanner.to_batches():
                for row in batch.to_pylist():
                    release_dt = parse_dt(row.get("release_published_at")) or parse_dt(row.get("created_at"))
                    repo_key = row.get("repo_key")
                    if repo_key and release_dt and (repo_key not in prior_release or release_dt > prior_release[repo_key]):
                        prior_release[repo_key] = release_dt

    for year, month in tqdm(months, desc="Aggregating months", unit="month", dynamic_ncols=True):
        files = parquet_files_for_month(args.parquet_root, year, month)
        repo_stats, rows_seen = aggregate_one_month(files, args.batch_size)
        total_rows += rows_seen
        last_day = calendar.monthrange(year, month)[1]
        month_end = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)

        output: dict[str, Any] = {}
        output_repos: Iterable[str]
        if args.zero_fill:
            output_repos = sorted(repos)
        else:
            output_repos = sorted(repo_stats)

        for repo_key in tqdm(
            output_repos,
            total=len(repos) if args.zero_fill else len(repo_stats),
            desc=f"Writing {month_key(year, month)}",
            unit="repo",
            leave=False,
            dynamic_ncols=True,
        ):
            stats = repo_stats.get(repo_key)
            if stats is None:
                value, carried = empty_month_dict(month_end, prior_release.get(repo_key))
            else:
                value, carried = stats.to_dict(month_end, prior_release.get(repo_key))
            if carried is not None:
                prior_release[repo_key] = carried
            output[repo_key] = value

        expected_files = calendar.monthrange(year, month)[1] * 24
        meta = {
            "schema_version": "parquet-monthly-v1",
            "pipeline_version": SCRIPT_VERSION,
            "month": month_key(year, month),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "parquet_files_with_matches": len(files),
            "expected_hourly_files": expected_files,
            "coverage_note": "Zero-match hourly files intentionally have no Parquet file; source coverage is recorded in the extraction checkpoint database.",
            "parquet_rows": rows_seen,
            "repositories_with_events": len(repo_stats),
            "repository_universe_size": len(repos),
            "zero_filled": bool(args.zero_fill),
            "note": "Missing repo-months represent zero events; do not forward-fill event counts.",
        }
        final = {"_meta": meta, **output}
        out_path = args.monthly_dir / f"delta_{month_key(year, month)}.json"
        atomic_json_dump(out_path, final, indent=None if args.compact_json else 2)
        manifest_months.append(meta)
        logging.info(
            "%s: %,d Parquet rows, %,d active repos, output=%s",
            month_key(year, month), rows_seen, len(repo_stats), out_path,
        )

    manifest = {
        "pipeline_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parquet_root": str(args.parquet_root.resolve()),
        "monthly_dir": str(args.monthly_dir.resolve()),
        "repo_list": str(args.repo_list.resolve()),
        "repository_count": len(repos),
        "months": manifest_months,
        "total_parquet_rows_read": total_rows,
        "zero_fill": bool(args.zero_fill),
    }
    atomic_json_dump(args.monthly_dir / "_manifest.json", manifest, indent=2)


# ---------------------------------------------------------------------------
# Validation and CLI
# ---------------------------------------------------------------------------

def validate_parquet(args: argparse.Namespace) -> None:
    files = sorted(args.parquet_root.glob("year=*/month=*/*.parquet"))
    if not files:
        raise RuntimeError(f"No Parquet files found below {args.parquet_root}")
    failed = 0
    rows = 0
    for path in tqdm(files, desc="Validating Parquet", unit="file"):
        try:
            parquet_file = pq.ParquetFile(path)
            rows += parquet_file.metadata.num_rows
            actual = set(parquet_file.schema_arrow.names)
            missing = set(PARQUET_COLUMNS) - actual
            if missing:
                failed += 1
                logging.error("%s missing columns: %s", path, sorted(missing))
        except Exception as exc:
            failed += 1
            logging.error("Invalid %s: %s", path, exc)
    logging.info("Validation: %,d files, %,d rows, %d failures", len(files), rows, failed)
    if failed:
        raise SystemExit(2)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-list", type=Path, required=True, help="JSON/TXT/CSV target repository list")
    parser.add_argument("--parquet-root", type=Path, required=True, help="Filtered Parquet root")
    parser.add_argument("--start-month", type=parse_month, help="First month, YYYY-MM")
    parser.add_argument("--end-month", type=parse_month, help="Last month, YYYY-MM")
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter local GH Archive into Parquet and build monthly maintenance features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Scan raw GH Archive once and write filtered Parquet")
    add_common(extract)
    extract.add_argument("--input-root", type=Path, required=True, help="Root containing year folders")
    extract.add_argument("--workers", type=int, default=max(1, min((os.cpu_count() or 2) - 1, 8)))
    extract.add_argument("--row-group-size", type=int, default=100_000)
    extract.add_argument("--state-db", type=Path, help="SQLite checkpoint path")
    extract.add_argument("--force", action="store_true", help="Reprocess even if checkpoint is current")

    aggregate = sub.add_parser("aggregate", help="Aggregate filtered Parquet into monthly JSON")
    add_common(aggregate)
    aggregate.add_argument("--monthly-dir", type=Path, required=True)
    aggregate.add_argument("--batch-size", type=int, default=100_000)
    aggregate.add_argument("--zero-fill", action="store_true", help="Include every target repo in every month")
    aggregate.add_argument("--compact-json", action="store_true", help="Write compact JSON")
    aggregate.add_argument(
        "--bootstrap-release-history", action="store_true",
        help="Scan earlier Parquet partitions to initialize last_release_at",
    )

    all_cmd = sub.add_parser("all", help="Run extract, then aggregate")
    add_common(all_cmd)
    all_cmd.add_argument("--input-root", type=Path, required=True)
    all_cmd.add_argument("--monthly-dir", type=Path, required=True)
    all_cmd.add_argument("--workers", type=int, default=max(1, min((os.cpu_count() or 2) - 1, 8)))
    all_cmd.add_argument("--row-group-size", type=int, default=100_000)
    all_cmd.add_argument("--state-db", type=Path)
    all_cmd.add_argument("--force", action="store_true")
    all_cmd.add_argument("--batch-size", type=int, default=100_000)
    all_cmd.add_argument("--zero-fill", action="store_true")
    all_cmd.add_argument("--compact-json", action="store_true")
    all_cmd.add_argument("--bootstrap-release-history", action="store_true")

    validate = sub.add_parser("validate", help="Validate all generated Parquet files")
    validate.add_argument("--parquet-root", type=Path, required=True)
    validate.add_argument("--verbose", action="store_true")

    return parser


def resolve_months(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.start_month and args.end_month:
        if args.start_month > args.end_month:
            raise ValueError("--start-month must not be after --end-month")
        return list(iter_months(args.start_month, args.end_month))
    if args.start_month or args.end_month:
        raise ValueError("Provide both --start-month and --end-month")

    if hasattr(args, "input_root"):
        available = infer_available_months(args.input_root)
    else:
        available = []
        for year_dir in args.parquet_root.glob("year=*"):
            try:
                year = int(year_dir.name.split("=", 1)[1])
            except (ValueError, IndexError):
                continue
            for month_dir in year_dir.glob("month=*"):
                try:
                    month = int(month_dir.name.split("=", 1)[1])
                except (ValueError, IndexError):
                    continue
                available.append((year, month))
        available = sorted(set(available))

    if not available:
        raise RuntimeError("Could not infer any available months")
    return list(iter_months(available[0], available[-1]))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(getattr(args, "verbose", False))

    if args.command == "validate":
        validate_parquet(args)
        return

    repos = load_repo_list(args.repo_list)
    logging.info("Loaded %,d unique target repositories", len(repos))
    months = resolve_months(args)
    logging.info("Selected months: %s to %s (%d)", month_dash(*months[0]), month_dash(*months[-1]), len(months))

    if args.command in {"extract", "all"}:
        if not args.input_root.exists():
            raise FileNotFoundError(args.input_root)
        run_extract(args, repos, months)

    if args.command in {"aggregate", "all"}:
        run_aggregate(args, repos, months)


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on Windows.
    main()
