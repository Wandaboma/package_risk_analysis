#!/usr/bin/env python3
"""
Download Rust package security advisories from the GitHub API.

This script uses GitHub's Global Security Advisories REST endpoint and writes
JSONL rows compatible with code/validate_combined_metric_correlation.py.

Examples:
  python helper/download_rust_advisories.py
  python helper/download_rust_advisories.py --output data/rust_advisories_stream.jsonl
  python helper/download_rust_advisories.py --since 2025-11-01 --token %GITHUB_TOKEN%
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://api.github.com/advisories"
DEFAULT_OUTPUT = "data/rust_advisories_stream.jsonl"
CVSS_KEYS = ("AV", "AC", "PR", "UI", "S", "C", "I", "A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Rust GitHub Security Advisories as JSONL."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--ecosystem",
        default="rust",
        help="GitHub advisory ecosystem to download. Default: rust",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only keep advisories published on or after this date/time, e.g. 2025-11-01.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"),
        help="GitHub token. Defaults to GITHUB_TOKEN or GH_TOKEN when set.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between API pages. Default: 0.2",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=100,
        help="GitHub page size, max 100. Default: 100",
    )
    parser.add_argument(
        "--state",
        default="published",
        choices=["published", "withdrawn", "all"],
        help="Advisory state to request. Default: published",
    )
    return parser.parse_args()


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if len(text) == 10:
        text = text + "T00:00:00+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_link_header(value: str | None) -> dict[str, str]:
    links: dict[str, str] = {}
    if not value:
        return links
    for part in value.split(","):
        section = part.strip().split(";")
        if len(section) < 2:
            continue
        url = section[0].strip()
        if url.startswith("<") and url.endswith(">"):
            url = url[1:-1]
        rel = None
        for item in section[1:]:
            item = item.strip()
            if item.startswith('rel="') and item.endswith('"'):
                rel = item[5:-1]
        if rel:
            links[rel] = url
    return links


def sleep_until_reset(headers: Any) -> None:
    reset = headers.get("X-RateLimit-Reset")
    if not reset:
        return
    try:
        reset_at = int(reset)
    except ValueError:
        return
    delay = max(0, reset_at - int(time.time()) + 2)
    if delay:
        print(f"GitHub rate limit reached. Sleeping {delay} seconds...", file=sys.stderr)
        time.sleep(delay)


def github_get_json(url: str, token: str | None) -> tuple[list[dict[str, Any]], dict[str, str], Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "lib-risk-analysis-rust-advisory-downloader",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
            links = parse_link_header(response.headers.get("Link"))
            return data, links, response.headers
    except HTTPError as exc:
        if exc.code in (403, 429):
            sleep_until_reset(exc.headers)
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API request failed: HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"GitHub API request failed: {exc}") from exc


def parse_cvss_vector(vector: str | None) -> dict[str, str]:
    parts: dict[str, str] = {}
    if not vector:
        return parts
    for token in vector.split("/"):
        if ":" not in token:
            continue
        key, value = token.split(":", 1)
        if key in CVSS_KEYS:
            parts[key] = value
    return parts


def advisory_packages(advisory: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for vuln in advisory.get("vulnerabilities") or []:
        package = vuln.get("package") or {}
        name = package.get("name")
        if name:
            names.append(str(name).strip().lower())
    return sorted(set(name for name in names if name))


def advisory_to_rows(advisory: dict[str, Any]) -> list[dict[str, Any]]:
    cvss = advisory.get("cvss") or {}
    cvss_score = cvss.get("score")
    cvss_vector = cvss.get("vector_string")
    cvss_parts = parse_cvss_vector(cvss_vector)

    base = {
        "ghsaId": advisory.get("ghsa_id"),
        "cveId": advisory.get("cve_id"),
        "severity": advisory.get("severity"),
        "publishedAt": advisory.get("published_at"),
        "updatedAt": advisory.get("updated_at"),
        "withdrawnAt": advisory.get("withdrawn_at"),
        "cvss_score": cvss_score,
        "cvss_vector": cvss_vector,
        **cvss_parts,
    }

    rows = []
    for package in advisory_packages(advisory):
        rows.append({"package": package, **base})
    return rows


def download_advisories(args: argparse.Namespace) -> list[dict[str, Any]]:
    params = {
        "ecosystem": args.ecosystem,
        "per_page": min(max(args.per_page, 1), 100),
        "sort": "published",
        "direction": "desc",
    }
    if args.state != "all":
        params["state"] = args.state

    url = f"{API_URL}?{urlencode(params)}"
    since_dt = parse_datetime(args.since)
    rows: list[dict[str, Any]] = []
    page = 0

    while url:
        page += 1
        advisories, links, headers = github_get_json(url, args.token)
        if not isinstance(advisories, list):
            raise RuntimeError(f"Unexpected GitHub response on page {page}: {type(advisories).__name__}")

        stop_after_page = False
        for advisory in advisories:
            published_at = parse_datetime(advisory.get("published_at"))
            if since_dt and published_at and published_at < since_dt:
                stop_after_page = True
                continue
            rows.extend(advisory_to_rows(advisory))

        remaining = headers.get("X-RateLimit-Remaining")
        print(f"Downloaded page {page}: {len(advisories)} advisories, {len(rows)} package rows total, remaining={remaining}")

        if stop_after_page:
            break
        url = links.get("next")
        if url and args.sleep > 0:
            time.sleep(args.sleep)

    return rows


def write_jsonl(rows: list[dict[str, Any]], output: str) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def main() -> None:
    args = parse_args()
    rows = download_advisories(args)
    write_jsonl(rows, args.output)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
