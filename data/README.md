# Data directory

This repository commits small, ten-record fixtures under `samples/` so that every
tabular and event input format can be inspected without downloading the full research
corpus. These fixtures illustrate schemas only; they are not large enough to reproduce
the reported measurements.

Run `package-risk-data all` after installing the project to detect and retrieve
missing public data. Existing non-empty files are retained unless `--force` is used.

## Fetch and workflow scripts

Run these commands from the repository root. Fetch all public crates.io and advisory
inputs (existing non-empty files are skipped):

```bash
python scripts/fetch_data.py
```

Common data options include:

```bash
python scripts/fetch_data.py --days 180 --sqlite
python scripts/fetch_data.py --token-file /path/to/github-token
python scripts/fetch_data.py \
  --gharchive-root /path/to/gharchive \
  --gharchive-start 2024-01-01 --gharchive-end 2025-12-31
```

After the required raw and derived inputs are present, run the complete analysis:

```bash
python scripts/run_all.py
```

Use `python scripts/run_all.py --dry-run` to inspect the commands, `--fetch` to run
public-data retrieval first, or `--stages criticality maintenance` to select stages.
The full replaceability stage also requires `crate_function_conclude.csv`,
`deprecated_pairs.csv`, and `embeddings_cache.npz`; these are derived model outputs,
not downloadable public source datasets.

The `crates.csv`, `versions.csv`, and `dependencies.csv` files originate in the
crates.io **PostgreSQL database dump**. They are raw table exports and retain
database-internal identifiers such as `crate_id` and `version_id`. Use
`package-risk-data sqlite` to create a local SQLite copy.

> **Download-history limitation:** the crates.io database dump intentionally contains
> only about the most recent 90 days of version-download history for volume reasons.
> The separate daily archive can provide a selected window; the data command defaults
> to 90 days to keep storage and transfer costs bounded.

GitHub activity inputs require the GH Archive collection pipeline documented in the
root README. Security advisory retrieval uses `GITHUB_TOKEN`/`GH_TOKEN` (recommended)
or `--token-file PATH`.

The two-column result used by the installed library is stored at
`result/crate_hazard_score.csv`. It contains exactly 10,000 CSV lines (one header plus
the first 9,999 experiment records) and the columns `crate_name` and `score`; it is an
output artifact rather than a raw input fixture.

## Included fixtures

Each fixture contains ten example records (plus a header for CSV files):

| Fixture | Role |
|---|---|
| `crates.csv`, `versions.csv`, `dependencies.csv` | raw crates.io relational tables |
| `version_downloads.csv`, `crate_downloads.csv` | recent version/crate download counts |
| `crates_with_stars.csv` | crate metadata joined with repository metadata |
| `keywords.csv`, `crates_keywords.csv` | keyword lookup and join table |
| `crate_function_conclude.csv` | derived functional summaries |
| `deprecated_pairs.csv` | derived deprecated/replacement pairs |
| `project_list.json` | repositories selected for activity collection |
| `monthly/delta_2023_11.json` | monthly repository-event features |
| `rust_advisories_stream.jsonl` | Rust security advisories |
