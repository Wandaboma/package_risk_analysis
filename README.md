# Rust Package Risk Analysis

This repository implements a multi-dimensional prioritization metric for third-party
Rust packages. It combines dependency criticality, predicted maintenance status, and
functional replaceability. A high score does not mean that a crate is unsafe; it
means that the crate may deserve more auditing, monitoring, or ecosystem support.

## Install

Python 3.10 or newer is required. Installing the project now installs the shared
analysis dependencies and the `package-risk-data` command in one reproducible step:

```bash
python -m pip install -e .
```

Install optional dependencies only for the relevant stage:

```bash
python -m pip install -e ".[maintenance]"  # PyTorch and Prophet
python -m pip install -e ".[gharchive]"    # PyArrow and orjson
```

### Library API

The installed package includes the reference importance results, so callers do not
need the repository checkout or a network request to read them:

```python
from package_risk_analysis import (
    get_crate_importance,
    get_reference_scores,
    get_reference_scores_csv,
)

top_100 = get_reference_scores(limit=100)
serde = get_crate_importance("serde")
csv_response_body = get_reference_scores_csv()
```

`get_reference_scores()` returns dictionaries with the stable schema
`{"crate_name": str, "importance": float}`. `get_crate_importance()` performs a
case-insensitive lookup and returns one such dictionary or `None`.
`get_reference_scores_csv()` returns the complete UTF-8 CSV text and can be used
directly as an HTTP or service response body with content type `text/csv`.

The readable source is committed as
[`result/crate_importance_reference.csv`](result/crate_importance_reference.csv). It
contains the `crate_name` and `importance_with_download_portion` fields exported from
the experiment result, with the latter renamed to the public `importance` field.

## Repository layout

```text
src/package_risk_analysis/                # installable API, data preparation, resources
scripts/fetch_data.py                     # fetches all public inputs
scripts/run_all.py                        # runs the complete analysis pipeline
code/                                     # analysis entry-point scripts
code/helper/                              # optional collection/enrichment scripts
data/README.md                            # provenance and limitations
data/samples/                             # ten-record fixtures for every input format
result/                                   # generated metrics and reference scores
```

## One-command workflow

From the repository root, fetch the public crates.io and advisory inputs with:

```bash
python scripts/fetch_data.py
```

The fetcher is idempotent: it detects non-empty outputs and only retrieves missing
files. It also derives `data/project_list.json` from the repository URLs in
`crates.csv`. Optional examples:

```bash
python scripts/fetch_data.py --days 180 --sqlite
python scripts/fetch_data.py --token-file /path/to/github-token
python scripts/fetch_data.py \
  --gharchive-root /path/to/gharchive \
  --gharchive-start 2024-01-01 --gharchive-end 2025-12-31
```

Run all analysis stages in dependency order with:

```bash
python scripts/run_all.py
```

The runner executes dependency criticality, maintenance prediction, replaceability,
metric combination, and advisory validation. It writes stable outputs below `result/`
and stops with a list of missing inputs before starting an unavailable stage.

Useful variants:

```bash
# Print the complete command plan without executing anything
python scripts/run_all.py --dry-run

# Fetch public inputs, then run all stages
python scripts/run_all.py --fetch

# Run or resume selected stages
python scripts/run_all.py --stages combine validate

# Fetch/process an explicit GH Archive period before the maintenance stage
python scripts/run_all.py --fetch \
  --gharchive-root /path/to/gharchive \
  --gharchive-start 2024-01-01 --gharchive-end 2025-12-31
```

The full workflow additionally requires the derived
`data/crate_function_conclude.csv`, `data/deprecated_pairs.csv`, and
`data/embeddings_cache.npz` inputs for replaceability. These are not public source
datasets and cannot be recreated by a generic downloader; compatible files may be
produced with a local model or the optional enrichment helpers. Maintenance likewise
requires monthly GH Archive features, either already under `data/monthly/` or produced
by supplying the GH Archive options above.

## Prepare data

The public-input command checks for non-empty files and downloads only those that are
missing. The following retrieves the crates.io core tables, the
download files, and Rust security advisories:

```bash
python scripts/fetch_data.py
# Equivalent installed command:
package-risk-data --data-dir data all
```

To also import the raw crates.io CSV tables into an easy-to-query SQLite database:

```bash
package-risk-data --data-dir data all --sqlite
# or, after the CSVs exist:
package-risk-data --data-dir data sqlite --output data/crates.sqlite3
```

The crates.io CSVs originate from the
[nightly crates.io database dump](https://crates.io/data-access). 
Consequently, they retain internal IDs (`crate_id`, `version_id`,
and similar columns) used to join tables. The optional SQLite file keeps those raw
columns and adds indexes to make joins inspectable with ordinary SQL. Daily counts
come from the separate
[version-download archive](https://static.crates.io/archive/version-downloads/).

> **Version-download coverage:** for volume reasons, crates.io database dumps include
> only roughly the most recent 90 days of version-download history. 

The committed files in `data/samples/` contain example data used in
the experiment corpus and demonstrate the expected formats. See
[`data/README.md`](data/README.md) for the complete provenance note.

### GitHub credentials

Anonymous advisory requests work at low volume. For authenticated access, use an
environment variable so a token does not enter shell history:

```bash
export GITHUB_TOKEN="..."                 # Linux/macOS
$env:GITHUB_TOKEN = "..."                 # PowerShell
package-risk-data --data-dir data all
```

Alternatively, put only the token in a permission-restricted file and pass its path:

```bash
package-risk-data --data-dir data all --token-file /path/to/github-token
```

### GitHub activity and semantic enrichment

Maintenance prediction uses monthly GitHub event features. Retrieve only the explicit
date range needed from [GH Archive](https://www.gharchive.org/):

```bash
package-risk-data --data-dir data gharchive \
  --start-date 2024-01-01 --end-date 2025-12-31 \
  --output-dir /path/to/gharchive
```

Then build the monthly inputs with:

```bash
python code/helper/gharchive_info_collect.py all \
  --input-root /path/to/gharchive \
  --repo-list data/project_list.json \
  --parquet-root data/gha_parquet \
  --monthly-dir data/monthly \
  --start-month 2024-01 --end-month 2025-12
```

Functional summaries and embeddings are derived inputs. The repository includes
their formats and downstream analysis. You may provide compatible summaries/embeddings from a local model or
another service.

## Analysis workflow

All commands below are run from the repository root.

### 1. Dependency criticality

```bash
python code/structual_importance.py \
  --crates data/crates.csv \
  --versions data/versions.csv \
  --deps data/dependencies.csv \
  --version-downloads data/version_downloads.csv \
  --out-dir result \
  --crate-name ""
```

The latest-version dependency graph produces `crate_importance_metric.csv` and
supporting graph statistics. Higher `importance` means greater structural criticality.

### 2. Maintenance prediction

```bash
python code/advanced_maintenance_prediction.py \
  --models Mamba --epochs 50 \
  --output-dir result/maintenance_model
```

The model consumes `data/monthly/delta_YYYY_MM.json` activity sequences and produces
`activity_prediction.csv`. Lower `activity_probability` means greater predicted
maintenance risk.

### 3. Replaceability

```bash
python code/similarity_eval.py
```

The script consumes functional summaries plus cached embeddings and writes
`crate_replacement_metric.csv`. Lower `replacement_metric` means harder to replace.

### 4. Combine metrics

```bash
python code/combine_metrics.py \
  --importance result/crate_importance_metric.csv \
  --activity result/maintenance_model/mamba_activity_prediction.csv \
  --replacement result/crate_replacement_metric.csv \
  --combine-method both \
  --output result/crate_combined_criticality.csv
```

Expected columns are `crate_name,importance`, `crate_name,activity_probability`, and
`crate_name,replacement_metric` for the three input metrics respectively.

### 5. Validate against advisories

```bash
python code/validate_combined_metric_correlation.py \
  --advisories data/rust_advisories_stream.jsonl \
  --crates data/crates.csv \
  --importance result/crate_importance_metric.csv \
  --activity result/maintenance_model/mamba_activity_prediction.csv \
  --replacement result/crate_replacement_metric.csv \
  --output result/combined_metric_validation.csv
```

## Resource guide

The figures below are estimations for a contemporary 8-core workstation and
vary with corpus date, network speed, selected period, and accelerator. 

| Step | Typical wall time | Peak working disk | Main scaling factor |
|---|---:|---:|---|
| Download/extract crates.io dump | 10–40 min | 8–15 GB | network and dump size |
| Convert core CSVs to SQLite (optional) | 15–45 min | 8–20 GB extra | CSV and index size |
| Filter 3 years of GH Archive | hours to days | 50–300+ GB | raw archive period and events |
| Dependency criticality | 30 min–several hours | 8–32 GB RAM, 2–10 GB disk | graph size and distance bounds |
| Maintenance training | 1–8 hours | 4–16 GB disk | model, epochs, CPU/GPU |
| Semantic embeddings | hours | 1–10 GB | model/service throughput |
| Combine and advisory validation | minutes | under 2 GB | number of crates/advisories |

