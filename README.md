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

## Repository layout

```text
src/package_risk_analysis/                # installable data-preparation library
code/                                     # analysis entry-point scripts
code/helper/                              # optional collection/enrichment scripts
data/README.md                            # provenance and limitations
data/samples/                             # ten-record fixtures for every input format
result/                                   # generated metrics and figures
```

## Prepare data

The public-input command checks for non-empty files and downloads only those that are
missing. The following retrieves the crates.io core tables, the
download files, and Rust security advisories:

```bash
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
  --out-dir result
```

The latest-version dependency graph produces `crate_importance_metric.csv` and
supporting graph statistics. Higher `importance` means greater structural criticality.

### 2. Maintenance prediction

```bash
python code/advanced_maintenance_prediction.py --models Mamba --epochs 50
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
  --importance result/criticality.csv \
  --activity result/metrics/maintenance.csv \
  --replacement result/replaceability.csv \
  --output result/combined_metric_validation.csv
```

## Resource guide

The figures below are planning estimates for a contemporary 8-core workstation and
vary with corpus date, network speed, selected period, and accelerator. Measure and
report the exact environment when reproducing results.

| Step | Typical wall time | Peak working disk | Main scaling factor |
|---|---:|---:|---|
| Download/extract crates.io dump | 10–40 min | 8–15 GB | network and dump size |
| Download + combine 90 daily download files | 10–30 min | 1–4 GB | network and date window |
| Convert core CSVs to SQLite (optional) | 15–45 min | 8–20 GB extra | CSV and index size |
| Filter 24 months of GH Archive | hours to days | 50–300+ GB | raw archive period and events |
| Dependency criticality | 30 min–several hours | 8–32 GB RAM, 2–10 GB disk | graph size and distance bounds |
| Maintenance training | 1–8 hours | 4–16 GB disk | model, epochs, CPU/GPU |
| Semantic enrichment/embeddings | hours | 1–10 GB | model/service throughput |
| Combine and advisory validation | minutes | under 2 GB | number of crates/advisories |

For a format-only inspection, use `data/samples/`; it needs only a few megabytes and
does not reproduce the study results.
