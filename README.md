# Rust Package Risk Analysis

This repository contains the code and generated artifacts for a multi-dimensional risk metric for third-party Rust packages. 

The core idea is that package risk cannot be captured by dependency popularity alone. The metric combines three complementary dimensions:

1. **Dependency criticality**: how much impact a package can have through downstream dependents and how much risk it can inherit from upstream dependencies.
2. **Maintenance status**: whether historical development activity suggests the package is likely to remain actively maintained.
3. **Functional replaceability**: whether similar packages exist that could serve as practical alternatives if a package becomes vulnerable, abandoned, or unavailable.

The combined score is intended for prioritization. A high score does not mean a package is unsafe; it means the package may deserve more security auditing, maintenance monitoring, or ecosystem governance attention.

## Repository Layout

```text
code/
  structual_importance.py                 # dependency criticality metric
  advanced_maintenance_prediction.py      # maintenance/activity prediction
  similarity_eval.py                      # embedding-based replaceability metric
  replacement_eval.py                     # alternate replaceability evaluation helper
  combine_metrics.py                      # combines criticality, maintenance, replaceability
  validate_combined_metric_correlation.py # validates metric against advisory severity
  helper/
    download_rust_advisories.py           # downloads GitHub Rust security advisories
    get_crate_download.py                 # downloads crates.io daily version downloads
    get_pkg_textual.py                    # extracts package functional summaries with an LLM
    deprecated_identify.py                # identifies deprecated packages/replacements with an LLM

data/
  rust_advisories_stream.jsonl            # advisory data used for validation

result/
  metrics/                                # generated metric CSVs
  graphs/                                 # generated validation plots
```

## Data Notice

The original crates.io dump, GitHub activity data, embeddings, and other raw intermediate files can be large, so they may not be included in the repository. The Rust package data can be reconstructed from the crates.io database dump, crates.io download archives, and the GitHub API.

Expected raw inputs include:

- `data/crates.csv`
- `data/versions.csv`
- `data/dependencies.csv`
- `data/version_downloads.csv`
- `data/monthly/*.json` or equivalent monthly GitHub activity files
- package text/function data for replaceability analysis

## Installation

Use Python 3.10+ and install the common scientific stack:

```bash
pip install numpy pandas scipy scikit-learn matplotlib tqdm requests
```

For maintenance prediction, PyTorch is also required:

```bash
pip install torch
```

## Workflow

### 1. Download Advisory Data

Download Rust advisories from GitHub's Global Security Advisories API:

```bash
python code/helper/download_rust_advisories.py --output data/rust_advisories_stream.jsonl
```

Optional date filter:

```bash
python code/helper/download_rust_advisories.py --since 2025-11-01 --output data/rust_advisories_stream.jsonl
```

If GitHub rate limits anonymous requests, set `GITHUB_TOKEN` or pass `--token`.

### 2. Dependency Criticality

Compute structural criticality from crates.io package, version, dependency, and download data:

```bash
python code/structual_importance.py \
  --crates data/crates.csv \
  --versions data/versions.csv \
  --deps data/dependencies.csv \
  --version-downloads data/version_downloads.csv \
  --out-dir result
```

This script builds the latest-version dependency graph, computes downstream and upstream structural scores, optionally weights nodes by recent downloads, and writes outputs such as:

- `result/crate_importance_metric.csv`
- `result/all_crates_importance.csv`
- `result/pagerank_global.csv`
- `result/summary.json`

The main metric expected by later steps is a CSV with:

```text
crate_name, importance
```

### 3. Maintenance Prediction

Predict future activity/maintenance probability from monthly GitHub activity sequences:

```bash
python code/advanced_maintenance_prediction.py --models Mamba --epochs 50
```

The script trains a temporal model using historical activity signals such as issues, pull requests, releases, contributors, and days since last release. It writes a timestamped folder under `result/advanced_prediction_*` containing model artifacts and:

```text
mamba_activity_prediction.csv
```

The maintenance metric expected by later steps is:

```text
crate_name, activity_probability
```


### 4. Replaceability Metric

The replaceability pipeline represents package functionality with text embeddings and estimates whether each package has similar alternatives.

```bash
python code/similarity_eval.py
```

The primary output expected by later steps is:

```text
crate_name, replacement_metric
```

`code/replacement_eval.py` is an alternate helper for evaluating replacement/substitutability using embedding and ground-truth files:

```bash
python code/replacement_eval.py \
  --embeddings data/package_embeddings-co.json \
  --groundtruth data/groundtruth_clean.json \
  --output result/package_substitutability.csv
```

### 5. Combine Metrics

Combine the three dimensions into a unified criticality/risk-prioritization score:

```bash
python code/combine_metrics.py --combine-method both --output result/crate_combined_criticality.csv
```

Expected input columns:

- criticality: `crate_name`, `importance`
- maintenance: `crate_name`, `activity_probability`
- replaceability: `crate_name`, `replacement_metric`

Ranking convention:

- Higher `importance` means more structurally critical.
- Lower `activity_probability` means higher maintenance risk.
- Lower `replacement_metric` means harder to replace.

The script can compute geometric-mean rank, average rank, or both.

### 6. Validate Against Security Advisories

Validate the metric against Rust security advisories and CVSS severity:

```bash
python code/validate_combined_metric_correlation.py \
  --advisories data/rust_advisories_stream.jsonl \
  --crates data/crates.csv \
  --importance result/criticality.csv \
  --activity result/metrics/maintenance.csv \
  --replacement result/replaceability.csv \
  --output result/combined_metric_validation.csv
```

The validation script maps advisories to crates, keeps advisories after the configured start date, computes Pearson and Spearman correlations, and plots the combined score and each dimension against CVSS severity.

## Generated Artifacts

The repository may contain generated outputs such as:

- `result/metrics/criticality.csv`
- `result/metrics/maintenance.csv`
- `result/metrics/replaceability.csv`
- `result/graphs/*.png`

These are analysis outputs rather than raw source data. If you regenerate them, keep column names consistent with the scripts above. In particular, downstream scripts expect the criticality column to be named `importance`.

