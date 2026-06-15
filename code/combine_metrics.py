#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine three crate metrics into a single criticality score.

Supported combination methods:
  1. geometric mean of ranks
  2. arithmetic average of ranks
  3. both

Ranking direction:
  Higher rank = more critical / higher final score

Metrics:
  importance:
      higher importance value → higher rank

  activity_probability:
      lower activity probability → higher risk rank
      because lower activity means more likely to become inactive

  replacement_metric:
      lower replacement metric → higher irreplaceability rank
      because lower replaceability means harder to replace

Final interpretation:
  A high combined score means the crate is important, likely to go inactive,
  and hard to replace.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ── Paths ──────────────────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(base_dir, "..", "result")

IMPORTANCE_CSV = os.path.join(result_dir, "criticality.csv")
REPLACEMENT_CSV = os.path.join(result_dir, "replaceability.csv")
ACTIVITY_CSV = os.path.join(result_dir, "maintenance.csv")
OUTPUT_CSV = os.path.join(result_dir, "crate_combined_criticality.csv")


IMPORTANCE_COL = "importance"
REPLACEMENT_COL = "replacement_metric"
ACTIVITY_COL = "activity_probability"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine crate importance, activity risk, and replaceability into criticality scores."
    )

    parser.add_argument(
        "--combine-method",
        choices=["geo", "avg", "both"],
        default="both",
        help=(
            "Combination method. "
            "'geo' uses geometric mean rank, "
            "'avg' uses average rank, "
            "'both' outputs both scores. Default: both."
        ),
    )

    parser.add_argument(
        "--output",
        default=OUTPUT_CSV,
        help=f"Output CSV path. Default: {OUTPUT_CSV}",
    )

    return parser.parse_args()


def normalize_to_0_1(values: np.ndarray) -> np.ndarray:
    """
    Normalize values to [0, 1].
    If all values are identical, return all zeros.
    """
    values = np.asarray(values, dtype=float)

    min_value = np.nanmin(values)
    max_value = np.nanmax(values)

    denominator = max_value - min_value

    if denominator == 0 or np.isnan(denominator):
        return np.zeros_like(values, dtype=float)

    return (values - min_value) / denominator


def load_and_merge() -> pd.DataFrame:
    imp = pd.read_csv(IMPORTANCE_CSV)
    rep = pd.read_csv(REPLACEMENT_CSV)
    act = pd.read_csv(ACTIVITY_CSV)

    required_imp_cols = {"crate_name", IMPORTANCE_COL}
    required_rep_cols = {"crate_name", REPLACEMENT_COL}
    required_act_cols = {"crate_name", ACTIVITY_COL}

    missing_imp_cols = required_imp_cols - set(imp.columns)
    missing_rep_cols = required_rep_cols - set(rep.columns)
    missing_act_cols = required_act_cols - set(act.columns)

    if missing_imp_cols:
        raise ValueError(f"Importance CSV missing required columns: {sorted(missing_imp_cols)}")

    if missing_rep_cols:
        raise ValueError(f"Replacement CSV missing required columns: {sorted(missing_rep_cols)}")

    if missing_act_cols:
        raise ValueError(f"Activity CSV missing required columns: {sorted(missing_act_cols)}")

    imp = imp[["crate_name", IMPORTANCE_COL]].copy()
    rep = rep[["crate_name", REPLACEMENT_COL]].copy()
    act = act[["crate_name", ACTIVITY_COL]].copy()

    imp["crate_name"] = imp["crate_name"].astype(str).str.strip().str.lower()
    rep["crate_name"] = rep["crate_name"].astype(str).str.strip().str.lower()
    act["crate_name"] = act["crate_name"].astype(str).str.strip().str.lower()

    df = imp.merge(rep, on="crate_name", how="inner").merge(act, on="crate_name", how="inner")

    print(f"Rows after inner join: {len(df)}")

    metric_cols = [
        IMPORTANCE_COL,
        REPLACEMENT_COL,
        ACTIVITY_COL,
    ]

    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)

    df = df.dropna(subset=metric_cols).copy()

    dropped = before - len(df)

    if dropped:
        print(f"Dropped {dropped} rows with missing metric values.")

    print(f"Crates with all three values: {len(df)}")

    return df


def compute_metric_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-metric ranks.

    Ranking convention:
      rank 1 = least critical
      rank N = most critical

    importance:
      higher value is better / more critical

    activity_probability:
      lower value is riskier, so rank by negative value

    replacement_metric:
      lower value means harder to replace, so rank by negative value
    """
    df = df.copy()

    rank_importance = rankdata(
        df[IMPORTANCE_COL].values,
        method="average",
    )

    rank_risk = rankdata(
        -df[ACTIVITY_COL].values,
        method="average",
    )

    rank_irreplaceability = rankdata(
        -df[REPLACEMENT_COL].values,
        method="average",
    )

    df["rank_importance"] = rank_importance
    df["rank_risk"] = rank_risk
    df["rank_irreplaceability"] = rank_irreplaceability

    return df


def compute_combined_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute both geometric mean rank score and average rank score.
    """
    df = df.copy()

    rank_importance = df["rank_importance"].values.astype(float)
    rank_risk = df["rank_risk"].values.astype(float)
    rank_irreplaceability = df["rank_irreplaceability"].values.astype(float)

    geo_rank_raw = (
        rank_importance
        * rank_risk
        * rank_irreplaceability
    ) ** (1.0 / 3.0)

    avg_rank_raw = (
        rank_importance
        + rank_risk
        + rank_irreplaceability
    ) / 3.0

    df["geo_rank_raw"] = geo_rank_raw
    df["avg_rank_raw"] = avg_rank_raw

    df["geo_rank_score"] = normalize_to_0_1(geo_rank_raw)
    df["avg_rank_score"] = normalize_to_0_1(avg_rank_raw)

    return df


def apply_selected_combination(df: pd.DataFrame, combine_method: str) -> pd.DataFrame:
    """
    Add final selected score column according to the chosen method.
    """
    df = df.copy()

    if combine_method == "geo":
        df["combined_score"] = df["geo_rank_score"]
        df["combined_score_raw"] = df["geo_rank_raw"]
        df["combined_method"] = "geometric_mean_rank"

    elif combine_method == "avg":
        df["combined_score"] = df["avg_rank_score"]
        df["combined_score_raw"] = df["avg_rank_raw"]
        df["combined_method"] = "average_rank"

    elif combine_method == "both":
        # Keep both scores and use geometric score as the default sorting score.
        # You can change this to avg_rank_score if you prefer average rank as default.
        df["combined_score"] = df["geo_rank_score"]
        df["combined_score_raw"] = df["geo_rank_raw"]
        df["combined_method"] = "both_geo_default"

    else:
        raise ValueError(f"Unsupported combine method: {combine_method}")

    return df


def sort_result(df: pd.DataFrame, combine_method: str) -> pd.DataFrame:
    if combine_method == "geo":
        sort_col = "geo_rank_score"
    elif combine_method == "avg":
        sort_col = "avg_rank_score"
    else:
        sort_col = "combined_score"

    return df.sort_values(sort_col, ascending=False).reset_index(drop=True)


def print_top_crates(result: pd.DataFrame, combine_method: str, top_k: int = 20):
    print(f"\nTop {top_k} most critical crates by method: {combine_method}")

    display_cols = [
        "crate_name",
        IMPORTANCE_COL,
        ACTIVITY_COL,
        REPLACEMENT_COL,
        "rank_importance",
        "rank_risk",
        "rank_irreplaceability",
        "geo_rank_score",
        "avg_rank_score",
        "combined_score",
        "combined_method",
    ]

    existing_cols = [col for col in display_cols if col in result.columns]

    print(result[existing_cols].head(top_k).to_string(index=False))


def main():
    args = parse_args()

    print("Loading metrics ...")
    df = load_and_merge()

    print("Computing metric ranks ...")
    df = compute_metric_ranks(df)

    print("Computing combined scores ...")
    result = compute_combined_scores(df)

    print(f"Applying selected combination method: {args.combine_method}")
    result = apply_selected_combination(result, args.combine_method)

    result = sort_result(result, args.combine_method)

    output_path = args.output

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result.to_csv(output_path, index=False)

    print(f"\nSaved {len(result)} rows -> {output_path}")

    print_top_crates(result, args.combine_method, top_k=20)


if __name__ == "__main__":
    main()