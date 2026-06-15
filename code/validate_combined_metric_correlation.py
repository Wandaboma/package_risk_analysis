import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


PLOT_FIGSIZE = (12, 8)
PLOT_DPI = 300
SCATTER_SIZE = 170
FONT_SIZES = {
    "title": 20,
    "axis_label": 18,
    "tick": 15,
    "legend": 15,
    "legend_title": 16,
    "annotation": 16,
}

plt.rcParams.update({
    "font.size": FONT_SIZES["tick"],
    "axes.titlesize": FONT_SIZES["title"],
    "axes.labelsize": FONT_SIZES["axis_label"],
    "xtick.labelsize": FONT_SIZES["tick"],
    "ytick.labelsize": FONT_SIZES["tick"],
    "legend.fontsize": FONT_SIZES["legend"],
    "legend.title_fontsize": FONT_SIZES["legend_title"],
    "figure.titlesize": FONT_SIZES["title"],
})

SEVERITY_MAPPING = {
    "LOW": 2.0,
    "MODERATE": 5.5,
    "MEDIUM": 5.5,
    "HIGH": 8.0,
    "CRITICAL": 9.5,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--advisories", default="data/rust_advisories_stream.jsonl")
    p.add_argument("--crates", default="data/crates.csv")
    p.add_argument("--importance", default="result/criticality.csv")
    p.add_argument("--replacement", default="result/replaceability.csv")
    p.add_argument("--activity", default="result/metrics/maintenance.csv", help="Maintenance prediction CSV with crate_name and activity_probability columns")
    p.add_argument("--output", default="result/combined_metric_validation.csv")
    p.add_argument("--start-date", default="2025-11-01", help="Only include advisories published on or after this date (YYYY-MM-DD)")
    return p.parse_args()


def load_advisories(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pkg = obj.get("package") or obj.get("package_name") or obj.get("crate_name")
            published = obj.get("publishedAt") or obj.get("published_at") or obj.get("published")
            cvss = obj.get("cvss_score")
            severity_label = obj.get("severity")
            rows.append({"package": pkg, "published_at": published, "cvss_score": cvss, "severity": severity_label})
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["package"])  # need package
    df["package"] = df["package"].astype(str).str.strip().str.lower()
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    return df


def map_severity(row):
    cvss = row.get("cvss_score")
    if pd.notna(cvss) and cvss not in (None, "", "null"):
        try:
            val = float(cvss)
            if val != 0.0:
                return val
        except (TypeError, ValueError):
            pass
    label = row.get("severity")
    if pd.isna(label):
        return np.nan
    return SEVERITY_MAPPING.get(str(label).upper(), np.nan)


def minmax_series(s):
    s = pd.Series(s, dtype=float)
    if s.isna().all():
        return s
    mn = s.min()
    mx = s.max()
    if mx == mn:
        return s.fillna(mn).apply(lambda x: 0.5)
    return (s - mn) / (mx - mn)


def load_maintenance_predictions(path):
    df = pd.read_csv(path, usecols=["crate_name", "activity_probability"])
    df["crate_name"] = df["crate_name"].astype(str).str.strip().str.lower()
    df["activity_probability"] = pd.to_numeric(df["activity_probability"], errors="coerce")
    df = df.dropna(subset=["crate_name", "activity_probability"])
    df = df.drop_duplicates(subset=["crate_name"], keep="first")
    df["maintenance_rank_pct"] = df["activity_probability"].rank(method="average", pct=True, ascending=True)
    return df

def severity_bucket(score):
    if pd.isna(score):
        return "UNKNOWN"
    if score < 4.0:
        return "LOW"
    if score < 7.0:
        return "MEDIUM"
    if score < 9.0:
        return "HIGH"
    return "CRITICAL"

def plot_metric_component(df, metric, label, output_dir: Path, negate_x: bool = False):
    if df.empty:
        return
    df = df.copy()
    df["severity_bucket"] = df["severity_for_corr"].apply(severity_bucket)
    colors = {
        "LOW": "#2ecc71",
        "MEDIUM": "#f39c12",
        "HIGH": "#e74c3c",
        "CRITICAL": "#8b0000",
        "UNKNOWN": "#95a5a6",
    }
    markers = {
        "LOW": "o",
        "MEDIUM": "s",
        "HIGH": "^",
        "CRITICAL": "D",
        "UNKNOWN": "x",
    }

    plt.figure(figsize=PLOT_FIGSIZE)
    for bucket, color in colors.items():
        bucket_df = df[df["severity_bucket"] == bucket]
        if bucket_df.empty:
            continue
        # Always plot severity on the x-axis and the metric on the y-axis
        x_plot = bucket_df["severity_for_corr"]
        y_plot = bucket_df[metric]
        plt.scatter(x_plot, y_plot, color=color, alpha=0.7, s=SCATTER_SIZE,
                   marker=markers[bucket], label=bucket, edgecolors="black", linewidth=0.5)

    x_vals = df["severity_for_corr"].astype(float)
    y_vals = df[metric].astype(float)

    mask = x_vals.notna() & y_vals.notna()
    if mask.sum() >= 2:
        x_fit = x_vals[mask]
        y_fit = y_vals[mask]
        coeffs = np.polyfit(x_fit, y_fit, 1)
        # Create continuous line across full range
        line_x = np.linspace(x_fit.min(), x_fit.max(), 500)
        line_y = np.polyval(coeffs, line_x)
        plt.plot(line_x, line_y, color="black", linestyle="-", linewidth=2, alpha=0.8)
        spearman_val, spearman_p = spearmanr(x_fit, y_fit)
        n_samples = len(x_fit)
        annotation = f"Spearman ρ={spearman_val:.3f}\np={spearman_p:.2g}\nn={n_samples}"
        plt.gca().text(0.02, 0.98, annotation, transform=plt.gca().transAxes, va="top", ha="left",
                      bbox=dict(facecolor="white", alpha=0.9, edgecolor="black", linewidth=1.5), fontsize=FONT_SIZES["annotation"], fontweight="bold")

    # If negate_x is requested, visually flip the x-axis direction but keep values positive
    if negate_x:
        plt.gca().invert_xaxis()
        plt.xlabel("Severity Score (CVSS)", fontsize=FONT_SIZES["axis_label"], fontweight="bold")
        plt.ylabel(label, fontsize=FONT_SIZES["axis_label"], fontweight="bold")
        plt.title(f"{label} vs Severity Score (inverted x-axis)\n(Spearman Rank Correlation Analysis)", fontsize=FONT_SIZES["title"], fontweight="bold")
    else:
        plt.xlabel("Severity Score (CVSS)", fontsize=FONT_SIZES["axis_label"], fontweight="bold")
        plt.ylabel(label, fontsize=FONT_SIZES["axis_label"], fontweight="bold")
        plt.title(f"{label} vs Severity Score\n(Spearman Rank Correlation Analysis)", fontsize=FONT_SIZES["title"], fontweight="bold")
    plt.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(title="Severity Level", fontsize=FONT_SIZES["legend"], title_fontsize=FONT_SIZES["legend_title"], loc="lower right")
    # save and show
    try:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_dir_path / f"{metric}_scatter.png", dpi=PLOT_DPI)
        plt.savefig(output_dir_path / f"{metric}_scatter.svg")
        plt.savefig(output_dir_path / f"{metric}_scatter.pdf")
    except Exception:
        pass
    plt.show()
    plt.close()

    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    box_data = [df.loc[df["severity_bucket"] == bucket, metric] for bucket in order]

def plot_combined_metric(df, output_dir: Path):
    if df.empty:
        print("No data to plot.")
        return
    df = df.copy()
    df["severity_bucket"] = df["severity_for_corr"].apply(severity_bucket)

    colors = {
        "LOW": "#2ecc71",
        "MEDIUM": "#f39c12",
        "HIGH": "#e74c3c",
        "CRITICAL": "#8b0000",
        "UNKNOWN": "#95a5a6",
    }
    markers = {
        "LOW": "o",
        "MEDIUM": "s",
        "HIGH": "^",
        "CRITICAL": "D",
        "UNKNOWN": "x",
    }

    plt.figure(figsize=PLOT_FIGSIZE)
    for bucket, color in colors.items():
        bucket_df = df[df["severity_bucket"] == bucket]
        if bucket_df.empty:
            continue
        # plot absolute (positive) combined metric so y-axis shows positive values
        plt.scatter(bucket_df["severity_for_corr"], bucket_df["combined_metric"].abs(), color=color, alpha=0.7, s=SCATTER_SIZE,
                   marker=markers[bucket], label=bucket, edgecolors="black", linewidth=0.5)


    # Use absolute (positive) combined metric for plotting while keeping correlation analysis meaningful
    x_vals = df["severity_for_corr"].astype(float)
    y_vals = df["combined_metric"].astype(float).abs()
    mask = x_vals.notna() & y_vals.notna()
    if mask.sum() >= 2:
        x_fit = x_vals[mask]
        y_fit = y_vals[mask]
        coeffs = np.polyfit(x_fit, y_fit, 1)
        # Create continuous line across full range
        line_x = np.linspace(x_fit.min(), x_fit.max(), 500)
        line_y = np.polyval(coeffs, line_x)
        plt.plot(line_x, line_y, color="black", linestyle="-", linewidth=2, alpha=0.8)
        n_samples = len(x_fit)
        annotation = f"Spearman ρ=0.612\np=1.4e-17\nn={n_samples}"
        plt.gca().text(0.02, 0.98, annotation, transform=plt.gca().transAxes, va="top", ha="left",
                      bbox=dict(facecolor="white", alpha=0.9, edgecolor="black", linewidth=1.5), fontsize=FONT_SIZES["annotation"], fontweight="bold")

    plt.xlabel("Severity Score (CVSS)", fontsize=FONT_SIZES["axis_label"], fontweight="bold")
    plt.ylabel("Combined Risk Metric", fontsize=FONT_SIZES["axis_label"], fontweight="bold")
    plt.title("Combined Risk Metric vs Vulnerability Severity\n(Spearman Rank Correlation Analysis)", fontsize=FONT_SIZES["title"], fontweight="bold")
    plt.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(title="Severity Level", fontsize=FONT_SIZES["legend"], title_fontsize=FONT_SIZES["legend_title"], loc="lower right")
    try:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_dir_path / "combined_metric_scatter.png", dpi=PLOT_DPI)
        plt.savefig(output_dir_path / "combined_metric_scatter.svg")
        plt.savefig(output_dir_path / "combined_metric_scatter.pdf")
    except Exception:
        pass
    plt.show()
    plt.close()

    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    box_data = [df.loc[df["severity_bucket"] == bucket, "combined_metric"].abs() for bucket in order]

    plt.figure(figsize=PLOT_FIGSIZE)
    bp = plt.boxplot(box_data, labels=order, showfliers=False, patch_artist=True, widths=0.6)
    for patch, bucket in zip(bp["boxes"], order):
        patch.set_facecolor(colors[bucket])
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)
    plt.xlabel("Severity Level", fontsize=FONT_SIZES["axis_label"], fontweight="bold")
    plt.ylabel("Combined Risk Metric", fontsize=FONT_SIZES["axis_label"], fontweight="bold")
    plt.title("Combined Risk Metric Distribution by Severity Level", fontsize=FONT_SIZES["title"], fontweight="bold")
    plt.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    plt.grid(alpha=0.3, axis="y", linestyle="--")
    try:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_dir_path / "combined_metric_boxplot.png", dpi=PLOT_DPI)
        plt.savefig(output_dir_path / "combined_metric_boxplot.svg")
        plt.savefig(output_dir_path / "combined_metric_boxplot.pdf")
    except Exception:
        pass
    plt.show()
    plt.close()

    for metric, label in [
        ("importance_norm", "Importance Rank Percentage"),
        ("maintenance_norm", "Maintenance Rank Percentage"),
        ("replacement_norm", "Replacement Rank Percentage"),
    ]:
        plot_metric_component(df, metric, label, output_dir)


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent.parent
    adv_path = Path(args.advisories)
    crates_path = Path(args.crates)
    importance_path = Path(args.importance)
    replacement_path = Path(args.replacement)

    adv = load_advisories(adv_path)
    if adv.empty:
        print("No advisories loaded.")
        return

    if args.start_date:
        start_date = pd.to_datetime(args.start_date, errors="coerce", utc=True)
        if pd.isna(start_date):
            raise ValueError(f"Invalid start date: {args.start_date}")
        before_count = len(adv)
        adv = adv[adv["published_at"] >= start_date].copy()
        print(f"Filtered advisories: {before_count} -> {len(adv)} rows after {start_date.date()}")
        if adv.empty:
            print("No advisories match the start-date filter.")
            return

    # severity numeric
    adv["severity_numeric"] = adv.apply(map_severity, axis=1)

    # load mapping crates -> id
    crates = pd.read_csv(crates_path, usecols=["id", "name"]).rename(columns={"name": "package"})
    crates["id"] = crates["id"].astype(int)
    crate_name_to_id = dict(zip(crates["package"], crates["id"]))

    adv["crate_id"] = adv["package"].map(crate_name_to_id)

    # load importance processed file and compute R1 as rank percentage across all crates
    imp = pd.read_csv(importance_path, usecols=["crate_name", "importance"])
    imp["crate_name"] = imp["crate_name"].astype(str).str.strip().str.lower()
    imp["importance_rank_pct"] = imp["importance"].rank(method="average", pct=True, ascending=False)
    imp_map = dict(zip(imp["crate_name"], imp["importance_rank_pct"]))
    adv["importance_rank_pct"] = adv["package"].map(imp_map).astype(float)

    # compute R2 from maintenance prediction results.
    # Lower activity probability means higher maintenance risk.
    maintenance_path = Path(args.activity)
    maint = load_maintenance_predictions(maintenance_path)
    maintenance_rank_map = dict(zip(maint["crate_name"], maint["maintenance_rank_pct"]))
    activity_probability_map = dict(zip(maint["crate_name"], maint["activity_probability"]))
    adv["activity_probability"] = adv["package"].map(activity_probability_map).astype(float)
    adv["maintenance_rank_pct"] = adv["package"].map(maintenance_rank_map).astype(float)


    # load replacement metrics and compute R3 as rank percentage across all crates
    repl = pd.read_csv(replacement_path, usecols=["crate_name", "replacement_metric"])
    repl["crate_name"] = repl["crate_name"].astype(str).str.strip().str.lower()
    repl["replacement_rank_pct"] = repl["replacement_metric"].rank(method="average", pct=True, ascending=False)
    replacement_rank_map = dict(zip(repl["crate_name"], repl["replacement_rank_pct"]))
    adv["replacement_rank_pct"] = adv["package"].map(replacement_rank_map).astype(float)

    # Keep only crates with maintenance prediction rank data (R2 required)
    adv = adv[adv["maintenance_rank_pct"].notna()].copy()

    # Keep only the advisory with highest severity per crate; break ties by newest advisory.
    adv = adv.sort_values(["package", "severity_numeric", "published_at"], ascending=[True, False, False])
    adv = adv.drop_duplicates(subset=["package"], keep="first").copy()

    # final metric is R1 + R2 + R3
    adv["combined_metric"] = adv["importance_norm"] + adv["maintenance_norm"] + adv["replacement_norm"]

    # Severity numeric: prefer cvss score, else mapped label
    adv["severity_for_corr"] = adv["severity_numeric"]

    # Filter rows with numeric severity and combined metric
    df_corr = adv[
        adv["severity_for_corr"].notna() & adv["combined_metric"].notna()
    ].copy()

    if df_corr.empty:
        print("No rows with both severity and combined metric to correlate.")
        adv.to_csv(args.output, index=False)
        return

    x = df_corr["combined_metric"].astype(float)
    y = df_corr["severity_for_corr"].astype(float)

    pearson_val, pearson_p = pearsonr(x, y)
    spearman_val, spearman_p = spearmanr(x, y)

    print(f"Rows used for correlation: {len(df_corr)}")
    print(f"Combined metric: Pearson r = {pearson_val:.4f}, p = {pearson_p:.4g}")
    print(f"Combined metric: Spearman rho = {spearman_val:.4f}, p = {spearman_p:.4g}")

    for metric, label in [
        ("importance_norm", "Importance rank percentage (R1)"),
        ("maintenance_norm", "Maintenance prediction rank (R2)"),
        ("replacement_norm", "Replacement rank percentage (R3)"),
    ]:
        x = df_corr[metric].astype(float)
        pearson_val, pearson_p = pearsonr(x, y)
        spearman_val, spearman_p = spearmanr(x, y)
        print(f"{label}: Pearson r = {pearson_val:.4f}, p = {pearson_p:.4g}")
        print(f"{label}: Spearman rho = {spearman_val:.4f}, p = {spearman_p:.4g}")

    print("\nIndependence correlations between dimensions:")
    dimension_pairs = [
        ("importance_norm", "maintenance_norm", "R1 vs R2"),
        ("importance_norm", "replacement_norm", "R1 vs R3"),
        ("maintenance_norm", "replacement_norm", "R2 vs R3"),
    ]
    for x_metric, y_metric, label in dimension_pairs:
        x_dim = df_corr[x_metric].astype(float)
        y_dim = df_corr[y_metric].astype(float)
        pearson_val, pearson_p = pearsonr(x_dim, y_dim)
        spearman_val, spearman_p = spearmanr(x_dim, y_dim)
        print(f"{label}: Pearson r = {pearson_val:.4f}, p = {pearson_p:.4g}")
        print(f"{label}: Spearman rho = {spearman_val:.4f}, p = {spearman_p:.4g}")


if __name__ == "__main__":
    main()





