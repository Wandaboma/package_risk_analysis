#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kwargs):  # fallback: no-op wrapper
        return it

try:
    import networkx as nx
except Exception:
    nx = None


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
dump_dir = os.path.join(base_dir, "..", "data")
result_dir = os.path.join(base_dir, "..", "result")

default_crates_path = os.path.join(dump_dir, "crates.csv")
default_versions_path = os.path.join(dump_dir, "versions.csv")
default_deps_path = os.path.join(dump_dir, "dependencies.csv")
default_version_downloads_path = os.path.join(dump_dir, "version_downloads.csv")


# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
def load_crates(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["id", "name"])
    df["id"] = df["id"].astype(int)
    df["name"] = df["name"].astype(str)
    return df


def load_versions(path: str):
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(int)
    df["crate_id"] = df["crate_id"].astype(int)

    for c in ["updated_at", "created_at"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "updated_at" in df.columns and df["updated_at"].notna().any():
        idx = df.sort_values(["crate_id", "updated_at", "id"]).groupby("crate_id").tail(1)
    elif "created_at" in df.columns and df["created_at"].notna().any():
        idx = df.sort_values(["crate_id", "created_at", "id"]).groupby("crate_id").tail(1)
    else:
        idx = df.sort_values(["crate_id", "id"]).groupby("crate_id").tail(1)

    latest_version = dict(zip(idx["crate_id"], idx["id"]))        # crate_id -> latest version_id
    version_to_crate = dict(zip(df["id"], df["crate_id"]))        # version_id -> crate_id
    return latest_version, version_to_crate


def load_directed_edges(deps_path: str, crate_set: set, version_to_crate: dict, latest_version: dict) -> pd.DataFrame:
    deps = pd.read_csv(deps_path)

    deps = deps[deps["kind"].astype(str) == "0"].copy()
    deps["version_id"] = deps["version_id"].astype(int)
    deps["crate_id"] = deps["crate_id"].astype(int)  # target crate id

    deps["from_crate"] = deps["version_id"].map(version_to_crate)
    deps = deps.dropna(subset=["from_crate"])
    deps["from_crate"] = deps["from_crate"].astype(int)

    # FIX: drop rows where from_crate has no latest_version before comparing,
    # otherwise .map() returns NaN and the equality check silently drops them.
    deps["latest_ver"] = deps["from_crate"].map(latest_version)
    deps = deps.dropna(subset=["latest_ver"])
    deps["latest_ver"] = deps["latest_ver"].astype(int)
    deps = deps[deps["version_id"] == deps["latest_ver"]]
    deps.drop(columns=["latest_ver"], inplace=True)

    deps = deps[deps["from_crate"].isin(crate_set) & deps["crate_id"].isin(crate_set)]
    deps = deps[deps["from_crate"] != deps["crate_id"]]

    edges = deps[["from_crate", "crate_id"]].rename(columns={"crate_id": "to_crate"})
    edges.drop_duplicates(inplace=True)
    return edges


# ----------------------------------------------------------------------
# Build adjacency
# ----------------------------------------------------------------------
def build_directed_adj(n: int, edges: pd.DataFrame, id2idx: dict) -> csr_matrix:
    src = edges["from_crate"].map(id2idx).to_numpy()
    dst = edges["to_crate"].map(id2idx).to_numpy()
    data = np.ones(len(src), dtype=np.float64)
    A = csr_matrix((data, (src, dst)), shape=(n, n))
    return A


# ----------------------------------------------------------------------
# PageRank (sparse power iteration) + cache
# ----------------------------------------------------------------------
def pagerank_power_iteration_sparse(
    A: csr_matrix,
    alpha: float = 0.85,
    tol: float = 1e-10,
    max_iter: int = 200,
    verbose: bool = True
) -> np.ndarray:
    """
    Standard PageRank via power iteration.

    A[i, j] = 1 means edge i -> j.
    PageRank accumulates rank from *incoming* edges, so we need A^T:
        x_new = alpha * (A^T @ x_over_out + dangling_mass / n) + teleport
    """
    n = A.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)

    A = A.tocsr().astype(np.float64)
    # A^T so that multiplication gathers incoming contributions
    A_T = A.T.tocsr()

    out_deg = np.asarray(A.sum(axis=1)).ravel()
    dangling = (out_deg == 0)

    x = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - alpha) / n

    for it in range(max_iter):
        # FIX: must copy, otherwise x_prev is just an alias to x
        # and the convergence check would always see zero error.
        x_prev = x.copy()

        # Normalise outgoing rank
        x_over_out = np.zeros(n, dtype=np.float64)
        non_dangling = ~dangling
        if non_dangling.any():
            x_over_out[non_dangling] = x_prev[non_dangling] / out_deg[non_dangling]

        # FIX: use A^T so each node collects rank from its *in*-neighbors.
        # Before this was `x_over_out @ A`, which spreads rank *forward*
        # along edges (i.e. computes outgoing, not incoming contributions).
        contrib = A_T @ x_over_out

        dangling_mass = x_prev[dangling].sum() if dangling.any() else 0.0
        x = alpha * (contrib + dangling_mass / n) + teleport

        err = np.abs(x - x_prev).sum()
        if verbose and (it % 10 == 0 or it == max_iter - 1):
            print(f"PageRank iter={it:03d}, L1_error={err:.3e}")
        if err < tol:
            if verbose:
                print(f"PageRank converged at iter={it}, L1_error={err:.3e}")
            break

    s = x.sum()
    if s > 0:
        x /= s
    return x


def load_pagerank_cache(cache_path: str, expected_ids: list) -> dict:
    if not os.path.exists(cache_path):
        return {}

    try:
        df = pd.read_csv(cache_path)
        if "crate_id" not in df.columns or "pagerank" not in df.columns:
            return {}
        df["crate_id"] = df["crate_id"].astype(int)
        df["pagerank"] = df["pagerank"].astype(float)
        m = dict(zip(df["crate_id"].tolist(), df["pagerank"].tolist()))

        exp_set = set(expected_ids)
        got_set = set(m.keys())
        cover = len(exp_set & got_set) / max(1, len(exp_set))
        if cover < 0.98:
            print(f"PageRank cache coverage too low: {cover:.2%}. Recompute.")
            return {}
        return m
    except Exception as e:
        print(f"Failed to read pagerank cache: {e}. Recompute.")
        return {}


def save_pagerank_cache(cache_path: str, crate_ids: list, pr: np.ndarray):
    df = pd.DataFrame({"crate_id": crate_ids, "pagerank": pr.astype(np.float64)})
    df.to_csv(cache_path, index=False)


# ----------------------------------------------------------------------
# Downloads: recent window aggregation
# ----------------------------------------------------------------------
def load_recent_downloads(version_dl_path: str, version_id_to_crate_id: dict, window_days: int = 90) -> dict:
    print(f"Loading recent version downloads from {version_dl_path} ...")
    vdl = pd.read_csv(version_dl_path)

    required_cols = {"date", "downloads", "version_id"}
    missing = required_cols - set(vdl.columns)
    if missing:
        raise ValueError(f"version_downloads.csv missing required columns: {missing}")

    vdl["date"] = pd.to_datetime(vdl["date"], errors="coerce")
    vdl = vdl.dropna(subset=["date"])
    vdl["version_id"] = vdl["version_id"].astype(int)
    vdl["downloads"] = vdl["downloads"].astype(float)

    max_date = vdl["date"].max()
    cutoff = max_date - pd.Timedelta(days=window_days)
    vdl = vdl[vdl["date"] >= cutoff].copy()

    print(f"Using download window: {cutoff.date()} to {max_date.date()}")

    vdl["crate_id"] = vdl["version_id"].map(version_id_to_crate_id)
    vdl = vdl.dropna(subset=["crate_id"])
    vdl["crate_id"] = vdl["crate_id"].astype(int)

    crate_recent_downloads = vdl.groupby("crate_id")["downloads"].sum().to_dict()
    print(f"Computed recent downloads for {len(crate_recent_downloads)} crates")
    return crate_recent_downloads


# ----------------------------------------------------------------------
# Importance helpers (EXACT shortest path dist, only for topK)
# ----------------------------------------------------------------------
def minmax_norm_dict(values: dict) -> dict:
    arr = np.array(list(values.values()), dtype=np.float64)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return {k: 0.0 for k in values.keys()}
    return {k: float((v - mn) / (mx - mn)) for k, v in values.items()}


def combine_scores(
    norm_down: float,
    norm_up: float,
    method: str = "linear",
    lam: float = 0.5,
) -> float:
    """
    Combine normalized S_down and S_up into final importance.

    method:
      - linear:   lam*down + (1-lam)*up
      - geo:      geometric mean sqrt(down*up)
      - harmonic: harmonic mean 2*d*u/(d+u)
      - max:      max(down, up)
      - min:      min(down, up)
    """
    d = float(norm_down)
    u = float(norm_up)

    if method == "linear":
        lam = float(lam)
        return lam * d + (1.0 - lam) * u
    if method == "geo":
        return float(np.sqrt(max(d, 0.0) * max(u, 0.0)))
    if method == "harmonic":
        denom = d + u
        return 0.0 if denom <= 0 else float(2.0 * d * u / denom)
    if method == "max":
        return float(max(d, u))
    if method == "min":
        return float(min(d, u))
    lam = float(lam)
    return lam * d + (1.0 - lam) * u

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import numpy as np

def importance_for_top_nodes_shortest_path(
    A_wcc: csr_matrix,
    top_local_indices: list,
    w_vec: np.ndarray,
    out_deg_wcc: np.ndarray, 
    alpha_down: float,
    beta_up: float,
    max_dist_down: int,
    max_dist_up: int,
) -> tuple[dict, dict, dict]:
    """
    New metric definition (per your figure), with edge i->j meaning i depends on j:

    Node weight:
      w(v) = log(1 + downloads(v))  (or w(v)=1 if disabled)

    Downstream (dependents of v):
      down(v) = { u | (u -> v) ∈ E }  (incoming neighbors of v)
      S_down(v) = Σ_{u} w(u) * exp(-alpha_down * d(u, v)),
                  where 0 < d(u,v) <= max_dist_down

    Upstream (dependencies of v):
      up(v) = { u | (v -> u) ∈ E }  (outgoing neighbors of v)
      S_up(v) = Σ_{u} w(u) * exp(-beta_up * d(v, u)),
                where 0 < d(v,u) <= max_dist_up

    NOTE:
    - Upstream uses the same distance-decay formula as downstream (no in-degree normalization).
    - Return 3 dicts for backward compatibility:
        (sdown_map, sup_map, a_map) where a_map == sup_map.
    """
    n = A_wcc.shape[0]
    w_vec = np.asarray(w_vec, dtype=np.float64).reshape(-1)
    if w_vec.size != n:
        raise ValueError("w_vec size mismatch A_wcc")

    # Reverse graph: if A[i,j]=1 (i->j), then A_rev[j,i]=1
    A_rev = A_wcc.transpose().tocsr()

    sdown_map = {}
    sup_map = {}
    a_map = {}

    for v_idx in _tqdm(top_local_indices, desc="Computing importance", unit="node"):
        # (kept as-is) distances d(v, u) in original graph
        dist_up = shortest_path(
            A_wcc, directed=True, unweighted=True, indices=v_idx, return_predecessors=False
        )
        # (kept as-is) distances d(u, v) in original graph (via reversed graph)
        dist_down = shortest_path(
            A_rev, directed=True, unweighted=True, indices=v_idx, return_predecessors=False
        )

        mask_up = np.isfinite(dist_up) & (dist_up > 0) & (dist_up <= float(max_dist_up))
        mask_down = np.isfinite(dist_down) & (dist_down > 0) & (dist_down <= float(max_dist_down))

        # S_up(v) = sum w(u) * exp(-beta*d(v,u))  [same decay formula as S_down]
        if mask_up.any():
            d = dist_up[mask_up]
            sup_val = float(np.sum(w_vec[mask_up] * np.exp(-beta_up * d)))
        else:
            sup_val = 0.0

        # S_down(v) = sum w(u) * exp(-alpha*d(u,v))
        if mask_down.any():
            d2 = dist_down[mask_down]
            sdown_val = float(np.sum(w_vec[mask_down] * np.exp(-alpha_down * d2)))
        else:
            sdown_val = 0.0

        sdown_map[v_idx] = sdown_val
        sup_map[v_idx] = sup_val
        a_map[v_idx] = sup_val 

    return sdown_map, sup_map, a_map


def compute_single_crate_scores_by_name(
    crate_name: str,
    crates_df: pd.DataFrame,
    id2idx: dict,
    A_wcc: csr_matrix,
    w_vec: np.ndarray,
    out_deg_wcc: np.ndarray,
    alpha_down: float,
    beta_up: float,
    max_dist_down: int,
    max_dist_up: int,
) -> tuple[dict, str]:
    """
    Compute S_down and S_up for a single crate identified by name,
    and provide per-distance-layer statistics within max distance.

    Returns:
      ({crate_id, crate_name, S_down, S_up, up_layer_stats, down_layer_stats}, "") on success
      ({}, error_message) on failure
    """
    def _layer_stats(dist_arr: np.ndarray, w_arr: np.ndarray, max_dist: int) -> list:
        stats = []
        for d in range(1, int(max_dist) + 1):
            mask = np.isfinite(dist_arr) & (dist_arr == float(d))
            cnt = int(mask.sum())
            if cnt > 0:
                ws = w_arr[mask]
                total_w = float(np.sum(ws))
                avg_w = float(np.mean(ws))
            else:
                total_w = 0.0
                avg_w = 0.0
            stats.append({
                "distance": int(d),
                "count": cnt,
                "avg_weight": avg_w,
                "sum_weight": total_w,
            })
        return stats

    q = str(crate_name).strip()
    if not q:
        return {}, "crate name is empty"

    # Prefer exact match first; fallback to case-insensitive exact match.
    exact = crates_df[crates_df["name"] == q]
    if len(exact) == 1:
        row = exact.iloc[0]
    elif len(exact) > 1:
        ids = exact["id"].astype(int).tolist()[:10]
        return {}, f"multiple exact matches for '{q}', example crate_ids={ids}"
    else:
        lower = crates_df[crates_df["name"].str.lower() == q.lower()]
        if len(lower) == 0:
            return {}, f"crate '{q}' not found"
        if len(lower) > 1:
            names = lower["name"].astype(str).head(10).tolist()
            return {}, f"ambiguous case-insensitive matches for '{q}': {names}"
        row = lower.iloc[0]

    crate_id = int(row["id"])
    crate_name_resolved = str(row["name"])
    local_idx = id2idx.get(crate_id)
    if local_idx is None:
        return {}, f"crate id {crate_id} not found in graph index"

    # Distances for layer statistics:
    # - dist_up[k] = d(v, k) in original graph (dependencies direction)
    # - dist_down[k] = d(k, v) in original graph via reversed graph
    A_rev = A_wcc.transpose().tocsr()
    dist_up = shortest_path(
        A_wcc, directed=True, unweighted=True, indices=int(local_idx), return_predecessors=False
    )
    dist_down = shortest_path(
        A_rev, directed=True, unweighted=True, indices=int(local_idx), return_predecessors=False
    )

    up_layer_stats = _layer_stats(dist_up, np.asarray(w_vec, dtype=np.float64), max_dist_up)
    down_layer_stats = _layer_stats(dist_down, np.asarray(w_vec, dtype=np.float64), max_dist_down)

    sdown_map_local, sup_map_local, _ = importance_for_top_nodes_shortest_path(
        A_wcc=A_wcc,
        top_local_indices=[int(local_idx)],
        w_vec=w_vec,
        out_deg_wcc=out_deg_wcc,
        alpha_down=alpha_down,
        beta_up=beta_up,
        max_dist_down=max_dist_down,
        max_dist_up=max_dist_up,
    )

    result = {
        "crate_id": crate_id,
        "crate_name": crate_name_resolved,
        "S_down": float(sdown_map_local.get(local_idx, 0.0)),
        "S_up": float(sup_map_local.get(local_idx, 0.0)),
        "up_layer_stats": up_layer_stats,
        "down_layer_stats": down_layer_stats,
    }
    return result, ""


def corr_report(x: np.ndarray, y: np.ndarray, name_x: str, name_y: str) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return {
            "pair": f"{name_x} vs {name_y}",
            "n": int(x.size),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }
    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    return {
        "pair": f"{name_x} vs {name_y}",
        "n": int(x.size),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata


def spearman_rank_scatter(
    x: np.ndarray,
    y: np.ndarray,
    name_x: str,
    name_y: str,
    out_path: str,
    dpi: int = 150,
    labels: list = None,
    csv_path: str = None,
) -> dict:
    """
    Rank–Rank scatter plot for Spearman correlation.

    If points are too many, we automatically shrink marker size to make the plot clearer.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if x.size < 3:
        return {
            "pair": f"rank({name_x}) vs rank({name_y})",
            "n": int(x.size),
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }

    # ---- convert to ranks ----
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")

    # ---- spearman (rank-rank) ----
    sr = spearmanr(rx, ry)

    # ---- plot ----
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # auto marker size for large n
    n = int(rx.size)
    if n >= 200000:
        ms = 0.2
    elif n >= 100000:
        ms = 0.3
    elif n >= 50000:
        ms = 0.5
    elif n >= 20000:
        ms = 0.8
    elif n >= 10000:
        ms = 1.2
    elif n >= 5000:
        ms = 2.0
    else:
        ms = 6.0

    plt.figure(figsize=(6, 5))
    plt.scatter(rx, ry, alpha=0.35, s=ms)
    plt.xlabel(f"rank({name_x})")
    plt.ylabel(f"rank({name_y})")
    plt.title(f"Spearman rank correlation")

    text = (
        f"n = {rx.size}\n"
        f"Spearman r = {sr.statistic:.3f}\n"
        f"p = {sr.pvalue:.2e}"
    )
    plt.text(
        0.02,
        0.98,
        text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

    if labels is not None and csv_path is not None:
        labels_arr = np.asarray(labels)
        df_out = pd.DataFrame({
            "crate_name": labels_arr[m],
            f"{name_x}": x,
            f"{name_y}": y,
            f"rank_{name_x}": rx.astype(int),
            f"rank_{name_y}": ry.astype(int),
        }).sort_values(f"rank_{name_x}")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_out.to_csv(csv_path, index=False)

    return {
        "pair": f"rank({name_x}) vs rank({name_y})",
        "n": int(rx.size),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def plot_score_distributions(
    sdown_values: list,
    sup_values: list,
    out_dir: str,
    dpi: int = 150,
):
    """
    Save histograms of the S_down and S_up score distributions to <out_dir>/graph/.
    """
    graph_dir = os.path.join(out_dir, "graph")
    os.makedirs(graph_dir, exist_ok=True)

    for values, name in [(sdown_values, "S_down"), (sup_values, "S_up")]:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]

        plt.figure(figsize=(7, 4))
        plt.hist(arr, bins=80, edgecolor="none", alpha=0.8)
        plt.xlabel(name)
        plt.ylabel("Count")
        plt.title(f"Distribution of {name} (n={len(arr)})")
        plt.tight_layout()
        out_path = os.path.join(graph_dir, f"dist_{name}.png")
        plt.savefig(out_path, dpi=dpi)
        plt.close()
        print(f"Saved distribution plot: {out_path}")


# ----------------------------------------------------------------------
# Plot (label includes importance)
# ----------------------------------------------------------------------
def plot_topk_subgraph_with_pr_importance_pydot(
    edges_all_wcc: pd.DataFrame,
    id_to_name: dict,
    top_ids: list,
    indeg_map_wcc: dict,
    pr_map_global: dict,
    downloads_map: dict,
    importance_map: dict,
    out_path_png: str,
    title: str,
):
    if nx is None:
        raise RuntimeError("networkx is not installed. pip install networkx")

    # REQUIREMENT: if node number > 100, do not draw
    if len(top_ids) > 100:
        print(f"[SKIP] plot_topk_subgraph_with_pr_importance_pydot: node_count={len(top_ids)} > 100, skip drawing.")
        return

    top_set = set(top_ids)
    e = edges_all_wcc[
        edges_all_wcc["from_crate"].isin(top_set) &
        edges_all_wcc["to_crate"].isin(top_set)
    ].copy()

    G = nx.DiGraph()
    for cid in top_ids:
        G.add_node(cid)
    for r in e.itertuples(index=False):
        G.add_edge(int(r.from_crate), int(r.to_crate))

    layout_used = None
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
        layout_used = "dot (pydot)"
    except Exception as e1:
        print(f"Graphviz(pydot) layout failed: {e1}. Fallback to kamada_kawai_layout.")
        pos = nx.kamada_kawai_layout(G)
        layout_used = "kamada_kawai"

    # REQUIREMENT: node size by IMPORTANCE (not PageRank)
    imp_vals = np.array([float(importance_map.get(cid, 0.0)) for cid in G.nodes()], dtype=np.float64)
    imp_vals = np.where(np.isfinite(imp_vals), imp_vals, 0.0)
    if imp_vals.max() > 0:
        imp_scaled = np.sqrt(imp_vals / imp_vals.max())
    else:
        imp_scaled = np.ones_like(imp_vals)
    node_sizes = (700 + 4500 * imp_scaled).tolist()

    plt.figure(figsize=(24, 14))
    ax = plt.gca()
    ax.set_title(f"{title}\n(layout={layout_used})")

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.92, ax=ax)
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.2,
        alpha=0.75,
        connectionstyle="arc3,rad=0.10",
        ax=ax
    )

    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    xspan = (max(xs) - min(xs)) if xs else 1.0
    yspan = (max(ys) - min(ys)) if ys else 1.0
    dx = 0.012 * (xspan if xspan > 0 else 1.0)
    dy = 0.012 * (yspan if yspan > 0 else 1.0)

    for cid in G.nodes():
        name = id_to_name.get(cid, str(cid))
        indeg = int(indeg_map_wcc.get(cid, 0))
        pr = float(pr_map_global.get(cid, 0.0))
        dl = float(downloads_map.get(cid, 0.0))
        imp = float(importance_map.get(cid, 0.0))

        name_show = (name[:26] + "…") if len(name) > 26 else name

        label = (
            f"{name_show}\n"
            f"indeg={indeg}  pr={pr:.3e}\n"
            f"dl={dl:.0f}  imp={imp:.4f}"
        )
        x, y = pos[cid]
        ax.text(
            x + dx, y + dy, label,
            fontsize=9,
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.20", alpha=0.18, linewidth=0.6)
        )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=240)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crates", default=default_crates_path)
    parser.add_argument("--versions", default=default_versions_path)
    parser.add_argument("--deps", default=default_deps_path)

    parser.add_argument("--out-dir", default=result_dir)

    parser.add_argument("--pagerank-cache", default="pagerank_global.csv")
    parser.add_argument("--force-pagerank", action="store_true")
    parser.add_argument("--pagerank-alpha", type=float, default=0.85)
    parser.add_argument("--pagerank-tol", type=float, default=1e-10)
    parser.add_argument("--pagerank-max-iter", type=int, default=200)

    parser.add_argument("--version-downloads", default=None)
    parser.add_argument("--download-window-days", type=int, default=90)

    parser.add_argument("--alpha-down", type=float, default=0.7)
    parser.add_argument("--beta-up", type=float, default=0.7)
    parser.add_argument("--importance-lambda", type=float, default=0.8,
                        help="Weight for S_down when --importance-combine=linear (S_up weight = 1 - lambda). Default 0.8.")
    parser.add_argument("--importance-combine", type=str, default="linear",
                        choices=["linear", "geo", "harmonic", "max", "min"],
                        help="How to combine normalized S_down and S_up. Default: linear (0.8*down + 0.2*up).")
    parser.add_argument("--no-dist-plot", dest="dist_plot", action="store_false", default=True,
                        help="Disable distribution histograms for S_down and S_up.")
    parser.add_argument("--importance-unweighted", action="store_true",
                        help="Ignore node weights and treat all nodes as weight 1 (w(v)=1)")

    parser.add_argument("--max-dist-up", type=int, default=6,
                        help="Only consider upstream nodes with 0<dist(v,u)<=max_dist_up")
    parser.add_argument("--max-dist-down", type=int, default=6,
                        help="Only consider downstream nodes with 0<dist(u,v)<=max_dist_down")
    parser.add_argument("--crate-name", type=str, default="serde",
                        help="If provided, compute only this crate's S_down and S_up and print to terminal.")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    crates = load_crates(args.crates)
    latest_version, version_to_crate = load_versions(args.versions)

    crate_ids = crates["id"].tolist()
    crate_set = set(crate_ids)
    id2idx = {cid: i for i, cid in enumerate(crate_ids)}
    idx2id = {i: cid for cid, i in id2idx.items()}
    id_to_name = dict(zip(crates["id"].astype(int).tolist(), crates["name"].astype(str).tolist()))

    edges = load_directed_edges(args.deps, crate_set, version_to_crate, latest_version)
    print(f"Directed edges (latest only): {len(edges)}")

    A = build_directed_adj(len(crate_ids), edges, id2idx)
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    in_deg = np.asarray(A.sum(axis=0)).ravel()

    print("\n=== Global Directed Graph Stats ===")
    print(f"Nodes: {len(crate_ids)}")
    print(f"Edges: {A.nnz}")
    print(f"Avg out-degree: {out_deg.mean():.3f}")
    print(f"Avg in-degree: {in_deg.mean():.3f}")

    n_wcc, labels_wcc = connected_components(A, directed=True, connection="weak")
    sizes_wcc = np.bincount(labels_wcc)
    largest_wcc_label = int(np.argmax(sizes_wcc))
    print(f"\nWeakly Connected Components: {n_wcc}")
    print(f"Largest WCC label: {largest_wcc_label}, size: {int(sizes_wcc[largest_wcc_label])}")

    n_scc, labels_scc = connected_components(A, directed=True, connection="strong")
    print(f"Strongly Connected Components: {n_scc}")

    node_df = pd.DataFrame({
        "crate_id": crate_ids,
        "name": crates["name"].astype(str).tolist(),
        "in_degree": in_deg.astype(np.int64),
        "out_degree": out_deg.astype(np.int64),
        "wcc_label": labels_wcc.astype(np.int64),
        "scc_label": labels_scc.astype(np.int64),
    })
    node_df.to_csv(os.path.join(args.out_dir, "nodes_degrees_wcc_scc.csv"), index=False)

    # Single-crate mode: compute only S_down and S_up, then exit.
    if args.crate_name:
        vdl_path = args.version_downloads or default_version_downloads_path
        if not os.path.isabs(vdl_path):
            vdl_path = os.path.abspath(vdl_path)

        if os.path.exists(vdl_path):
            recent_downloads = load_recent_downloads(
                version_dl_path=vdl_path,
                version_id_to_crate_id=version_to_crate,
                window_days=args.download_window_days
            )
        else:
            print(f"[WARN] version_downloads.csv not found: {vdl_path}")
            print("[WARN] downloads treated as 0 => w(u)=0, scores may degenerate.")
            recent_downloads = {}

        downloads_arr = np.array([float(recent_downloads.get(cid, 0.0)) for cid in crate_ids], dtype=np.float64)
        if args.importance_unweighted:
            w_vec = np.ones_like(downloads_arr, dtype=np.float64)
        else:
            w_vec = np.log1p(downloads_arr)

        one_result, err = compute_single_crate_scores_by_name(
            crate_name=args.crate_name,
            crates_df=crates,
            id2idx=id2idx,
            A_wcc=A.tocsr(),
            w_vec=w_vec,
            out_deg_wcc=out_deg.astype(np.float64),
            alpha_down=args.alpha_down,
            beta_up=args.beta_up,
            max_dist_down=args.max_dist_down,
            max_dist_up=args.max_dist_up,
        )
        if err:
            print(f"[ERROR] {err}")
            return

        print("\n=== Single Crate Structural Scores ===")
        print(f"crate_name: {one_result['crate_name']}")
        print(f"crate_id: {one_result['crate_id']}")
        print(f"S_down: {one_result['S_down']:.12f}")
        print(f"S_up: {one_result['S_up']:.12f}")

        print("\n--- Upstream Layer Stats (distance d(v,u)) ---")
        print("distance\tcount\tavg_weight\tsum_weight")
        for row in one_result.get("up_layer_stats", []):
            print(
                f"{row['distance']}\t{row['count']}\t{row['avg_weight']:.6f}\t{row['sum_weight']:.6f}"
            )

        print("\n--- Downstream Layer Stats (distance d(u,v)) ---")
        print("distance\tcount\tavg_weight\tsum_weight")
        for row in one_result.get("down_layer_stats", []):
            print(
                f"{row['distance']}\t{row['count']}\t{row['avg_weight']:.6f}\t{row['sum_weight']:.6f}"
            )
        return

    # PageRank with cache
    cache_path = args.pagerank_cache
    if not os.path.isabs(cache_path):
        cache_path = os.path.join(args.out_dir, cache_path)

    pr_map = {}
    if not args.force_pagerank:
        pr_map = load_pagerank_cache(cache_path, expected_ids=crate_ids)

    if pr_map:
        print(f"\nLoaded PageRank from cache: {cache_path}")
        pr = np.array([pr_map.get(cid, 0.0) for cid in crate_ids], dtype=np.float64)
    else:
        print("\nComputing PageRank on the whole directed graph...")
        pr = pagerank_power_iteration_sparse(
            A,
            alpha=args.pagerank_alpha,
            tol=args.pagerank_tol,
            max_iter=args.pagerank_max_iter,
            verbose=True
        )
        save_pagerank_cache(cache_path, crate_ids, pr)
        pr_map = dict(zip(crate_ids, pr.tolist()))
        print(f"Saved PageRank cache to: {cache_path}")

    pd.DataFrame({"crate_id": crate_ids, "pagerank": pr}).to_csv(
        os.path.join(args.out_dir, "pagerank_global.csv"),
        index=False
    )

    # All nodes (global graph — not restricted to largest WCC)
    group_ids = crate_ids
    group_set = crate_set

    wcc_edges = edges.copy()
    indeg_wcc_map = wcc_edges["to_crate"].value_counts().to_dict()
    for cid in group_ids:
        indeg_wcc_map.setdefault(cid, 0)

    A_wcc = A.tocsr()
    out_deg_wcc = out_deg.astype(np.float64)

    group_df = pd.DataFrame({
        "crate_id": group_ids,
        "name": [id_to_name.get(cid, str(cid)) for cid in group_ids],
        "in_degree_wcc": [int(indeg_wcc_map.get(cid, 0)) for cid in group_ids],
        "pagerank": [float(pr_map.get(cid, 0.0)) for cid in group_ids],
    })

    active_df = group_df.copy()
    active_ids = active_df["crate_id"].astype(int).tolist()
    print(f"\nComputing importance for all {len(active_ids)} nodes.")

    # Downloads
    vdl_path = args.version_downloads or default_version_downloads_path
    if not os.path.isabs(vdl_path):
        vdl_path = os.path.abspath(vdl_path)

    if os.path.exists(vdl_path):
        recent_downloads = load_recent_downloads(
            version_dl_path=vdl_path,
            version_id_to_crate_id=version_to_crate,
            window_days=args.download_window_days
        )
    else:
        print(f"[WARN] version_downloads.csv not found: {vdl_path}")
        print("[WARN] downloads treated as 0 => w(u)=0, importance may degenerate.")
        recent_downloads = {}

    downloads_wcc = np.array([float(recent_downloads.get(cid, 0.0)) for cid in group_ids], dtype=np.float64)
    if args.importance_unweighted:
        w_vec = np.ones_like(downloads_wcc, dtype=np.float64)
    else:
        w_vec = np.log1p(downloads_wcc)

    wcc_id_to_local = {cid: i for i, cid in enumerate(group_ids)}
    active_local_indices = [wcc_id_to_local[cid] for cid in active_ids]

    # Importance (EXACT shortest path + threshold) — all nodes
    print(f"Computing importance for {len(active_ids)} nodes ...")
    sdown_map_local, sup_map_local, _ = importance_for_top_nodes_shortest_path(
        A_wcc=A_wcc,
        top_local_indices=active_local_indices,
        w_vec=w_vec,
        out_deg_wcc=out_deg_wcc,
        alpha_down=args.alpha_down,
        beta_up=args.beta_up,
        max_dist_down=args.max_dist_down,
        max_dist_up=args.max_dist_up,
    )

    sdown_all = {li: sdown_map_local[li] for li in active_local_indices}
    sup_all = {li: sup_map_local[li] for li in active_local_indices}
    norm_sdown = minmax_norm_dict(sdown_all)
    norm_sup = minmax_norm_dict(sup_all)

    lam = float(args.importance_lambda)
    comb_method = str(args.importance_combine).lower()
    importance_local = {
        li: combine_scores(norm_sdown[li], norm_sup[li], method=comb_method, lam=lam)
        for li in active_local_indices
    }

    sdown_map = {group_ids[li]: float(sdown_map_local[li]) for li in active_local_indices}
    sup_map = {group_ids[li]: float(sup_map_local[li]) for li in active_local_indices}
    importance_map = {group_ids[li]: float(importance_local[li]) for li in active_local_indices}

    # Distribution plots for S_down and S_up
    if args.dist_plot:
        plot_score_distributions(
            sdown_values=list(sdown_map.values()),
            sup_values=list(sup_map.values()),
            out_dir=args.out_dir,
        )

    # Save results for all nodes
    all_downloads = np.array([float(recent_downloads.get(cid, 0.0)) for cid in active_ids], dtype=np.float64)
    all_imp = np.array([float(importance_map.get(cid, 0.0)) for cid in active_ids], dtype=np.float64)
    all_sdown = np.array([sdown_map.get(cid, 0.0) for cid in active_ids], dtype=np.float64)
    all_sup = np.array([sup_map.get(cid, 0.0) for cid in active_ids], dtype=np.float64)

    active_df["downloads_window"] = all_downloads
    active_df["importance"] = all_imp
    active_df["S_down"] = all_sdown
    active_df["S_up"] = all_sup

    all_csv = os.path.join(args.out_dir, "all_crates_importance.csv")
    active_df.sort_values("importance", ascending=False).to_csv(all_csv, index=False)

    # Correlation (all nodes)
    all_pr = np.array([float(pr_map.get(cid, 0.0)) for cid in active_ids], dtype=np.float64)
    all_names = [id_to_name.get(cid, str(cid)) for cid in active_ids]

    corr_rows = []
    corr_rows.append(corr_report(all_imp, all_pr, "importance", "pagerank_global"))
    corr_rows.append(corr_report(all_imp, all_downloads, "importance", "downloads_window"))
    corr_rows.append(corr_report(all_imp, np.log1p(all_downloads), "importance", "log1p(downloads_window)"))
    corr_df = pd.DataFrame(corr_rows)

    corr_path = os.path.join(args.out_dir, "all_correlations.csv")
    corr_df.to_csv(corr_path, index=False)

    spearman_rank_scatter(
        all_imp,
        all_pr,
        name_x="importance",
        name_y="pagerank_global",
        out_path=os.path.join(args.out_dir, "rank_importance_vs_rank_pagerank_global.png"),
        labels=all_names,
        csv_path=os.path.join(args.out_dir, "rank_importance_vs_rank_pagerank_global.csv"),
    )

    spearman_rank_scatter(
        all_imp,
        all_downloads,
        name_x="importance",
        name_y="downloads_window",
        out_path=os.path.join(args.out_dir, "rank_importance_vs_rank_downloads_window.png"),
        labels=all_names,
        csv_path=os.path.join(args.out_dir, "rank_importance_vs_rank_downloads_window.csv"),
    )

    # Save simple metric CSV: crate name + importance score
    metric_csv = os.path.join(args.out_dir, "crate_importance_metric.csv")
    pd.DataFrame({
        "crate_name": [id_to_name.get(cid, str(cid)) for cid in active_ids],
        "importance": [float(importance_map.get(cid, 0.0)) for cid in active_ids],
    }).sort_values("importance", ascending=False).to_csv(metric_csv, index=False)
    print(f"Saved importance metric CSV: {metric_csv}")

    print("\n=== Correlation Results (all nodes) ===")
    print(corr_df.to_string(index=False))
    print(f"Saved correlation CSV: {corr_path}")

    summary = {
        "nodes": int(len(crate_ids)),
        "edges_latest_only": int(A.nnz),
        "wcc_count": int(n_wcc),
        "scc_count": int(n_scc),
        "biggest_wcc_label": int(largest_wcc_label),
        "biggest_wcc_size": int(sizes_wcc[largest_wcc_label]),
        "pagerank_cache": os.path.basename(cache_path),
        "all_csv": os.path.basename(all_csv),
        "corr_csv": os.path.basename(corr_path),
        "importance_params": {
            "alpha_down": float(args.alpha_down),
            "beta_up": float(args.beta_up),
            "lambda": float(args.importance_lambda),
            "combine": str(args.importance_combine),
            "unweighted": bool(args.importance_unweighted),
            "max_dist_up": int(args.max_dist_up),
            "max_dist_down": int(args.max_dist_down),
            "download_window_days": int(args.download_window_days),
            "w(u)": "1 (unweighted)" if args.importance_unweighted else "log(1+downloads(u))",
            "dist": "shortest path hop count (exact) with cutoff",
            "scope": "all nodes",
            "total_nodes": int(len(active_ids)),
        }
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(f"- {os.path.join(args.out_dir, 'nodes_degrees_wcc_scc.csv')}")
    print(f"- {os.path.join(args.out_dir, 'pagerank_global.csv')}")
    print(f"- {all_csv}")
    print(f"- {corr_path}")
    print(f"- {os.path.join(args.out_dir, 'summary.json')}")
    print(f"- {metric_csv}")


if __name__ == "__main__":
    main()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   