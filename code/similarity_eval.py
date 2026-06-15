"""
Crate Replacement Similarity Evaluation

Pipeline:
  Phase 1 - Build embedding index
    * Join crate_function_conclude.csv with crates_with_stars.csv to attach crate names
    * Serialize each crate into a text string (name + all 9 functional fields)
    * Batch-embed with CloudGPT text-embedding-3-large
    * Cache embeddings to result/embeddings_cache.npz (checkpoint/resume supported)

  Phase 2 - Evaluate on deprecated_pairs.csv
    * For each pair (deprecated -> replacement) where BOTH crates are in the index,
      compute cosine similarity ranking against ALL indexed crates
    * Report Hit@k (k=1,3,5,10,20) and MRR
    * Save per-pair results to result/similarity_eval_results.csv

Usage:
    python src/similarity_eval.py
"""

import os
import csv
import sys
import time
import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# from cloudgpt_azure_openai import get_openai_client

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / ".." / "data"
RESULT_DIR  = Path(__file__).parent / ".." / "result"

STARS_CSV      = DATA_DIR   / "crates.csv"
CONCLUDE_CSV   = DATA_DIR / "crate_function_conclude.csv"
PAIRS_CSV      = DATA_DIR / "deprecated_pairs.csv"
CACHE_NPZ      = DATA_DIR / "embeddings_cache.npz"
EVAL_OUT_CSV   = RESULT_DIR / "similarity_eval_results.csv"

EMBED_MODEL  = "text-embedding-3-large"
EMBED_DIM    = 3072        # dimensionality of text-embedding-3-large
BATCH_SIZE   = 64          # items per embedding API call
MAX_WORKERS  = 3           # parallel embedding batches
TOP_K_LIST   = [1, 3, 5, 10, 20, 50]

REPLACEMENT_OUT_CSV = RESULT_DIR / "crate_replacement_metric.csv"
REPLACEMENT_TOP_K   = 10   # nearest neighbours used to compute the metric

csv.field_size_limit(50_000_000)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ── Text serialisation ────────────────────────────────────────────────────────
FUNCTIONAL_FIELDS = [
    "functional_role",
    "capability_set",
    "api_intent",
    "architectural_position",
    "integration_pattern",
    "key_inputs",
    "key_outputs",
    "external_interfaces",
    "domain_tags",
]

def row_to_text(name: str, row: dict) -> str:
    """Serialize a crate's functional identity to a single embedding-ready string."""
    parts = [f"Crate: {name}"]
    labels = {
        "functional_role":        "Role",
        "capability_set":         "Capabilities",
        "api_intent":             "API",
        "architectural_position": "Position",
        "integration_pattern":    "Pattern",
        "key_inputs":             "Inputs",
        "key_outputs":            "Outputs",
        "external_interfaces":    "Interfaces",
        "domain_tags":            "Tags",
    }
    for f in FUNCTIONAL_FIELDS:
        v = row.get(f, "").strip()
        if v and v not in ("PARSE_ERROR", "LLM_ERROR", "ERROR"):
            parts.append(f"{labels[f]}: {v}")
    return ". ".join(parts)


# ── Embedding helpers ─────────────────────────────────────────────────────────
def embed_batch(client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts; returns list of float vectors."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    # resp.data is sorted by index
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


def cosine_similarity_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    query : (D,)   or (N, D)
    corpus: (M, D)
    returns (M,) or (N, M) cosine similarities
    """
    q = query / (np.linalg.norm(query, axis=-1, keepdims=True) + 1e-10)
    c = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
    return q @ c.T


# ── Replacement metric ───────────────────────────────────────────────────────
def compute_replacement_metrics(
    names: list,
    emb_matrix: np.ndarray,
    top_k: int = REPLACEMENT_TOP_K,
) -> dict:
    """
    Replacement metric for every crate.

    Definition
    ----------
    For crate c, compute cosine similarity to every other crate in the index,
    then average the top-K highest similarities (self excluded).

        replacement_metric(c) = mean_{u ∈ TopK_neighbours(c)} cos(emb_c, emb_u)

    Interpretation
    --------------
    * High score  →  many functionally similar crates exist  →  easy to replace.
    * Low score   →  the crate is functionally unique in the ecosystem  →  hard to replace.

    Parameters
    ----------
    names      : list of crate names, length N  (index-aligned with emb_matrix)
    emb_matrix : (N, D) float32 embeddings
    top_k      : number of nearest neighbours whose similarity is averaged

    Returns
    -------
    dict  crate_name -> float
    """
    N = emb_matrix.shape[0]
    k = min(top_k, N - 1)

    # L2-normalise once
    norms  = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-10
    normed = (emb_matrix / norms).astype(np.float32)

    metrics: dict = {}
    CHUNK = 512  # rows per batch — keeps peak memory bounded

    logger.info("Computing replacement metric (top-%d neighbours) for %d crates …", k, N)
    for start in range(0, N, CHUNK):
        end   = min(start + CHUNK, N)
        chunk = normed[start:end]        # (C, D)
        sims  = chunk @ normed.T         # (C, N)  — cosine similarities

        for local_i, global_i in enumerate(range(start, end)):
            row = sims[local_i].copy()
            row[global_i] = -2.0         # exclude self
            # np.partition gives the k largest values without full sort
            top_sims = np.partition(row, -k)[-k:]
            metrics[names[global_i]] = float(np.mean(top_sims))

    return metrics


# ── Phase 1: build / load embedding index ────────────────────────────────────
def build_or_load_index():
    """
    Returns:
        names      : list[str]   – crate names in index order
        crate_ids  : list[str]   – matching crate_ids
        emb_matrix : np.ndarray  – shape (N, EMBED_DIM)
        name_to_idx: dict        – name (lower) -> row index
    """
    # ── Step 1: build a name lookup from crates_with_stars ────────────────────
    logger.info("Loading crates_with_stars.csv ...")
    id_to_name: dict = {}
    with open(STARS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid  = row.get("id", "").strip()
            name = row.get("name", "").strip()
            if cid and name:
                id_to_name[cid] = name
    logger.info("  %d crates loaded.", len(id_to_name))

    # ── Step 2: load conclude rows, build text ────────────────────────────────
    logger.info("Loading crate_function_conclude.csv ...")
    records: list[dict] = []           # {crate_id, name, text}
    name_set: set = set()
    with open(CONCLUDE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid  = row.get("crate_id", "").strip()
            name = id_to_name.get(cid, "").strip()
            if not name:
                continue
            nl = name.lower()
            if nl in name_set:
                continue   # deduplicate
            name_set.add(nl)
            records.append({
                "crate_id": cid,
                "name":     name,
                "text":     row_to_text(name, row),
            })
    logger.info("  %d unique crates with functional descriptions.", len(records))

    # ── Step 3: load existing cache ───────────────────────────────────────────
    cached_names:  list = []
    cached_ids:    list = []
    cached_vecs:   list = []   # list of 1-D numpy arrays

    if CACHE_NPZ.exists():
        logger.info("Loading existing embedding cache from %s ...", CACHE_NPZ)
        data = np.load(str(CACHE_NPZ), allow_pickle=True)
        cached_names = data["names"].tolist()
        cached_ids   = data["crate_ids"].tolist()
        cached_vecs  = list(data["embeddings"])   # (N, D) -> list of rows
        logger.info("  %d embeddings already cached.", len(cached_names))

    done_names = set(cached_names)

    # ── Step 4: embed missing crates ─────────────────────────────────────────
    pending = [r for r in records if r["name"] not in done_names]
    logger.info("%d crates need embedding.", len(pending))

    if pending:
        # client = get_openai_client(use_azure_cli=True)
        # batches = [pending[i:i+BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]

        # def embed_one_batch(batch):
        #     texts = [r["text"] for r in batch]
        #     vecs  = embed_batch(client, texts)
        #     return batch, vecs

        # new_names: list = []
        # new_ids:   list = []
        # new_vecs:  list = []

        # SAVE_EVERY = 50   # save checkpoint every N batches
        # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        #     futures = {executor.submit(embed_one_batch, b): b for b in batches}
        #     with tqdm(total=len(batches), desc="Embedding batches", unit="batch", dynamic_ncols=True) as pbar:
        #         for i, future in enumerate(as_completed(futures)):
        #             try:
        #                 batch, vecs = future.result()
        #                 for r, v in zip(batch, vecs):
        #                     new_names.append(r["name"])
        #                     new_ids.append(r["crate_id"])
        #                     new_vecs.append(np.array(v, dtype=np.float32))
        #             except Exception as exc:
        #                 logger.error("Batch embedding failed: %s", exc)
        #             finally:
        #                 pbar.update(1)

        #             # Periodic checkpoint save
        #             if (i + 1) % SAVE_EVERY == 0 and new_vecs:
        #                 _save_cache(cached_names + new_names,
        #                             cached_ids   + new_ids,
        #                             cached_vecs  + new_vecs)
        #                 logger.info("  Checkpoint saved (%d total).", len(cached_vecs) + len(new_vecs))

        # # Final save
        # all_names = cached_names + new_names
        # all_ids   = cached_ids   + new_ids
        # all_vecs  = cached_vecs  + new_vecs
        # _save_cache(all_names, all_ids, all_vecs)
        # logger.info("Embedding index saved: %d entries.", len(all_names))
        pass
    # else:
    all_names = cached_names
    all_ids   = cached_ids
    all_vecs  = cached_vecs

    emb_matrix = np.stack(all_vecs, axis=0).astype(np.float32)   # (N, D)
    name_to_idx = {n.lower(): i for i, n in enumerate(all_names)}
    return all_names, all_ids, emb_matrix, name_to_idx


def _save_cache(names, crate_ids, vecs):
    CACHE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    mat = np.stack(vecs, axis=0).astype(np.float32)
    np.savez_compressed(
        str(CACHE_NPZ),
        names=np.array(names),
        crate_ids=np.array(crate_ids),
        embeddings=mat,
    )


# ── Phase 2: evaluation ───────────────────────────────────────────────────────
def evaluate(names, emb_matrix, name_to_idx):
    logger.info("Loading deprecated_pairs.csv ...")
    pairs = []
    with open(PAIRS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dep         = row.get("name", "").strip()
            replacement = row.get("replacement", "").strip()
            if not dep or not replacement:
                continue
            if dep.lower() == replacement.lower():
                continue   # self-referential, skip
            if dep.lower() in name_to_idx and replacement.lower() in name_to_idx:
                pairs.append({
                    "deprecated":   dep,
                    "replacement":  replacement,
                    "crate_id":     row.get("crate_id", ""),
                    "github_owner": row.get("github_owner", ""),
                    "github_repo":  row.get("github_repo", ""),
                })

    logger.info("Valid evaluation pairs (both crates in index): %d", len(pairs))
    if not pairs:
        logger.warning("No valid pairs found — nothing to evaluate.")
        return

    # Normalise the full matrix once
    norms  = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-10
    normed = emb_matrix / norms   # (N, D)

    hits     = {k: 0 for k in TOP_K_LIST}
    rr_total = 0.0
    results  = []

    for pair in tqdm(pairs, desc="Evaluating pairs", unit="pair", dynamic_ncols=True):
        dep_idx  = name_to_idx[pair["deprecated"].lower()]
        rep_idx  = name_to_idx[pair["replacement"].lower()]

        query  = normed[dep_idx]                   # (D,)
        scores = normed @ query                    # (N,) cosine sims

        # Exclude the query crate itself from ranking
        scores[dep_idx] = -2.0

        ranked = np.argsort(-scores)               # descending
        rank   = int(np.where(ranked == rep_idx)[0][0]) + 1   # 1-based

        rr_total += 1.0 / rank
        for k in TOP_K_LIST:
            if rank <= k:
                hits[k] += 1

        top5_names = [names[i] for i in ranked[:5]]
        results.append({
            "deprecated":        pair["deprecated"],
            "replacement":       pair["replacement"],
            "crate_id":          pair["crate_id"],
            "github_owner":      pair["github_owner"],
            "github_repo":       pair["github_repo"],
            "rank_of_replacement": rank,
            "top5_recommendations": ", ".join(top5_names),
            **{f"hit@{k}": int(rank <= k) for k in TOP_K_LIST},
        })

    n = len(pairs)
    mrr = rr_total / n
    logger.info("── Evaluation Results ─────────────────────────────")
    logger.info("  Total pairs evaluated: %d", n)
    for k in TOP_K_LIST:
        logger.info("  Hit@%-3d : %.4f  (%d/%d)", k, hits[k]/n, hits[k], n)
    logger.info("  MRR     : %.4f", mrr)
    logger.info("───────────────────────────────────────────────────")

    # Save per-pair results
    fieldnames = [
        "deprecated", "replacement", "crate_id", "github_owner", "github_repo",
        "rank_of_replacement", "top5_recommendations",
        *[f"hit@{k}" for k in TOP_K_LIST],
    ]
    with open(EVAL_OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Save summary
    summary_path = RESULT_DIR / "similarity_eval_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Model       : {EMBED_MODEL}\n")
        f.write(f"Index size  : {len(names)}\n")
        f.write(f"Pairs eval  : {n}\n")
        for k in TOP_K_LIST:
            f.write(f"Hit@{k:<3}    : {hits[k]/n:.4f}  ({hits[k]}/{n})\n")
        f.write(f"MRR         : {mrr:.4f}\n")

    logger.info("Per-pair results saved to %s", EVAL_OUT_CSV)
    logger.info("Summary saved to %s", summary_path)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    logger.info("=== Phase 1: Build embedding index ===")
    names, crate_ids, emb_matrix, name_to_idx = build_or_load_index()
    logger.info("Index ready: %d crates, embedding dim=%d", len(names), emb_matrix.shape[1])

    logger.info("=== Phase 2: Evaluate on deprecated_pairs ===")
    evaluate(names, emb_matrix, name_to_idx)

    logger.info("=== Phase 3: Compute & save replacement metrics ===")
    replacement_metrics = compute_replacement_metrics(names, emb_matrix, top_k=REPLACEMENT_TOP_K)

    # Save sorted by metric descending (most replaceable first)
    rows = sorted(replacement_metrics.items(), key=lambda x: x[1], reverse=True)
    REPLACEMENT_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(REPLACEMENT_OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["crate_name", "replacement_metric"])
        writer.writerows(rows)
    logger.info("Replacement metric CSV saved to %s  (%d crates)", REPLACEMENT_OUT_CSV, len(rows))


if __name__ == "__main__":
    main()
