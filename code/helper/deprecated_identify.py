"""
Identify deprecated crates and their recommended replacements using CloudGPT Azure OpenAI.

Reads crates_with_stars.csv, uses regex pre-filtering to find candidates that likely
contain deprecation signals, then calls the LLM to confirm and extract the replacement
crate name. Results are saved to result/deprecated_pairs.csv.

Supports checkpoint/resume: already-processed crate names are skipped on restart.

Usage:
    python src/deprecated_identify.py
"""

import csv
import re
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from cloudgpt_azure_openai import get_chat_completion

# -- Configuration
INPUT_CSV  = os.path.join(os.path.dirname(__file__), "..", "data-new", "crates_with_stars.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "result", "deprecated_pairs.csv")

ENGINE      = "gpt-4o-20241120"
MAX_WORKERS = 10
RETRY_TIMES = 3
RETRY_DELAY = 5

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

# -- Deprecation regex patterns (pre-filter before LLM call)
RE_DEPRECATED_PATTERNS = [
    r"\bdeprec(ated|ation|ate)\b",
    r"\bDeprecationWarning\b",
    r"\bno longer (maintained|supported|available|in use)\b",
    r"\bunmaintained\b",
    r"\bhas been deprecated\b",
    r"\breplaced (by|with)\b",
    r"\bsuperseded (by|with)\b",
    r"\bmoved to\b",
    r"\brecommend(ed)? (to use|using)\b",
    r"\bsuggest(ed)? (to use|using)\b",
    r"\bmaintenance mode\b",
    r"\barchived\b",
    r"\babandoned\b",
    r"\bno active development\b",
    r"\buse alternative\b",
    r"\bconsider switching to\b",
    r"\bwill not receive updates\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in RE_DEPRECATED_PATTERNS]


def extract_deprecated_snippet(text: str, window: int = 5):
    """Return a sentence window around the first deprecation signal, or None."""
    if not text:
        return None
    text = text.strip().replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for i, sentence in enumerate(sentences):
        if any(p.search(sentence) for p in _COMPILED):
            start = max(0, i - window)
            end   = min(len(sentences), i + window + 1)
            return " ".join(sentences[start:end]).strip()
    return None


# -- LLM prompt
SYSTEM_PROMPT = (
    "You are an assistant that identifies deprecated Rust crates and their replacements.\n"
    "Given a text snippet from a crate README or description, determine:\n"
    "1. Whether the crate is deprecated, abandoned, or replaced.\n"
    "2. If so, the exact name of the recommended replacement crate.\n\n"
    "Respond with ONLY one of:\n"
    "- The exact crate name of the replacement (e.g. tokio)\n"
    "- None if there is no clear replacement mentioned\n\n"
    "Do not include any explanation, punctuation, or extra text."
)


def call_llm(snippet: str) -> str:
    """Return replacement crate name, or 'None'."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f'Snippet:\n"""\n{snippet[:1000]}\n"""'},
    ]
    for attempt in range(1, RETRY_TIMES + 1):
        try:
            response = get_chat_completion(
                engine=ENGINE,
                messages=messages,
                use_azure_cli=True,
                temperature=0.0,
                max_tokens=50,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("LLM attempt %d/%d failed: %s", attempt, RETRY_TIMES, exc)
            if attempt < RETRY_TIMES:
                time.sleep(RETRY_DELAY)
    return "None"


# -- Resume support
def load_done_names(output_path: str) -> set:
    """Return set of crate names already written to the output CSV."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip()
            if name:
                done.add(name)
    logger.info("Resuming: %d crates already processed.", len(done))
    return done


# -- Main
def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    logger.info("Loading crate name index ...")
    all_pkg_names: set = set()
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip()
            if name:
                all_pkg_names.add(name.lower())

    done_names  = load_done_names(OUTPUT_CSV)
    file_exists = os.path.exists(OUTPUT_CSV)

    out_f  = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        out_f,
        fieldnames=["crate_id", "name", "github_owner", "github_repo", "replacement", "snippet"],
    )
    if not file_exists:
        writer.writeheader()
        out_f.flush()

    logger.info("Scanning for deprecation candidates ...")
    candidates = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip()
            if not name or name in done_names:
                continue
            combined = " ".join([
                row.get("description", "") or "",
                row.get("readme", "") or "",
            ])
            snippet = extract_deprecated_snippet(combined)
            if snippet:
                candidates.append({
                    "crate_id":     row.get("id", "").strip(),
                    "name":         name,
                    "github_owner": row.get("github_owner", "").strip(),
                    "github_repo":  row.get("github_repo",  "").strip(),
                    "snippet":      snippet,
                })

    total = len(candidates)
    logger.info("Candidates with deprecation signals: %d", total)

    if total == 0:
        logger.info("No new candidates to process.")
        out_f.close()
        return

    def handle(item):
        replacement = call_llm(item["snippet"])
        return item, replacement

    found = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(handle, item): item for item in candidates}
        with tqdm(total=total, desc="Identifying deprecated crates", unit="crate", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                item = futures[future]
                try:
                    item, replacement = future.result()
                    replacement = replacement.strip()
                    if replacement.lower() not in ("none", "") and replacement.lower() in all_pkg_names:
                        writer.writerow({
                            "crate_id":     item["crate_id"],
                            "name":         item["name"],
                            "github_owner": item["github_owner"],
                            "github_repo":  item["github_repo"],
                            "replacement":  replacement,
                            "snippet":      item["snippet"],
                        })
                        out_f.flush()
                        found += 1
                        pbar.set_postfix(found=found, last=f"{item['name']}->{replacement}")
                    else:
                        pbar.set_postfix(found=found, last=item["name"])
                except Exception as exc:
                    logger.error("Failed for %s: %s", item["name"], exc)
                    pbar.set_postfix(found=found, last=f"{item['name']} ERROR")
                finally:
                    pbar.update(1)

    out_f.close()
    logger.info("Done. %d replacement pairs saved to %s", found, OUTPUT_CSV)


if __name__ == "__main__":
    main()
