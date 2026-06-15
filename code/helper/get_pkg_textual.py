"""
Read crates_with_stars.csv, call CloudGPT Azure OpenAI to conclude each crate's
functional identity, and save results to a CSV.
Usage:
    python src/get_pkg_textual.py
"""

import os
import csv
import sys
import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── CloudGPT helper ──────────────────────────────────────────────────────────
from cloudgpt_azure_openai import get_chat_completion

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_CSV  = os.path.join(os.path.dirname(__file__), "..", "data-new", "crates_with_stars.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "result", "crate_function_conclude.csv")

ENGINE           = "gpt-4o-20241120"  # change to another available CloudGPT model if needed
MAX_WORKERS      = 5             # parallel threads; lower if rate-limited
RETRY_TIMES      = 3             # retries on transient errors
RETRY_DELAY      = 5             # seconds between retries

csv.field_size_limit(50_000_000)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logs from the openai / httpx internals
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ── JSON field names (also become CSV column names) ──────────────────────────
JSON_FIELDS = [
    "functional_role",        # primary system problem this package solves
    "capability_set",         # comma-separated list of key functional abilities
    "api_intent",             # library | CLI | framework | runtime | SDK | middleware | tool
    "architectural_position", # infrastructure | integration | abstraction | adapter | orchestration
    "integration_pattern",    # embedded | standalone | dependency | extension | pipeline-component
    "key_inputs",             # what data / types / resources it consumes
    "key_outputs",            # what data / types / artifacts it produces
    "external_interfaces",    # external systems / protocols it touches
    "domain_tags",            # 3–6 domain keywords for similarity bucketing
]

# ── Prompt template ───────────────────────────────────────────────────────────
# Design rationale (see CrossSim TOSEM-2022, LibMigrate SANER-2021, CoCoSoDa ICSE-2023):
# Decomposing a package's identity into orthogonal semantic axes before embedding
# removes README noise (marketing, tutorials, install steps) and produces higher-
# precision cosine similarity for replaceability detection than raw-text embedding.
SYSTEM_PROMPT = """You are a software architecture analyst specialising in library replaceability research.

Your task is NOT to summarise a repository.
Your task is to extract its FUNCTIONAL IDENTITY — the minimal, noise-free semantic
descriptor that can be used for cross-package similarity comparison.

Strict filtering rules — completely IGNORE:
- marketing language, taglines, badges
- installation / build instructions
- tutorials, code examples, Quick-Start guides
- changelog, history, motivation sections
- community / contribution / license info
- anything not directly describing WHAT the package does to data or systems

Extract the following nine fields and return ONLY a single valid JSON object.
No markdown fences, no prose, no keys outside the schema.

Schema (all values must be concise strings — no nested objects, no arrays):
{
  "functional_role":        "<one sentence: what system problem this package solves>",
  "capability_set":         "<comma-separated list of 3-7 core functional abilities>",
  "api_intent":             "<exactly one of: library | CLI | framework | runtime | SDK | middleware | tool>",
  "architectural_position": "<exactly one of: infrastructure | integration | abstraction | adapter | orchestration | runtime | other>",
  "integration_pattern":    "<exactly one of: embedded | standalone | dependency | extension | pipeline-component>",
  "key_inputs":             "<comma-separated data types / resources consumed>",
  "key_outputs":            "<comma-separated data types / artifacts produced>",
  "external_interfaces":    "<comma-separated external systems, protocols, or none>",
  "domain_tags":            "<3-6 lowercase domain keywords suitable for similarity bucketing>"
}

Total output token budget: ≤150 tokens. Be maximally concise."""


def build_user_prompt(name: str, description: str, documentation: str,
                      readme: str = "", keywords: str = "") -> str:
    parts = [f"Crate name: {name}"]
    if description and description.strip():
        parts.append(f"Description: {description.strip()[:600]}")
    if keywords and keywords.strip():
        parts.append(f"Keywords: {keywords.strip()}")
    if documentation and documentation.strip():
        parts.append(f"Documentation URL: {documentation.strip()[:200]}")
    if readme and readme.strip():
        # Truncate readme heavily — we only want the first meaningful block
        parts.append(f"README (excerpt): {readme.strip()[:1200]}")
    parts.append("\nReturn only the JSON object as specified. No extra text.")
    return "\n".join(parts)


def parse_llm_json(raw: str) -> dict:
    """Parse LLM response as JSON; fall back to a single error field on failure."""
    # Strip accidental markdown fences
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        data = json.loads(text)
        # Ensure all expected fields are present
        return {f: str(data.get(f, "")).strip() for f in JSON_FIELDS}
    except json.JSONDecodeError:
        return {f: "PARSE_ERROR" for f in JSON_FIELDS} | {"functional_role": f"PARSE_ERROR: {text[:120]}"}


# ── LLM call with retry ───────────────────────────────────────────────────────
def call_llm(name: str, description: str, documentation: str,
             readme: str = "", keywords: str = "") -> dict:
    """Call the LLM and return a parsed dict with JSON_FIELDS keys."""
    user_msg = build_user_prompt(name, description, documentation, readme, keywords)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    for attempt in range(1, RETRY_TIMES + 1):
        try:
            response = get_chat_completion(
                engine=ENGINE,
                messages=messages,
                use_azure_cli=True,   # ← change auth method here if needed
                # use_broker_login=True,
                # use_managed_identity=True,
                # use_device_code=True,
                temperature=0.0,      # deterministic for structured extraction
                max_tokens=250,
            )
            raw = response.choices[0].message.content
            return parse_llm_json(raw)
        except Exception as exc:
            logger.warning("Attempt %d/%d failed for '%s': %s", attempt, RETRY_TIMES, name, exc)
            if attempt < RETRY_TIMES:
                time.sleep(RETRY_DELAY)
    return {f: "LLM_ERROR" for f in JSON_FIELDS}


# ── Keyword lookup ───────────────────────────────────────────────────────────
def build_keyword_lookup(data_dir: str) -> dict:
    """Return a dict mapping crate_id (str) -> comma-separated keyword string."""
    logger.info("Loading keywords lookup ...")
    kw_map: dict = {}  # keyword_id -> keyword text
    with open(os.path.join(data_dir, "keywords.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kw_map[row["id"]] = row["keyword"]

    crate_kws: dict = {}  # crate_id -> list of keyword strings
    with open(os.path.join(data_dir, "crates_keywords.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["crate_id"]
            kw  = kw_map.get(row["keyword_id"], "")
            if kw:
                crate_kws.setdefault(cid, []).append(kw)

    # Collapse to comma-separated strings
    result = {cid: ", ".join(kws) for cid, kws in crate_kws.items()}
    logger.info("Keywords loaded for %d crates.", len(result))
    return result


# ── Resume support ────────────────────────────────────────────────────────────
def load_done_keys(output_path: str) -> set:
    """Return a set of (github_owner, github_repo) already written to the output CSV."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            owner = row.get("github_owner", "").strip()
            repo  = row.get("github_repo", "").strip()
            if owner or repo:
                done.add((owner, repo))
    logger.info("Resuming: %d rows already done.", len(done))
    return done


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    data_dir    = os.path.join(os.path.dirname(__file__), "..", "data-new")
    keyword_lut = build_keyword_lookup(data_dir)

    done_keys   = load_done_keys(OUTPUT_CSV)
    file_exists = os.path.exists(OUTPUT_CSV)

    # Output columns: crate id + identity keys + one column per JSON field
    CSV_COLUMNS = ["crate_id", "github_owner", "github_repo"] + JSON_FIELDS

    # Open output CSV in append mode
    out_f  = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=CSV_COLUMNS)
    if not file_exists:
        writer.writeheader()
        out_f.flush()

    # Read input CSV and collect rows to process
    pending = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            owner = row.get("github_owner", "").strip()
            repo  = row.get("github_repo",  "").strip()
            if not owner or not repo:
                continue
            if (owner, repo) in done_keys:
                continue
            crate_id = row.get("id", "").strip()
            pending.append({
                "crate_id":      crate_id,
                "name":          row.get("name", "").strip(),
                "description":   row.get("description", "") or "",
                "documentation": row.get("documentation", "") or "",
                "readme":        row.get("readme", "") or "",
                "keywords":      keyword_lut.get(crate_id, ""),
                "github_owner":  owner,
                "github_repo":   repo,
            })

    total = len(pending)
    logger.info("Rows to process: %d", total)

    if total == 0:
        logger.info("All rows already processed. Nothing to do.")
        out_f.close()
        return

    processed = 0

    def handle(item):
        fields = call_llm(item["name"], item["description"], item["documentation"],
                          item["readme"], item["keywords"])
        return item["crate_id"], item["github_owner"], item["github_repo"], fields

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(handle, item): item for item in pending}
        with tqdm(total=total, desc="Processing crates", unit="crate", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                item = futures[future]
                try:
                    crate_id, owner, repo, fields = future.result()
                    row = {"crate_id": crate_id, "github_owner": owner, "github_repo": repo}
                    row.update(fields)
                    writer.writerow(row)
                    out_f.flush()
                    processed += 1
                    pbar.set_postfix(last=f"{owner}/{repo}")
                except Exception as exc:
                    logger.error(
                        "Failed for %s/%s: %s",
                        item["github_owner"], item["github_repo"], exc,
                    )
                    # Write an error placeholder so re-runs skip and don't retry broken rows
                    err_row = {"crate_id": item["crate_id"], "github_owner": item["github_owner"], "github_repo": item["github_repo"]}
                    err_row.update({f: f"ERROR: {exc}" if f == "functional_role" else "ERROR" for f in JSON_FIELDS})
                    writer.writerow(err_row)
                    out_f.flush()
                    pbar.set_postfix(last=f"{item['github_owner']}/{item['github_repo']} ERROR")
                finally:
                    pbar.update(1)

    out_f.close()
    logger.info("Done. Results saved to %s", OUTPUT_CSV)


if __name__ == "__main__":
    main()
