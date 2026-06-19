import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_embeddings(embedding_path):
    embedding_data = {}
    with open(embedding_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading embeddings"):
            record = json.loads(line)
            embedding_data[record["package_name"]] = np.array(record["embedding"], dtype=np.float32)
    return embedding_data


def load_groundtruth(groundtruth_path):
    with open(groundtruth_path, "r", encoding="utf-8") as f:
        return json.load(f)  # List[Dict], each has "name" and "replacement"


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-10)


def evaluate_within_groundtruth_area(embedding_data, deduped_groundtruth):

    relevant_pkgs = set()
    for pair in deduped_groundtruth:
        relevant_pkgs.add(pair["name"])
        relevant_pkgs.add(pair["replacement"])

    filtered_names = [name for name in embedding_data if name in relevant_pkgs]
    filtered_embeddings = np.array([embedding_data[name] for name in filtered_names], dtype=np.float32)
    filtered_embeddings = normalize_embeddings(filtered_embeddings)

    hit_at_1 = hit_at_5 = hit_at_10 = hit_at_20 = hit_at_50 = hit_at_100 = mrr_total = total = 0

    for pair in tqdm(deduped_groundtruth, desc="Evaluating within groundtruth area"):
        dep_pkg = pair["name"]
        repl_pkg = pair["replacement"]

        if dep_pkg not in embedding_data or repl_pkg not in embedding_data:
            continue

        dep_vec = embedding_data[dep_pkg].astype(np.float32)
        dep_vec = dep_vec / np.maximum(np.linalg.norm(dep_vec), 1e-10)
        sims = np.dot(filtered_embeddings, dep_vec)

        top_k = 100
        top_k_idx = np.argpartition(-sims, top_k)[:top_k]
        ranked_idx = top_k_idx[np.argsort(-sims[top_k_idx])]

        total += 1
        for rank, idx in enumerate(ranked_idx):
            candidate = filtered_names[idx]
            if candidate == repl_pkg:
                if rank == 0: hit_at_1 += 1
                if rank < 5: hit_at_5 += 1
                if rank < 10: hit_at_10 += 1
                if rank < 20: hit_at_20 += 1
                if rank < 50: hit_at_50 += 1
                if rank < 100: hit_at_100 += 1
                mrr_total += 1 / (rank + 1)
                break

    print(f"Restricted to groundtruth-relevant packages ({len(filtered_names)} candidates)")
    print(f"Hit@1:  {hit_at_1 / total:.4f}")
    print(f"Hit@5:  {hit_at_5 / total:.4f}")
    print(f"Hit@10: {hit_at_10 / total:.4f}")
    print(f"Hit@20: {hit_at_20 / total:.4f}")
    print(f"Hit@50: {hit_at_50 / total:.4f}")
    print(f"Hit@100: {hit_at_100 / total:.4f}")
    print(f"MRR:    {mrr_total / total:.4f}")


def compute_substitutability(all_names, all_embeddings, output_csv_path, threshold=0.5):
    normalized_embeddings = normalize_embeddings(all_embeddings)
    substitute_counts = []

    for i in tqdm(range(len(normalized_embeddings)), desc="Computing substitutability"):
        vec = normalized_embeddings[i].reshape(1, -1)
        sims = np.dot(normalized_embeddings, vec.T).flatten()
        count = int(np.sum(sims > threshold)) - 1  # exclude self
        substitute_counts.append((all_names[i], count))

    df = pd.DataFrame(substitute_counts, columns=["package_name", "num_substitutes"])
    df.to_csv(output_csv_path, index=False)
    print(f"Substitutability results saved to: {output_csv_path}")

def compute_avg_top20_similarity(all_names, all_embeddings, output_csv_path, top_k=20):

    normalized_embeddings = normalize_embeddings(all_embeddings)
    avg_sim_scores = []

    for i in tqdm(range(len(normalized_embeddings)), desc="Computing avg top-20 similarity"):
        vec = normalized_embeddings[i].reshape(1, -1)
        sims = np.dot(normalized_embeddings, vec.T).flatten()

        sims[i] = -1.0
        top_k_sims = np.partition(sims, -top_k)[-top_k:]
        avg_sim = np.mean(top_k_sims)
        avg_sim_scores.append((all_names[i], avg_sim))
        # if i > 100: break

    df = pd.DataFrame(avg_sim_scores, columns=["package_name", "avg_top20_similarity"])
    df.to_csv(output_csv_path, index=False)
    print(f"Average top-{top_k} similarity results saved to: {output_csv_path}")

def show_substitutability_distribution(csv_path):
    df = pd.read_csv(csv_path)

    print("\nSubstitutability Distribution Summary:")
    print(df["num_substitutes"].describe())


    plt.figure(figsize=(10, 6))
    plt.hist(df["num_substitutes"], bins=50, edgecolor='black')
    plt.title("Distribution of Substitute Counts (cosine > 0.5)")
    plt.xlabel("Number of Substitutes")
    plt.ylabel("Number of Packages")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_substitute_score_log(df, column="num_substitutes"):
    log_scores = np.log1p(df[column])  # log(1 + x)
    return (log_scores - log_scores.min()) / (log_scores.max() - log_scores.min())


def calculate_substitute_score_linear(df, column="num_substitutes"):
    return (df[column] - df[column].min()) / (df[column].max() - df[column].min())


def calculate_substitute_score_percentile(df, column="num_substitutes"):
    return df[column].rank(pct=True)


def add_substitute_score(csv_path, method="log", output_path=None):
    df = pd.read_csv(csv_path)

    if method == "log":
        df["substitute_score"] = calculate_substitute_score_log(df)
    elif method == "linear":
        df["substitute_score"] = calculate_substitute_score_linear(df)
    elif method == "percentile":
        df["substitute_score"] = calculate_substitute_score_percentile(df)
    else:
        raise ValueError("Invalid method. Use 'log', 'linear', or 'percentile'.")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved scored CSV to: {output_path}")
    return df


def main():
    path = "db-dump/data/"
    embedding_path = path + "package_embeddings-co.json"
    groundtruth_path = path + "groundtruth_clean.json"
    output_csv_path = path + "package_substitutability.csv"

    embedding_data = load_embeddings(embedding_path)
    deduped_groundtruth = load_groundtruth(groundtruth_path)

    all_names = list(embedding_data.keys())
    all_embeddings = np.array([embedding_data[name] for name in all_names], dtype=np.float32)

    evaluate_within_groundtruth_area(embedding_data, deduped_groundtruth)

    compute_avg_top20_similarity(all_names, all_embeddings, output_csv_path)

    # show_substitutability_distribution(output_csv_path)


    # scored_csv_path = path + "package_substitutability_scored.csv"
    # add_substitute_score(output_csv_path, method="log", output_path=scored_csv_path)


if __name__ == "__main__":
    main()
