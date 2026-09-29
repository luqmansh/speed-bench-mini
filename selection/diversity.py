"""Reproduce Algorithm 1 (greedy selection + local swap refinement) vs random selection."""
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

K = 20            # prompts to select per category
N_SEEDS = 20      # random baselines / greedy restarts to average over
MODEL = "BAAI/bge-small-en-v1.5"


def avg_pairwise_sim(X):
    """Mean cosine similarity over all distinct pairs (rows of X are unit vectors)."""
    k = len(X)
    G = X @ X.T
    return (G.sum() - np.trace(G)) / (k * (k - 1))


def greedy_swap(G, k, seed, max_iter=1000):
    """Algorithm 1 from the paper. G is the full cosine-similarity matrix."""
    rng = np.random.default_rng(seed)
    N = len(G)
    first = int(rng.integers(N))
    S = [first]
    in_S = np.zeros(N, dtype=bool)
    in_S[first] = True
    m = G[:, first].copy()  # m[j] = total similarity of j to everything in S

    # Greedy phase: repeatedly add the candidate least similar to the current set
    while len(S) < k:
        cand = np.where(in_S, np.inf, m)
        i = int(np.argmin(cand))
        S.append(i)
        in_S[i] = True
        m += G[:, i]

    # Local swap refinement: swap in/out pairs while it lowers total similarity
    for _ in range(max_iter):
        outside = np.where(~in_S)[0]
        best_delta, best = 0.0, None
        for o in S:
            # delta = sum_{j in S, j != o} (sim(new, j) - sim(o, j))
            deltas = (m[outside] - G[outside, o]) - (m[o] - G[o, o])
            idx = int(np.argmin(deltas))
            if deltas[idx] < best_delta:
                best_delta, best = deltas[idx], (o, int(outside[idx]))
        if best is None:
            break
        o, new = best
        S.remove(o)
        S.append(new)
        in_S[o], in_S[new] = False, True
        m += G[:, new] - G[:, o]
    return S


def main():
    df = load_dataset("nvidia/SPEED-Bench", "qualitative")["test"].to_pandas()
    df["text"] = df["turns"].apply(lambda t: "\n".join(str(x) for x in t))

    print(f"Embedding {len(df)} prompts with {MODEL}...")
    model = SentenceTransformer(MODEL)
    emb = model.encode(df["text"].tolist(), normalize_embeddings=True,
                       show_progress_bar=True, batch_size=32)

    rows = []
    for cat in sorted(df["category"].unique()):
        X = emb[(df["category"] == cat).to_numpy()]
        G = X @ X.T

        rand = [avg_pairwise_sim(X[np.random.default_rng(s).choice(len(X), K, replace=False)])
                for s in range(N_SEEDS)]
        greedy = [avg_pairwise_sim(X[greedy_swap(G, K, s)]) for s in range(N_SEEDS)]

        r, g = float(np.mean(rand)), float(np.mean(greedy))
        rows.append({"category": cat,
                     "all_80": round(avg_pairwise_sim(X), 4),
                     "random_k": round(r, 4),
                     "greedy_swap_k": round(g, 4),
                     "reduction_%": round(100 * (r - g) / r, 1)})

    out = pd.DataFrame(rows)
    avg = {"category": "AVERAGE",
           "all_80": round(out["all_80"].mean(), 4),
           "random_k": round(out["random_k"].mean(), 4),
           "greedy_swap_k": round(out["greedy_swap_k"].mean(), 4),
           "reduction_%": round(100 * (out["random_k"].mean() - out["greedy_swap_k"].mean())
                                / out["random_k"].mean(), 1)}
    out = pd.concat([out, pd.DataFrame([avg])], ignore_index=True)

    print(out.to_string(index=False))
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/selection_results.csv", index=False)
    print("\nSaved results/selection_results.csv")


if __name__ == "__main__":
    main()
