"""Plot Algorithm 1 vs random selection (our version of the paper's Figure 2)."""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/selection_results.csv")
df = df[df["category"] != "AVERAGE"].sort_values("reduction_%", ascending=False)

x = range(len(df))
width = 0.4
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar([i - width / 2 for i in x], df["random_k"], width, label="Random selection")
ax.bar([i + width / 2 for i in x], df["greedy_swap_k"], width, label="Greedy + swap (Algorithm 1)")

ax.set_xticks(list(x))
ax.set_xticklabels(df["category"], rotation=45, ha="right")
ax.set_ylabel("Avg pairwise cosine similarity\n(lower = more diverse)")
ax.set_title("Algorithm 1 vs random selection (k=20 of 80 prompts per category)")
ax.legend()
fig.tight_layout()
fig.savefig("results/selection_plot.png", dpi=150)
print("Saved results/selection_plot.png")