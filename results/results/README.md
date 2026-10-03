# Results

## selection_results.csv
Algorithm 1 (greedy selection + swap refinement) vs random selection, per category.
- `all_80`: average pairwise similarity of all 80 prompts in the category
- `random_k`: average similarity of 20 randomly picked prompts (mean of 20 seeds)
- `greedy_swap_k`: average similarity of 20 prompts picked by Algorithm 1 (mean of 20 seeds)
- `reduction_%`: how much more diverse Algorithm 1's picks are vs random

## selection_plot.png
Bar chart of `random_k` vs `greedy_swap_k` per category. Lower = more diverse.
