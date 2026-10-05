# Notes

## 2026-09-28
- Set up repo structure and data exploration script.
- 880 prompts, 11 categories x 80, 0 empty prompts.
- Multi-turn: 120 prompts (max 5 turns). Paper reports 167 (~20%). Discrepancy: 120 vs 167.
- Algorithm 1 (greedy + swap), k=20 of 80 per category, 20 seeds, bge-small-en-v1.5:
  18.1% avg similarity reduction vs random; wins in all 11 categories.
  Lowest: summarization (3.1%), likely shared instruction template. Highest: math (32.4%).
  - Raw similarity values (~0.7) are much higher than the paper's (~0.14) because
  bge-small uses a different similarity scale than OpenAI's embedder.
  Compare relative reduction, not raw values.

## 2026-10-01
- Added README with progress and Algorithm 1 results.
- Updated requirements.txt (sentence-transformers, matplotlib).
- Next: speculative decoding on Colab (Qwen3-0.6B draft, Qwen3-1.7B target).
## 2026-10-03
- Wrote specdec/PLAN.md (draft/target models, verification approach, correctness check).
- Documented results file columns.
## 2026-10-04
- Added reproduction steps to README and usage docs to diversity.py.