# Notes

## 2026-09-28
- Set up repo structure and data exploration script.
- 880 prompts, 11 categories x 80, 0 empty prompts.
- Multi-turn: 120 prompts (max 5 turns). Paper reports 167 (~20%). Discrepancy: 120 vs 167.
- Algorithm 1 (greedy + swap), k=20 of 80 per category, 20 seeds, bge-small-en-v1.5:
  18.1% avg similarity reduction vs random; wins in all 11 categories.
  Lowest: summarization (3.1%), likely shared instruction template. Highest: math (32.4%).
