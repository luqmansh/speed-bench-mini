# SPEED-Bench Mini

A small-scale reproduction of **SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding** (NVIDIA, ICML 2026, [arXiv:2604.09557](https://arxiv.org/abs/2604.09557)).

> 🚧 Work in progress

## Progress
- [x] Load and validate the released benchmark (880 prompts, 11 categories)
- [x] Diversity selection: greedy selection + local swap refinement (Algorithm 1) vs random selection
- [ ] Speculative decoding from scratch (PyTorch, small open models)
- [ ] Acceptance length per category and vs draft length

## Results so far
**Algorithm 1 vs random selection:** k=20 of 80 prompts per category, 20 seeds, bge-small-en-v1.5 embeddings.
- **18.1% average reduction** in intra-category semantic similarity vs random sampling
- Greedy selection wins in **all 11 categories**
- Full table: `results/selection_results.csv`

**Dataset note:** the released split has 120 multi-turn prompts vs 167 reported in the paper.
