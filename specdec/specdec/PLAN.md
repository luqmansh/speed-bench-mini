# Speculative Decoding Plan

## Setup
- Draft model: Qwen3-0.6B (small, fast guesser)
- Target model: Qwen3-1.7B (the model being sped up)
- Both share the same tokenizer, which speculative decoding requires
- Greedy decoding, run on Google Colab (T4 GPU)

## How it works
1. Draft model guesses k tokens ahead
2. Target model checks all k guesses in one forward pass
3. Keep the guesses that match what the target would have picked, plus the target's own next token

## Correctness check
With greedy decoding, speculative output must be identical to normal target output.
If they differ, the implementation is wrong.

## Experiments
- Acceptance Length (AL) per category (paper's Table 1 trend)
- AL vs draft length k = 3 to 7 (paper's Figure 3)
