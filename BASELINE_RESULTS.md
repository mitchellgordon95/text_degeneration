# Baseline Results from Holtzman et al. 2019

## Original Paper Results (GPT-2 Large, 762M parameters)

This table presents the core results from "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019).
These serve as our baseline for comparison with modern models.

### Table 1: Decoding Methods Comparison

| Method | Self-BLEU4 ↓ | Repetition % ↓ | Perplexity | HUSE ↑ |
|--------|--------------|----------------|------------|--------|
| **Human** | 0.31 | 0.28 | 12.38 | - |
| **Greedy** | 0.50 | 73.66 | 1.50 | - |
| **Beam Search (b=16)** | 0.44 | 28.94 | 1.48 | - |
| **Pure Sampling** | 0.28 | 0.22 | 22.73 | 0.67 |
| **Top-k (k=40)** | 0.39 | 0.78 | 6.88 | 0.19 |
| **Nucleus (p=0.95)** | 0.32 | 0.36 | 13.13 | 0.97 |

*↓ = lower is better, ↑ = higher is better*

## Metric Explanations

### Self-BLEU4
- Measures diversity by computing BLEU scores between generated texts
- Range: 0-1 (lower = more diverse)
- Human baseline: 0.31
- **Our implementation**: We compute this on 200 generated samples

### Repetition %
- Percentage of text consisting of repeated 4-grams
- Range: 0-100% (lower = less repetitive)
- Human baseline: 0.28%
- **Key finding**: Greedy showed 73.66% repetition!

### Perplexity
- Model's confidence in the generated text
- Lower = more confident
- Human text perplexity: 12.38
- Generated text perplexity: ~1.5 (8.4x overconfidence!)
- **Note**: We measure perplexity gap (human_ppl / generated_ppl)

### Zipf Coefficient
- Word frequency distribution following Zipf's law
- Human text: typically ~1.0-1.2
- Higher = less diverse vocabulary
- **Note**: Not shown in original table but discussed in paper

### HUSE (Human Unified with Statistical Evaluation)
- Requires human evaluation
- Range: 0-1 (higher = better quality)
- **We do not implement this** (requires human annotators)

## Our Reproduction Goals

### Models to Test
We aim to reproduce these metrics for modern models:

**Baseline**:
- GPT-2 Large (same as paper)

**Modern LLMs**:
- GPT-4, GPT-5
- Claude-3.5 Sonnet, Claude-4 Opus
- Llama-3 70B
- Mixtral 8x7B
- Qwen2.5 (7B, 72B)
- Mistral (7B, Small-3-24B)

### Research Questions

1. **Has the degeneration problem been solved?**
   - Do modern models still show high repetition with greedy/beam search?
   - Expected: Significant improvement but not completely solved

2. **Is nucleus sampling still necessary?**
   - Does greedy decoding work better in RLHF models?
   - Expected: Gap between methods has narrowed

3. **How has the perplexity gap changed?**
   - Are modern models better calibrated?
   - Expected: Reduced overconfidence (closer to 1x ratio)

4. **Do different architectures behave differently?**
   - Compare standard vs MoE architectures
   - Compare base vs instruction-tuned models

## Success Criteria

We consider the degeneration problem "largely solved" if modern models show:
- Repetition rate < 10% with greedy decoding (vs 73.66% in GPT-2)
- Self-BLEU4 < 0.4 with greedy (vs 0.50 in GPT-2)
- Perplexity gap < 3x (vs 8.4x in GPT-2)

## Implementation Notes

- We use 200 samples per model/method combination
- All metrics implemented in `src/metrics/core/`
- Experiments configured in `config/experiments.yaml`
- Results will be compared directly to this baseline table