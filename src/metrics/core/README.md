# Core Metrics from Holtzman et al. 2019

These are the core metrics from "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019).

## Implemented Metrics

### 1. Repetition Rate (`repetition.py`)
- Measures the percentage of text consisting of repeated n-grams
- Default: 4-gram repetition as used in the paper
- Key finding: GPT-2 with beam search showed ~29% repetition

### 2. Perplexity Gap (`perplexity.py`)
- Measures model overconfidence by comparing perplexity on generated vs human text
- Computes overconfidence ratio: human_ppl / generated_ppl
- Key finding: GPT-2 showed 8.4x overconfidence (very miscalibrated)

### 3. Self-BLEU (`self_bleu.py`)
- Measures diversity between multiple generations
- Lower Self-BLEU indicates more diverse outputs
- Used to quantify the diversity benefits of nucleus sampling

### 4. Zipf Coefficient (`zipf.py`)
- Analyzes word frequency distribution following Zipf's law
- Used in the paper to show that beam search produces unnatural distributions
- Human text typically has coefficient ~1.0-1.2
- Higher coefficients indicate less diverse vocabulary usage

## Not Implemented (Requires Human Evaluation)

### 5. HUSE (Human Unified with Statistical Evaluation)
- Requires human annotators to rate text quality
- Combined human judgments with automatic metrics in the original paper
- Cannot be automated

## Usage

These metrics are designed to evaluate the degeneration problem in neural text generation,
particularly comparing deterministic methods (greedy, beam search) with stochastic methods
(nucleus sampling, top-k).

## Reference

Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019).
The Curious Case of Neural Text Degeneration.
ICLR 2020.