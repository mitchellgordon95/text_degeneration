# Testing Modern LLM Decoding Methods: Complete Experimental Framework

## Background
This project tests whether the findings from "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019) still apply to modern language models trained with RLHF and instruction-tuning.

### Original Paper's Key Findings (GPT-2, 2019)
1. **Beam search produces degenerate text**: 28.94% repetition rate
2. **Massive perplexity gap**: Generated text PPL=1.48 vs Human PPL=12.38
3. **Unreliable tail**: Low-probability tokens lead to incoherence
4. **Nucleus sampling (p=0.95) works best**: Balances quality and diversity

### Core Hypothesis
Modern LLMs (GPT-4, Claude-3.5, Llama-3) have fundamentally different probability distributions due to RLHF, potentially invalidating these findings.

---

## Final Experiment Plan

### Models to Test
- **Baseline**: GPT-2-large (exact model from paper)
- **Modern**: GPT-4, Claude-3.5, Llama-3-70B
- **Intermediate**: GPT-3.5-turbo (pre-heavy-RLHF)

### Decoding Methods
- **Greedy**: Always pick highest probability token
- **Beam Search**: k=5 and k=10
- **Nucleus (Top-p)**: p=0.95 (Holtzman's recommended value)

---

## Core Experiments

### 1. Beam Search Degeneration Test
**Question**: Does beam search still produce repetitive text?

```python
def test_degeneration(model, method, prompts):
    texts = generate(model, method, prompts[:200], max_length=256)

    metrics = {
        'repetition_rate': measure_4gram_repetition(texts),
        'self_bleu': calculate_self_bleu(texts),
        'distinct_1': len(unique_unigrams) / len(total_unigrams),
        'distinct_2': len(unique_bigrams) / len(total_bigrams)
    }
    return metrics
```

**Expected Results Table**:
| Model | Method | Repetition % | Self-BLEU | Distinct-1 | Distinct-2 |
|-------|--------|-------------|-----------|------------|------------|
| GPT-2 | Greedy | ~20% | High | Low | Low |
| GPT-2 | Beam-10 | ~29% | High | Low | Low |
| GPT-2 | Nucleus | ~0.4% | Low | High | High |
| GPT-4 | Greedy | ??? | ??? | ??? | ??? |
| GPT-4 | Beam-10 | ??? | ??? | ??? | ??? |
| GPT-4 | Nucleus | ??? | ??? | ??? | ??? |

**Success Criteria**: If GPT-4 beam search shows <10% repetition, problem largely solved

---

### 2. Perplexity Calibration Analysis
**Question**: Do models assign realistic probabilities to their own outputs?

```python
def test_perplexity_calibration(model):
    # Generate texts with different methods
    beam_texts = generate(model, "beam_10", n=500)
    nucleus_texts = generate(model, "nucleus_0.95", n=500)
    human_texts = load_human_continuations(n=500)

    # Measure model's perplexity on each
    results = {
        'beam_ppl': model.perplexity(beam_texts),
        'nucleus_ppl': model.perplexity(nucleus_texts),
        'human_ppl': model.perplexity(human_texts),
        'overconfidence_ratio': human_ppl / beam_ppl
    }
    return results
```

**Key Metric**: Overconfidence ratio (human_ppl / beam_ppl)
- GPT-2: 8.4x (very overconfident)
- Well-calibrated: <2x

---

### 3. Unreliable Tail Analysis
**Question**: Are low-probability tokens still problematic?

```python
def test_tail_reliability(model, prompt):
    probs = model.get_token_probabilities(prompt)

    # Split probability distribution
    regions = {
        'head': get_tokens_by_cumulative_prob(probs, 0, 0.1),      # Top 10%
        'nucleus': get_tokens_by_cumulative_prob(probs, 0, 0.95),  # Top 95%
        'tail': get_tokens_by_cumulative_prob(probs, 0.95, 1.0)    # Bottom 5%
    }

    # Sample from each region and measure coherence
    for region_name, tokens in regions.items():
        coherence_scores = []
        for _ in range(20):
            token = sample(tokens)
            continuation = model.generate(prompt + token, length=50)
            coherence_scores.append(measure_coherence(continuation))

        results[region_name] = mean(coherence_scores)
```

**Additional Test - Recovery Ability**:
```python
def test_recovery(model, prompt):
    # Force a bad token from tail
    bad_token = sample_from_tail(model, prompt)
    text = model.generate(prompt + bad_token, length=100)

    # Can model recover?
    early_coherence = coherence(text[:50])
    late_coherence = coherence(text[50:100])

    return late_coherence > early_coherence  # Did it improve?
```

---

### 4. Task-Specific Optimal Decoding
**Question**: Do different tasks need different strategies?

```python
tasks = {
    'creative': {
        'prompts': ["Once upon a time...", "In a world where..."],
        'metric': lambda x: diversity_score(x)
    },
    'factual': {
        'prompts': ["The capital of France is", "Water boils at"],
        'metric': lambda x: accuracy_score(x)
    },
    'code': {
        'prompts': ["def fibonacci(n):", "# Sort a list in Python"],
        'metric': lambda x: syntax_valid(x) and passes_tests(x)
    }
}

for task_name, task_config in tasks.items():
    for method in ['greedy', 'beam_10', 'nucleus_0.95']:
        outputs = generate(model, method, task_config['prompts'])
        score = task_config['metric'](outputs)
        results[task_name][method] = score
```

---

### 5. Beam Search Curse Deep Dive
**Question**: Does quality decrease with larger beam sizes?

```python
beam_sizes = [1, 2, 5, 10, 20, 50]

for size in beam_sizes:
    outputs = generate(model, f"beam_{size}", prompts)

    quality_metrics = {
        'repetition': measure_repetition(outputs),
        'perplexity': model.perplexity(outputs),
        'human_likeness': compare_to_human_distribution(outputs)
    }

    plot(size, quality_metrics)
```

**Expected Pattern**:
- GPT-2: Quality peaks around beam=5, then decreases
- GPT-4: Quality plateaus or continues improving?

---

## Implementation Details

### Prompt Dataset
```python
# Use same data as Holtzman for direct comparison
prompts = {
    'webtext': load_gpt2_webtext_validation()[:500],
    'news': load_cc_news_prompts()[:200],
    'stories': load_writing_prompts()[:200]
}
```

### Evaluation Metrics
```python
class Metrics:
    @staticmethod
    def repetition_rate(texts):
        """Percentage of text that is repeated 4-grams"""

    @staticmethod
    def self_bleu(texts):
        """Diversity between multiple generations"""

    @staticmethod
    def perplexity(model, texts):
        """Model's confidence in text"""

    @staticmethod
    def coherence(text):
        """Semantic coherence score using fine-tuned model"""
```

### Statistical Analysis
- Bootstrap confidence intervals for all metrics
- Paired t-tests for method comparisons
- Effect sizes (Cohen's d) to quantify improvements

---

## Execution Timeline

### Week 1: Setup & Core Testing
- **Day 1-2**: Setup infrastructure, implement metrics
- **Day 3**: Run degeneration test (Experiment 1)
- **Day 4**: Run perplexity analysis (Experiment 2)
- **Day 5**: Run tail analysis (Experiment 3)

### Week 2: Task-Specific & Analysis
- **Day 6-7**: Task-specific testing (Experiment 4)
- **Day 8**: Beam search curse (Experiment 5)
- **Day 9-10**: Analysis, visualization, report

---

## Resource Requirements

### Computational
- **API Costs**: ~$50 total
  - GPT-4: ~$20
  - Claude: ~$10
  - GPT-3.5: ~$5
  - Local models: Free
- **Storage**: ~10GB for outputs
- **GPU**: 8GB VRAM for local models

### Human Evaluation (Optional - Currently Tabled)
If findings are surprising, add:
- **Minimal**: $2,000 for basic quality ratings
- **Comprehensive**: $5,000 for multi-dimensional evaluation

---

## Expected Outcomes

### Most Likely
- Beam search repetition: ~10-15% (improved but not solved)
- Perplexity gap: ~3x (better calibrated)
- Tail more reliable but sampling still beneficial
- Task-specific differences less pronounced

### Would Be Surprising
- Beam search completely fixed (<5% repetition)
- Greedy produces creative text
- No task-specific differences
- Tail as reliable as head

### Would Change Everything
- Greedy optimal for all tasks
- Beam search curse reversed
- Temperature=0 more creative than sampling

---

## Key Deliverables

1. **Repetition Rate Comparison Table**: Shows improvement (or not) from GPT-2 to GPT-4
2. **Perplexity Calibration Chart**: Visual of overconfidence ratios
3. **Tail Reliability Heatmap**: Coherence by probability region and model
4. **Task-Specific Recommendations**: Which method for which use case
5. **Beam Search Curse Curves**: Quality vs beam size plots

---

## Code Structure

```
sampling_experiments/
├── experiments/
│   ├── degeneration.py      # Repetition analysis
│   ├── perplexity.py        # Calibration testing
│   ├── tail_analysis.py     # Reliability testing
│   ├── task_specific.py     # Task comparisons
│   └── beam_curse.py        # Beam size analysis
├── metrics/
│   ├── repetition.py        # N-gram overlap
│   ├── diversity.py         # Self-BLEU, distinct-n
│   ├── coherence.py         # Semantic coherence
│   └── perplexity.py        # Model confidence
├── models/
│   ├── model_wrapper.py     # Unified interface
│   ├── api_clients.py       # OpenAI, Anthropic
│   └── local_models.py      # HuggingFace models
├── data/
│   ├── prompts/             # Test prompts
│   └── outputs/             # Generated texts
├── analysis/
│   ├── statistics.py        # Significance tests
│   └── visualization.py     # Charts and plots
└── run_all.py               # Main experiment runner
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

# Run core experiments
python run_all.py --experiments core --models gpt2,gpt4,claude

# Analyze results
python analyze.py --input outputs/ --output results.md

# Generate plots
python visualize.py --data outputs/ --save plots/
```

---

## What We're NOT Doing (Explicitly Tabled)
- ❌ Optimal nucleus hyperparameter search
- ❌ RLHF impact analysis (too ambiguous as noted)
- ❌ Adaptive nucleus testing
- ❌ Contrastive decoding experiments
- ❌ Human detectability studies
- ❌ Human evaluation (unless results are very surprising)

---

## Decision Tree After Results

```mermaid
graph TD
    A[Run Core Experiments] --> B{Beam Search Fixed?}
    B -->|Yes <10% repetition| C[Write Paper: "Decoding Strategies Post-RLHF"]
    B -->|No >20% repetition| D[Investigate Why RLHF Didn't Help]
    B -->|Partial 10-20%| E[Add Human Evaluation]

    E --> F{Humans Agree?}
    F -->|Yes| G[Publish Findings]
    F -->|No| H[Investigate Automatic Metrics]

    D --> I[Test More Models]
    I --> J[Test Base vs RLHF Versions]
```

---

## Contact & Repository

- GitHub: [Will be created after experiments]
- Questions: [Your contact]
- Data: Will be released with paper