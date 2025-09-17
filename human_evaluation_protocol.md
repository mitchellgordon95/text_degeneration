# Human Evaluation Protocol for Decoding Experiments

## Overview
Human evaluation protocol for validating automatic metrics findings. Currently tabled but preserved for potential future use.

---

## Background: What Holtzman Did (2019)

### Original Protocol
- **Scale**: 1,000 texts (200 per method × 5 methods)
- **Annotators**: 20 per text via MTurk
- **Question**: Single "typicality" rating (6-point scale)
- **Categories**: Very Typical → Typical → Average → Specific → Rare → Invalid
- **Cost**: Estimated $2-3K

### Limitations
- Single dimension only
- No inter-annotator agreement reported
- No quality control described
- No pairwise comparisons

---

## Recommended Protocol (If Needed)

### Option A: Minimal Validation ($2,000, 1 week)
**When to use**: Quick validation if automatic metrics show surprising results

```python
protocol = {
    "texts": 500,  # 100 per method
    "annotators_per_text": 3,
    "questions": ["Overall, is this good text? (1-5)"],
    "platform": "Surge AI or Scale AI"
}
```

**Deliverables**:
- Basic quality rankings
- Correlation with automatic metrics
- Validation of surprising findings

---

### Option B: Balanced Multi-Dimensional ($5,000, 3 weeks)
**When to use**: If you need actionable insights for deployment

#### Three-Stage Pipeline

**Stage 1: Automated Filtering (Free)**
```python
# Remove obviously broken outputs
for text in all_texts:
    if repetition_rate(text) > 0.5:
        exclude()
    if grammar_errors(text) > threshold:
        exclude()
```

**Stage 2: Crowd Evaluation ($3,000)**
```python
dimensions = {
    "fluency": "How natural is the language? (1-5)",
    "coherence": "Do ideas flow logically? (1-5)",
    "relevance": "Does it match the prompt? (1-5)",
    "interesting": "Would you keep reading? (1-5)"
}

# 2,000 texts × 3 annotators × 4 dimensions
```

**Stage 3: Expert Pairwise Comparisons ($2,000)**
```python
# Top 100 prompt sets
# Compare methods head-to-head
# Use Bradley-Terry model for global ranking
comparisons = ["Which text is better overall?",
               "A much better | A slightly better | Equal | B slightly better | B much better"]
```

---

## Quality Control Measures

### 1. Attention Checks
```python
attention_checks = [
    # Obvious quality difference
    ("The cat sat on the mat.", "The the the the the"),

    # Identical texts (should rate equal)
    ("Sample text", "Sample text"),

    # Clear repetition
    ("Normal text here", "Text text text text text")
]
```

### 2. Inter-Annotator Agreement
- Calculate Krippendorff's α
- Require α > 0.6 for acceptable agreement
- Re-annotate high disagreement cases

### 3. Annotator Qualification
- Native English speakers
- Pass 10-question qualification test
- Maintain >80% agreement with gold standards

---

## Task-Specific Evaluation Criteria

### Creative Writing
- Creativity/originality (1-5)
- Would you want to read more? (Y/N)
- Vocabulary diversity
- Plot coherence

### Factual QA
- Accuracy (correct/incorrect)
- Completeness (1-5)
- Trustworthiness (1-5)
- Any hallucinations? (Y/N)

### Code Generation
- Syntax valid? (Y/N)
- Readable/well-structured? (1-5)
- Appears correct? (1-5)
- Well-commented? (1-5)

---

## Cost-Saving Alternative: LLM-as-Judge

### Hybrid Approach (~70% cost reduction)
```python
def hybrid_evaluation(texts):
    # Step 1: GPT-4 evaluates all texts ($500)
    llm_scores = gpt4_evaluate_batch(texts, criteria)

    # Step 2: Humans evaluate subset ($1,500)
    # - Top/bottom 10% from LLM
    # - Random 10% sample
    # - Cases where LLM uncertain
    human_subset = select_for_human_eval(llm_scores)
    human_scores = human_evaluate(human_subset)

    # Step 3: Calibrate and apply
    calibration = learn_calibration(llm_scores, human_scores)
    final_scores = apply_calibration(llm_scores, calibration)

    return final_scores
```

**Validation Required**:
- Correlation between GPT-4 and human judgments
- Check for bias toward certain methods
- Consistency across multiple runs

---

## Modern Annotation Interface

### Not MTurk - Custom Web Interface
```javascript
// React-based annotation platform
function AnnotationInterface({ textA, textB, prompt }) {
  return (
    <div>
      <PromptDisplay prompt={prompt} />
      <TextComparison textA={textA} textB={textB} />
      <RatingScale
        options={["A much better", "A slightly better",
                 "Equal", "B slightly better", "B much better"]}
      />
      <DimensionRatings
        dimensions={["fluency", "coherence", "creativity"]}
      />
      <OptionalComments maxLength={200} />
    </div>
  );
}
```

### Features
- Real-time progress tracking
- Keyboard shortcuts for efficiency
- Auto-save every rating
- Mobile-responsive
- Annotator leaderboards

---

## Statistical Analysis Plan

### Required Sample Size
```python
from statsmodels.stats.power import ttest_power

# For detecting medium effect size (d=0.5)
# With 80% power and α=0.05
n_per_group = ttest_power(0.5, power=0.8, alpha=0.05)
# Result: ~64 texts per method minimum

# Add 20% buffer for exclusions
recommended_n = int(n_per_group * 1.2)  # ~77 per method
```

### Analysis Methods
- Bootstrap confidence intervals
- Pairwise t-tests with Bonferroni correction
- Bradley-Terry model for pairwise comparisons
- Correlation with automatic metrics

---

## When to Trigger Human Evaluation

### Definitely Needed If:
- Automatic metrics show contradictory results
- Beam search appears completely "fixed" (<5% repetition)
- One method dominates all tasks (suspicious)
- Planning to publish findings

### Probably Not Needed If:
- Results align with expectations (10-15% improvement)
- Clear task-specific patterns emerge
- Strong correlation between automatic metrics
- Just validating for internal use

---

## Quick Budget Options

### Ultra-Minimal ($500)
- 50 texts per method
- 3 annotators each
- Single overall quality rating
- GPT-4 pre-screening

### Standard ($2,000)
- 100 texts per method
- 3 annotators each
- 3 dimensions rated
- Basic quality control

### Comprehensive ($5,000)
- Multi-stage pipeline
- Pairwise comparisons
- Expert annotations
- Full statistical analysis

---

## Timeline Estimates

### Minimal (3 days)
- Day 1: Setup and pilot
- Day 2: Annotation
- Day 3: Analysis

### Standard (1 week)
- Days 1-2: Setup and qualification
- Days 3-4: Annotation
- Days 5-6: Quality control
- Day 7: Analysis

### Comprehensive (3 weeks)
- Week 1: Setup, pilot, refinement
- Week 2: Main annotation
- Week 3: Expert review and analysis

---

## Bottom Line

**For this project**: Human evaluation is tabled unless automatic metrics show very surprising results. If needed, the $2,000 standard option provides good validation with reasonable cost.