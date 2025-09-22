# Extended Metrics

Additional metrics beyond the core Holtzman et al. 2019 paper for evaluating text generation quality.

## Currently Implemented

(None yet - all metrics from the original paper have been moved to the core directory)

## TODO: Future Metrics to Implement

### NoveltyBench Metrics
**Paper**: NoveltyBench: Evaluating Creativity and Diversity in Language Models
**Link**: https://www.researchgate.net/publication/390571066_NoveltyBench_Evaluating_Creativity_and_Diversity_in_Language_Models

TODO:
- [ ] Implement novelty score based on semantic similarity
- [ ] Add creativity metrics from the benchmark
- [ ] Implement diversity measures beyond self-BLEU

### MAUVE Score
**Paper**: MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers
**Links**:
- https://arxiv.org/pdf/2102.01454
- https://www.jmlr.org/papers/volume24/23-0023/23-0023.pdf

TODO:
- [ ] Implement MAUVE score for comparing text distributions
- [ ] Add support for computing divergence frontiers
- [ ] Create visualization tools for MAUVE curves

### Additional Metrics from Literature

#### From "LLM evaluation metrics: A comprehensive guide"
**Link**: https://wandb.ai/onlineinference/genai-research/reports/LLM-evaluation-metrics-A-comprehensive-guide-for-large-language-models--VmlldzoxMjU5ODA4NA

TODO:
- [ ] Review and implement relevant metrics from the guide
- [ ] Add semantic coherence metrics
- [ ] Consider task-specific evaluation metrics

#### From "A Survey on Evaluation of Large Language Models"
**Link**: https://arxiv.org/pdf/1802.01886

TODO:
- [ ] Implement robustness metrics
- [ ] Add factuality checking metrics (if applicable)
- [ ] Consider bias and fairness metrics

## Implementation Guidelines

When adding new metrics:
1. Create a new Python file in this directory
2. Follow the existing pattern of simple, focused functions
3. Add comprehensive docstrings
4. Update the `__init__.py` file
5. Add tests if applicable
6. Document any external dependencies