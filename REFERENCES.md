# Reference Papers

## Core Papers

### 1. The Curious Case of Neural Text Degeneration (2019)
**Authors**: Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi
**Link**: https://arxiv.org/abs/1904.09751
**Key Findings**:
- Beam search produces ~29% repetition rate in GPT-2
- Perplexity gap: Generated text PPL=1.48 vs Human PPL=12.38 (8x difference)
- Introduced nucleus (top-p) sampling with p=0.95 as optimal
- Identified "unreliable tail" problem in probability distributions
- Established that maximization-based decoding leads to degeneration

**Our Relevance**: This is the baseline we're comparing against - testing if these findings still hold for modern models.

---

### 2. A Thorough Examination of Decoding Methods in the Era of LLMs (2024)
**Authors**: Chufan Shi, Haoran Yang, Deng Cai, Zhisong Zhang, Yifan Wang, Yujiu Yang, Wai Lam
**Link**: https://arxiv.org/abs/2402.06925
**Key Findings**:
- Tested on Llama2 family (7B, 13B, 70B) both aligned and unaligned
- Greedy/beam search still produce "considerable repetitive content" even in modern models
- Alignment reduces variation between methods but doesn't eliminate problems
- Task dependency is critical: beam for closed-ended, sampling for open-ended
- Advanced methods require extensive hyperparameter tuning
- Model size affects optimal decoding strategy

**Our Relevance**: Recent comprehensive study showing degeneration persists in modern models, validates our experimental approach, suggests we'll find reduced but not eliminated repetition.

---

## Evaluation Metrics Papers

### 3. NoveltyBench: Evaluating Creativity and Diversity in Language Models (2024)
**Link**: https://www.researchgate.net/publication/390571066_NoveltyBench_Evaluating_Creativity_and_Diversity_in_Language_Models
**Key Contributions**:
- Framework for evaluating creativity and diversity in LLMs
- Novel metrics for measuring semantic novelty
- Benchmark dataset for creativity evaluation
**Our Relevance**: Provides additional diversity metrics beyond self-BLEU

### 4. MAUVE: Measuring the Gap Between Neural Text and Human Text (2021/2023)
**Authors**: Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, Zaid Harchaoui
**Links**:
- https://arxiv.org/pdf/2102.01454
- https://www.jmlr.org/papers/volume24/23-0023/23-0023.pdf
**Key Contributions**:
- Information-theoretic measure comparing text distributions
- Captures both Type I and Type II errors in generation
- More nuanced than perplexity or n-gram overlap
**Our Relevance**: Advanced metric for distribution comparison, could replace/supplement perplexity gap

### 5. LLM Evaluation Metrics: A Comprehensive Guide
**Link**: https://wandb.ai/onlineinference/genai-research/reports/LLM-evaluation-metrics-A-comprehensive-guide-for-large-language-models--VmlldzoxMjU5ODA4NA
**Key Contributions**:
- Comprehensive overview of evaluation metrics
- Practical guidance on metric selection
- Implementation considerations
**Our Relevance**: Reference for additional metrics to implement

### 6. A Survey on Evaluation of Large Language Models
**Link**: https://arxiv.org/pdf/1802.01886
**Key Contributions**:
- Systematic review of LLM evaluation approaches
- Categorization of evaluation metrics
- Discussion of evaluation challenges
**Our Relevance**: Framework for thinking about evaluation comprehensively

## Related Papers to Consider

### Contrastive Decoding Papers
- **Contrastive Search**: Su et al. (2022) - https://arxiv.org/abs/2202.06417
- **Contrastive Decoding**: Li et al. (2023) - https://arxiv.org/abs/2210.15097

### Other Decoding Methods
- **Typical Decoding**: Meister et al. (2022) - https://arxiv.org/abs/2202.00666
- **Frustratingly Simple Decoding**: (Add citation when found)

### Beam Search Analysis
- **On the Inadequacy of the Mode in Neural Text Generation**: Welleck et al. (2019)
- **If Beam Search is the Answer, What was the Question?**: Meister et al. (2020)

---

## Key Metrics to Track Across Papers

| Paper | Model | Method | Repetition Rate | Perplexity Gap | Notes |
|-------|-------|--------|----------------|----------------|-------|
| Holtzman 2019 | GPT-2 | Beam-10 | 28.94% | 8.4x | Baseline |
| Holtzman 2019 | GPT-2 | Nucleus-0.95 | 0.36% | ~2x | Proposed solution |
| Shi 2024 | Llama2-70B | Beam | Not specified | N/A | Still repetitive |
| **Ours** | GPT-4 | TBD | TBD | TBD | In progress |

---

## Questions These Papers Raise

1. **Why does repetition persist even after RLHF?** Both papers show deterministic methods cause repetition, even in aligned models.

2. **Is there a fundamental limit?** Maybe some repetition is inherent to autoregressive generation?

3. **Task-specific optimization**: Should we have different models or just different decoding for different tasks?

4. **Hyperparameter sensitivity**: Is p=0.95 still optimal for nucleus sampling in modern models?

5. **Scale effects**: How does model size (GPT-2 1.5B vs GPT-4 >1T) affect optimal decoding?

---

## Citation Format

```bibtex
@article{holtzman2019curious,
  title={The Curious Case of Neural Text Degeneration},
  author={Holtzman, Ari and Buys, Jan and Du, Li and Forbes, Maxwell and Choi, Yejin},
  journal={arXiv preprint arXiv:1904.09751},
  year={2019}
}

@article{shi2024thorough,
  title={A Thorough Examination of Decoding Methods in the Era of LLMs},
  author={Shi, Chufan and Yang, Haoran and Cai, Deng and Zhang, Zhisong and Wang, Yifan and Yang, Yujiu and Lam, Wai},
  journal={arXiv preprint arXiv:2402.06925},
  year={2024}
}
```