# Testing Modern LLM Decoding Methods

This repository contains experiments testing whether findings from "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019) still apply to modern language models.

## Core Question

**Do modern LLMs (GPT-4, Claude-3.5, Llama-3) trained with RLHF still exhibit the text degeneration problems identified in 2019?**

## Key Experiments

1. **Beam Search Degeneration**: Does beam search still cause ~29% repetition?
2. **Perplexity Calibration**: Is there still an 8x gap between generated and human text?
3. **Unreliable Tail**: Are low-probability tokens still problematic?
4. **Task-Specific Decoding**: Do different tasks need different strategies?
5. **Beam Search Curse**: Does quality decrease with larger beam sizes?

## Repository Structure

```
├── EXPERIMENTS.md              # Detailed experimental protocol
├── implementation_plan.md      # Technical implementation details
├── human_evaluation_protocol.md # Human eval protocol (if needed)
├── research_report.md          # Original analysis being tested
└── src/                       # Implementation (coming soon)
```

## Status

- [x] Experimental design complete
- [x] Implementation architecture planned
- [ ] Code implementation
- [ ] Experiments run
- [ ] Results analyzed

## Expected Timeline

- Week 1: Implementation and setup
- Week 2: Run experiments and analysis
- Total cost: ~$50 in API calls

## References

Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019). The Curious Case of Neural Text Degeneration. arXiv:1904.09751