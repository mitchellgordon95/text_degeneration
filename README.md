# Testing Modern LLM Decoding Methods

This repository contains experiments testing whether findings from "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019) still apply to modern language models.

## Core Question

**Do modern LLMs (GPT-4, Claude-3.5, Llama-3) trained with RLHF still exhibit the text degeneration problems identified in 2019?**

📊 **[See baseline results from the original paper](BASELINE_RESULTS.md)** - We aim to reproduce these metrics for modern models.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY=sk-your-openai-key-here
#   ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# 3. Verify everything works
python verify_setup.py

# 4. Run experiments
python run.py --experiment degeneration --models gpt2-large llama3-70b gpt-5

# Dry run to see what would be executed
python run.py --experiment degeneration --dry-run
```

## Setup Verification

Before running experiments, verify your setup:

```bash
python verify_setup.py
```

This script checks:
- ✅ Dependencies installed
- ✅ GPU availability
- ✅ API keys configured
- ✅ Models load correctly
- ⚡ Speed benchmarks

The script will tell you exactly what's missing and how to fix it.

## Important: API Limitations

**OpenAI and Anthropic APIs don't support beam search.** See [API_LIMITATIONS.md](API_LIMITATIONS.md) for details.
- OpenAI only provides top-5 logprobs
- Anthropic provides no logprobs at all
- Beam search experiments use only open source models

## Key Experiments

1. **Beam Search Degeneration**: Does beam search still cause ~29% repetition?
2. **Perplexity Calibration**: Is there still an 8x gap between generated and human text?
3. **Unreliable Tail**: Are low-probability tokens still problematic?
4. **Task-Specific Decoding**: Do different tasks need different strategies?
5. **Beam Search Curse**: Does quality decrease with larger beam sizes?

## Repository Structure

```
├── config/
│   ├── experiments.yaml       # Experiment configurations
│   ├── models.yaml           # Model settings
│   └── prompts.yaml          # Test prompts
├── src/
│   ├── models/               # Unified model interface
│   ├── metrics/              # Evaluation metrics
│   ├── experiments/          # Experiment implementations
│   └── utils/                # Utilities
├── outputs/                  # Results (auto-created)
├── EXPERIMENTS.md            # Detailed protocol
└── run.py                    # Main runner
```

## Usage

### Basic Run
```bash
# Run with default settings
python run.py --experiment degeneration

# Specify models
python run.py --experiment degeneration --models gpt2-large llama3-70b gpt-5 claude-4-opus

# Specify methods
python run.py --experiment degeneration --methods greedy beam_10 nucleus_0.95

# Limit samples (for testing)
python run.py --experiment degeneration --num-samples 10
```

### Configuration

Models and experiments are configured in YAML files:
- `config/experiments.yaml` - Experiment parameters
- `config/models.yaml` - Model configurations
- `config/prompts.yaml` - Test prompts

### Outputs

Results are saved to `outputs/`:
- `raw/` - Generated texts
- `metrics/` - Computed metrics
- `degeneration_results.json` - Aggregated results
- `degeneration_summary.csv` - Summary table

## Requirements

- Python 3.8+
- CUDA GPU (optional, for local models)
- API keys for OpenAI/Anthropic

## Status

- [x] Experimental design complete
- [x] Implementation complete
- [ ] Experiments run
- [ ] Results analyzed

## Expected Costs

- GPT-4: ~$20
- Claude-3.5: ~$10
- GPT-3.5: ~$5
- **Total: ~$35-50**

## References

Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019). The Curious Case of Neural Text Degeneration. arXiv:1904.09751