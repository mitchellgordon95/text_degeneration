# Testing Modern LLM Decoding Methods

This repository contains experiments testing whether findings from "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019) still apply to modern language models.

## Core Question

**Do modern LLMs (GPT-4, Claude-3.5, Llama-3) trained with RLHF still exhibit the text degeneration problems identified in 2019?**

## 📚 Documentation

- 📊 **[Baseline Results](BASELINE_RESULTS.md)** - Holtzman's original GPT-2 results we're reproducing
- ⚠️ **[API Limitations](API_LIMITATIONS.md)** - Why certain experiments only work with local models
- 📖 **[References](REFERENCES.md)** - All relevant papers and citations
- 🎓 **[Academic Principles](ACADEMIC_PRINCIPLES.md)** - Research rigor guidelines
- 🤖 **[Claude Instructions](CLAUDE.md)** - Guidelines for AI assistants working on this project
- 👥 **[Human Evaluation Protocol](human_evaluation_protocol.md)** - Guidelines for human evaluation (future)
- 📄 **[Research Report](research_report.md)** - Detailed analysis and findings (in progress)

## Key Metrics (Reproducing Holtzman et al. 2019)

1. **Repetition Rate**: % of repeated 4-grams (GPT-2: 73.66% with greedy)
2. **Perplexity Gap**: Overconfidence ratio (GPT-2: 8.4x)
3. **Self-BLEU**: Diversity measure (GPT-2: 0.50 with greedy)
4. **Zipf Coefficient**: Word frequency distribution

## Setup Instructions

### System Requirements

- **Disk Space**: ~500GB available for model weights
- **GPU Memory**: 16GB+ recommended (80GB+ for largest models)
- **RAM**: 32GB+ recommended
- **Python**: 3.8+

### Quick Start

```bash
# 1. Clone and enter directory
git clone https://github.com/yourusername/text_degeneration.git
cd text_degeneration

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set up API keys
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY=sk-your-openai-key-here
#   ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
#   HUGGINGFACE_TOKEN=hf_...  # For gated models

# 5. Generate prompts with human continuations (for perplexity gap)
pip install datasets  # If not already installed
python scripts/get_webtext_prompts.py

# 6. Verify everything works
python verify_setup.py

# 7. Run experiments
python run.py --experiment degeneration_local    # Local models
python run.py --experiment degeneration_openai   # OpenAI models
python run.py --experiment degeneration_anthropic # Anthropic models
```

### Detailed Setup Steps

#### 1. Get Access to Gated Models

Many models require explicit access approval on HuggingFace:

```bash
# Run verification to see which models need access
python verify_setup.py

# Visit links shown for each gated model:
# - https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
# - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
# - https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
# - https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501

# Note: Access approval usually takes minutes to hours
# Warning: Models will download ~500GB total!
```

#### 2. Test the Setup

```bash
# Do a dry run first
python run.py --experiment degeneration_local --dry-run

# Test with small sample
python run.py --experiment degeneration_local \
  --models gpt2 \
  --num-samples 5 \
  --methods greedy
```

#### 3. Run Full Experiments

```bash
# Run specific experiment types
python run.py --experiment degeneration_local    # All local models
python run.py --experiment degeneration_openai   # GPT-4, GPT-5
python run.py --experiment degeneration_anthropic # Claude models

# Or override specific models/methods
python run.py --experiment degeneration_local \
  --models gpt2-large llama3-70b \
  --methods greedy beam_16 nucleus_0.95

# Limit samples for testing
python run.py --experiment degeneration_local --num-samples 10
```

## Experiment Types

- **degeneration_local**: Full methods (greedy, beam, top-k, nucleus) on local models
- **degeneration_openai**: Limited methods on GPT-4/GPT-5 (no beam search due to API limitations)
- **degeneration_anthropic**: Limited methods on Claude models (no beam search, no perplexity)

## Repository Structure

```
├── config/
│   ├── experiments.yaml       # Experiment configurations
│   ├── models.yaml           # Model settings
│   └── prompts.yaml          # Test prompts with human continuations
├── src/
│   ├── models/               # Unified model interface
│   ├── metrics/
│   │   ├── core/            # Metrics from Holtzman paper
│   │   └── extended/        # Additional metrics (future)
│   ├── experiments/          # Experiment implementations
│   └── utils/                # Utilities
├── scripts/
│   ├── get_webtext_prompts.py  # Generate prompts with continuations
│   └── analyze_results.py      # Analysis tools
├── outputs/                  # Results (auto-created)
│   ├── raw/                 # Generated texts
│   └── metrics/             # Computed metrics
└── run.py                    # Main runner
```

## Configuration

Models and experiments are configured in YAML files:
- `config/experiments.yaml` - Experiment parameters and methods
- `config/models.yaml` - Model configurations and capabilities
- `config/prompts.yaml` - Test prompts and human continuations

## Outputs

Results are saved to `outputs/`:
- `raw/` - Generated texts for each model/method
- `metrics/` - Computed metrics (repetition, self-BLEU, perplexity)
- Results displayed in Holtzman's table format for easy comparison

## Troubleshooting

### No module named 'torch'
```bash
# Install PyTorch CPU version (no GPU needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### API key not found
```bash
# Check .env file exists and has keys
cat .env

# Or set directly in terminal (temporary)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### Out of memory
```bash
# Edit config/models.yaml and set load_in_8bit: true
# Or use smaller models (gpt2 instead of gpt2-large)
```

### Permission denied
```bash
chmod +x run.py
```

## Expected Costs

- GPT-4: ~$20
- Claude-3.5: ~$10
- GPT-5: ~$25
- **Total: ~$50-75** for full reproduction

## Status

- [x] Experimental design complete
- [x] Core metrics implementation
- [x] Human continuations for perplexity gap
- [x] Output format matching Holtzman table
- [ ] Full experiments run
- [ ] Results analyzed

## Citation

If you use this code, please cite:

```bibtex
@article{gordon2024revisiting,
  title={Revisiting the Curious Case of Neural Text Degeneration},
  author={Gordon, Mitchell},
  journal={GitHub},
  year={2024}
}
```

Original paper:
```bibtex
@article{holtzman2019curious,
  title={The Curious Case of Neural Text Degeneration},
  author={Holtzman, Ari and Buys, Jan and Du, Li and Forbes, Maxwell and Choi, Yejin},
  journal={arXiv preprint arXiv:1904.09751},
  year={2019}
}
```