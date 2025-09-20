# Setup Instructions for Virtual Environment

## 0. System Requirements

- **Disk Space**: ~500GB available for model weights
- **GPU Memory**: 16GB+ recommended (99GB+ for largest models)
- **RAM**: 32GB+ recommended for large model loading

## 1. Create and Activate Virtual Environment

```bash
# Navigate to project directory
cd /Users/mitchg/Desktop/sampling

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

## 2. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# If you encounter issues with torch, install CPU version:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# For GPU support (if you have CUDA):
# pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 3. Set Up API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor

# Add your keys:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# HUGGINGFACE_TOKEN=hf_...  # Required for gated models
```

## 4. Get Access to Gated Models

Many state-of-the-art models require explicit access approval:

```bash
# First, run verification to see which models need access
python verify_setup.py

# The script will show error messages with links for gated models.
# Visit each link and request access:
# - https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
# - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
# - https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
# - https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501

# Note: Access approval usually takes a few minutes to a few hours
# Warning: Models will download ~500GB total - ensure adequate disk space!
```

## 5. Test the Setup

```bash
# Run comprehensive verification first
python verify_setup.py

# Do a dry run first to verify everything is configured
python run.py --experiment degeneration_local --models gpt2-large --dry-run

# Test with a small sample
python run.py --experiment degeneration_local --models gpt2-large --num-samples 5 --methods greedy
```

## 6. Run Full Experiments

```bash
# Run the main degeneration experiments
python run.py --experiment degeneration_local    # Local models only
python run.py --experiment degeneration_openai   # OpenAI models
python run.py --experiment degeneration_anthropic # Anthropic models

# Or run with specific models
python run.py --experiment degeneration_local --models gpt2-large qwen2.5-7b

# Monitor costs with limit
python run.py --experiment degeneration_openai --cost-limit 10.0
```

## 7. Deactivate When Done

```bash
deactivate
```

## Common Issues and Solutions

### Issue: No module named 'torch'
```bash
# Install PyTorch CPU version (lighter, no GPU needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: API key not found
```bash
# Make sure .env file exists and has keys
cat .env

# Or set directly in terminal (temporary)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### Issue: Out of memory with GPT-2
```bash
# Edit config/models.yaml and set load_in_8bit: true for gpt2-large
# Or use smaller model like gpt2-medium
```

### Issue: Permission denied on run.py
```bash
chmod +x run.py
```

## Quick Test Commands

```bash
# Activate venv and test with minimal settings
source venv/bin/activate
export OPENAI_API_KEY="your-key"  # If not using .env

# Test with just 2 samples to verify it works
python run.py --experiment degeneration \
  --models gpt-3.5-turbo-instruct \
  --methods greedy nucleus_0.95 \
  --num-samples 2

# Check outputs
ls -la outputs/
cat outputs/degeneration_summary.csv
```