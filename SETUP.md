# Setup Instructions for Virtual Environment

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
```

## 4. Test the Setup

```bash
# Do a dry run first to verify everything is configured
python run.py --experiment degeneration --models gpt2-large --dry-run

# Test with a small sample
python run.py --experiment degeneration --models gpt2-large --num-samples 5 --methods greedy
```

## 5. Run Full Experiments

```bash
# Run the main degeneration experiment
python run.py --experiment degeneration --models gpt2-large gpt-4 claude-3-5-sonnet-20241022

# Or run with specific methods
python run.py --experiment degeneration --models gpt-4 --methods greedy beam_10 nucleus_0.95

# Monitor costs with limit
python run.py --experiment degeneration --cost-limit 10.0
```

## 6. Deactivate When Done

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