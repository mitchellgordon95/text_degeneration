# Implementation Architecture for Decoding Experiments

## Overview
After thinking through various options (HuggingFace evaluate, EleutherAI's lm-evaluation-harness, notebooks), I recommend a **lightweight custom framework** that's simple but systematic. Here's why and how.

---

## Architecture Decision

### Options Considered

1. **EleutherAI's lm-evaluation-harness**
   - ✅ Battle-tested, handles many models
   - ❌ Heavyweight, hard to customize for specific decoding experiments
   - ❌ Not great for sampling method comparisons

2. **Jupyter Notebooks**
   - ✅ Great for exploration
   - ❌ Hard to reproduce, track, and scale
   - ❌ Messy for multiple experiments

3. **Custom Framework** ← **Recommended**
   - ✅ Exactly what we need, nothing more
   - ✅ Easy to understand and modify
   - ✅ Can integrate best parts of other tools
   - ❌ Need to build it (but it's not that complex)

---

## Proposed Implementation Structure

```
sampling/
├── config/
│   ├── experiments.yaml      # Experiment configurations
│   ├── models.yaml          # Model endpoints and settings
│   └── prompts.yaml         # Test prompts by category
│
├── src/
│   ├── models/
│   │   ├── base.py         # Abstract base class
│   │   ├── openai_model.py # GPT-3.5, GPT-4
│   │   ├── anthropic_model.py # Claude
│   │   ├── huggingface_model.py # GPT-2, Llama
│   │   └── unified.py      # Unified interface
│   │
│   ├── decoding/
│   │   ├── methods.py      # Greedy, beam, nucleus implementations
│   │   └── sampler.py      # Sampling utilities
│   │
│   ├── metrics/
│   │   ├── repetition.py   # N-gram metrics
│   │   ├── diversity.py    # Self-BLEU, distinct-n
│   │   ├── perplexity.py   # Model confidence
│   │   └── coherence.py    # Semantic coherence
│   │
│   ├── experiments/
│   │   ├── base_experiment.py
│   │   ├── degeneration.py
│   │   ├── perplexity_calibration.py
│   │   ├── tail_analysis.py
│   │   ├── task_specific.py
│   │   └── beam_curse.py
│   │
│   └── utils/
│       ├── data_loader.py  # Load prompts
│       ├── cache.py        # Cache responses
│       ├── cost_tracker.py # Track API costs
│       └── logger.py       # Structured logging
│
├── outputs/
│   ├── raw/               # Raw model outputs
│   ├── metrics/           # Computed metrics
│   └── figures/           # Visualizations
│
├── notebooks/
│   ├── explore.ipynb      # For exploration
│   └── analysis.ipynb     # For final analysis
│
├── run.py                 # Main runner
├── analyze.py            # Analysis script
└── requirements.txt
```

---

## Core Components

### 1. Unified Model Interface
```python
# src/models/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np

class BaseModel(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.total_tokens = 0
        self.total_cost = 0.0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        method: str = "greedy",
        max_length: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        num_beams: int = 5,
        **kwargs
    ) -> str:
        """Generate text using specified decoding method"""
        pass

    @abstractmethod
    def get_token_probabilities(
        self,
        prompt: str,
        next_token: Optional[str] = None
    ) -> Dict[str, float]:
        """Get probability distribution over next tokens"""
        pass

    @abstractmethod
    def compute_perplexity(
        self,
        texts: List[str]
    ) -> float:
        """Compute perplexity of texts under this model"""
        pass

# src/models/openai_model.py
import openai
from typing import List, Dict
import numpy as np

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str, method: str = "greedy", **kwargs) -> str:
        # Map our method names to OpenAI parameters
        params = self._get_params(method, kwargs)

        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            **params
        )

        # Track costs
        self.total_tokens += response.usage.total_tokens
        self.total_cost += self._calculate_cost(response.usage)

        return response.choices[0].text

    def _get_params(self, method: str, kwargs: dict) -> dict:
        if method == "greedy":
            return {"temperature": 0, "max_tokens": kwargs.get("max_length", 256)}
        elif method == "nucleus":
            return {
                "temperature": kwargs.get("temperature", 1.0),
                "top_p": kwargs.get("top_p", 0.95),
                "max_tokens": kwargs.get("max_length", 256)
            }
        elif method.startswith("beam"):
            # OpenAI doesn't support beam search directly
            # We'll implement our own using logprobs
            return {"temperature": 0, "max_tokens": 1, "logprobs": 10}

# src/models/unified.py
class UnifiedModel:
    """Factory for creating models with consistent interface"""

    @staticmethod
    def create(model_name: str, **kwargs) -> BaseModel:
        if "gpt" in model_name.lower():
            return OpenAIModel(model_name, **kwargs)
        elif "claude" in model_name.lower():
            return AnthropicModel(model_name, **kwargs)
        elif model_name in ["gpt2", "llama", "mistral"]:
            return HuggingFaceModel(model_name, **kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")
```

### 2. Experiment Configuration
```yaml
# config/experiments.yaml
experiments:
  degeneration:
    models: ["gpt2-large", "gpt-3.5-turbo", "gpt-4", "claude-3.5-sonnet"]
    methods: ["greedy", "beam_5", "beam_10", "nucleus_0.95"]
    num_samples: 200
    max_length: 256
    metrics: ["repetition_rate", "self_bleu", "distinct_n"]

  perplexity_calibration:
    models: ["gpt2-large", "gpt-4"]
    methods: ["beam_10", "nucleus_0.95"]
    num_samples: 500
    max_length: 256
    metrics: ["perplexity", "overconfidence_ratio"]

  tail_analysis:
    models: ["gpt2-large", "gpt-4", "claude-3.5-sonnet"]
    probability_ranges:
      head: [0.0, 0.1]
      middle: [0.1, 0.5]
      tail: [0.95, 1.0]
    samples_per_range: 20
    metrics: ["coherence", "recovery_ability"]

# config/models.yaml
models:
  gpt2-large:
    type: huggingface
    model_id: gpt2-large
    device: cuda

  gpt-3.5-turbo:
    type: openai
    model_id: gpt-3.5-turbo-instruct
    cost_per_1k_tokens: 0.002

  gpt-4:
    type: openai
    model_id: gpt-4
    cost_per_1k_tokens: 0.03

  claude-3.5-sonnet:
    type: anthropic
    model_id: claude-3-5-sonnet-20241022
    cost_per_1k_tokens: 0.003
```

### 3. Experiment Base Class
```python
# src/experiments/base_experiment.py
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm
import pickle

class BaseExperiment:
    def __init__(self, config: Dict[str, Any], output_dir: str = "outputs"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories
        self.raw_dir = self.output_dir / "raw" / self.__class__.__name__
        self.metrics_dir = self.output_dir / "metrics" / self.__class__.__name__
        self.raw_dir.mkdir(exist_ok=True, parents=True)
        self.metrics_dir.mkdir(exist_ok=True, parents=True)

        # For resuming interrupted experiments
        self.checkpoint_file = self.output_dir / f"{self.__class__.__name__}_checkpoint.pkl"

    def run(self):
        """Main experiment loop with checkpointing"""
        results = self.load_checkpoint()

        for model_name in tqdm(self.config["models"], desc="Models"):
            if model_name in results:
                print(f"Skipping {model_name} (already completed)")
                continue

            model = UnifiedModel.create(model_name)
            results[model_name] = {}

            for method in tqdm(self.config["methods"], desc="Methods", leave=False):
                if method in results.get(model_name, {}):
                    continue

                outputs = self.generate_outputs(model, method)
                metrics = self.compute_metrics(outputs)

                results[model_name][method] = {
                    "outputs": outputs,
                    "metrics": metrics
                }

                # Save checkpoint after each method
                self.save_checkpoint(results)

                # Save raw outputs
                self.save_outputs(model_name, method, outputs)

        # Final save
        self.save_results(results)
        return results

    def generate_outputs(self, model: BaseModel, method: str) -> List[str]:
        """Generate outputs for this experiment"""
        raise NotImplementedError

    def compute_metrics(self, outputs: List[str]) -> Dict[str, float]:
        """Compute metrics for outputs"""
        raise NotImplementedError

    def save_checkpoint(self, results: Dict):
        """Save intermediate results for resuming"""
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(results, f)

    def load_checkpoint(self) -> Dict:
        """Load previous results if they exist"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return {}
```

### 4. Specific Experiment Implementation
```python
# src/experiments/degeneration.py
from .base_experiment import BaseExperiment
from ..metrics import repetition, diversity
import numpy as np

class DegenerationExperiment(BaseExperiment):
    def __init__(self, config, prompts):
        super().__init__(config)
        self.prompts = prompts[:config["num_samples"]]

    def generate_outputs(self, model, method):
        outputs = []
        for prompt in self.prompts:
            text = model.generate(
                prompt=prompt,
                method=method,
                max_length=self.config["max_length"]
            )
            outputs.append(text)
        return outputs

    def compute_metrics(self, outputs):
        return {
            "repetition_rate": repetition.measure_ngram_repetition(outputs, n=4),
            "self_bleu": diversity.compute_self_bleu(outputs),
            "distinct_1": diversity.distinct_n_grams(outputs, n=1),
            "distinct_2": diversity.distinct_n_grams(outputs, n=2),
            "avg_length": np.mean([len(o.split()) for o in outputs])
        }
```

### 5. Smart Caching Layer
```python
# src/utils/cache.py
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Optional

class ResponseCache:
    """Cache model responses to avoid redundant API calls"""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, model: str, prompt: str, params: dict) -> str:
        """Generate unique cache key"""
        key_data = {
            "model": model,
            "prompt": prompt,
            "params": params
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, model: str, prompt: str, params: dict) -> Optional[str]:
        """Retrieve cached response if it exists"""
        key = self._get_cache_key(model, prompt, params)
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def set(self, model: str, prompt: str, params: dict, response: str):
        """Cache a response"""
        key = self._get_cache_key(model, prompt, params)
        cache_file = self.cache_dir / f"{key}.pkl"

        with open(cache_file, 'wb') as f:
            pickle.dump(response, f)
```

### 6. Main Runner
```python
# run.py
import argparse
import yaml
from pathlib import Path
from src.experiments import (
    DegenerationExperiment,
    PerplexityCalibrationExperiment,
    TailAnalysisExperiment,
    TaskSpecificExperiment,
    BeamCurseExperiment
)
from src.utils import load_prompts
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["all", "degeneration", "perplexity", "tail", "task", "beam"])
    parser.add_argument("--config", default="config/experiments.yaml")
    parser.add_argument("--models", nargs="+", help="Override config models")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load configurations
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load prompts
    prompts = load_prompts("config/prompts.yaml")

    # Run experiments
    experiments = {
        "degeneration": DegenerationExperiment,
        "perplexity": PerplexityCalibrationExperiment,
        "tail": TailAnalysisExperiment,
        "task": TaskSpecificExperiment,
        "beam": BeamCurseExperiment
    }

    if args.experiment == "all":
        to_run = experiments.keys()
    else:
        to_run = [args.experiment]

    for exp_name in to_run:
        logger.info(f"Running {exp_name} experiment...")
        exp_config = config["experiments"][exp_name]

        if args.models:
            exp_config["models"] = args.models

        if args.dry_run:
            logger.info(f"Would run: {exp_config}")
            continue

        experiment = experiments[exp_name](exp_config, prompts)
        results = experiment.run()
        logger.info(f"Completed {exp_name}, cost: ${experiment.total_cost:.2f}")

if __name__ == "__main__":
    main()
```

---

## Key Design Decisions

### 1. Why Not Use Existing Frameworks?

**lm-evaluation-harness**: Great for standard benchmarks, but our experiments need fine control over decoding methods that it doesn't provide well.

**HuggingFace evaluate**: Good for metrics, and we'll use some of their implementations, but doesn't provide the full experimental pipeline.

**Weights & Biases / MLflow**: Good for tracking, but adds complexity. We'll use simple JSON/pickle for now, can add W&B later if needed.

### 2. Why This Structure?

- **Separation of concerns**: Models, experiments, metrics all separate
- **Resumability**: Checkpoint after each method/model combination
- **Caching**: Avoid redundant API calls
- **Extensibility**: Easy to add new experiments or models
- **Reproducibility**: All outputs saved, configs tracked

### 3. Smart Implementation Tricks

**Unified beam search for API models**: Since OpenAI/Anthropic don't expose beam search, implement our own using their logprobs:
```python
def beam_search_via_logprobs(model, prompt, beam_size=5):
    beams = [(prompt, 0.0)]  # (text, log_prob)

    for _ in range(max_length):
        candidates = []

        for text, score in beams:
            # Get top k next tokens with logprobs
            response = model.complete(text, logprobs=beam_size*2)

            for token, logprob in response.logprobs:
                candidates.append((text + token, score + logprob))

        # Keep top beam_size candidates
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    return beams[0][0]  # Return highest scoring
```

**Cost tracking built-in**:
```python
class CostTracker:
    def __init__(self):
        self.costs = {"gpt-4": 0, "gpt-3.5": 0, "claude": 0}

    def track(self, model, tokens):
        rates = {
            "gpt-4": 0.03/1000,
            "gpt-3.5-turbo": 0.002/1000,
            "claude-3-5-sonnet": 0.003/1000
        }
        self.costs[model] += tokens * rates.get(model, 0)

    def report(self):
        total = sum(self.costs.values())
        print(f"Total cost: ${total:.2f}")
        for model, cost in self.costs.items():
            print(f"  {model}: ${cost:.2f}")
```

---

## Alternative: Lighter Notebook-Based Approach

If the framework feels too heavy, here's a notebook-based approach:

```python
# single_file_experiments.py
"""
Minimal implementation - everything in one file for simplicity
"""

import openai
import anthropic
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from collections import Counter
from typing import List, Dict
import json
import pickle
from tqdm import tqdm

# Setup
openai.api_key = "..."
anthropic_client = anthropic.Anthropic(api_key="...")

class SimpleExperiments:
    def __init__(self):
        self.results = {}
        self.cache = {}

    def generate(self, model: str, prompt: str, method: str, **kwargs) -> str:
        """Simple generation wrapper"""
        cache_key = f"{model}_{prompt[:50]}_{method}_{kwargs}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if model == "gpt-4":
            result = self._generate_openai(prompt, method, **kwargs)
        elif model == "claude":
            result = self._generate_claude(prompt, method, **kwargs)
        elif model == "gpt2":
            result = self._generate_hf(prompt, method, **kwargs)

        self.cache[cache_key] = result
        return result

    def run_degeneration_test(self):
        """Test 1: Repetition rates"""
        models = ["gpt2", "gpt-4"]
        methods = ["greedy", "beam_10", "nucleus_0.95"]

        for model in models:
            for method in methods:
                texts = []
                for prompt in prompts[:100]:
                    text = self.generate(model, prompt, method)
                    texts.append(text)

                rep_rate = self.measure_repetition(texts)
                print(f"{model} + {method}: {rep_rate:.2%} repetition")

    def measure_repetition(self, texts: List[str]) -> float:
        """Simple 4-gram repetition measure"""
        total_4grams = 0
        repeated_4grams = 0

        for text in texts:
            words = text.split()
            four_grams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
            counter = Counter(four_grams)

            total_4grams += len(four_grams)
            repeated_4grams += sum(count - 1 for count in counter.values() if count > 1)

        return repeated_4grams / max(total_4grams, 1)

# Run
exp = SimpleExperiments()
exp.run_degeneration_test()
```

---

## Recommendation

**Start with the lightweight framework** - it's not that complex and will save time in the long run. The key benefits:

1. **Resumability**: Can interrupt and resume expensive experiments
2. **Caching**: Won't repeat API calls if you need to rerun
3. **Organized outputs**: Everything saved systematically
4. **Easy to extend**: Add new experiments without refactoring

The initial setup might take a day, but it'll make the actual experiments much smoother. Plus, if findings are interesting, you'll have clean code ready for open-sourcing.