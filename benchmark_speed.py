#!/usr/bin/env python3
"""Speed benchmark for all available models and methods."""

import sys
import os
import time
import statistics
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Mock dependencies that aren't installed for testing
from unittest.mock import MagicMock

# Mock libraries we don't have installed
sys.modules['openai'] = MagicMock()
sys.modules['anthropic'] = MagicMock()

# Import after mocking
from models.unified import UnifiedModel
from models.base import UnsupportedMethodError


def benchmark_model(model_name: str, config: Dict, num_samples: int = 5) -> Dict:
    """
    Benchmark a single model with all its supported methods.

    Args:
        model_name: Name of the model to test
        config: Model configuration
        num_samples: Number of test runs per method

    Returns:
        Dictionary of benchmark results
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking {model_name}")
    print(f"{'='*70}")

    # Get model capabilities first
    caps = UnifiedModel.get_model_capabilities(model_name, **config)
    print(f"Model type: {caps['type']}")
    print(f"Beam search support: {caps['beam_search']}")
    print(f"Logprobs support: {caps['logprobs']}")
    print(f"Limitations: {', '.join(caps['limitations'][:2])}")

    # Test prompts of varying lengths
    test_prompts = [
        "The weather today is",                                    # Short
        "Once upon a time in a land far away, there lived",      # Medium
        "The key to success in life is understanding that every challenge presents an opportunity"  # Long
    ]

    # Determine methods to test based on model type
    if caps['type'] == 'openai':
        methods = ["greedy", "nucleus_0.95", "temperature"]
    elif caps['type'] == 'anthropic':
        methods = ["greedy", "nucleus_0.95", "temperature"]
    elif caps['type'] == 'huggingface':
        methods = ["greedy", "beam_5", "beam_10", "nucleus_0.95", "top_k_50"]
    else:
        methods = ["greedy"]

    results = {
        "model_name": model_name,
        "model_type": caps['type'],
        "methods": {}
    }

    # Skip actual model loading for API models (would require keys)
    if caps['type'] in ['openai', 'anthropic']:
        print("\n⚠️  Skipping API model (requires authentication)")
        print("   Would test methods:", ", ".join(methods))

        # Return mock results for API models
        for method in methods:
            results["methods"][method] = {
                "mean_time": 2.5,  # Typical API latency
                "std_time": 0.5,
                "min_time": 1.8,
                "max_time": 3.2,
                "tokens_per_second": 20.0,
                "status": "mock_api_results"
            }
        return results

    # For HuggingFace models, actually load and benchmark
    try:
        print(f"\nLoading {model_name}...")
        load_start = time.time()
        model = UnifiedModel.create(model_name, **config)
        load_time = time.time() - load_start
        print(f"✓ Loaded in {load_time:.2f}s")

        # Warm up the model
        print("Warming up model...")
        model.generate("Hello", method="greedy", max_length=10)
        print("✓ Warmed up")

    except Exception as e:
        print(f"✗ Failed to load {model_name}: {e}")
        return {"model_name": model_name, "error": str(e)}

    # Benchmark each method
    for method in methods:
        print(f"\n  Testing {method}:")

        if not model.can_use_method(method):
            print(f"    ✗ Method not supported")
            continue

        method_times = []
        total_tokens = 0

        for i in range(num_samples):
            prompt = test_prompts[i % len(test_prompts)]
            print(f"    Run {i+1}/{num_samples}: '{prompt[:30]}...'", end=" ")

            try:
                start_time = time.time()

                # Generate with consistent parameters
                if method == "greedy":
                    output = model.generate(prompt, method="greedy", max_length=50)
                elif method.startswith("beam_"):
                    beam_size = int(method.split("_")[1])
                    output = model.generate(prompt, method=method, max_length=50, num_beams=beam_size)
                elif method.startswith("nucleus_"):
                    output = model.generate(prompt, method=method, max_length=50)
                elif method.startswith("top_k_"):
                    output = model.generate(prompt, method=method, max_length=50)
                elif method == "temperature":
                    output = model.generate(prompt, method="temperature", max_length=50, temperature=0.7)
                else:
                    output = model.generate(prompt, method=method, max_length=50)

                gen_time = time.time() - start_time

                if output and len(output.strip()) > 0:
                    # Estimate tokens (rough approximation)
                    tokens = len(output.split())
                    total_tokens += tokens
                    method_times.append(gen_time)
                    print(f"✓ {gen_time:.2f}s ({tokens} tokens)")
                else:
                    print(f"✗ Empty output")

            except UnsupportedMethodError as e:
                print(f"✗ Unsupported: {e}")
                break
            except Exception as e:
                print(f"✗ Error: {e}")
                continue

        # Calculate statistics
        if method_times:
            mean_time = statistics.mean(method_times)
            std_time = statistics.stdev(method_times) if len(method_times) > 1 else 0
            min_time = min(method_times)
            max_time = max(method_times)
            avg_tokens = total_tokens / len(method_times)
            tokens_per_sec = avg_tokens / mean_time if mean_time > 0 else 0

            results["methods"][method] = {
                "mean_time": mean_time,
                "std_time": std_time,
                "min_time": min_time,
                "max_time": max_time,
                "avg_tokens": avg_tokens,
                "tokens_per_second": tokens_per_sec,
                "status": "success"
            }

            print(f"    Results: {mean_time:.2f}±{std_time:.2f}s, {tokens_per_sec:.1f} tok/s")
        else:
            results["methods"][method] = {"status": "failed"}

    return results


def main():
    """Run comprehensive speed benchmark."""

    print("🚀 SPEED BENCHMARK FOR TEXT GENERATION MODELS")
    print(f"GPU: {os.popen('nvidia-smi --query-gpu=name --format=csv,noheader,nounits').read().strip()}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Models to benchmark (only test local models that don't require API keys)
    models_to_test = [
        # Local models that we can actually load
        ("gpt2", {"type": "huggingface", "model_id": "gpt2"}),
        ("gpt2-large", {"type": "huggingface", "model_id": "gpt2-large"}),

        # API models (mock results)
        ("gpt-4", {"type": "openai"}),
        ("claude-3-5-sonnet-20241022", {"type": "anthropic"}),
    ]

    # Optional: Add larger models if user wants
    if len(sys.argv) > 1 and sys.argv[1] == "--include-large":
        models_to_test.extend([
            ("llama3-8b", {"type": "huggingface", "model_id": "meta-llama/Meta-Llama-3-8B-Instruct"}),
            ("qwen2.5-7b", {"type": "huggingface", "model_id": "Qwen/Qwen2.5-7B-Instruct"}),
        ])
        print("Including large models (requires HuggingFace token for some models)")

    all_results = []

    for model_name, config in models_to_test:
        try:
            result = benchmark_model(model_name, config, num_samples=3)
            all_results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠️  Benchmark interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Failed to benchmark {model_name}: {e}")
            all_results.append({"model_name": model_name, "error": str(e)})

    # Generate summary report
    print("\n" + "="*80)
    print("📊 BENCHMARK SUMMARY")
    print("="*80)

    # Create summary table
    summary_data = []

    for result in all_results:
        if "error" in result:
            continue

        model_name = result["model_name"]
        model_type = result.get("model_type", "unknown")

        for method, stats in result.get("methods", {}).items():
            if stats.get("status") == "success":
                summary_data.append({
                    "Model": model_name,
                    "Type": model_type,
                    "Method": method,
                    "Mean Time (s)": f"{stats['mean_time']:.2f}",
                    "Std Dev (s)": f"{stats['std_time']:.2f}",
                    "Tokens/sec": f"{stats['tokens_per_second']:.1f}",
                    "Min Time (s)": f"{stats['min_time']:.2f}",
                    "Max Time (s)": f"{stats['max_time']:.2f}"
                })

    if summary_data:
        df = pd.DataFrame(summary_data)
        print("\nSpeed Comparison:")
        print(df.to_string(index=False))

        # Save to file
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        df.to_csv(output_dir / "speed_benchmark.csv", index=False)
        print(f"\n💾 Results saved to {output_dir / 'speed_benchmark.csv'}")
    else:
        print("\n⚠️  No successful benchmarks to summarize")

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()