#!/usr/bin/env python3
"""Test Claude models with all decoding methods."""

import sys
import os
import time
from typing import List, Tuple
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.unified import UnifiedModel

# Load environment variables
load_dotenv()


def test_claude_model(model_name: str, num_samples: int = 3) -> bool:
    """Test a single Claude model with all decoding methods."""

    methods = ["greedy", "beam_5", "beam_10", "nucleus_0.95"]
    test_prompts = [
        "The weather today is",
        "Once upon a time in a land far away",
        "The key to success in life is"
    ][:num_samples]

    print("=" * 70)
    print(f"Testing {model_name}")
    print("=" * 70)

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not found")
        return False

    # Load model once
    print(f"\nLoading {model_name}...", end=" ")
    start = time.time()
    try:
        model = UnifiedModel.create(model_name)
        load_time = time.time() - start
        print(f"✓ ({load_time:.2f}s)")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return False

    results = []
    total_success = True

    for method in methods:
        print(f"\n{method}:")
        method_success = True

        for i, prompt in enumerate(test_prompts):
            print(f"  Prompt {i+1}: '{prompt[:30]}...'", end=" ")

            try:
                start = time.time()

                # Parse method parameters
                if method == "greedy":
                    output = model.generate(prompt, method="greedy", max_length=50)
                elif method.startswith("beam_"):
                    beam_size = int(method.split("_")[1])
                    output = model.generate(prompt, method="beam", max_length=50, num_beams=beam_size)
                elif method.startswith("nucleus_"):
                    p_value = float(method.split("_")[1])
                    output = model.generate(prompt, method="nucleus", max_length=50, top_p=p_value)
                else:
                    output = model.generate(prompt, method=method, max_length=50)

                gen_time = time.time() - start

                if output and len(output.strip()) > 0:
                    print(f"✓ ({gen_time:.2f}s)")
                    results.append({
                        "model": model_name,
                        "method": method,
                        "prompt": prompt,
                        "output": output[:100],
                        "time": gen_time
                    })
                else:
                    print(f"✗ Empty output")
                    method_success = False

            except Exception as e:
                print(f"✗ Error: {e}")
                method_success = False
                total_success = False
                break

        if method_success:
            model_results = [r for r in results if r["model"] == model_name and r["method"] == method]
            avg_time = sum(r["time"] for r in model_results) / len(model_results)
            print(f"  Average time: {avg_time:.2f}s")

    # Model summary
    if total_success:
        model_results = [r for r in results if r["model"] == model_name]
        total_time = sum(r["time"] for r in model_results)
        print(f"\n✅ {model_name}: All methods working")
        print(f"   Total time: {total_time:.2f}s for {len(model_results)} generations")
        print(f"   Average: {total_time/len(model_results):.2f}s per generation")
    else:
        print(f"\n❌ {model_name}: Some methods failed")

    return total_success


def main(num_samples: int = 3) -> bool:
    """Test all Claude models."""

    models = [
        "claude-3-5-sonnet-20241022"
    ]

    print("\n" + "=" * 70)
    print("Claude Models Test Suite")
    print("=" * 70)

    all_success = True
    for model_name in models:
        success = test_claude_model(model_name, num_samples)
        all_success = all_success and success

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    if all_success:
        print(f"✅ All {len(models)} Claude models tested successfully")
    else:
        print(f"❌ Some Claude models had failures")

    return all_success


if __name__ == "__main__":
    # Parse arguments
    num_samples = 3
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])

    # Run tests
    success = main(num_samples)
    sys.exit(0 if success else 1)