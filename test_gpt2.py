#!/usr/bin/env python3
"""Test GPT-2-large model with all decoding methods."""

import sys
import os
import time
from typing import List, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.unified import UnifiedModel


def test_gpt2(num_samples: int = 3) -> bool:
    """Test GPT-2-large with all decoding methods."""

    model_name = "gpt2-large"
    methods = ["greedy", "beam_5", "beam_10", "nucleus_0.95"]
    test_prompts = [
        "The weather today is",
        "Once upon a time in a land far away",
        "The key to success in life is"
    ][:num_samples]

    print("=" * 70)
    print(f"Testing {model_name}")
    print("=" * 70)

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
            avg_time = sum(r["time"] for r in results if r["method"] == method) / len(test_prompts)
            print(f"  Average time: {avg_time:.2f}s")

    # Summary
    print("\n" + "=" * 70)
    if total_success:
        print(f"✅ All {len(methods)} methods working for {model_name}")
        total_time = sum(r["time"] for r in results)
        print(f"Total generation time: {total_time:.2f}s for {len(results)} generations")
        print(f"Average time per generation: {total_time/len(results):.2f}s")
    else:
        print(f"❌ Some methods failed for {model_name}")
    print("=" * 70)

    return total_success


if __name__ == "__main__":
    # Parse arguments
    num_samples = 3
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])

    # Run test
    success = test_gpt2(num_samples)
    sys.exit(0 if success else 1)