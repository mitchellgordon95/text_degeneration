#!/usr/bin/env python3
"""
Comprehensive verification script for text degeneration experiments.

This script tests:
1. All dependencies are installed
2. GPU availability and performance
3. API authentication works
4. All model types and decoding methods function correctly
5. Speed benchmarks for planning experiments

Run this before starting experiments to ensure everything works.
"""

import sys
import os
import time
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_dependencies():
    """Check that all required packages are installed."""
    print("🔍 Checking dependencies...")

    missing = []
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        missing.append("numpy")

    try:
        import pandas
        print("  ✓ pandas")
    except ImportError:
        missing.append("pandas")

    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError:
        missing.append("torch")

    try:
        import transformers
        print(f"  ✓ transformers {transformers.__version__}")
    except ImportError:
        missing.append("transformers")

    try:
        import openai
        print(f"  ✓ openai {openai.__version__}")
    except ImportError:
        missing.append("openai")

    try:
        import anthropic
        print(f"  ✓ anthropic {anthropic.__version__}")
    except ImportError:
        missing.append("anthropic")

    try:
        from dotenv import load_dotenv
        print("  ✓ python-dotenv")
    except ImportError:
        missing.append("python-dotenv")

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("✅ All dependencies installed\n")
    return True


def check_gpu():
    """Check GPU availability and specs."""
    print("🖥️  Checking GPU...")

    try:
        import torch

        if not torch.cuda.is_available():
            print("  ⚠️  CUDA not available - will use CPU (slower)")
            return {"available": False, "device": "cpu"}

        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"  ✓ CUDA available")
        print(f"  ✓ Device: {device_name}")
        print(f"  ✓ Memory: {memory_total:.1f} GB")
        print(f"  ✓ Device count: {device_count}")

        return {
            "available": True,
            "device": "cuda",
            "name": device_name,
            "memory_gb": memory_total,
            "count": device_count
        }
    except Exception as e:
        print(f"  ❌ Error checking GPU: {e}")
        return {"available": False, "device": "cpu"}


def check_api_keys():
    """Check API key availability."""
    print("🔑 Checking API keys...")

    # Load .env if available
    try:
        from dotenv import load_dotenv
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv()
            print("  ✓ Loaded .env file")
        else:
            print("  ⚠️  No .env file found (using environment variables)")
    except ImportError:
        pass

    api_status = {}

    # OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        if openai_key.startswith("sk-") and len(openai_key) > 20:
            print("  ✓ OpenAI API key found")
            api_status["openai"] = True
        else:
            print("  ⚠️  OpenAI API key format looks incorrect")
            api_status["openai"] = False
    else:
        print("  ❌ OpenAI API key not found")
        api_status["openai"] = False

    # Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        if anthropic_key.startswith("sk-ant-") and len(anthropic_key) > 20:
            print("  ✓ Anthropic API key found")
            api_status["anthropic"] = True
        else:
            print("  ⚠️  Anthropic API key format looks incorrect")
            api_status["anthropic"] = False
    else:
        print("  ❌ Anthropic API key not found")
        api_status["anthropic"] = False

    # HuggingFace (optional)
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("  ✓ HuggingFace token found (for gated models)")
        api_status["huggingface"] = True
    else:
        print("  ⚠️  HuggingFace token not found (some models may be inaccessible)")
        api_status["huggingface"] = False

    print()
    return api_status


def test_model(model_name: str, config: Dict, api_status: Dict) -> Dict:
    """Test a single model with supported methods."""

    # Import here to avoid issues if dependencies missing
    from models.unified import UnifiedModel
    from models.base import UnsupportedMethodError

    print(f"🧪 Testing {model_name}...")

    # Check if we can test this model type
    model_type = config.get("type", "huggingface")

    if model_type in ["openai", "anthropic"] and not api_status.get(model_type, False):
        print(f"  ⚠️  Skipping {model_type} model (no API key)")
        return {"status": "skipped", "reason": "no_api_key"}

    try:
        # Get capabilities
        caps = UnifiedModel.get_model_capabilities(model_name, **config)
        print(f"  📋 Type: {caps['type']}")
        print(f"  📋 Beam search: {caps['beam_search']}")
        print(f"  📋 Logprobs: {caps['logprobs']}")

        # Load model
        print(f"  🔄 Loading model...", end=" ")
        load_start = time.time()
        model = UnifiedModel.create(model_name, **config)
        load_time = time.time() - load_start
        print(f"✓ ({load_time:.1f}s)")

        # Test methods
        test_prompt = "The weather today is"
        results = {"load_time": load_time, "methods": {}}

        # Get appropriate methods for this model type
        if caps['type'] == 'openai':
            methods = ["greedy", "nucleus_0.95"]
        elif caps['type'] == 'anthropic':
            methods = ["greedy", "nucleus_0.95"]
        else:  # huggingface
            methods = ["greedy", "beam_5", "nucleus_0.95"]

        for method in methods:
            print(f"    Testing {method}...", end=" ")

            if not model.can_use_method(method):
                print("❌ Unsupported")
                continue

            try:
                start = time.time()
                output = model.generate(
                    test_prompt,
                    method=method,
                    max_length=20  # Short for speed
                )
                gen_time = time.time() - start

                if output and len(output.strip()) > 0:
                    tokens = len(output.split())
                    tok_per_sec = tokens / gen_time if gen_time > 0 else 0
                    print(f"✓ {gen_time:.2f}s ({tok_per_sec:.1f} tok/s)")

                    results["methods"][method] = {
                        "time": gen_time,
                        "tokens": tokens,
                        "tokens_per_sec": tok_per_sec,
                        "output_sample": output[:50] + "..." if len(output) > 50 else output
                    }
                else:
                    print("❌ Empty output")
                    results["methods"][method] = {"error": "empty_output"}

            except UnsupportedMethodError as e:
                print(f"❌ {e}")
                results["methods"][method] = {"error": "unsupported"}
            except Exception as e:
                print(f"❌ Error: {e}")
                results["methods"][method] = {"error": str(e)}

        results["status"] = "success"

        # Clean up GPU memory after testing
        if model_type == "huggingface":
            try:
                import gc
                import torch
                del model
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  🧹 GPU memory cleaned")
            except Exception as cleanup_error:
                print(f"  ⚠️  Memory cleanup warning: {cleanup_error}")

        return results

    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return {"status": "failed", "error": str(e)}


def generate_report(results: Dict, gpu_info: Dict):
    """Generate summary report."""
    print("\n" + "="*80)
    print("📊 VERIFICATION REPORT")
    print("="*80)

    # System info
    print(f"\n💻 System Information:")
    print(f"  GPU: {gpu_info.get('name', 'None')} ({gpu_info.get('memory_gb', 0):.1f} GB)")
    print(f"  Device: {gpu_info.get('device', 'cpu')}")

    # Model results
    print(f"\n🤖 Model Test Results:")

    successful_models = []
    failed_models = []

    for model_name, result in results.items():
        status = result.get("status", "unknown")

        if status == "success":
            successful_models.append(model_name)
            methods_with_time = [m for m in result["methods"].values() if "time" in m]
            methods_tested = len(methods_with_time)
            if methods_with_time:
                avg_time = statistics.mean([m["time"] for m in methods_with_time])
                print(f"  ✅ {model_name}: {methods_tested} methods, avg {avg_time:.2f}s")
            else:
                print(f"  ✅ {model_name}: {methods_tested} methods (no timing data)")

        elif status == "skipped":
            reason = result.get("reason", "unknown")
            print(f"  ⚠️  {model_name}: skipped ({reason})")

        else:
            failed_models.append(model_name)
            error = result.get("error", "unknown")
            print(f"  ❌ {model_name}: {error}")

    # Speed summary
    if successful_models:
        print(f"\n⚡ Speed Summary:")
        for model_name in successful_models:
            result = results[model_name]
            print(f"\n  {model_name} (load: {result['load_time']:.1f}s):")

            for method, stats in result["methods"].items():
                if "time" in stats:
                    print(f"    {method:12}: {stats['time']:.2f}s ({stats['tokens_per_sec']:.1f} tok/s)")

    # Recommendations
    print(f"\n💡 Recommendations:")

    if not gpu_info["available"]:
        print("  - Install CUDA-capable PyTorch for GPU acceleration")
        print("  - CPU-only execution will be much slower")

    if failed_models:
        print(f"  - Fix issues with: {', '.join(failed_models)}")

    missing_apis = []
    if "openai" not in [m for m in results.keys() if results[m].get("status") == "success"]:
        missing_apis.append("OpenAI")
    if "claude" not in [m for m in results.keys() if results[m].get("status") == "success"]:
        missing_apis.append("Anthropic")

    if missing_apis:
        print(f"  - Add API keys for: {', '.join(missing_apis)}")
        print("  - Copy .env.example to .env and add your keys")

    if len(successful_models) >= 2:
        print("  ✅ Ready to run experiments!")
    else:
        print("  ⚠️  Need at least 2 working models for meaningful experiments")


def main():
    """Run complete verification."""
    print("🚀 TEXT DEGENERATION EXPERIMENTS - SETUP VERIFICATION")
    print("="*60)
    print("This script verifies that everything is working for experiments.")
    print("It will test dependencies, GPU, API keys, and model performance.\n")

    # Check dependencies first
    if not check_dependencies():
        print("❌ Please install missing dependencies first")
        return 1

    # Check GPU
    gpu_info = check_gpu()

    # Check API keys
    api_status = check_api_keys()

    # Test models
    print("🧪 Testing models...")

    # Define models to test
    models_to_test = [
        # Small local model (should work on most systems)
        ("gpt2", {"type": "huggingface", "model_id": "gpt2"}),

        # API models (if keys available)
        ("gpt-4", {"type": "openai", "model_id": "gpt-4"}),
        ("claude-3-5-sonnet-20241022", {"type": "anthropic", "model_id": "claude-3-5-sonnet-20241022"}),
    ]

    # Add larger local models based on GPU memory
    gpu_memory = gpu_info.get("memory_gb", 0)

    if gpu_info["available"]:
        if gpu_memory > 10:
            models_to_test.extend([
                ("gpt2-large", {"type": "huggingface", "model_id": "gpt2-large"}),
                ("qwen2.5-7b", {"type": "huggingface", "model_id": "Qwen/Qwen2.5-7B-Instruct"}),
                ("mistral-7b", {"type": "huggingface", "model_id": "mistralai/Mistral-7B-Instruct-v0.3"}),
            ])
            print(f"  💪 GPU with {gpu_memory:.1f}GB detected - including medium models")

        if gpu_memory > 30:
            models_to_test.extend([
                ("mistral-small-3-24b", {"type": "huggingface", "model_id": "mistralai/Mistral-Small-24B-Instruct-2501", "load_in_8bit": True}),
            ])
            print(f"  🔥 Including 24B model")

        if gpu_memory > 70:
            models_to_test.extend([
                ("llama3-70b", {"type": "huggingface", "model_id": "meta-llama/Llama-3.3-70B-Instruct", "load_in_8bit": True}),
                ("qwen2.5-72b", {"type": "huggingface", "model_id": "Qwen/Qwen2.5-72B-Instruct", "load_in_8bit": True}),
                ("mixtral-8x7b", {"type": "huggingface", "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "load_in_8bit": True}),
            ])
            print(f"  🚀 Including 70B+ models")

        # Show what models we're skipping due to memory limitations
        if gpu_memory <= 10:
            print(f"  ⚠️  Skipping medium models (need >10GB, have {gpu_memory:.1f}GB)")
        if gpu_memory <= 30:
            print(f"  ⚠️  Skipping large models (need >30GB, have {gpu_memory:.1f}GB)")
        if gpu_memory <= 70:
            print(f"  ⚠️  Skipping 70B+ models (need >70GB, have {gpu_memory:.1f}GB)")
    else:
        print("  ❌ No GPU detected - skipping all local models except gpt2")

    results = {}

    for model_name, config in models_to_test:
        try:
            result = test_model(model_name, config, api_status)
            results[model_name] = result
        except KeyboardInterrupt:
            print(f"\n⚠️  Testing interrupted")
            break
        except Exception as e:
            print(f"  ❌ Unexpected error testing {model_name}: {e}")
            results[model_name] = {"status": "failed", "error": str(e)}

    # Generate report
    generate_report(results, gpu_info)

    # Save results
    try:
        import json
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_info": gpu_info,
            "api_status": api_status,
            "model_results": results
        }

        with open(output_dir / "verification_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Full report saved to outputs/verification_report.json")

    except Exception as e:
        print(f"\n⚠️  Could not save report: {e}")

    # Return code
    successful_count = len([r for r in results.values() if r.get("status") == "success"])
    if successful_count >= 2:
        print("\n🎉 Setup verification complete - ready for experiments!")
        return 0
    else:
        print("\n⚠️  Setup incomplete - please fix issues above")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)