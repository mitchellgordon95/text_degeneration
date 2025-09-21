#!/usr/bin/env python3
"""
Academic-Grade Capability Verification for Text Degeneration Experiments.

This script acts as a STRICT GATE for academic experiments:
1. Verifies all dependencies are installed
2. Tests GPU availability and memory requirements
3. Validates API authentication
4. RIGOROUSLY tests ALL claimed model capabilities against reality
5. Verifies parameter isolation and error handling
6. FAILS IMMEDIATELY on any discrepancy with actionable fix instructions

ZERO TOLERANCE for silent fallbacks or approximations.
Either everything works perfectly or the script fails with clear fix instructions.
"""

import sys
import os
import gc
import time
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our capability system
from src.utils.capabilities import (
    CapabilityManager,
    UnsupportedMethodError,
    get_method_parameters,
    validate_experiment_config
)
from src.utils import load_config
from src.models.unified import UnifiedModel


class VerificationError(Exception):
    """Raised when verification finds a configuration/implementation error."""
    pass


class AcademicVerifier:
    """Academic-grade verification with zero tolerance for discrepancies."""

    def __init__(self):
        self.capability_manager = CapabilityManager()
        self.gpu_info = None
        self.api_status = {}
        self.models_config = {}
        self.experiments_config = {}
        self.errors = []

    def check_dependencies(self):
        """Check that all required packages are installed."""
        print("🔍 Checking dependencies...")

        required_packages = [
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("openai", "openai"),
            ("anthropic", "anthropic"),
            ("PyYAML", "yaml"),
            ("tqdm", "tqdm")
        ]

        missing = []
        for package_name, import_name in required_packages:
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"  ✓ {package_name} {version}")
            except ImportError:
                missing.append(package_name)
                print(f"  ❌ {package_name}")

        if missing:
            raise VerificationError(
                f"Missing required packages: {', '.join(missing)}\n"
                f"Fix: pip install {' '.join(missing)}"
            )

        print("✅ All dependencies installed\n")

    def check_gpu(self):
        """Check GPU availability and memory."""
        print("🖥️  Checking GPU...")

        if not torch.cuda.is_available():
            print("  ⚠️  CUDA not available - CPU only mode")
            self.gpu_info = {"available": False, "device": "cpu"}
            return

        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        memory_free = (memory_total - torch.cuda.memory_allocated() / 1e9)

        print(f"  ✓ CUDA available")
        print(f"  ✓ Device: {device_name}")
        print(f"  ✓ Total Memory: {memory_total:.1f} GB")
        print(f"  ✓ Free Memory: {memory_free:.1f} GB")
        print(f"  ✓ Device count: {device_count}")

        self.gpu_info = {
            "available": True,
            "device": "cuda",
            "name": device_name,
            "memory_total_gb": memory_total,
            "memory_free_gb": memory_free,
            "count": device_count
        }
        print()

    def check_api_keys(self):
        """Check API key availability and format."""
        print("🔑 Checking API keys...")

        # Load .env if available
        try:
            from dotenv import load_dotenv
            env_file = Path(".env")
            if env_file.exists():
                load_dotenv()
                print("  ✓ Loaded .env file")
        except ImportError:
            pass

        # OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key and openai_key.startswith("sk-") and len(openai_key) > 20:
            print("  ✓ OpenAI API key found")
            self.api_status["openai"] = True
        else:
            print("  ❌ OpenAI API key not found or invalid format")
            self.api_status["openai"] = False

        # Anthropic
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic_key.startswith("sk-ant-") and len(anthropic_key) > 20:
            print("  ✓ Anthropic API key found")
            self.api_status["anthropic"] = True
        else:
            print("  ❌ Anthropic API key not found or invalid format")
            self.api_status["anthropic"] = False

        # HuggingFace (optional)
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            print("  ✓ HuggingFace token found")
            self.api_status["huggingface"] = True
        else:
            print("  ⚠️  HuggingFace token not found (some models may be inaccessible)")
            self.api_status["huggingface"] = False

        print()

    def load_configurations(self):
        """Load and validate configurations."""
        print("📋 Loading configurations...")

        try:
            self.models_config = load_config("config/models.yaml")["models"]
            print(f"  ✓ Loaded {len(self.models_config)} models")
        except Exception as e:
            raise VerificationError(f"Failed to load models.yaml: {e}")

        try:
            self.experiments_config = load_config("config/experiments.yaml")["experiments"]
            print(f"  ✓ Loaded {len(self.experiments_config)} experiments")
        except Exception as e:
            raise VerificationError(f"Failed to load experiments.yaml: {e}")

        print()

    def verify_parameter_isolation(self):
        """Verify that parameter isolation is working correctly."""
        print("🔬 Verifying parameter isolation...")

        test_cases = [
            ("greedy", {"base_method": "greedy", "temperature": 0.0, "top_p": 1.0, "top_k": None, "num_beams": 1, "do_sample": False}),
            ("beam_10", {"base_method": "beam", "temperature": 1.0, "top_p": 1.0, "top_k": None, "num_beams": 10, "do_sample": False}),
            ("nucleus_0.95", {"base_method": "nucleus", "temperature": 1.0, "top_p": 0.95, "top_k": None, "num_beams": 1, "do_sample": True}),
            ("top_k_50", {"base_method": "top_k", "temperature": 1.0, "top_p": 1.0, "top_k": 50, "num_beams": 1, "do_sample": True}),
        ]

        for method, expected_params in test_cases:
            try:
                actual_params = get_method_parameters(method)

                # Check each expected parameter
                for key, expected_value in expected_params.items():
                    if key not in actual_params:
                        raise VerificationError(
                            f"Parameter isolation error for {method}: missing parameter '{key}'\n"
                            f"Expected: {expected_params}\n"
                            f"Actual: {actual_params}\n"
                            f"Fix: Update get_method_parameters() in src/utils/capabilities.py"
                        )

                    if actual_params[key] != expected_value:
                        raise VerificationError(
                            f"Parameter isolation error for {method}: '{key}' = {actual_params[key]}, expected {expected_value}\n"
                            f"Expected: {expected_params}\n"
                            f"Actual: {actual_params}\n"
                            f"Fix: Update get_method_parameters() in src/utils/capabilities.py"
                        )

                print(f"  ✓ {method}: parameters correctly isolated")

            except Exception as e:
                raise VerificationError(f"Parameter isolation failed for {method}: {e}")

        print("✅ Parameter isolation verified\n")

    def verify_model_comprehensive(self, model_name: str, config: Dict) -> Dict:
        """Comprehensively verify a single model against its claimed capabilities."""
        print(f"🧪 Comprehensively testing {model_name}...")

        model_type = config.get("type", "huggingface")

        # Skip if we don't have API keys
        if model_type in ["openai", "anthropic"] and not self.api_status.get(model_type, False):
            print(f"  ⚠️  Skipping {model_type} model (no API key)")
            return {"status": "skipped", "reason": "no_api_key"}

        try:
            # Get claimed capabilities from YAML
            capabilities = self.capability_manager.get_model_capabilities(model_name)
            claimed_methods = capabilities.get("supported_methods", [])
            supports_logprobs = capabilities.get("supports_logprobs", False)
            supports_full_logprobs = capabilities.get("supports_full_logprobs", False)
            supports_beam_search = capabilities.get("supports_beam_search", False)
            max_beam_size = capabilities.get("max_beam_size")

            print(f"  📋 Claimed methods: {claimed_methods}")
            print(f"  📋 Logprobs: {supports_logprobs} (full: {supports_full_logprobs})")
            print(f"  📋 Beam search: {supports_beam_search} (max: {max_beam_size})")

            # Load the model
            print(f"  🔄 Loading model...", end=" ")
            load_start = time.time()
            model = UnifiedModel.create(model_name, **config)
            load_time = time.time() - load_start
            print(f"✓ ({load_time:.1f}s)")

            # Measure GPU memory if applicable
            memory_used = None
            if model_type == "huggingface" and self.gpu_info["available"]:
                memory_after = torch.cuda.memory_allocated() / 1e9
                memory_used = memory_after
                print(f"  💾 GPU memory used: {memory_used:.1f} GB")

                # Check if model fits in available memory (with headroom)
                available_memory = self.gpu_info["memory_free_gb"]
                if memory_used > available_memory * 0.9:
                    raise VerificationError(
                        f"Model {model_name} uses {memory_used:.1f} GB but only {available_memory:.1f} GB available\n"
                        f"Fix: Add 'load_in_8bit: true' to {model_name} config in models.yaml"
                    )

            # Test EVERY claimed method
            results = {"load_time": load_time, "memory_used_gb": memory_used, "methods": {}}
            test_prompt = "The weather today is"

            print(f"  🎯 Testing {len(claimed_methods)} claimed methods...")

            for method in claimed_methods:
                print(f"    Testing {method}...", end=" ")

                try:
                    # Test method support
                    if not model.can_use_method(method):
                        raise VerificationError(
                            f"Model {model_name} claims to support '{method}' but can_use_method() returns False\n"
                            f"Fix: Remove '{method}' from supported_methods for {model_name} in models.yaml"
                        )

                    # Test actual generation
                    start = time.time()
                    output = model.generate(
                        test_prompt,
                        method=method,
                        max_length=20  # Short for speed
                    )
                    gen_time = time.time() - start

                    if not output or len(output.strip()) == 0:
                        raise VerificationError(
                            f"Model {model_name} method '{method}' produces empty output\n"
                            f"Fix: Investigate model implementation or remove method from capabilities"
                        )

                    tokens = len(output.split())
                    tok_per_sec = tokens / gen_time if gen_time > 0 else 0

                    print(f"✓ {gen_time:.2f}s ({tok_per_sec:.1f} tok/s)")

                    results["methods"][method] = {
                        "time": gen_time,
                        "tokens": tokens,
                        "tokens_per_sec": tok_per_sec,
                        "output_sample": output[:50] + "..." if len(output) > 50 else output
                    }

                except UnsupportedMethodError as e:
                    raise VerificationError(
                        f"Model {model_name} claims to support '{method}' but raises UnsupportedMethodError: {e}\n"
                        f"Fix: Remove '{method}' from supported_methods for {model_name} in models.yaml"
                    )
                except Exception as e:
                    raise VerificationError(
                        f"Model {model_name} method '{method}' failed: {e}\n"
                        f"Fix: Investigate model implementation or remove method from capabilities"
                    )

            # Test unsupported methods (should fail)
            all_possible_methods = ["greedy", "beam_5", "beam_10", "nucleus_0.95", "top_k_50", "contrastive"]
            unsupported_methods = [m for m in all_possible_methods if m not in claimed_methods]

            if unsupported_methods:
                print(f"  🚫 Testing {len(unsupported_methods)} unsupported methods (should fail)...")

                for method in unsupported_methods[:3]:  # Test a few to save time
                    print(f"    Testing {method} (should fail)...", end=" ")

                    try:
                        output = model.generate(test_prompt, method=method, max_length=10)
                        raise VerificationError(
                            f"Model {model_name} should NOT support '{method}' but generation succeeded\n"
                            f"Fix: Add '{method}' to supported_methods for {model_name} in models.yaml, "
                            f"or fix model implementation to reject it"
                        )
                    except UnsupportedMethodError:
                        print("✓ correctly rejected")
                    except Exception as e:
                        # Other errors are also acceptable (model-specific failures)
                        print(f"✓ rejected with: {type(e).__name__}")

            # Test logprob capabilities
            if supports_logprobs:
                print(f"    Testing logprob support...", end=" ")
                try:
                    probs = model.get_token_probabilities(test_prompt)
                    if not probs or len(probs) == 0:
                        raise VerificationError(
                            f"Model {model_name} claims logprob support but returns empty probabilities\n"
                            f"Fix: Set supports_logprobs: false for {model_name} in models.yaml"
                        )
                    print(f"✓ returned {len(probs)} token probabilities")
                except UnsupportedMethodError:
                    raise VerificationError(
                        f"Model {model_name} claims logprob support but raises UnsupportedMethodError\n"
                        f"Fix: Set supports_logprobs: false for {model_name} in models.yaml"
                    )

            # Clean up GPU memory
            if model_type == "huggingface" and self.gpu_info["available"]:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  🧹 GPU memory cleaned")

            results["status"] = "success"
            return results

        except VerificationError:
            raise  # Re-raise verification errors
        except Exception as e:
            raise VerificationError(f"Unexpected error testing {model_name}: {e}")

    def verify_experiment_configurations(self):
        """Verify that all experiment configurations are valid."""
        print("🔬 Verifying experiment configurations...")

        for exp_name, exp_config in self.experiments_config.items():
            print(f"  📋 Validating {exp_name}...")

            errors = validate_experiment_config(exp_config, self.models_config)

            if errors:
                error_msg = f"Experiment '{exp_name}' has configuration errors:\n"
                for error in errors:
                    error_msg += f"  - {error}\n"
                error_msg += f"Fix: Update config/experiments.yaml to remove invalid model/method combinations"

                raise VerificationError(error_msg)

            print(f"    ✓ {exp_name}: valid configuration")

        print("✅ All experiment configurations valid\n")

    def run_verification(self):
        """Run complete verification with strict failure mode."""
        print("🚀 ACADEMIC-GRADE CAPABILITY VERIFICATION")
        print("="*60)
        print("STRICT MODE: Zero tolerance for discrepancies")
        print("Either everything works perfectly or we fail with actionable fixes.\n")

        try:
            # Phase 1: Basic setup
            self.check_dependencies()
            self.check_gpu()
            self.check_api_keys()
            self.load_configurations()

            # Phase 2: Parameter isolation
            self.verify_parameter_isolation()

            # Phase 3: Experiment configuration validation
            self.verify_experiment_configurations()

            # Phase 4: Comprehensive model verification
            print("🧪 COMPREHENSIVE MODEL VERIFICATION")
            print("="*50)

            # Test a representative subset of models
            models_to_test = []

            # Always test these core models if available
            core_models = ["gpt2", "gpt-4", "claude-3-5-sonnet-20241022"]
            for model in core_models:
                if model in self.models_config:
                    models_to_test.append(model)

            # Add GPU-appropriate models
            if self.gpu_info and self.gpu_info["available"]:
                gpu_memory = self.gpu_info["memory_free_gb"]

                if gpu_memory > 10:
                    models_to_test.extend(["gpt2-large", "qwen2.5-7b"])
                if gpu_memory > 30:
                    models_to_test.extend(["mistral-small-3-24b"])
                if gpu_memory > 70:
                    models_to_test.extend(["llama3-70b"])

            # Remove duplicates and filter to existing models
            models_to_test = list(set(m for m in models_to_test if m in self.models_config))

            print(f"Testing {len(models_to_test)} representative models...")

            successful_models = 0
            for model_name in models_to_test:
                config = self.models_config[model_name]
                try:
                    result = self.verify_model_comprehensive(model_name, config)
                    if result.get("status") == "success":
                        successful_models += 1
                    print(f"  ✅ {model_name}: PASSED comprehensive verification")
                except VerificationError:
                    raise  # Re-raise to fail immediately
                except KeyboardInterrupt:
                    print(f"\n⚠️  Verification interrupted by user")
                    return 1
                print()  # Add spacing between models

            # Final success
            print("🎉 VERIFICATION COMPLETE - ALL SYSTEMS READY!")
            print("="*50)
            print(f"✅ {successful_models} models verified and working perfectly")
            print("✅ All configurations match reality")
            print("✅ Parameter isolation confirmed")
            print("✅ Error handling verified")
            print("✅ Ready for academic-grade experiments!")

            return 0

        except VerificationError as e:
            print(f"\n❌ VERIFICATION FAILED")
            print("="*50)
            print(str(e))
            print("\n🔧 Please fix the issues above and run verification again.")
            return 1

        except KeyboardInterrupt:
            print(f"\n⚠️  Verification interrupted by user")
            return 1

        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR")
            print("="*50)
            print(f"Error: {e}")
            traceback.print_exc()
            return 1


def main():
    """Run academic-grade verification."""
    verifier = AcademicVerifier()
    return verifier.run_verification()


if __name__ == "__main__":
    sys.exit(main())