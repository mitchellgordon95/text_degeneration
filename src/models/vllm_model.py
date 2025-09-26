"""vLLM model implementation for high-performance inference."""

from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams

from .base import BaseModel
from ..utils.capabilities import UnsupportedMethodError


class VLLMModel(BaseModel):
    """
    vLLM-based model implementation for fast inference.

    Provides the same interface as HuggingFaceModel but with vLLM's
    optimized PagedAttention and continuous batching for better performance.
    """

    def __init__(self, model_name: str, model_id: str = None, **kwargs):
        super().__init__(model_name)

        # Use model_name as model_id if not provided
        if model_id is None:
            model_id = kwargs.get("model_id", model_name)

        # Filter out HuggingFace-specific parameters that vLLM doesn't need
        vllm_kwargs = self._filter_vllm_kwargs(kwargs)

        # Set max_logprobs to support beam search
        # vLLM beam search needs 2 * beam_width logprobs
        # Set to 32 to support beam sizes up to 16 (matching Holtzman's experiments)
        if "max_logprobs" not in vllm_kwargs:
            vllm_kwargs["max_logprobs"] = 32

        # Initialize vLLM engine
        try:
            self.llm = LLM(model=model_id, **vllm_kwargs)
            print(f"✓ Loaded {model_id} with vLLM")
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_id} with vLLM: {e}")

    def _filter_vllm_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out HuggingFace-specific kwargs that vLLM doesn't support."""
        # Map HuggingFace parameters to vLLM equivalents
        vllm_kwargs = {}

        # Map load_in_8bit to quantization (only if no explicit quantization is set)
        if kwargs.get("load_in_8bit", False) and "quantization" not in kwargs:
            vllm_kwargs["quantization"] = "fp8"  # Use FP8 quantization in vLLM

        # Map device parameter
        if "device" in kwargs and kwargs["device"] != "auto":
            # vLLM handles device placement automatically
            pass

        # Pass through vLLM-compatible parameters
        vllm_params = [
            "tensor_parallel_size", "pipeline_parallel_size", "max_model_len",
            "block_size", "swap_space", "gpu_memory_utilization", "max_num_batched_tokens",
            "max_num_seqs", "max_paddings", "quantization", "enforce_eager",
            "max_context_len_to_capture", "disable_custom_all_reduce", "cpu_offload_gb"
        ]

        for param in vllm_params:
            if param in kwargs:
                vllm_kwargs[param] = kwargs[param]

        return vllm_kwargs

    def generate_greedy(self, prompt: str, max_length: int = 256) -> str:
        """Generate text using greedy decoding (temperature=0)."""
        self.validate_method("greedy")

        sampling_params = SamplingParams(
            temperature=0.0,  # Greedy = deterministic
            max_tokens=max_length,
            top_p=1.0,
            top_k=-1
        )

        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def generate_beam(self, prompt: str, beam_size: int, max_length: int = 256) -> str:
        """Generate text using beam search decoding."""
        method_name = f"beam_{beam_size}"
        self.validate_method(method_name)

        # Validate beam size against model capabilities
        self._capability_manager.validate_beam_size(self.model_name, method_name, beam_size)

        # Use the correct vLLM V1 beam search API
        beam_params = BeamSearchParams(
            beam_width=beam_size,
            max_tokens=max_length
        )

        # vLLM beam search expects prompts as dictionaries
        prompt_dict = [{"prompt": prompt}]
        outputs = self.llm.beam_search(prompt_dict, beam_params)
        # Return the best (first) sequence from beam search
        return outputs[0].sequences[0].text

    def generate_nucleus(self, prompt: str, top_p: float, max_length: int = 256) -> str:
        """Generate text using nucleus (top-p) sampling."""
        method_name = f"nucleus_{top_p}"
        self.validate_method(method_name)

        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=max_length,
            top_p=top_p,
            top_k=-1
        )

        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def generate_top_k(self, prompt: str, top_k: int, max_length: int = 256) -> str:
        """Generate text using top-k sampling."""
        method_name = f"top_k_{top_k}"
        self.validate_method(method_name)

        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=max_length,
            top_p=1.0,
            top_k=top_k
        )

        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def generate_temperature(self, prompt: str, temperature: float, max_length: int = 256) -> str:
        """Generate text using temperature sampling."""
        # This handles both "temperature" and "pure_sampling" methods
        if temperature == 1.0:
            method_name = "pure_sampling"
        else:
            method_name = f"temperature_{temperature}"
        self.validate_method(method_name)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_length,
            top_p=1.0,  # No truncation for temperature/pure sampling
            top_k=-1    # No top-k truncation
        )

        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def get_token_probabilities(
        self,
        prompt: str,
        next_token: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get probability distribution over next tokens.

        Args:
            prompt: Input prompt
            next_token: If provided, return only probability for this token

        Note: vLLM has limited support for extracting token probabilities
        compared to HuggingFace. This method provides basic functionality.
        """
        if not self.supports_logprobs:
            raise UnsupportedMethodError(
                f"Model {self.model_name} does not support log probabilities"
            )

        # Generate with logprobs enabled (vLLM max is 20)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,  # vLLM max is 20
            prompt_logprobs=1
        )

        try:
            outputs = self.llm.generate([prompt], sampling_params)
            output = outputs[0].outputs[0]

            # Extract token probabilities from the first generated token
            if output.logprobs and len(output.logprobs) > 0:
                token_logprobs = output.logprobs[0]
                # Convert log probabilities to probabilities
                probs = {token: np.exp(logprob.logprob) for token, logprob in token_logprobs.items()}
                return probs
            else:
                return {}

        except Exception as e:
            raise RuntimeError(f"Failed to get token probabilities: {e}")

    def compute_perplexity(self, texts: List[str]) -> float:
        """
        Compute perplexity for a list of texts.

        Note: This is a simplified implementation. vLLM doesn't directly
        support perplexity computation like HuggingFace does.
        """
        if not self.supports_logprobs:
            raise UnsupportedMethodError(
                f"Model {self.model_name} does not support perplexity computation "
                "(requires log probabilities)"
            )

        try:
            total_log_prob = 0.0
            total_tokens = 0

            for text in texts:
                # Generate with prompt logprobs to get likelihood
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=1,  # Minimum required by vLLM (we ignore the generated token)
                    prompt_logprobs=len(text.split()),  # Get logprobs for input tokens
                )

                outputs = self.llm.generate([text], sampling_params)

                # Sum up the log probabilities
                if hasattr(outputs[0], 'prompt_logprobs') and outputs[0].prompt_logprobs:
                    for token_logprob in outputs[0].prompt_logprobs:
                        if token_logprob is not None:
                            # Extract logprob value from Logprob object
                            logprob_obj = list(token_logprob.values())[0]
                            if hasattr(logprob_obj, 'logprob'):
                                total_log_prob += logprob_obj.logprob
                            else:
                                total_log_prob += float(logprob_obj)
                            total_tokens += 1

            if total_tokens == 0:
                return float('inf')

            # Calculate perplexity: exp(-average_log_prob)
            avg_log_prob = total_log_prob / total_tokens
            perplexity = np.exp(-avg_log_prob)
            return float(perplexity)

        except Exception as e:
            print(f"Warning: Could not compute perplexity with vLLM: {e}")
            return float('inf')

    def _get_token_probabilities_impl(
        self,
        prompt: str,
        next_token: Optional[str] = None
    ) -> Dict[str, float]:
        """Implementation of getting token probabilities."""
        # Generate with logprobs enabled (vLLM max is 20)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,  # vLLM max is 20
            prompt_logprobs=1
        )

        try:
            outputs = self.llm.generate([prompt], sampling_params)
            output = outputs[0].outputs[0]

            # Extract token probabilities from the first generated token
            if output.logprobs and len(output.logprobs) > 0:
                token_logprobs = output.logprobs[0]
                # Convert log probabilities to probabilities
                probs = {token: np.exp(logprob.logprob) for token, logprob in token_logprobs.items()}

                if next_token:
                    return {next_token: probs.get(next_token, 0.0)}
                return probs
            else:
                return {}

        except Exception as e:
            raise RuntimeError(f"Failed to get token probabilities: {e}")

    def _compute_perplexity_impl(self, texts: List[str]) -> float:
        """Implementation of perplexity calculation."""
        try:
            total_log_prob = 0.0
            total_tokens = 0

            for text in texts:
                # Generate with prompt logprobs to get likelihood
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=1,  # Minimum required by vLLM (we ignore the generated token)
                    prompt_logprobs=len(text.split()),  # Get logprobs for input tokens
                )

                outputs = self.llm.generate([text], sampling_params)

                # Sum up the log probabilities
                if hasattr(outputs[0], 'prompt_logprobs') and outputs[0].prompt_logprobs:
                    for token_logprob in outputs[0].prompt_logprobs:
                        if token_logprob is not None:
                            # Extract logprob value from Logprob object
                            logprob_obj = list(token_logprob.values())[0]
                            if hasattr(logprob_obj, 'logprob'):
                                total_log_prob += logprob_obj.logprob
                            else:
                                total_log_prob += float(logprob_obj)
                            total_tokens += 1

            if total_tokens == 0:
                return float('inf')

            # Calculate perplexity: exp(-average_log_prob)
            avg_log_prob = total_log_prob / total_tokens
            perplexity = np.exp(-avg_log_prob)
            return float(perplexity)

        except Exception as e:
            print(f"Warning: Could not compute perplexity with vLLM: {e}")
            return float('inf')

    def _analyze_tail_distribution_impl(
        self,
        prompt: str,
        percentile_ranges: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Implementation of tail distribution analysis."""
        # vLLM has limited support for full vocabulary distribution analysis
        # This is a simplified implementation
        try:
            # Get token probabilities (vLLM default max is 20)
            logprobs_count = min(top_k, 20)  # vLLM max is 20
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                logprobs=20,  # vLLM max is 20
                prompt_logprobs=1
            )

            outputs = self.llm.generate([prompt], sampling_params)
            output = outputs[0].outputs[0]

            if not output.logprobs or len(output.logprobs) == 0:
                return {f"range_{i}": {"tokens": [], "avg_prob": 0.0, "count": 0}
                       for i, _ in enumerate(percentile_ranges)}

            # Convert log probabilities to probabilities and sort
            token_logprobs = output.logprobs[0]
            token_probs = [(token, np.exp(logprob.logprob)) for token, logprob in token_logprobs.items()]
            token_probs.sort(key=lambda x: x[1], reverse=True)

            # Analyze each percentile range
            results = {}
            total_tokens = len(token_probs)

            for i, (min_pct, max_pct) in enumerate(percentile_ranges):
                start_idx = int(min_pct * total_tokens)
                end_idx = int(max_pct * total_tokens)

                range_tokens = token_probs[start_idx:end_idx]
                avg_prob = np.mean([prob for _, prob in range_tokens]) if range_tokens else 0.0

                results[f"range_{i}"] = {
                    "tokens": [token for token, _ in range_tokens[:10]],  # Sample first 10
                    "avg_prob": float(avg_prob),
                    "count": len(range_tokens),
                    "percentile_range": (min_pct, max_pct)
                }

            return results

        except Exception as e:
            print(f"Warning: Could not analyze tail distribution with vLLM: {e}")
            return {f"range_{i}": {"tokens": [], "avg_prob": 0.0, "count": 0, "error": str(e)}
                   for i, _ in enumerate(percentile_ranges)}