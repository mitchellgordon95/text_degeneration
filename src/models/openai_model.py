"""OpenAI API model wrapper with strict limitations."""

import os
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from openai import OpenAI

from .base import BaseModel, UnsupportedMethodError


class OpenAIModel(BaseModel):
    """
    OpenAI API model wrapper.

    Limitations:
    - No native beam search support
    - Only top-5 logprobs available
    - Cannot analyze full vocabulary distribution
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)

        # Get API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize client
        self.client = OpenAI(api_key=self.api_key)

        # Model-specific settings
        self.model_id = kwargs.get("model_id", model_name)
        self.cost_per_1k = kwargs.get("cost_per_1k", 0.002)


    def _generate_impl(
        self,
        prompt: str,
        method: str,
        max_length: int,
        **kwargs
    ) -> str:
        """
        Generate text using OpenAI API with strict parameter enforcement.

        NO SILENT FALLBACKS: Unsupported methods will raise errors.
        """
        # Extract parameters from kwargs
        base_method = kwargs.get("base_method", method)
        temperature = kwargs.get("temperature", 1.0)
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k")
        num_beams = kwargs.get("num_beams")

        # Fail fast for unsupported methods - NO SILENT FALLBACKS
        if base_method == "beam" or num_beams and num_beams > 1:
            raise UnsupportedMethodError(
                f"OpenAI API does not support beam search. "
                f"Model: {self.model_name}, Method: {method}, "
                f"Base method: {base_method}, Beams: {num_beams}"
            )

        if base_method == "top_k" or top_k is not None:
            raise UnsupportedMethodError(
                f"OpenAI API does not support top_k sampling natively. "
                f"Model: {self.model_name}, Method: {method}, "
                f"Base method: {base_method}, top_k: {top_k}. "
                f"Use nucleus sampling instead."
            )

        # Only supported methods reach here
        if base_method not in ["greedy", "temperature", "nucleus"]:
            raise UnsupportedMethodError(
                f"OpenAI model {self.model_name} does not support method {base_method}. "
                f"Supported: greedy, temperature, nucleus"
            )

        # Use Chat Completions API
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p
            )

            # Track usage
            if hasattr(response, 'usage'):
                self.total_tokens += response.usage.total_tokens
                self.total_cost += (response.usage.total_tokens / 1000) * self.cost_per_1k

            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def _get_token_probabilities_impl(
        self,
        prompt: str,
        next_token: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get token probabilities from OpenAI using Chat Completions API.

        Uses logprobs parameter (top-5 tokens).
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5  # Maximum allowed
            )

            if not response.choices[0].logprobs:
                return {}

            # Extract logprobs from chat completions response
            choice_logprobs = response.choices[0].logprobs
            if not choice_logprobs.content or len(choice_logprobs.content) == 0:
                return {}

            # Get the first token's logprobs
            token_logprobs = choice_logprobs.content[0]
            if not hasattr(token_logprobs, 'top_logprobs') or not token_logprobs.top_logprobs:
                return {}

            # Convert to probability dict
            prob_dict = {}
            for logprob_obj in token_logprobs.top_logprobs:
                if hasattr(logprob_obj, 'token') and hasattr(logprob_obj, 'logprob'):
                    token = logprob_obj.token
                    logprob = logprob_obj.logprob
                    prob_dict[token] = np.exp(logprob)

            # Track usage
            if hasattr(response, 'usage'):
                self.total_tokens += response.usage.total_tokens
                self.total_cost += (response.usage.total_tokens / 1000) * self.cost_per_1k

            return prob_dict

        except Exception as e:
            raise RuntimeError(f"OpenAI API error getting probabilities: {e}")

    def _compute_perplexity_impl(self, texts: List[str]) -> float:
        """
        Compute perplexity using OpenAI's limited logprobs.

        WARNING: This will be approximate due to top-5 limitation!
        """
        print(
            f"WARNING: Computing perplexity with OpenAI is approximate "
            f"(only top-5 tokens available)"
        )

        total_log_prob = 0.0
        total_tokens = 0

        for text in texts:
            # Split into tokens (approximate)
            words = text.split()

            for i in range(len(words) - 1):
                context = " ".join(words[:i+1])
                next_word = words[i+1]

                try:
                    response = self.client.completions.create(
                        model=self.model_id,
                        prompt=context,
                        max_tokens=1,
                        temperature=0.0,
                        logprobs=5,
                        echo=False
                    )

                    if response.choices[0].logprobs:
                        # Check if next word is in top-5
                        tokens = response.choices[0].logprobs.tokens[0]
                        logprobs = response.choices[0].logprobs.token_logprobs[0]

                        for token, logprob in zip(tokens, logprobs):
                            if token.strip() == next_word:
                                total_log_prob += logprob
                                total_tokens += 1
                                break
                        else:
                            # Token not in top-5, assign low probability
                            total_log_prob += -20  # Arbitrary low value
                            total_tokens += 1

                    # Track usage
                    if hasattr(response, 'usage'):
                        self.total_tokens += response.usage.total_tokens
                        self.total_cost += (response.usage.total_tokens / 1000) * self.cost_per_1k

                except Exception as e:
                    print(f"Error computing perplexity for context: {e}")
                    continue

        if total_tokens == 0:
            return float('inf')

        avg_log_prob = total_log_prob / total_tokens
        perplexity = np.exp(-avg_log_prob)

        return perplexity

    def _analyze_tail_distribution_impl(
        self,
        prompt: str,
        percentile_ranges: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        This is NOT supported for OpenAI models.

        Raises:
            UnsupportedMethodError: Always, as OpenAI doesn't provide full distribution
        """
        raise UnsupportedMethodError(
            f"OpenAI models cannot analyze tail distribution "
            f"(only top-5 tokens available, not full vocabulary)"
        )