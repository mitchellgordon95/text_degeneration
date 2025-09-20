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

    @property
    def supported_methods(self) -> List[str]:
        """OpenAI supports only sampling methods, no beam search."""
        return [
            "greedy",
            "temperature",
            "nucleus_0.9", "nucleus_0.95", "nucleus_0.99",
            "top_k_10", "top_k_50", "top_k_100"
        ]

    @property
    def supports_logprobs(self) -> bool:
        """OpenAI provides limited logprobs (top-5 only)."""
        return True

    @property
    def supports_full_logprobs(self) -> bool:
        """OpenAI does NOT provide full vocabulary probabilities."""
        return False

    def _generate_impl(
        self,
        prompt: str,
        method: str,
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        num_beams: Optional[int],
        **kwargs
    ) -> str:
        """
        Generate text using OpenAI API.

        Note: Beam search will raise an error as it's not supported.
        """
        # Fail fast if beam search is requested
        if method.startswith("beam"):
            raise UnsupportedMethodError(
                f"OpenAI API does not support beam search. "
                f"Model: {self.model_name}, Method: {method}"
            )

        # Configure parameters based on method
        if method == "greedy":
            api_temperature = 0.0
            api_top_p = 1.0
        elif method == "temperature":
            api_temperature = temperature
            api_top_p = 1.0
        elif method.startswith("nucleus"):
            api_temperature = temperature
            api_top_p = top_p
        elif method.startswith("top_k"):
            # OpenAI doesn't have native top_k, simulate with temperature
            print(
                f"WARNING: OpenAI doesn't support top_k natively. "
                f"Using temperature={temperature} as approximation."
            )
            api_temperature = temperature
            api_top_p = 1.0
        else:
            raise ValueError(f"Unknown method: {method}")

        # Use Chat Completions API (modern, all models supported)
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=api_temperature,
                top_p=api_top_p
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
        Get token probabilities from OpenAI.

        Limited to top-5 tokens only!
        """
        try:
            response = self.client.completions.create(
                model=self.model_id,
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
                logprobs=5  # Maximum allowed
            )

            if not response.choices[0].logprobs:
                return {}

            # Extract top-5 token probabilities
            logprobs = response.choices[0].logprobs
            tokens = logprobs.tokens[0] if logprobs.tokens else []
            token_logprobs = logprobs.token_logprobs[0] if logprobs.token_logprobs else []

            # Convert to probability dict
            prob_dict = {}
            for token, logprob in zip(tokens[:5], token_logprobs[:5]):
                if logprob is not None:
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