"""HuggingFace Inference API model implementation."""

import os
import time
import requests
from typing import Dict, List, Optional
import numpy as np

from .base import BaseModel


class HuggingFaceAPIModel(BaseModel):
    """HuggingFace Inference API wrapper for fast cloud inference."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name)

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            print("Warning: No HuggingFace API key provided. Using public inference (may be rate limited)")

        # Map model names to HuggingFace model IDs
        self.model_id = self._get_model_id(model_name)
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"

        # Headers for API requests
        self.headers = {}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        # Wait for model to load if needed
        self._wait_for_model()

    def _get_model_id(self, model_name: str) -> str:
        """Map our model names to HuggingFace model IDs."""
        mapping = {
            "gpt2": "gpt2",
            "gpt2-medium": "gpt2-medium",
            "gpt2-large": "gpt2-large",
            "gpt2-xl": "gpt2-xl",
        }
        return mapping.get(model_name, model_name)

    def _wait_for_model(self, max_retries: int = 5):
        """Wait for model to be loaded in HuggingFace's infrastructure."""
        for attempt in range(max_retries):
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": "test", "parameters": {"max_new_tokens": 1}}
            )

            if response.status_code == 200:
                print(f"✓ {self.model_name} ready on HuggingFace API")
                return
            elif response.status_code == 503:
                # Model is loading
                estimated_time = response.json().get("estimated_time", 20)
                print(f"Model {self.model_name} is loading... (estimated: {estimated_time:.0f}s)")
                time.sleep(min(estimated_time + 2, 30))
            else:
                print(f"Warning: Unexpected status {response.status_code}: {response.text}")
                break

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
        """Generate text using HuggingFace Inference API."""
        # NO CACHING - for scientific rigor, each generation should be fresh

        # Prepare generation parameters
        parameters = self._get_generation_parameters(
            method, max_length, temperature, top_p, top_k, num_beams
        )

        # Make API request
        payload = {
            "inputs": prompt,
            "parameters": parameters,
            "options": {
                "use_cache": False,  # Don't use HF's cache, we have our own
                "wait_for_model": True
            }
        }

        # Make API request - no retries, fail fast
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload
        )

        if response.status_code == 200:
            result = response.json()

            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            else:
                generated_text = result.get("generated_text", "")

            # Remove the prompt from the output
            if generated_text.startswith(prompt):
                output = generated_text[len(prompt):].strip()
            else:
                output = generated_text

            # Track tokens (approximate)
            self.total_tokens += len(output.split())

            # NO CACHING - return result directly
            return output

        elif response.status_code == 503:
            # Model is loading
            raise RuntimeError(f"Model is still loading. Please wait and try again. Response: {response.text}")
        else:
            # Any other error - fail immediately
            raise RuntimeError(f"HuggingFace API error {response.status_code}: {response.text}")

    def _get_generation_parameters(
        self,
        method: str,
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        num_beams: int
    ) -> dict:
        """Get generation parameters for different methods."""
        base_params = {
            "max_new_tokens": max_length,
            "return_full_text": True,
        }

        if method == "greedy":
            return {
                **base_params,
                "do_sample": False,
                "num_beams": 1,
            }

        elif method == "nucleus" or method.startswith("nucleus_"):
            # Extract p value if specified like "nucleus_0.95"
            if "_" in method:
                p_value = float(method.split("_")[1])
            else:
                p_value = top_p

            return {
                **base_params,
                "do_sample": True,
                "temperature": temperature,
                "top_p": p_value,
            }

        elif method == "top_k" or method.startswith("top_k_"):
            # Extract k value if specified
            if "_" in method:
                k_value = int(method.split("_")[1])
            else:
                k_value = top_k

            return {
                **base_params,
                "do_sample": True,
                "temperature": temperature,
                "top_k": k_value,
            }

        elif method.startswith("beam"):
            # Extract beam size if specified like "beam_10"
            if "_" in method:
                beam_size = int(method.split("_")[1])
            else:
                beam_size = num_beams

            return {
                **base_params,
                "do_sample": False,
                "num_beams": beam_size,
                "early_stopping": True,
            }

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_token_probabilities(self, prompt: str, next_token: Optional[str] = None) -> Dict[str, float]:
        """Get probability distribution over next tokens."""
        # This would require a different API endpoint that's not commonly available
        # Return empty dict for now
        return {}

    def compute_perplexity(self, texts: List[str]) -> float:
        """Compute perplexity of texts."""
        # This would require token-level probabilities which aren't available via the API
        # Return a placeholder
        return -1.0

    def get_logprobs(self, prompt: str, continuation: str) -> List[float]:
        """Get log probabilities for each token in continuation."""
        # Not available via standard HuggingFace Inference API
        return []