"""HuggingFace model wrapper with full decoding capabilities."""

import torch
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

from .base import BaseModel, UnsupportedMethodError


class HuggingFaceModel(BaseModel):
    """
    HuggingFace model wrapper for local models.

    Full capabilities:
    - All decoding methods including beam search
    - Full vocabulary logprobs access
    - Complete perplexity calculation
    - Full distribution analysis
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

        # Model configuration
        self.model_id = kwargs.get("model_id", model_name)
        self.device = kwargs.get("device", "auto")
        self.load_in_8bit = kwargs.get("load_in_8bit", False)

        # Set device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {self.model_id} on {self.device}...")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            if self.load_in_8bit:
                # Requires bitsandbytes library
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        quantization_config=quantization_config,
                        device_map="auto"
                    )
                except ImportError:
                    print("8-bit loading requires bitsandbytes. Falling back to normal loading.")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        dtype=torch.float16 if self.device == "cuda" else torch.float32
                    ).to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)

            self.model.eval()
            print(f"✓ Loaded {self.model_id}")

        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model {self.model_id}: {e}")

    @property
    def supported_methods(self) -> List[str]:
        """HuggingFace models support ALL decoding methods."""
        return [
            "greedy",
            "beam", "beam_2", "beam_5", "beam_10", "beam_20",
            "temperature",
            "nucleus_0.9", "nucleus_0.95", "nucleus_0.99",
            "top_k_10", "top_k_50", "top_k_100",
            "contrastive"  # Advanced method
        ]

    @property
    def supports_logprobs(self) -> bool:
        """HuggingFace provides full logprobs access."""
        return True

    @property
    def supports_full_logprobs(self) -> bool:
        """HuggingFace provides complete vocabulary probabilities."""
        return True

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
        Generate text using HuggingFace transformers.

        Supports all decoding methods.
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        # Configure generation parameters
        gen_kwargs = {
            "max_new_tokens": max_length,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Set method-specific parameters
        if method == "greedy":
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = 1

        elif method.startswith("beam"):
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = num_beams or 5
            gen_kwargs["early_stopping"] = True

        elif method == "temperature":
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 1.0
            gen_kwargs["top_k"] = 0

        elif method.startswith("nucleus"):
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = 0

        elif method.startswith("top_k"):
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_k"] = top_k
            gen_kwargs["top_p"] = 1.0

        elif method == "contrastive":
            # Contrastive search (requires transformers >= 4.24)
            gen_kwargs["penalty_alpha"] = kwargs.get("penalty_alpha", 0.6)
            gen_kwargs["top_k"] = kwargs.get("top_k", 5)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode and return only the new tokens
        generated = outputs[0][input_length:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        # Track token usage
        self.total_tokens += len(generated)

        return text

    def _get_token_probabilities_impl(
        self,
        prompt: str,
        next_token: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get full vocabulary probability distribution.
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Get model logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last position logits

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # If specific token requested
        if next_token:
            token_id = self.tokenizer.encode(next_token, add_special_tokens=False)[0]
            return {next_token: probs[token_id].item()}

        # Return top-100 for efficiency (full vocab would be huge)
        top_k = 100
        top_probs, top_indices = torch.topk(probs, top_k)

        prob_dict = {}
        for i in range(top_k):
            token = self.tokenizer.decode([top_indices[i].item()])
            prob_dict[token] = top_probs[i].item()

        return prob_dict

    def _compute_perplexity_impl(self, texts: List[str]) -> float:
        """
        Compute exact perplexity with full vocabulary access.
        """
        total_loss = 0.0
        total_tokens = 0

        for text in texts:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            # Get model outputs with labels for loss calculation
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss.item()

            # Accumulate
            num_tokens = input_ids.shape[1] - 1  # Exclude first token
            total_loss += loss * num_tokens
            total_tokens += num_tokens

        if total_tokens == 0:
            return float('inf')

        # Perplexity = exp(average loss)
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        return perplexity

    def _analyze_tail_distribution_impl(
        self,
        prompt: str,
        percentile_ranges: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Analyze full probability distribution tail.
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Get model logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last position

        # Convert to probabilities and sort
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Calculate cumulative distribution
        cumsum = torch.cumsum(sorted_probs, dim=0)

        results = {}
        for min_pct, max_pct in percentile_ranges:
            # Find tokens in this percentile range
            mask = (cumsum >= min_pct) & (cumsum <= max_pct)
            range_indices = sorted_indices[mask]
            range_probs = sorted_probs[mask]

            # Decode tokens in this range
            tokens = []
            for idx in range_indices[:20]:  # Limit to 20 for readability
                token = self.tokenizer.decode([idx.item()])
                tokens.append(token)

            results[f"{min_pct:.0%}-{max_pct:.0%}"] = {
                "num_tokens": len(range_indices),
                "total_probability": float(range_probs.sum()),
                "avg_probability": float(range_probs.mean()) if len(range_probs) > 0 else 0,
                "sample_tokens": tokens
            }

        return results