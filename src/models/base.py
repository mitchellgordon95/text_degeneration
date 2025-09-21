"""Base model class with strict method compatibility checking."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

# Import our capability system
from ..utils.capabilities import CapabilityManager, UnsupportedMethodError, get_method_parameters


class BaseModel(ABC):
    """
    Abstract base class for all models.

    Enforces strict method compatibility using YAML-based capabilities.
    No silent fallbacks allowed.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.total_tokens = 0
        self.total_cost = 0.0

        # Initialize capability manager for this model
        self._capability_manager = CapabilityManager()

    @property
    def supported_methods(self) -> List[str]:
        """
        Return list of supported decoding methods from YAML config.
        """
        return self._capability_manager.get_supported_methods(self.model_name)

    @property
    def supports_logprobs(self) -> bool:
        """Whether this model can return log probabilities."""
        return self._capability_manager.supports_logprobs(self.model_name)

    @property
    def supports_full_logprobs(self) -> bool:
        """Whether this model can return probabilities for all vocabulary tokens."""
        return self._capability_manager.supports_full_logprobs(self.model_name)

    def validate_method(self, method: str) -> None:
        """
        Check if method is supported. Raises exception if not.

        Args:
            method: Decoding method name (e.g., "greedy", "beam_10", "nucleus_0.95")

        Raises:
            UnsupportedMethodError: If method is not supported
        """
        self._capability_manager.validate_method(self.model_name, method)

    def can_use_method(self, method: str) -> bool:
        """
        Check if this model supports a given decoding method.

        Args:
            method: Method name to check

        Returns:
            True if method is supported, False otherwise
        """
        return self._capability_manager.supports_method(self.model_name, method)

    def generate(
        self,
        prompt: str,
        method: str = "greedy",
        max_length: int = 256,
        **kwargs
    ) -> str:
        """
        Generate text using specified decoding method with strict parameter isolation.

        Args:
            prompt: Input prompt
            method: Decoding method (must be in supported_methods)
            max_length: Maximum tokens to generate
            **kwargs: Additional method-specific parameters (will be overridden by method defaults)

        Returns:
            Generated text

        Raises:
            UnsupportedMethodError: If method is not supported
        """
        # Strict validation - no silent fallbacks
        self.validate_method(method)

        # Get canonical parameters for this method (enforces parameter isolation)
        method_params = get_method_parameters(method)

        # Override with any kwargs if provided, but prioritize method defaults
        combined_params = {**kwargs, **method_params}

        # Delegate to implementation with strict parameters
        return self._generate_impl(
            prompt=prompt,
            method=method,
            max_length=max_length,
            **combined_params
        )

    @abstractmethod
    def _generate_impl(
        self,
        prompt: str,
        method: str,
        max_length: int,
        **kwargs
    ) -> str:
        """
        Actual implementation of text generation.

        Args:
            prompt: Input prompt
            method: Base method name (e.g., "greedy", "beam", "nucleus")
            max_length: Maximum tokens to generate
            **kwargs: Method-specific parameters (temperature, top_p, top_k, num_beams, etc.)

        To be implemented by subclasses.
        """
        pass

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

        Returns:
            Dictionary mapping tokens to probabilities

        Raises:
            UnsupportedMethodError: If model doesn't support logprobs
        """
        if not self.supports_logprobs:
            raise UnsupportedMethodError(
                f"Model {self.model_name} does not support getting token probabilities"
            )

        return self._get_token_probabilities_impl(prompt, next_token)

    @abstractmethod
    def _get_token_probabilities_impl(
        self,
        prompt: str,
        next_token: Optional[str] = None
    ) -> Dict[str, float]:
        """Implementation of getting token probabilities."""
        pass

    def compute_perplexity(self, texts: List[str]) -> float:
        """
        Compute perplexity of texts under this model.

        Args:
            texts: List of texts to evaluate

        Returns:
            Perplexity score

        Raises:
            UnsupportedMethodError: If model doesn't support perplexity calculation
        """
        if not self.supports_logprobs:
            raise UnsupportedMethodError(
                f"Model {self.model_name} cannot compute perplexity (no logprob access)"
            )

        return self._compute_perplexity_impl(texts)

    @abstractmethod
    def _compute_perplexity_impl(self, texts: List[str]) -> float:
        """Implementation of perplexity calculation."""
        pass

    def analyze_tail_distribution(
        self,
        prompt: str,
        percentile_ranges: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Analyze token probability distribution tail.

        Args:
            prompt: Input prompt
            percentile_ranges: List of (min, max) percentile ranges to analyze

        Returns:
            Analysis results for each range

        Raises:
            UnsupportedMethodError: If model doesn't support full distribution access
        """
        if not self.supports_full_logprobs:
            raise UnsupportedMethodError(
                f"Model {self.model_name} cannot analyze tail distribution "
                f"(no full vocabulary access)"
            )

        return self._analyze_tail_distribution_impl(prompt, percentile_ranges)

    @abstractmethod
    def _analyze_tail_distribution_impl(
        self,
        prompt: str,
        percentile_ranges: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Implementation of tail distribution analysis."""
        pass