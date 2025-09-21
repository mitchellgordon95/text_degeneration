"""Model capability validation and management.

This module provides strict capability checking for academic experiments,
eliminating silent fallbacks and ensuring rigorous testing standards.
"""

import yaml
from typing import Dict, List, Optional, Set, Any, Tuple
from pathlib import Path


class UnsupportedMethodError(Exception):
    """Raised when attempting to use an unsupported decoding method."""
    pass


class CapabilityManager:
    """Manages model capabilities from YAML configuration."""

    def __init__(self, models_config_path: str = "config/models.yaml"):
        """Initialize capability manager with models config."""
        self.models_config_path = Path(models_config_path)
        self.models_config = self._load_models_config()

    def _load_models_config(self) -> Dict[str, Any]:
        """Load models configuration from YAML file."""
        try:
            with open(self.models_config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('models', {})
        except Exception as e:
            raise RuntimeError(f"Failed to load models config from {self.models_config_path}: {e}")

    def get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """Get capabilities for a specific model."""
        if model_name not in self.models_config:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        model_config = self.models_config[model_name]
        capabilities = model_config.get('capabilities', {})

        if not capabilities:
            raise ValueError(
                f"Model '{model_name}' has no capabilities defined. "
                f"Please add capabilities section to {self.models_config_path}"
            )

        return capabilities

    def get_supported_methods(self, model_name: str) -> List[str]:
        """Get list of supported decoding methods for a model."""
        capabilities = self.get_model_capabilities(model_name)
        return capabilities.get('supported_methods', [])

    def supports_method(self, model_name: str, method: str) -> bool:
        """Check if a model supports a specific decoding method."""
        supported_methods = self.get_supported_methods(model_name)

        # Handle parameterized methods properly
        if method in supported_methods:
            return True

        # Extract base method name for parameterized methods
        if method.startswith("beam_"):
            base_method = "beam"
            # Special validation for beam search - check if size is within limits
            try:
                beam_size = int(method.split("_")[1])
                max_beam = self.get_max_beam_size(model_name)
                if max_beam is not None and beam_size > max_beam:
                    return False  # Beam size exceeds model's maximum
            except (IndexError, ValueError):
                return False  # Invalid beam method format
        elif method.startswith("nucleus_"):
            base_method = "nucleus"
        elif method.startswith("top_k_"):
            base_method = "top_k"
        else:
            # For non-parameterized methods, use the method as-is
            base_method = method

        return base_method in supported_methods

    def supports_logprobs(self, model_name: str) -> bool:
        """Check if a model supports log probabilities."""
        capabilities = self.get_model_capabilities(model_name)
        return capabilities.get('supports_logprobs', False)

    def supports_full_logprobs(self, model_name: str) -> bool:
        """Check if a model supports full vocabulary log probabilities."""
        capabilities = self.get_model_capabilities(model_name)
        return capabilities.get('supports_full_logprobs', False)

    def supports_beam_search(self, model_name: str) -> bool:
        """Check if a model supports beam search."""
        capabilities = self.get_model_capabilities(model_name)
        return capabilities.get('supports_beam_search', False)

    def get_max_beam_size(self, model_name: str) -> Optional[int]:
        """Get maximum beam size for a model."""
        capabilities = self.get_model_capabilities(model_name)
        return capabilities.get('max_beam_size')

    def get_limitations(self, model_name: str) -> List[str]:
        """Get list of model limitations."""
        capabilities = self.get_model_capabilities(model_name)
        return capabilities.get('limitations', [])

    def validate_method(self, model_name: str, method: str) -> None:
        """
        Validate that a model supports a method. Raises exception if not.

        Args:
            model_name: Name of the model
            method: Decoding method to check

        Raises:
            UnsupportedMethodError: If method is not supported
        """
        if not self.supports_method(model_name, method):
            supported_methods = self.get_supported_methods(model_name)
            limitations = self.get_limitations(model_name)

            error_msg = (
                f"Model '{model_name}' does not support method '{method}'. "
                f"Supported methods: {supported_methods}. "
                f"Limitations: {limitations}"
            )
            raise UnsupportedMethodError(error_msg)

    def validate_beam_size(self, model_name: str, method: str, beam_size: int) -> None:
        """
        Validate beam size for beam search methods.

        Args:
            model_name: Name of the model
            method: Decoding method
            beam_size: Requested beam size

        Raises:
            UnsupportedMethodError: If beam size is not supported
        """
        if not method.startswith('beam'):
            return  # Not a beam search method

        if not self.supports_beam_search(model_name):
            raise UnsupportedMethodError(
                f"Model '{model_name}' does not support beam search"
            )

        max_beam_size = self.get_max_beam_size(model_name)
        if max_beam_size is not None and beam_size > max_beam_size:
            raise UnsupportedMethodError(
                f"Model '{model_name}' maximum beam size is {max_beam_size}, "
                f"requested {beam_size}"
            )



def validate_experiment_config(experiment_config: Dict[str, Any],
                               models_config: Dict[str, Any]) -> List[str]:
    """
    Validate an experiment configuration against model capabilities.

    Args:
        experiment_config: Experiment configuration
        models_config: Models configuration

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    capability_manager = CapabilityManager()

    models = experiment_config.get('models', [])
    methods = experiment_config.get('methods', [])

    for model_name in models:
        # Check if model exists
        if model_name not in models_config:
            errors.append(f"Model '{model_name}' not found in models configuration")
            continue

        for method in methods:
            try:
                # Validate method support
                capability_manager.validate_method(model_name, method)

                # Validate beam size if applicable
                if method.startswith('beam') and '_' in method:
                    beam_size = int(method.split('_')[1])
                    capability_manager.validate_beam_size(model_name, method, beam_size)

            except (UnsupportedMethodError, ValueError) as e:
                errors.append(f"Model '{model_name}', method '{method}': {str(e)}")

    return errors


def get_compatible_experiments(models_config: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Get compatible model/method combinations for experiment planning.

    Args:
        models_config: Models configuration

    Returns:
        Dictionary mapping experiment types to compatible models
    """
    capability_manager = CapabilityManager()

    experiments = {
        'beam_search_capable': [],
        'logprobs_capable': [],
        'full_logprobs_capable': [],
        'api_models': [],
        'local_models': []
    }

    for model_name in models_config.keys():
        try:
            capabilities = capability_manager.get_model_capabilities(model_name)
            model_type = models_config[model_name].get('type', 'unknown')

            if capability_manager.supports_beam_search(model_name):
                experiments['beam_search_capable'].append(model_name)

            if capability_manager.supports_logprobs(model_name):
                experiments['logprobs_capable'].append(model_name)

            if capability_manager.supports_full_logprobs(model_name):
                experiments['full_logprobs_capable'].append(model_name)

            if model_type in ['openai', 'anthropic']:
                experiments['api_models'].append(model_name)
            elif model_type == 'huggingface':
                experiments['local_models'].append(model_name)

        except Exception as e:
            # Skip models with capability issues
            print(f"Warning: Skipping model {model_name} due to capability error: {e}")
            continue

    return experiments