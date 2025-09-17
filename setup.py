"""Setup script for text degeneration experiments."""

from setuptools import setup, find_packages

setup(
    name="text-degeneration-experiments",
    version="0.1.0",
    description="Testing modern LLM decoding methods",
    author="Mitchell Gordon",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "nltk>=3.8",
        "scikit-learn>=1.3.0",
        "python-dotenv>=1.0.0",
        "tenacity>=8.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "analysis": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "jupyter>=1.0.0",
        ]
    }
)