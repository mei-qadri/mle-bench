"""Setup script for MLE-bench Multi-Agent System."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="mle-bench-agents",
    version="1.0.0",
    author="Anthropic MLE-bench Team",
    description="Multi-agent system for solving MLE-bench competitions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anthropics/mle-bench",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        # Core dependencies
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        
        # ML libraries
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2.0",
        
        # Deep learning (optional, install separately)
        # "torch>=2.0.0",
        # "transformers>=4.30.0",
        # "timm>=0.9.0",
        
        # HPO
        "optuna>=3.0.0",
        
        # Monitoring
        "psutil>=5.9.0",
        "GPUtil>=1.4.0",
        
        # Development
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
    ],
    extras_require={
        "torch": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
        ],
        "transformers": [
            "transformers>=4.30.0",
            "sentencepiece>=0.1.99",
        ],
        "vision": [
            "opencv-python>=4.8.0",
            "albumentations>=1.3.0",
            "timm>=0.9.0",
        ],
        "audio": [
            "librosa>=0.10.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mle-bench-agent=mle_bench_agents.cli:main",
        ],
    },
)
