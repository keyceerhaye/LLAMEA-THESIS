"""
Setup script for LLAMEA-MADA
LLM-based Metaheuristic Algorithm Design and Adaptation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="llamea-mada",
    version="1.0.0",
    description="LLM-based Metaheuristic Algorithm Design and Adaptation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LLAMEA Research Team",
    author_email="",
    url="https://github.com/yourrepo/llamea-mada",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26.3,<2",
        "pandas>=2.0.3",
        "openai>=1.99.1,<2",
        "tqdm>=4.66.4,<5",
        "ollama>=0.2.1,<0.3",
        "jsonlines>=4.0.0,<5",
        "configspace>=1.2.0,<2",
        "google-generativeai>=0.8.1,<0.9",
        "joblib>=1.4.2,<2",
        "ioh>=0.3.18",
        "python-dotenv>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.2,<0.14",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pytest-cov>=3.0.0",
        ],
        "hpo": [
            "smac>=2.2.0,<3",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="evolutionary-algorithms llm metaheuristics optimization bbob",
)






