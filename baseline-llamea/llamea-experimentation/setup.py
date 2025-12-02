"""
Simple setup script for LLaMEA Experimentation
This allows the llamea package to be imported easily
"""

from setuptools import setup, find_packages

setup(
    name="llamea-experimentation",
    version="1.0.0",
    description="LLaMEA Experimentation Setup",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26.3,<2",
        "pandas==2.0.3",
        "openai>=1.99.1,<2",
        "tqdm>=4.66.4,<5",
        "ollama>=0.2.1,<0.3",
        "jsonlines>=4.0.0,<5",
        "configspace>=1.2.0,<2",
        "google-generativeai>=0.8.1,<0.9",
        "joblib>=1.4.2,<2",
        "ioh>=0.3.18",
        "python-dotenv>=1.0.0",
    ],
)






