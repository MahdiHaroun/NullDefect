"""
setup.py — makes src/ importable as a package.
Install in dev mode: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="defect-detection-mlops",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
