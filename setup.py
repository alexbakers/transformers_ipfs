#!/usr/bin/env python
"""
Setup script for transformers_ipfs package.
Note: The PyPI package name uses hyphens (transformers-ipfs), but the Python module uses underscores (transformers_ipfs).
"""

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Use a minimal setup.py for compatibility
# Dependencies are defined in pyproject.toml
setup(
    long_description=long_description,
    long_description_content_type="text/markdown",
) 