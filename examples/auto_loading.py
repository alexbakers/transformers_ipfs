#!/usr/bin/env python
"""
Demonstrates the auto-loading feature of transformers_patch.

Before running this:
1. Install transformers_patch with: pip install -e .
2. Run: transformers-patch install
3. Then run this script: python auto_loading.py

The patch should be automatically applied when importing transformers!
"""

# Notice that we don't import transformers_patch first!
# The .pth file should handle the auto-loading.

print("Importing transformers...")
from transformers import AutoTokenizer, AutoModel

print("\nLoading a tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("\nLoading a model...")
model = AutoModel.from_pretrained("gpt2")

print("\nIf you saw loading messages with the 🤗 emoji, auto-loading is working!") 