#!/usr/bin/env python
"""Basic usage example for transformers_patch."""

# First, import and apply the patch
import transformers_patch
transformers_patch.apply_patch()

# Then import and use transformers as normal
from transformers import AutoTokenizer, AutoModel

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

print("Loading model...")
model = AutoModel.from_pretrained("distilbert-base-uncased")

print("\nAll loaded successfully! You should have seen loading messages above.")
print("Try with other models to see the patch in action.") 