"""
Transformers IPFS Integration

Enable IPFS model loading for the Hugging Face Transformers library
when loading models and tokenizers.
"""

import atexit
import builtins
import os
import sys

# Set the environment variable before any other imports to prevent llama_ipfs messages
os.environ["TRANSFORMERS_IPFS_ACTIVE"] = "1"

# Suppress any other patchers' output during our initialization
original_print = builtins.print


def filtered_print(*args, **kwargs):
    # Filter out llama_ipfs messages
    if (
        args
        and isinstance(args[0], str)
        and "Llama-cpp-python IPFS integration active" in args[0]
    ):
        return
    return original_print(*args, **kwargs)


# Replace print with our filtered version
builtins.print = filtered_print

# Import version
from ._version import __version__

# Import patch functionality
from .patcher import TransformersPatcher


# Clean-up function
def _cleanup():
    # Restore original print function
    builtins.print = original_print


# Register the cleanup function
atexit.register(_cleanup)

# Create a patcher instance
_patcher = TransformersPatcher()

# Apply the patch
_patcher.apply_patch()


# Function to check if IPFS integration is active
def is_active():
    """Check if transformers IPFS integration is active"""
    # Check for patched classes in transformers
    try:
        import transformers

        for name in ["AutoModel", "AutoConfig"]:
            cls = getattr(transformers, name, None)
            if (
                cls
                and hasattr(cls, "from_pretrained")
                and hasattr(cls.from_pretrained, "_is_patched")
            ):
                return True
    except (ImportError, AttributeError):
        pass
    return _patcher.is_applied


# CLI-compatible function names
def activate():
    """Activate the transformers IPFS integration"""
    return _patcher.apply_patch()


def status():
    """Check if transformers IPFS integration is active (CLI-compatible)"""
    return is_active()


# Ensure cleanup happens on exit
def _ensure_cleanup():
    """Make sure cleanup happens on exit"""
    _patcher._cleanup()
    # Clear our environment variable
    if "TRANSFORMERS_IPFS_ACTIVE" in os.environ:
        del os.environ["TRANSFORMERS_IPFS_ACTIVE"]


# Register cleanup
atexit.register(_ensure_cleanup)

# Export patcher for tests
patcher = _patcher


# Expose apply_patch function for tests
def apply_patch():
    """Apply the transformers IPFS patch"""
    return _patcher.apply_patch()
