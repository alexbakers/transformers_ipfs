"""Basic tests for transformers_ipfs package."""

import importlib
import sys

import pytest


def test_import():
    """Test that the package can be imported."""
    try:
        import transformers_ipfs

        assert transformers_ipfs.__version__ is not None
    except ImportError:
        pytest.fail("Failed to import transformers_ipfs")


def test_apply_patch():
    """Test that the apply_patch function exists."""
    import transformers_ipfs

    assert hasattr(transformers_ipfs, "apply_patch")
    assert callable(transformers_ipfs.apply_patch)


def test_is_active():
    """Test that the is_active function exists and can be called."""
    import transformers_ipfs

    assert hasattr(transformers_ipfs, "is_active")
    assert callable(transformers_ipfs.is_active)
    # Just check that it can be called without errors
    transformers_ipfs.is_active()


def test_patcher():
    """Test that patcher instance exists."""
    import transformers_ipfs

    assert hasattr(transformers_ipfs, "patcher")
    assert transformers_ipfs.patcher is not None


if __name__ == "__main__":
    # Run the tests
    test_import()
    test_apply_patch()
    test_is_active()
    test_patcher()
    print("All tests passed!")
