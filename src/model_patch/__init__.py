"""
Base Patcher Module

This module provides a base class for creating patchers for different model libraries.
"""

import atexit
import functools
import importlib.util
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Type

# Track original import function and patched methods for cleanup
_ORIGINAL_IMPORT = __import__
_PATCHED_METHODS: Dict[str, Callable] = {}


class BasePatcher(ABC):
    """Base class for creating patchers for model libraries.

    This abstract class provides the infrastructure for patching model loading
    methods in various ML libraries.
    """

    def __init__(self):
        """Initialize the patcher."""
        self.is_applied = False
        self.patched_methods = {}

        # Register cleanup on exit
        atexit.register(self._cleanup)

    @abstractmethod
    def get_target_module_name(self) -> str:
        """Get the name of the module to patch.

        Returns:
            The name of the module to patch
        """
        pass

    @abstractmethod
    def patch_module(self, module) -> bool:
        """Patch the target module.

        Args:
            module: The module to patch

        Returns:
            True if patching was successful, False otherwise
        """
        pass

    def get_emoji(self) -> str:
        """Get the emoji to use for model loading announcements.

        Returns:
            The emoji as a string
        """
        return "👉"

    def create_patched_function(self, original: Callable, class_name: str) -> Callable:
        """Create a patched version of a function.

        Args:
            original: The original function to patch
            class_name: The name of the class the function belongs to

        Returns:
            The patched function
        """
        emoji = self.get_emoji()

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            # Get the model name from args or kwargs
            pretrained_model_name = None
            if args and isinstance(args[0], str):
                pretrained_model_name = args[0]
            elif "pretrained_model_name_or_path" in kwargs:
                pretrained_model_name = kwargs["pretrained_model_name_or_path"]

            # Print loading message
            if pretrained_model_name:
                print(f"{emoji} Loading {class_name} from {pretrained_model_name}...")
            else:
                print(f"{emoji} Loading {class_name}...")

            # Call the original function
            result = original(*args, **kwargs)
            return result

        # Mark as patched to avoid re-patching
        wrapped._is_patched = True
        return wrapped

    def apply_patch(self) -> bool:
        """Apply the patch to the target module.

        Returns:
            True if patching was successful, False otherwise
        """
        # If already applied, return True
        if self.is_applied:
            return True

        # Get the module name
        module_name = self.get_target_module_name()

        # Check if the module is already imported
        if module_name in sys.modules:
            # Module is already imported, patch it
            module = sys.modules[module_name]
            success = self.patch_module(module)
            self.is_applied = success
            return success

        # Check if the module is available
        if importlib.util.find_spec(module_name) is None:
            # Module is not installed
            return False

        # Get current import function
        current_import = __import__

        # Create a safer import hook that preserves existing patches
        def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Use current import function which might already be patched by other libraries
            module = current_import(name, globals, locals, fromlist, level)

            # If this is our target module, patch it
            if name == module_name:
                try:
                    self.patch_module(module)
                    self.is_applied = True
                except Exception:
                    # Don't crash if patching fails
                    pass

            return module

        # Store the original import for potential cleanup
        global _ORIGINAL_IMPORT
        _ORIGINAL_IMPORT = current_import

        # Mark this import hook
        patched_import._model_patch_hook = True

        # Replace the import function
        __builtins__["__import__"] = patched_import

        return True

    def _cleanup(self):
        """Clean up patches when Python exits"""
        # Restore original methods if needed
        module_name = self.get_target_module_name()
        if module_name in sys.modules and self.patched_methods:
            try:
                for class_name, methods in self.patched_methods.items():
                    for method_name, original in methods.items():
                        # Try to restore the original method
                        pass  # Implementation depends on how patching is done
            except Exception:
                # Silently fail during cleanup
                pass


# Function to restore original import when Python exits
def _cleanup_import_hooks():
    """Restore original import function when Python exits"""
    if hasattr(__builtins__["__import__"], "_model_patch_hook"):
        __builtins__["__import__"] = _ORIGINAL_IMPORT


# Register cleanup function
atexit.register(_cleanup_import_hooks)
