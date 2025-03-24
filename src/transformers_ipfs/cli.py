#!/usr/bin/env python
"""CLI for transformers_ipfs package."""

import argparse
import os
import site
import sys
from importlib import import_module
from pathlib import Path

from ._version import __version__


def apply_patch():
    """Load the transformers patch module."""
    # Try to import the module, installing if necessary
    try:
        # Check if transformers is installed
        import_module("transformers")
    except ImportError:
        print("❌ Transformers is not installed. Please install it first:")
        print("   pip install transformers")
        return False

    # Now try to apply the patch
    try:
        import transformers_ipfs

        # The module auto-applies the patch when imported
        return True
    except ImportError:
        print("❌ Could not import transformers_ipfs. Make sure it's installed:")
        print("   pip install transformers-ipfs")
        return False


def activate_command():
    """Activate transformers-ipfs integration."""
    # Create a .pth file in the site-packages directory
    site_packages = site.getsitepackages()
    user_site = site.getusersitepackages()

    # Try system site-packages first
    success = False
    for sp in site_packages:
        try:
            sp_path = Path(sp)
            pth_file = sp_path / "transformers_ipfs.pth"

            with open(pth_file, "w") as f:
                f.write("import transformers_ipfs")

            print(f"✅ Created {pth_file}")
            success = True
            break
        except (PermissionError, OSError):
            continue

    # Try user site-packages if system site-packages didn't work
    if not success:
        try:
            os.makedirs(user_site, exist_ok=True)
            pth_file = Path(user_site) / "transformers_ipfs.pth"

            with open(pth_file, "w") as f:
                f.write("import transformers_ipfs")

            print(f"✅ Created {pth_file}")
            success = True
        except (PermissionError, OSError) as e:
            print(f"❌ Error: {e}")

    if success:
        print("\n🎉 transformers-ipfs integration activated successfully!")
        print("   It will be automatically loaded in all Python sessions.")
    else:
        print("\n❌ Failed to activate transformers-ipfs integration.")


def deactivate_command():
    """Deactivate transformers-ipfs integration."""
    site_packages = site.getsitepackages()
    user_site = site.getusersitepackages()
    site_packages.append(user_site)

    any_removed = False
    for sp in site_packages:
        sp_path = Path(sp)
        pth_file = sp_path / "transformers_ipfs.pth"

        if pth_file.exists():
            try:
                pth_file.unlink()
                print(f"✅ Removed {pth_file}")
                any_removed = True
            except (PermissionError, OSError) as e:
                print(f"❌ Could not remove {pth_file}: {e}")

    if any_removed:
        print("\n🎉 transformers-ipfs integration deactivated successfully!")
    else:
        print("\n❌ No transformers-ipfs integration files found to deactivate.")


# Function to handle the status command
def status_command():
    """Check the status of transformers-ipfs integration."""
    # Check if any .pth files exist
    site_packages = site.getsitepackages()
    user_site = site.getusersitepackages()
    site_packages.append(user_site)

    activated_files = []
    for sp in site_packages:
        sp_path = Path(sp)
        pth_file = sp_path / "transformers_ipfs.pth"
        if pth_file.exists():
            activated_files.append(str(pth_file))

    if activated_files:
        print(f"✅ transformers-ipfs integration is activated")
        print(f"   Version: {__version__}")
        print("   Activation files:")
        for file in activated_files:
            print(f"   - {file}")
    else:
        print("❌ transformers-ipfs integration is not activated")
        print("   Run 'transformers-ipfs activate' to activate it")


# Function to handle the test command
def test_command():
    """Test transformers-ipfs integration."""
    import importlib.util

    # Check if transformers is installed
    if importlib.util.find_spec("transformers") is None:
        print("❌ transformers is not installed")
        print("   Please install it with: pip install transformers")
        return False

    print(f"✅ transformers-ipfs version: {__version__}")

    # Try to import and see if our patch is active
    import transformers

    if hasattr(transformers, "ipfs_enabled"):
        print("✅ transformers-ipfs integration is active")
    else:
        print("❌ transformers-ipfs integration is not active")
        print("   Run 'transformers-ipfs activate' to enable it")
        return False

    print("\nTesting IPFS functionality...")
    print("Attempting to load a small test file from IPFS...")

    try:
        # Try to load a very small model config from IPFS
        transformers.AutoConfig.from_pretrained(
            "ipfs://bafkreihs7wnevyvkdh33qo6vbp6d72b33zgqzlexvl4l5s5ygku3ixpm24"
        )
        print("🎉 transformers-ipfs is working correctly.")
        return True
    except Exception as e:
        print(f"❌ Error loading from IPFS: {e}")
        print("   This may indicate a problem with the transformers-ipfs integration.")
        return False


def main():
    """Main entry point for transformers-ipfs CLI."""
    parser = argparse.ArgumentParser(
        description="transformers-ipfs command line interface"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Activate command
    activate_parser = subparsers.add_parser(
        "activate", help="Activate transformers-ipfs integration"
    )

    # Deactivate command
    deactivate_parser = subparsers.add_parser(
        "deactivate", help="Deactivate transformers-ipfs integration"
    )

    # Status command
    status_parser = subparsers.add_parser(
        "status", help="Check status of transformers-ipfs integration"
    )

    # Test command
    test_parser = subparsers.add_parser(
        "test", help="Test transformers-ipfs integration"
    )

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")

    # Parse arguments
    args = parser.parse_args()

    # Handle commands
    if args.command == "activate":
        activate_command()
    elif args.command == "deactivate":
        deactivate_command()
    elif args.command == "status":
        status_command()
    elif args.command == "test":
        result = test_command()
        sys.exit(0 if result else 1)
    elif args.command == "version":
        print(f"transformers-ipfs version {__version__}")
    else:
        # Default to showing help if no command provided
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
