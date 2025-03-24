#!/usr/bin/env python
"""
Demonstrates how to use the transformers-patch CLI.

This script shows how to use the transformers-patch command-line tool
to activate, check status, test, and deactivate the patch.
"""

import subprocess
import sys

def run_command(cmd):
    """Run a command and print its output."""
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    return result.returncode

def main():
    """Run the CLI examples."""
    # Check if transformers-patch is installed
    if run_command("which transformers-patch") != 0:
        print("transformers-patch is not installed or not in PATH.")
        print("Install it first with: pip install -e .")
        return 1
        
    # Show the version
    run_command("transformers-patch --version")
    
    # Show the current status
    run_command("transformers-patch status")
    
    # Run a test to verify patch functionality
    run_command("transformers-patch test")
    
    # Activate the auto-loading feature
    print("\nActivating auto-loading feature...")
    run_command("transformers-patch activate")
    
    # Show status again to verify activation
    run_command("transformers-patch status")
    
    # Deactivate (commented out to not disrupt your installation)
    # print("\nDeactivating auto-loading feature...")
    # run_command("transformers-patch deactivate")
    
    # Show status one more time
    # run_command("transformers-patch status")
    
    print("\nCLI demonstration complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 