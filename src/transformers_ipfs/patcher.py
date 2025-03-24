"""
Transformers patcher implementation.

This module contains the implementation of the TransformersPatcher class
that patches the Hugging Face Transformers library to display
information when loading models and tokenizers.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union
from urllib.parse import urlparse

import requests

from model_patch import BasePatcher

# Add a module-level flag to prevent duplicate patching
_PATCHING_IN_PROGRESS = False


class TransformersPatcher(BasePatcher):
    """Patcher for the Hugging Face Transformers library.

    This class patches the from_pretrained methods of Transformers classes
    to display information when loading models and tokenizers.
    """

    def __init__(self):
        """Initialize the patcher."""
        super().__init__()
        # Track original methods for proper cleanup
        self.original_methods = {}
        # Create a cache directory for IPFS downloads
        self.ipfs_cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "ipfs", "models"
        )
        os.makedirs(self.ipfs_cache_dir, exist_ok=True)

    def get_target_module_name(self) -> str:
        """Get the name of the module to patch.

        Returns:
            The name of the module to patch
        """
        return "transformers"

    def get_emoji(self) -> str:
        """Get the emoji to use for model loading announcements.

        Returns:
            The emoji as a string
        """
        return "🤗"

    def download_ipfs(self, cid: str, cache_dir: Optional[str] = None) -> str:
        """Download a model from IPFS.

        Args:
            cid: IPFS CID
            cache_dir: Directory to store cached models

        Returns:
            Path to the downloaded model directory
        """
        if cache_dir is None:
            cache_dir = self.ipfs_cache_dir

        # Create model directory in cache
        model_dir = os.path.join(cache_dir, cid)

        # If model already exists in cache and has config.json, use it
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            # Validate that it contains essential files like config.json
            if os.path.exists(os.path.join(model_dir, "config.json")):
                # For sharded models, check if index and all shards exist
                index_file = os.path.join(model_dir, "model.safetensors.index.json")
                if os.path.exists(index_file):
                    try:
                        with open(index_file, "r") as f:
                            index_data = json.load(f)
                        # Check if all shards exist
                        weight_map = index_data.get("weight_map", {})
                        shard_files = set(weight_map.values())
                        missing_shards = [
                            shard
                            for shard in shard_files
                            if not os.path.exists(os.path.join(model_dir, shard))
                        ]

                        if missing_shards:
                            print(
                                f"🔶 Cached model at {model_dir} is missing {len(missing_shards)} shards, redownloading..."
                            )
                            # Rename the incomplete directory
                            incomplete_dir = (
                                f"{model_dir}_incomplete_{int(time.time())}"
                            )
                            if os.path.exists(incomplete_dir):
                                shutil.rmtree(incomplete_dir)
                            shutil.move(model_dir, incomplete_dir)
                        else:
                            print(f"🔄 Using cached model from {model_dir}")
                            return model_dir
                    except Exception:
                        print(f"🔶 Error validating sharded model")
                        # Continue with redownload
                else:
                    print(f"🔄 Using cached model from {model_dir}")
                    return model_dir
            else:
                print(
                    f"🔶 Cached model at {model_dir} appears incomplete, redownloading..."
                )
                # Rename the incomplete directory
                incomplete_dir = f"{model_dir}_incomplete_{int(time.time())}"
                if os.path.exists(incomplete_dir):
                    shutil.rmtree(incomplete_dir)
                shutil.move(model_dir, incomplete_dir)

        # Create a temp directory for staging the download
        temp_model_dir = f"{model_dir}_temp_{int(time.time())}"
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)
        os.makedirs(temp_model_dir, exist_ok=True)

        # First check if local IPFS daemon is running
        ipfs_daemon_available = False
        try:
            # Try to connect to the IPFS API
            ipfs_api_url = "http://127.0.0.1:5001/api/v0/id"
            response = requests.post(ipfs_api_url, timeout=10)

            if response.status_code == 200:
                ipfs_daemon_available = True
                print(f"🌐 Local IPFS node detected, using it to download {cid}")

                # Use subprocess to call ipfs get
                try:
                    cmd = ["ipfs", "get", "-o", os.path.join(temp_model_dir, cid), cid]
                    print(f"   Running: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=600
                    )

                    if result.returncode == 0:
                        # Find the downloaded content - should be in a subfolder named with the CID
                        downloaded_path = os.path.join(temp_model_dir, cid)

                        # Verify download was successful
                        if os.path.exists(downloaded_path) and os.path.isdir(
                            downloaded_path
                        ):
                            # Check for config.json which is essential for transformer models
                            if os.path.exists(
                                os.path.join(downloaded_path, "config.json")
                            ):
                                # Make the final directory
                                os.makedirs(model_dir, exist_ok=True)

                                # Copy all files to the model directory
                                for item in os.listdir(downloaded_path):
                                    src = os.path.join(downloaded_path, item)
                                    dst = os.path.join(model_dir, item)
                                    if os.path.isdir(src):
                                        shutil.copytree(src, dst, dirs_exist_ok=True)
                                    else:
                                        shutil.copy2(src, dst)

                                print(
                                    f"✅ Model downloaded successfully using local IPFS node to {model_dir}"
                                )

                                # Clean up
                                shutil.rmtree(temp_model_dir)

                                # Check if we need to add required tokenizer files
                                # GPT-2 tokenizers require merges.txt, but sometimes it's missing
                                if os.path.exists(
                                    os.path.join(model_dir, "vocab.json")
                                ) and not os.path.exists(
                                    os.path.join(model_dir, "merges.txt")
                                ):
                                    print(
                                        "   🔶 Model has a vocab.json but is missing merges.txt, creating an empty one (required for GPT-2 tokenizers)"
                                    )
                                    with open(
                                        os.path.join(model_dir, "merges.txt"), "w"
                                    ) as f:
                                        # Write an empty file - this is often enough for GPT-2 tokenizers
                                        pass

                                return model_dir
                            else:
                                print(
                                    f"🔶 Local IPFS download succeeded but missing config.json"
                                )
                                # Fall back to HTTP gateway download
                        else:
                            print(
                                f"🔶 Local IPFS download succeeded but directory structure is unexpected"
                            )
                            # Fall back to HTTP gateway download
                    else:
                        print(f"🔶 Local IPFS download failed")
                        # Fall back to HTTP gateway download
                except (subprocess.SubprocessError, TimeoutError) as e:
                    print(f"🔶 Error using local IPFS node")
                    # Fall back to HTTP gateway download
        except requests.RequestException as e:
            # Don't print a message here, we'll print it below
            ipfs_daemon_available = False

        # Fall back to HTTP gateway download if daemon isn't available or download failed
        if not ipfs_daemon_available:
            print(f"🔵 Local IPFS node not available, using IPFS gateway")

        # Fall back to HTTP gateway download
        print(f"📥 Downloading model files from IPFS: {cid}")

        # Essential files to check for in a Hugging Face model
        essential_files = [
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
        ]
        model_files = ["model.safetensors", "pytorch_model.bin"]

        # Try different IPFS gateways
        gateways = [f"https://ipfs.io/ipfs/{cid}", f"https://dweb.link/ipfs/{cid}"]

        successful_gateway = None
        for gateway in gateways:
            try:
                # First check if the gateway can respond with config.json
                config_url = f"{gateway}/config.json"
                ipfs_uri = f"ipfs://{cid}/config.json"
                response = requests.head(config_url, timeout=10)

                if response.status_code == 200:
                    successful_gateway = gateway
                    print(f"   Using gateway via: {ipfs_uri}")
                    break
                else:
                    print(
                        f"   Gateway via {ipfs_uri} returned {response.status_code}, trying next gateway..."
                    )
            except requests.RequestException as e:
                print(f"   Error with gateway via {ipfs_uri}")

        if not successful_gateway:
            print("❌ Could not find a working gateway for this CID")
            shutil.rmtree(temp_model_dir)
            raise RuntimeError(f"Failed to find working gateway for CID: {cid}")

        # Create model directory to store downloaded files
        os.makedirs(model_dir, exist_ok=True)

        # Download essential files
        download_success = True
        for filename in essential_files:
            file_url = f"{successful_gateway}/{filename}"
            file_path = os.path.join(model_dir, filename)

            try:
                response = requests.get(file_url, timeout=30)
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    print(f"   ✓ Downloaded {filename}")
                else:
                    print(
                        f"   ✗ File {filename} not available (status: {response.status_code})"
                    )
                    # Not all models have all files, so continue unless it's config.json
                    if filename == "config.json":
                        download_success = False
                        break
            except requests.RequestException as e:
                print(f"   ✗ Error downloading {filename}")
                if filename == "config.json":
                    download_success = False
                    break

        if not download_success:
            print("❌ Failed to download essential file config.json")
            shutil.rmtree(model_dir)
            shutil.rmtree(temp_model_dir)
            raise RuntimeError(f"Failed to download config.json for CID: {cid}")

        # Check for model files
        # First check if there's a model.safetensors.index.json file indicating a sharded model
        index_url = f"{successful_gateway}/model.safetensors.index.json"
        is_sharded = False

        try:
            response = requests.head(index_url, timeout=10)
            if response.status_code == 200:
                is_sharded = True
                print("   Model uses sharded weights, downloading index file...")

                # Download the index file
                response = requests.get(index_url, timeout=30)
                index_path = os.path.join(model_dir, "model.safetensors.index.json")
                with open(index_path, "wb") as f:
                    f.write(response.content)

                try:
                    # Parse the index to get shard filenames
                    index_data = response.json()
                    weight_map = index_data.get("weight_map", {})
                    shard_files = set(weight_map.values())

                    print(f"   Found {len(shard_files)} shards to download")

                    # Download each shard
                    for shard in shard_files:
                        shard_url = f"{successful_gateway}/{shard}"
                        shard_path = os.path.join(model_dir, shard)

                        print(f"   Downloading model shard: {shard}")
                        try:
                            # Stream download with progress
                            with requests.get(shard_url, stream=True, timeout=300) as r:
                                r.raise_for_status()
                                total_size = int(r.headers.get("content-length", 0))

                                with open(shard_path, "wb") as f:
                                    downloaded = 0
                                    for chunk in r.iter_content(chunk_size=8192 * 1024):
                                        f.write(chunk)
                                        downloaded += len(chunk)

                                        # Print progress
                                        if total_size > 0:
                                            percent = int(100 * downloaded / total_size)
                                            print(
                                                f"\r      {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({percent}%)",
                                                end="",
                                            )

                                    print()  # New line after progress

                            # Verify shard was downloaded correctly
                            if os.path.getsize(shard_path) == 0:
                                print(f"   ✗ Downloaded shard {shard} is empty")
                                download_success = False
                                break

                            print(f"   ✓ Downloaded shard: {shard}")
                        except requests.RequestException as e:
                            print(f"   ✗ Failed to download shard {shard}")
                            download_success = False
                            break
                except Exception as e:
                    print(f"   ✗ Error processing index file")
                    download_success = False
        except requests.RequestException:
            is_sharded = False

        # If not sharded or shard download failed, try to download a single model file
        if not is_sharded or not download_success:
            download_success = False
            for model_file in model_files:
                model_url = f"{successful_gateway}/{model_file}"
                model_path = os.path.join(model_dir, model_file)

                try:
                    # Check if the file exists
                    response = requests.head(model_url, timeout=10)
                    if response.status_code != 200:
                        print(
                            f"   ✗ Model file {model_file} not found, trying next option..."
                        )
                        continue

                    print(f"   Downloading {model_file}...")
                    # Stream download with progress
                    with requests.get(model_url, stream=True, timeout=300) as r:
                        r.raise_for_status()
                        total_size = int(r.headers.get("content-length", 0))

                        with open(model_path, "wb") as f:
                            downloaded = 0
                            for chunk in r.iter_content(chunk_size=8192 * 1024):
                                f.write(chunk)
                                downloaded += len(chunk)

                                # Print progress
                                if total_size > 0:
                                    percent = int(100 * downloaded / total_size)
                                    print(
                                        f"\r      {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({percent}%)",
                                        end="",
                                    )

                            print()  # New line after progress

                    # Verify file was downloaded correctly
                    if os.path.getsize(model_path) > 0:
                        print(f"   ✓ Downloaded {model_file}")
                        download_success = True
                        break
                    else:
                        print(f"   ✗ Downloaded file {model_file} is empty")
                except requests.RequestException as e:
                    print(f"   ✗ Error downloading {model_file}")

        # Final validation
        if not download_success:
            print("❌ Failed to download model weights")
            shutil.rmtree(model_dir)
            shutil.rmtree(temp_model_dir)
            raise RuntimeError(f"Failed to download model weights for CID: {cid}")

        # Clean up temp directory
        shutil.rmtree(temp_model_dir)

        # Final sanity check - make sure we have config.json in the final directory
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            print("❌ Final model directory is missing config.json")
            raise RuntimeError("Model directory is incomplete after download")

        # Check if we need to add required tokenizer files
        # GPT-2 tokenizers require merges.txt, but sometimes it's missing
        if os.path.exists(os.path.join(model_dir, "vocab.json")) and not os.path.exists(
            os.path.join(model_dir, "merges.txt")
        ):
            print(
                "   🔶 Model has a vocab.json but is missing merges.txt, creating an empty one (required for GPT-2 tokenizers)"
            )
            with open(os.path.join(model_dir, "merges.txt"), "w") as f:
                # Write an empty file - this is often enough for GPT-2 tokenizers
                pass

        print(f"✅ Model downloaded successfully to {model_dir}")
        return model_dir

    def handle_ipfs_model(
        self, model_id: str, cache_dir: Optional[str] = None, **kwargs
    ) -> str:
        """Handle IPFS model references.

        Args:
            model_id: Model ID, expected to start with "ipfs://"
            cache_dir: Cache directory to store models

        Returns:
            Local path to the model directory
        """
        # Extract CID from model_id
        parsed = urlparse(model_id)
        if parsed.scheme != "ipfs":
            raise ValueError(f"Expected 'ipfs://' scheme, got: {model_id}")

        cid = parsed.netloc
        if not cid:
            # Handle ipfs://CID format (without netloc)
            cid = parsed.path.lstrip("/")

        # Check if the path contains a directory path after the CID
        subpath = ""
        if "/" in cid:
            parts = cid.split("/", 1)
            cid = parts[0]
            subpath = parts[1]
            print(f"Note: IPFS path includes subdirectory: {subpath}")

        # Download or get from cache
        return self.download_ipfs(cid, cache_dir)

    def create_patched_function(self, original: callable, class_name: str) -> callable:
        """Create a patched version of a function.

        Args:
            original: The original function to patch
            class_name: The name of the class the function belongs to

        Returns:
            The patched function
        """
        emoji = self.get_emoji()

        def wrapped(*args, **kwargs):
            # Get the model name from args or kwargs
            pretrained_model_name = None
            if args and isinstance(args[0], str):
                pretrained_model_name = args[0]
            elif "pretrained_model_name_or_path" in kwargs:
                pretrained_model_name = kwargs["pretrained_model_name_or_path"]

            # Check if it's an IPFS model
            if pretrained_model_name and pretrained_model_name.startswith("ipfs://"):
                try:
                    print(
                        f"{emoji} Loading {class_name} from IPFS: {pretrained_model_name}..."
                    )

                    # Get cache directory from kwargs or use default
                    cache_dir = kwargs.get("cache_dir", None)

                    # Download or get from cache
                    local_model_path = self.handle_ipfs_model(
                        pretrained_model_name, cache_dir
                    )

                    # Replace the IPFS URL with the local path
                    if args and args[0] == pretrained_model_name:
                        args = list(args)
                        args[0] = local_model_path
                        args = tuple(args)
                    elif "pretrained_model_name_or_path" in kwargs:
                        kwargs["pretrained_model_name_or_path"] = local_model_path

                    # Print success message
                    print(
                        f"✅ Successfully prepared IPFS model: {pretrained_model_name}"
                    )
                except Exception as e:
                    print(f"❌ Error loading IPFS model: {pretrained_model_name}")
                    raise
            else:
                # Print loading message for non-IPFS models
                if pretrained_model_name:
                    print(
                        f"{emoji} Loading {class_name} from {pretrained_model_name}..."
                    )
                else:
                    print(f"{emoji} Loading {class_name}...")

            # Call the original function with potentially modified args
            result = original(*args, **kwargs)
            return result

        # Mark as patched to avoid re-patching
        wrapped._is_patched = True
        return wrapped

    def patch_module(self, module) -> bool:
        """Patch the transformers module.

        Args:
            module: The transformers module to patch

        Returns:
            True if patching was successful, False otherwise
        """
        # Check if patching is already in progress to prevent recursive patching
        global _PATCHING_IN_PROGRESS
        if _PATCHING_IN_PROGRESS:
            return False

        # Set patching flag to prevent recursive calls
        _PATCHING_IN_PROGRESS = True

        # First check if we have already tried to patch this module
        # to avoid infinite recursion when importing transformers again
        # Set a flag to avoid double patching
        os.environ["TRANSFORMERS_IPFS_ACTIVE"] = "1"

        # Check if already patched
        if hasattr(module, "_transformers_ipfs_attempted"):
            _PATCHING_IN_PROGRESS = False
            return getattr(module, "_transformers_ipfs_result", False)

        # Mark as attempted to prevent duplicate messages
        setattr(module, "_transformers_ipfs_attempted", True)

        try:
            # Classes to patch - more comprehensive list based on the original implementation
            classes_to_patch = [
                "AutoBackbone",
                "AutoConfig",
                "AutoFeatureExtractor",
                "AutoImageProcessor",
                "AutoModel",
                "AutoModelForAudioClassification",
                "AutoModelForAudioFrameClassification",
                "AutoModelForAudioXVector",
                "AutoModelForCausalLM",
                "AutoModelForCTC",
                "AutoModelForDepthEstimation",
                "AutoModelForDocumentQuestionAnswering",
                "AutoModelForImageClassification",
                "AutoModelForMaskedImageModeling",
                "AutoModelForMaskedLM",
                "AutoModelForMultipleChoice",
                "AutoModelForNextSentencePrediction",
                "AutoModelForObjectDetection",
                "AutoModelForPreTraining",
                "AutoModelForQuestionAnswering",
                "AutoModelForSemanticSegmentation",
                "AutoModelForSequenceClassification",
                "AutoModelForSpeechSeq2Seq",
                "AutoModelForTableQuestionAnswering",
                "AutoModelForTokenClassification",
                "AutoModelForVideoClassification",
                "AutoModelForVision2Seq",
                "AutoModelForVisualQuestionAnswering",
                "AutoModelForZeroShotImageClassification",
                "AutoModelForZeroShotObjectDetection",
                "AutoProcessor",
                "AutoTokenizer",
                "TFAutoModel",
                "TFAutoModelForMaskedLM",
            ]

            # We'll patch these even if we skip others with dependency issues
            core_classes = ["AutoModel", "AutoTokenizer", "AutoConfig", "AutoProcessor"]

            # Track patched classes
            patched_classes = []
            skipped_classes = []

            # Initialize method storage if needed
            if not hasattr(self, "original_methods"):
                self.original_methods = {}

            # Patch each class if it exists
            for class_name in classes_to_patch:
                try:
                    # Safely check if the class exists without triggering import errors
                    cls = None
                    if hasattr(module, class_name):
                        # Get the class cautiously
                        try:
                            cls = getattr(module, class_name)
                        except (ImportError, AttributeError):
                            # Skip if there's an import error (missing dependency)
                            skipped_classes.append(class_name)
                            continue

                        # Check if from_pretrained is available
                        if hasattr(cls, "from_pretrained"):
                            # Check if already patched by transformers_ipfs
                            if hasattr(cls.from_pretrained, "_is_patched"):
                                continue

                            # Get the original method
                            original = cls.from_pretrained

                            # Store the original method for cleanup
                            if class_name not in self.original_methods:
                                self.original_methods[class_name] = {}
                            self.original_methods[class_name][
                                "from_pretrained"
                            ] = original

                            # Create a patched version
                            wrapped = self.create_patched_function(original, class_name)

                            # Replace the original method but preserve llama-patch patching
                            if hasattr(original, "_llama_patch_hook"):
                                wrapped._llama_patch_hook = True

                            # Store the patched method
                            setattr(cls, "from_pretrained", wrapped)

                            # Track patched classes
                            patched_classes.append(class_name)
                except Exception:
                    # Skip any other errors and continue
                    skipped_classes.append(class_name)
                    continue

            # Print summary
            if patched_classes:
                transformers_version = getattr(module, "__version__", "unknown")
                print(
                    f"✅ Patched {len(patched_classes)} transformer classes in version {transformers_version}"
                )
                print(f"   Examples: {', '.join(patched_classes[:5])}...")

                # If we didn't patch any core classes, that's a problem
                if not any(cls in patched_classes for cls in core_classes):
                    print(
                        "🔶 Warning: Failed to patch core classes - patch may not be fully functional"
                    )

                # Mark as successful
                setattr(module, "_transformers_ipfs_result", True)
                return True
            else:
                print("❌ Failed to patch any transformer classes")
                return False
        finally:
            # Clear the environment flag when done
            if "TRANSFORMERS_IPFS_ACTIVE" in os.environ:
                del os.environ["TRANSFORMERS_IPFS_ACTIVE"]

            # Reset patching flag
            _PATCHING_IN_PROGRESS = False

    def _cleanup(self):
        """Clean up patches when Python exits"""
        module_name = self.get_target_module_name()
        if (
            module_name in sys.modules
            and hasattr(self, "original_methods")
            and self.original_methods
        ):
            try:
                module = sys.modules[module_name]
                for class_name, methods in self.original_methods.items():
                    if hasattr(module, class_name):
                        cls = getattr(module, class_name)
                        for method_name, original in methods.items():
                            if hasattr(cls, method_name):
                                setattr(cls, method_name, original)
            except Exception:
                # Silently fail during cleanup
                pass
