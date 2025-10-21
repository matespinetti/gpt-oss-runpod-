#!/usr/bin/env python3
"""
OpenWebUI Mode - Direct vLLM OpenAI API Server
This mode exposes vLLM's OpenAI-compatible API directly for OpenWebUI
No RunPod serverless wrapper - just pure vLLM API server
"""

import os
import subprocess
import time
import signal
import sys

# Environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
HF_HOME = os.getenv("HF_HOME", "/runpod-volume/huggingface")
HF_TOKEN = os.getenv("HF_TOKEN", "")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.95")
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "8192")
DTYPE = os.getenv("DTYPE", "bfloat16")
VLLM_PORT = os.getenv("VLLM_PORT", "8000")

# Set HF cache
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = HF_HOME
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN

print("=" * 70)
print("vLLM OpenAI API Server - OpenWebUI Mode")
print("=" * 70)
print(f"Model: {MODEL_NAME}")
print(f"HF Home: {HF_HOME}")
print(f"Port: {VLLM_PORT}")
print("")

# Find model path
model_arg = MODEL_NAME
found_model = False

# Check for MODEL_PATH override
model_path_override = os.getenv("MODEL_PATH")
if model_path_override and os.path.exists(model_path_override):
    model_arg = model_path_override
    found_model = True
    print(f"✓ Using MODEL_PATH override: {model_arg}")

# Check custom directories
if not found_model:
    custom_dirs = [
        os.path.join(HF_HOME, "model-gpt-oss"),
        os.path.join(HF_HOME, "gpt-oss-120b"),
        os.path.join(HF_HOME, "model-gpt-oss-120b"),
        os.path.join(HF_HOME, MODEL_NAME.split('/')[-1]),
    ]

    for custom_dir in custom_dirs:
        if os.path.exists(custom_dir) and os.path.exists(os.path.join(custom_dir, "config.json")):
            model_arg = custom_dir
            found_model = True
            print(f"✓ Found model in: {model_arg}")
            break

# Check HuggingFace cache
if not found_model:
    possible_cache_paths = [
        os.path.join(HF_HOME, "hub", f"models--{MODEL_NAME.replace('/', '--')}"),
        os.path.join(HF_HOME, f"models--{MODEL_NAME.replace('/', '--')}")
    ]

    for model_cache_path in possible_cache_paths:
        if os.path.exists(model_cache_path):
            snapshots_dir = os.path.join(model_cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = [d for d in os.listdir(snapshots_dir)
                           if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshots:
                    snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                    found_model = True
                    print(f"✓ Found model in HF cache: {snapshot_path}")

                    # Create symlink to avoid vLLM repo_id parsing issues
                    symlink_path = os.path.join(HF_HOME, "model")
                    try:
                        if os.path.islink(symlink_path):
                            os.unlink(symlink_path)
                        if not os.path.exists(symlink_path):
                            os.symlink(snapshot_path, symlink_path)
                            print(f"✓ Created symlink: {symlink_path}")
                            model_arg = symlink_path
                        else:
                            model_arg = snapshot_path
                    except Exception as e:
                        print(f"⚠ Symlink creation failed: {e}")
                        model_arg = snapshot_path
                    break

if not found_model:
    print(f"→ Using model name (will download): {MODEL_NAME}")

# Build vLLM command
use_serve_command = os.getenv("USE_VLLM_SERVE", "false").lower() == "true"

if use_serve_command:
    cmd = [
        "vllm", "serve", model_arg,
        "--host", "0.0.0.0",
        "--port", VLLM_PORT,
        "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
        "--max-model-len", MAX_MODEL_LEN,
        "--trust-remote-code",
        "--dtype", DTYPE,
    ]
else:
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_arg,
        "--host", "0.0.0.0",
        "--port", VLLM_PORT,
        "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
        "--max-model-len", MAX_MODEL_LEN,
        "--trust-remote-code",
        "--dtype", DTYPE,
    ]

print("")
print("=" * 70)
print("Starting vLLM Server...")
print("=" * 70)
print(f"Command: {' '.join(cmd)}")
print("")
print("OpenWebUI Configuration:")
print(f"  API Base URL: http://<your-runpod-id>-{VLLM_PORT}.proxy.runpod.net/v1")
print(f"  Model Name: {MODEL_NAME}")
print(f"  API Key: (not required, use any value)")
print("=" * 70)
print("")

# Start vLLM server (foreground)
def signal_handler(signum, frame):
    print("\n\nShutting down...")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Run vLLM directly in foreground
try:
    subprocess.run(cmd)
except KeyboardInterrupt:
    print("\n\nShutting down...")
    sys.exit(0)
