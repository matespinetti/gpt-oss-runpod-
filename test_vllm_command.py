#!/usr/bin/env python3
"""
Test script to verify vLLM command works correctly
Run this to debug vLLM startup issues
"""
import os
import subprocess
import sys

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
HF_HOME = os.getenv("HF_HOME", "/runpod-volume/huggingface")
VLLM_PORT = "8000"

print("=" * 60)
print("vLLM Command Test")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"HF_HOME: {HF_HOME}")
print(f"Port: {VLLM_PORT}")
print("")

# Test 1: Check if vllm command exists
print("Test 1: Checking if 'vllm' command is available...")
try:
    result = subprocess.run(["which", "vllm"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ vllm found at: {result.stdout.strip()}")
    else:
        print("✗ vllm command not found")
        print("Trying 'vllm --version'...")
        version_result = subprocess.run(["vllm", "--version"], capture_output=True, text=True)
        print(version_result.stdout)
        print(version_result.stderr)
except Exception as e:
    print(f"✗ Error: {e}")

print("")

# Test 2: Check vllm serve help
print("Test 2: Checking 'vllm serve --help'...")
try:
    result = subprocess.run(["vllm", "serve", "--help"], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("✓ 'vllm serve' command is available")
        print("\nFirst 20 lines of help:")
        print("\n".join(result.stdout.split("\n")[:20]))
    else:
        print("✗ Error running vllm serve --help")
        print(result.stderr)
except Exception as e:
    print(f"✗ Error: {e}")

print("")

# Test 3: Try the actual command (dry run)
print("Test 3: Testing actual vLLM serve command...")
cmd = [
    "vllm", "serve", MODEL_NAME,
    "--host", "0.0.0.0",
    "--port", VLLM_PORT,
    "--gpu-memory-utilization", "0.95",
    "--max-model-len", "8192",
    "--trust-remote-code",
    "--dtype", "bfloat16",
]

print(f"Command: {' '.join(cmd)}")
print("")
print("This will actually try to start vLLM. Press Ctrl+C to stop.")
print("Starting in 3 seconds...")

import time
time.sleep(3)

try:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    print("\n" + "=" * 60)
    print("vLLM Output:")
    print("=" * 60)

    # Stream output for 30 seconds or until error
    start_time = time.time()
    while time.time() - start_time < 30:
        line = process.stdout.readline()
        if line:
            print(f"[vLLM] {line.rstrip()}")

        # Check if process exited
        if process.poll() is not None:
            print(f"\n✗ Process exited with code: {process.returncode}")
            break
    else:
        print("\n✓ vLLM is running! Terminating test...")
        process.terminate()

except KeyboardInterrupt:
    print("\n\nTest interrupted by user")
    if 'process' in locals():
        process.terminate()
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
