"""
RunPod vLLM Handler - OpenAI Compatible (Proxy Mode)

This handler starts a vLLM OpenAI-compatible server and proxies requests to it.
Simpler than the official worker-vllm which uses AsyncLLMEngine directly.
Perfect for use with OpenWebUI and other OpenAI-compatible clients.
"""

import os
import runpod
import subprocess
import time
import json
import signal
import sys
import aiohttp
import asyncio
from typing import Dict, Any, AsyncGenerator

# Environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
HF_HOME = os.getenv("HF_HOME", "/runpod-volume/huggingface")
HF_TOKEN = os.getenv("HF_TOKEN", "")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.95")
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "8192")
DTYPE = os.getenv("DTYPE", "bfloat16")
VLLM_PORT = os.getenv("VLLM_PORT", "8000")

# Set HF cache environment variables
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = HF_HOME
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN

# Global vLLM process
vllm_process = None


def find_model_path():
    """Find model in various possible locations"""
    # Check MODEL_PATH override
    model_path_override = os.getenv("MODEL_PATH")
    if model_path_override and os.path.exists(model_path_override):
        return model_path_override, True

    # Check custom directories
    custom_dirs = [
        os.path.join(HF_HOME, "model-gpt-oss"),
        os.path.join(HF_HOME, "gpt-oss-120b"),
        os.path.join(HF_HOME, "model-gpt-oss-120b"),
        os.path.join(HF_HOME, MODEL_NAME.split('/')[-1]),
    ]

    for custom_dir in custom_dirs:
        if os.path.exists(custom_dir) and os.path.exists(os.path.join(custom_dir, "config.json")):
            return custom_dir, True

    # Check HuggingFace cache
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
                    # Create symlink to avoid repo ID parsing issues
                    symlink_path = os.path.join(HF_HOME, "model")
                    try:
                        if os.path.islink(symlink_path):
                            os.unlink(symlink_path)
                        if not os.path.exists(symlink_path):
                            os.symlink(snapshot_path, symlink_path)
                            return symlink_path, True
                    except Exception:
                        pass
                    return snapshot_path, True

    # Use model name (will download)
    return MODEL_NAME, False


def start_vllm_server():
    """Start vLLM OpenAI-compatible server"""
    global vllm_process

    print(f"Searching for model {MODEL_NAME}...")
    model_arg, found_local = find_model_path()

    if found_local:
        print(f"✓ Found local model: {model_arg}")
    else:
        print(f"→ Will download model: {MODEL_NAME}")

    # Build vLLM command
    # For GPT-OSS models, all OpenAI endpoints should be enabled by default
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_arg,
        "--host", "0.0.0.0",
        "--port", VLLM_PORT,
        "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
        "--max-model-len", MAX_MODEL_LEN,
        "--trust-remote-code",
        "--dtype", DTYPE,
        "--served-model-name", MODEL_NAME,  # Ensure consistent model name
    ]

    print(f"\nStarting vLLM server...")
    print(f"Command: {' '.join(cmd)}\n")

    # Start vLLM server
    vllm_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # Stream output
    def stream_output():
        for line in iter(vllm_process.stdout.readline, ''):
            if line:
                print(f"[vLLM] {line.rstrip()}")

    import threading
    threading.Thread(target=stream_output, daemon=True).start()

    # Wait for server to be ready
    print("Waiting for vLLM server to initialize...")
    import requests
    max_wait = 300
    start_time = time.time()

    while time.time() - start_time < max_wait:
        if vllm_process.poll() is not None:
            raise RuntimeError(f"vLLM process exited with code {vllm_process.returncode}")

        try:
            response = requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=2)
            if response.status_code == 200:
                print("✅ vLLM server is ready!\n")
                return
        except:
            pass

        time.sleep(5)

    raise RuntimeError("vLLM server startup timeout")


class JobInput:
    """Parse job input - compatible with official worker-vllm"""
    def __init__(self, input_data: Dict[str, Any]):
        self.input_data = input_data
        self.openai_route = input_data.get("openai_route", "")

        # Detect request type
        if "messages" in input_data:
            self.request_type = "chat"
        elif "prompt" in input_data:
            self.request_type = "completion"
        else:
            self.request_type = "chat"  # default


async def generate_openai(job_input: JobInput) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate responses from vLLM OpenAI API
    Yields batches like the official worker-vllm pattern

    Supports GPT-OSS endpoints:
    - /v1/responses (for tool use and reasoning)
    - /v1/chat/completions (familiar chat interface)
    - /v1/completions (simple text completion)
    - /v1/models (list models)
    """
    # Determine endpoint based on openai_route or request content
    openai_route = job_input.openai_route

    if openai_route == "/v1/models":
        endpoint = f"http://localhost:{VLLM_PORT}/v1/models"
        method = "GET"
        payload = None
    elif openai_route == "/v1/responses":
        # GPT-OSS specific responses endpoint for tool use
        endpoint = f"http://localhost:{VLLM_PORT}/v1/responses"
        method = "POST"
        payload = job_input.input_data.copy()
    elif openai_route == "/v1/completions" or (job_input.request_type == "completion" and "prompt" in job_input.input_data):
        # Text completions endpoint
        endpoint = f"http://localhost:{VLLM_PORT}/v1/completions"
        method = "POST"
        payload = job_input.input_data.copy()
    elif openai_route == "/v1/chat/completions" or "messages" in job_input.input_data:
        # Chat completions endpoint (default for OpenWebUI)
        endpoint = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
        method = "POST"
        payload = job_input.input_data.copy()
    else:
        # Default fallback - try chat completions
        endpoint = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
        method = "POST"
        payload = job_input.input_data.copy()

        # If we have a prompt but no messages, convert it
        if "prompt" in payload and "messages" not in payload:
            prompt_text = payload.pop("prompt")
            payload["messages"] = [{"role": "user", "content": prompt_text}]

    # Clean up payload
    if payload:
        payload.pop("openai_route", None)
        # Ensure model is set
        if "model" not in payload:
            payload["model"] = MODEL_NAME

    stream = payload.get("stream", False) if payload else False

    # Log the request for debugging
    print("[Handler] Request:")
    print(f"  Endpoint: {endpoint}")
    print(f"  Method: {method}")
    if payload:
        print(f"  Payload keys: {list(payload.keys())}")
        print(f"  Model: {payload.get('model', 'not set')}")
        print(f"  Stream: {stream}")

    try:
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        result = await response.json()
                        yield result
                    else:
                        error_text = await response.text()
                        print(f"[Handler] GET Error {response.status}: {error_text}")
                        yield {"error": f"vLLM API error ({response.status}): {error_text}"}
            else:
                async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"[Handler] POST Error {response.status} to {endpoint}")
                        print(f"[Handler] Error body: {error_text}")
                        yield {"error": f"vLLM API error ({response.status}): {error_text}"}
                        return

                    if stream:
                        # Stream SSE responses
                        async for line in response.content:
                            if line:
                                decoded = line.decode('utf-8').strip()
                                if decoded.startswith("data: "):
                                    data_str = decoded[6:]  # Remove "data: " prefix
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data = json.loads(data_str)
                                        yield data
                                    except json.JSONDecodeError:
                                        continue
                    else:
                        # Non-streaming response
                        result = await response.json()
                        yield result

    except Exception as e:
        print(f"Error in generate_openai: {str(e)}")
        import traceback
        traceback.print_exc()
        yield {"error": str(e)}


async def handler(job):
    """
    RunPod serverless handler
    Follows the official worker-vllm async generator pattern
    """
    try:
        job_input = JobInput(job.get("input", {}))

        # Generate responses
        async for batch in generate_openai(job_input):
            yield batch

    except Exception as e:
        print(f"Handler error: {str(e)}")
        import traceback
        traceback.print_exc()
        yield {"error": str(e)}


def cleanup_handler(signum, frame):
    """Cleanup on shutdown"""
    global vllm_process
    print("\nShutting down...")
    if vllm_process:
        vllm_process.terminate()
        try:
            vllm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            vllm_process.kill()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, cleanup_handler)
signal.signal(signal.SIGINT, cleanup_handler)

# Start vLLM server
print("=" * 70)
print(f"RunPod vLLM Worker - OpenAI Compatible")
print("=" * 70)
print(f"Model: {MODEL_NAME}")
print(f"HF Home: {HF_HOME}")
print("=" * 70)

try:
    start_vllm_server()
except Exception as e:
    print(f"Failed to start vLLM server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Start RunPod serverless
print("=" * 70)
print("Starting RunPod Serverless Handler")
print("=" * 70)
print("\nOpenWebUI Configuration:")
print(f"  Base URL: https://api.runpod.ai/v2/YOUR-ENDPOINT-ID/openai/v1")
print(f"  API Key: Your RunPod API key")
print(f"  Model: {MODEL_NAME}")
print("=" * 70)
print("")

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
