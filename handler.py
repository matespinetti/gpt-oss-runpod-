import os
import runpod
import subprocess
import time
import json
import signal
import sys

# Environment variables - RunPod best practices
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
HF_HOME = os.getenv("HF_HOME", "/runpod-volume/huggingface")
HF_TOKEN = os.getenv("HF_TOKEN", "")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.95")
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "8192")
DTYPE = os.getenv("DTYPE", "bfloat16")
VLLM_PORT = os.getenv("VLLM_PORT", "8000")

# Set HF cache - important for RunPod network volumes
os.environ["HF_HOME"] = HF_HOME
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# Global vLLM process
vllm_process = None

def start_vllm_server():
    """Start vLLM OpenAI-compatible server"""
    global vllm_process

    # Check if model exists locally - support multiple directory structures
    print(f"Searching for model...")
    print(f"HF_HOME: {HF_HOME} (exists: {os.path.exists(HF_HOME)})")

    model_arg = MODEL_NAME  # Default to model name
    found_model = False

    # Strategy 0: Check for explicit model path override
    model_path_override = os.getenv("MODEL_PATH")
    if model_path_override and os.path.exists(model_path_override):
        model_arg = model_path_override
        found_model = True
        print(f"✓ Using MODEL_PATH override: {model_arg}")

    # Strategy 1: Check for custom model directory (e.g., /runpod-volume/huggingface/model-gpt-oss)
    if not found_model:
        # Common patterns: model-gpt-oss, gpt-oss-120b, model-gpt-oss-120b
        custom_dirs = [
            os.path.join(HF_HOME, "model-gpt-oss"),
            os.path.join(HF_HOME, "gpt-oss-120b"),
            os.path.join(HF_HOME, "model-gpt-oss-120b"),
            os.path.join(HF_HOME, MODEL_NAME.split('/')[-1]),  # Last part of model name
        ]

        for custom_dir in custom_dirs:
            if os.path.exists(custom_dir):
                # Verify it's a valid model directory (has config.json or model files)
                try:
                    files = os.listdir(custom_dir)
                    if os.path.exists(os.path.join(custom_dir, "config.json")) or \
                       os.path.exists(os.path.join(custom_dir, "pytorch_model.bin")) or \
                       any(f.endswith('.safetensors') for f in files if os.path.isfile(os.path.join(custom_dir, f))):
                        model_arg = custom_dir
                        found_model = True
                        print(f"✓ Found model in custom directory: {model_arg}")
                        break
                except PermissionError:
                    print(f"⚠ Permission denied accessing: {custom_dir}")
                    continue

    # Strategy 2: Check HuggingFace cache structure (if not found in custom dir)
    if not found_model:
        # Check both hub/ subdirectory and direct models-- directory
        possible_cache_paths = [
            os.path.join(HF_HOME, "hub", f"models--{MODEL_NAME.replace('/', '--')}"),
            os.path.join(HF_HOME, f"models--{MODEL_NAME.replace('/', '--')}")
        ]
        
        for model_cache_path in possible_cache_paths:
            if os.path.exists(model_cache_path):
                snapshots_dir = os.path.join(model_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    # Get the first snapshot directory
                    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        model_arg = os.path.join(snapshots_dir, snapshots[0])
                        found_model = True
                        print(f"✓ Found model in HF cache: {model_arg}")
                        break
                # Also check if the cache directory itself contains model files (some cache structures)
                elif os.path.exists(os.path.join(model_cache_path, "config.json")):
                    model_arg = model_cache_path
                    found_model = True
                    print(f"✓ Found model in HF cache (direct): {model_arg}")
                    break

    # If still not found, use model name (will download)
    if not found_model:
        print(f"→ Model not found locally, will use model name: {MODEL_NAME}")
        print(f"   vLLM will download the model on first run")

    # Build vLLM command
    # The vllm/vllm-openai Docker image uses the older syntax by default
    # Set USE_VLLM_SERVE=true only if you know your vLLM version supports it
    use_serve_command = os.getenv("USE_VLLM_SERVE", "false").lower() == "true"

    if use_serve_command:
        # Modern vllm serve command (v0.4.0+)
        # Note: May not work with vllm/vllm-openai:latest Docker image
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
        # Older python -m syntax - more compatible with vllm/vllm-openai Docker images
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

    print(f"\nStarting vLLM server on port {VLLM_PORT}...")
    print(f"Command style: {'vllm serve' if use_serve_command else 'python -m vllm.entrypoints.openai.api_server'}")
    print(f"Model argument: {model_arg}")
    print(f"Full command: {' '.join(cmd)}")
    print(f"\nConfiguration:")
    print(f"  - GPU Memory Utilization: {GPU_MEMORY_UTILIZATION}")
    print(f"  - Max Model Length: {MAX_MODEL_LEN}")
    print(f"  - Data Type: {DTYPE}")
    print(f"  - Trust Remote Code: True")
    print(f"\nTip: Set USE_VLLM_SERVE=false to use older command syntax if needed")
    print("")

    # Start process with output streaming for debugging
    vllm_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # Stream output in background thread for debugging
    def stream_output():
        try:
            for line in iter(vllm_process.stdout.readline, ''):
                if line:
                    print(f"[vLLM] {line.rstrip()}")
        except Exception as e:
            print(f"[vLLM] Output stream error: {e}")

    import threading
    output_thread = threading.Thread(target=stream_output, daemon=True)
    output_thread.start()

    # Wait for server to be ready
    print("Waiting for vLLM server to initialize (this may take several minutes)...")
    max_wait = 300  # 5 minutes for large model loading
    start_time = time.time()
    check_interval = 5

    while time.time() - start_time < max_wait:
        # Check if process crashed
        if vllm_process.poll() is not None:
            print(f"❌ vLLM process exited with code {vllm_process.returncode}")
            raise RuntimeError(f"vLLM server failed to start (exit code: {vllm_process.returncode})")

        try:
            import requests
            response = requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=2)
            if response.status_code == 200:
                print("✅ vLLM server is ready!")
                # Warm up with a test request
                print("Warming up model with test request...")
                return
        except requests.exceptions.RequestException:
            pass
        except Exception as e:
            print(f"Health check error: {e}")

        elapsed = int(time.time() - start_time)
        if elapsed % 30 == 0 and elapsed > 0:
            print(f"Still waiting... ({elapsed}s elapsed)")

        time.sleep(check_interval)

    # If we timeout, raise an error instead of continuing
    raise RuntimeError("vLLM server startup timeout - server did not become healthy in time")

def call_vllm(messages, max_tokens=256, temperature=1.0, stream=False):
    """Call local vLLM OpenAI endpoint using requests"""
    import requests
    
    url = f"http://localhost:{VLLM_PORT}/v1/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    try:
        response = requests.post(url, json=payload, timeout=300)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"vLLM API error ({response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"vLLM API request failed: {str(e)}")

def handler(job):
    """RunPod handler following RunPod best practices"""
    try:
        job_input = job.get("input", {})

        # Extract parameters with defaults
        messages = job_input.get("messages", [])
        max_tokens = job_input.get("max_tokens", 256)
        temperature = job_input.get("temperature", 1.0)
        stream = job_input.get("stream", False)

        # Validation
        if not messages:
            return {"error": "No messages provided in input"}

        # Call vLLM
        result = call_vllm(messages, max_tokens, temperature, stream)
        return result

    except Exception as e:
        print(f"Handler error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def cleanup_handler(signum, frame):
    """Cleanup vLLM process on shutdown"""
    global vllm_process
    print("\nShutting down vLLM server...")
    if vllm_process:
        vllm_process.terminate()
        try:
            vllm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Force killing vLLM process...")
            vllm_process.kill()
    sys.exit(0)


# Register cleanup handlers
signal.signal(signal.SIGTERM, cleanup_handler)
signal.signal(signal.SIGINT, cleanup_handler)

# Start vLLM server on initialization
print("=" * 50)
print(f"Initializing vLLM Server for {MODEL_NAME}")
print("=" * 50)
try:
    start_vllm_server()
except Exception as e:
    print(f"Failed to start vLLM server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Start RunPod serverless handler
print("=" * 50)
print("Starting RunPod Serverless Handler")
print("=" * 50)
runpod.serverless.start({"handler": handler})