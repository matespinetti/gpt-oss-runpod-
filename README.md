# RunPod Serverless vLLM Handler - OpenAI Compatible

This handler runs vLLM models on RunPod Serverless with **full OpenAI API compatibility**. Perfect for use with **OpenWebUI** and other OpenAI-compatible clients.

Based on the official [runpod-workers/worker-vllm](https://github.com/runpod-workers/worker-vllm) pattern.

## Features

- ✅ **Full OpenAI API compatibility** - Drop-in replacement for OpenAI API
- ✅ **Streaming support** - Real-time response streaming
- ✅ **OpenWebUI ready** - Seamless integration with OpenWebUI
- ✅ **Smart model loading** - Automatic detection from HuggingFace cache
- ✅ **Async/await support** - Proper async handling with RunPod
- ✅ **RunPod optimized** - Network volume support and best practices
- ✅ **Configurable via environment variables**

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `openai/gpt-oss-120b` | HuggingFace model name |
| `MODEL_PATH` | *(auto-detect)* | **Override:** Explicit path to model directory |
| `HF_HOME` | `/runpod-volume/huggingface` | HuggingFace cache directory |
| `HF_TOKEN` | *(empty)* | HuggingFace API token (for gated models) |
| `GPU_MEMORY_UTILIZATION` | `0.95` | GPU memory utilization (0.0-1.0) |
| `MAX_MODEL_LEN` | `8192` | Maximum context length |
| `DTYPE` | `bfloat16` | Model data type (bfloat16, float16, etc.) |
| `VLLM_PORT` | `8000` | vLLM server port |
| `USE_VLLM_SERVE` | `false` | Use `vllm serve` (true) or legacy `python -m` syntax (false) |

### Model Path Detection

The handler automatically searches for your model in this order:

1. **`MODEL_PATH`** - If set, uses this exact path (highest priority)
2. **Custom directories** in `HF_HOME`:
   - `model-gpt-oss`
   - `gpt-oss-120b`
   - `model-gpt-oss-120b`
   - Last part of `MODEL_NAME`
3. **HuggingFace cache** - Standard cache structure at `HF_HOME/hub/models--{org}--{model}/snapshots/{hash}/`
4. **Model name** - Falls back to `MODEL_NAME` (vLLM will download)

**Example:** If your model is at `/runpod-volume/huggingface/model-gpt-oss`, it will be auto-detected. Or set:
```bash
MODEL_PATH=/runpod-volume/huggingface/model-gpt-oss
```

## Deployment

### 1. Build Docker Image

```bash
docker build --platform linux/amd64 -t your-username/gpt-oss-runpod .
docker push your-username/gpt-oss-runpod
```

### 2. Create RunPod Network Volume

1. Go to RunPod Storage
2. Create a new network volume (e.g., 200GB+)
3. Note the volume ID

### 3. Deploy to RunPod Serverless

1. Go to RunPod Serverless
2. Create new endpoint
3. Use your Docker image
4. Configure environment variables
5. Attach the network volume to `/runpod-volume`
6. Select GPU (A100 80GB or H100 recommended for 120B model)

### 4. Connect with OpenWebUI

In OpenWebUI settings:

1. **Connection Type**: OpenAI API
2. **Base URL**: `https://api.runpod.ai/v2/YOUR-ENDPOINT-ID/openai/v1`
3. **API Key**: Your RunPod API key
4. **Model Name**: Match your `MODEL_NAME` env var (e.g., `openai/gpt-oss-120b`)

### 5. Test with Python OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_RUNPOD_API_KEY",
    base_url="https://api.runpod.ai/v2/YOUR-ENDPOINT-ID/openai/v1"
)

# Non-streaming
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=256,
    temperature=0.7
)
print(response.choices[0].message.content)

# Streaming
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": "Tell me a story"}
    ],
    max_tokens=512,
    temperature=0.7,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 6. Test with cURL

```bash
curl https://api.runpod.ai/v2/YOUR-ENDPOINT-ID/openai/v1/chat/completions \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

## Supported OpenAI Endpoints

**For GPT-OSS Models:**
- ✅ `/v1/responses` - Tool use with reasoning chains (GPT-OSS recommended)
- ✅ `/v1/chat/completions` - Chat completions (OpenWebUI default)
- ✅ `/v1/completions` - Text completions
- ✅ `/v1/models` - List available models

All endpoints support both streaming and non-streaming modes.

**Note:** The handler automatically routes requests based on content:
- `messages` → `/v1/chat/completions`
- `prompt` → `/v1/completions`
- `openai_route` parameter overrides automatic detection

## Troubleshooting

### Model Path Errors

**Error:**
```
ERROR retrieving safetensors: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/runpod-volume/huggingface/models--openai--gpt-oss-120b/snapshots/...'
```

**Solution:** The handler now automatically creates a symlink at `{HF_HOME}/model` to avoid vLLM's repo ID parsing issues. If this still fails:
- Set `MODEL_PATH` environment variable to point directly to your model directory
- Ensure the directory contains `config.json` and model weights

### AsyncIO Errors

**Error:**
```
RuntimeError: asyncio.run() cannot be called from a running event loop
```

**Solution:** ✅ **Fixed in current version!** The handler now uses proper async/await with RunPod's event loop.

### OpenWebUI Connection Issues

If OpenWebUI can't connect:

1. **Check endpoint URL format:**
   - Must include `/openai/v1` suffix
   - Example: `https://api.runpod.ai/v2/abc123xyz/openai/v1`

2. **Verify API key:**
   - Use your RunPod API key (not OpenAI key)
   - Find it at: https://www.runpod.io/console/user/settings

3. **Check endpoint status:**
   - Endpoint must be active (not scaled to zero)
   - Wait for worker to initialize (~2-5 minutes on first start)

4. **Model name must match:**
   - Use the same value as `MODEL_NAME` environment variable
   - Default: `openai/gpt-oss-120b`

### Model Not Loading

1. **Check HuggingFace token:** Required for gated/private models
2. **Verify GPU memory:** 120B model needs ~70GB+ VRAM
3. **Check network volume:** Model should be cached at `/runpod-volume/huggingface`
4. **Check logs:** Look for "✓ Found model in..." messages

### Timeout During Startup

- Increase timeout in code (currently 5 minutes)
- Large models take time to load on first run
- Subsequent runs will be faster with cached model

## Model Requirements

| Model | VRAM Needed | Recommended GPU |
|-------|-------------|-----------------|
| `openai/gpt-oss-20b` | ~16GB | A100 40GB, L40S |
| `openai/gpt-oss-120b` | ~70GB | A100 80GB, H100 |

## How It Works

This worker follows the official RunPod vLLM worker pattern:

1. **Startup**: Launches vLLM OpenAI API server on container initialization
2. **Request Handling**: Async handler proxies RunPod requests to vLLM
3. **Response Streaming**: Uses `return_aggregate_stream=True` for streaming support
4. **OpenAI Compatibility**: vLLM provides native OpenAI API endpoints

```
OpenWebUI/Client → RunPod API → Handler (proxy) → vLLM OpenAI API → Model
```

## Performance Tips

1. **Use Network Volumes:** Pre-download model to avoid cold start delays
2. **Enable Flashboot:** Configure in RunPod for faster cold starts
3. **Adjust GPU Memory:** Lower `GPU_MEMORY_UTILIZATION` if OOM errors occur
4. **Set Max Context:** Lower `MAX_MODEL_LEN` to fit more in memory

## Files

- `handler.py` - Main RunPod handler with vLLM integration
- `Dockerfile` - Container definition
- `test_vllm_command.py` - Diagnostic tool for vLLM command testing
- `README.md` - This file

## API Usage Examples

### Direct RunPod API (Alternative)

You can also call RunPod's native API directly:

```python
import requests

response = requests.post(
    "https://api.runpod.ai/v2/YOUR-ENDPOINT-ID/runsync",
    headers={
        "Authorization": "Bearer YOUR_RUNPOD_API_KEY",
        "Content-Type": "application/json"
    },
    json={
        "input": {
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 256,
            "temperature": 0.7
        }
    }
)
print(response.json())
```

But the **OpenAI-compatible endpoint is recommended** for better compatibility with existing tools.

## References

- [Official RunPod vLLM Worker](https://github.com/runpod-workers/worker-vllm)
- [RunPod vLLM OpenAI Compatibility Docs](https://docs.runpod.io/serverless/vllm/openai-compatibility)
- [OpenAI GPT-OSS Cookbook](https://cookbook.openai.com/articles/gpt-oss/run-vllm)
- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenWebUI Documentation](https://docs.openwebui.com/)
