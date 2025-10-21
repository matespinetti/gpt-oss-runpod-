# Handler Updates - Following Official worker-vllm Pattern

## What Changed

The handler has been completely rewritten to follow the official [runpod-workers/worker-vllm](https://github.com/runpod-workers/worker-vllm) pattern while maintaining your proxy-based architecture.

## Key Improvements

### 1. **Proper Async Pattern** ✅
```python
async def handler(job):
    job_input = JobInput(job.get("input", {}))
    async for batch in generate_openai(job_input):
        yield batch
```

This matches the official worker structure:
- Async generator function
- `JobInput` class for parsing requests
- Yields batches from generation function
- Uses `return_aggregate_stream=True`

### 2. **Request Type Detection** ✅
```python
class JobInput:
    def __init__(self, input_data):
        self.openai_route = input_data.get("openai_route", "")
        # Auto-detect: chat vs completion
        if "messages" in input_data:
            self.request_type = "chat"
        elif "prompt" in input_data:
            self.request_type = "completion"
```

Properly handles:
- `/v1/chat/completions` - When `messages` field present
- `/v1/completions` - When `prompt` field present
- `/v1/models` - When `openai_route` is `/v1/models`

### 3. **Streaming Support** ✅
```python
if stream:
    # Parse SSE format: "data: {...}"
    async for line in response.content:
        decoded = line.decode('utf-8').strip()
        if decoded.startswith("data: "):
            data_str = decoded[6:]  # Remove "data: " prefix
            if data_str == "[DONE]":
                break
            data = json.loads(data_str)
            yield data
```

Properly parses Server-Sent Events (SSE) format from vLLM.

### 4. **Simplified Model Path Logic** ✅
Moved to a clean `find_model_path()` function:
- Checks `MODEL_PATH` override
- Checks custom directories
- Checks HuggingFace cache with symlink creation
- Falls back to model name

## Architecture

```
┌─────────────┐
│  OpenWebUI  │
│   / Client  │
└──────┬──────┘
       │ HTTPS
       ▼
┌─────────────────────────────────────┐
│  RunPod API Gateway                 │
│  /v2/{endpoint-id}/openai/v1/*      │
└──────┬──────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  RunPod Serverless Worker (Container)│
│  ┌────────────────────────────────┐  │
│  │  handler.py (async)            │  │
│  │  - JobInput parsing            │  │
│  │  - Request routing             │  │
│  │  - Async proxy to vLLM         │  │
│  └───────┬────────────────────────┘  │
│          │ HTTP                       │
│  ┌───────▼────────────────────────┐  │
│  │  vLLM OpenAI Server            │  │
│  │  (subprocess on port 8000)     │  │
│  │  - /v1/chat/completions        │  │
│  │  - /v1/completions             │  │
│  │  - /v1/models                  │  │
│  └───────┬────────────────────────┘  │
│          │                            │
│  ┌───────▼────────────────────────┐  │
│  │  vLLM AsyncLLMEngine           │  │
│  │  (GPU inference)               │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

## Differences from Official worker-vllm

| Aspect | Official Worker | This Handler |
|--------|----------------|--------------|
| **vLLM Integration** | Uses `AsyncLLMEngine` directly | Proxies to vLLM OpenAI server |
| **Complexity** | More complex (engine management) | Simpler (proxy pattern) |
| **Base Image** | Custom build from scratch | `vllm/vllm-openai:latest` |
| **Setup Time** | Faster (no subprocess) | Slightly slower (subprocess startup) |
| **Code Maintenance** | More code to maintain | Less code to maintain |
| **Flexibility** | More control over engine | Limited to vLLM server features |

**Why use proxy pattern?**
- Simpler to understand and maintain
- Leverages official vLLM Docker image
- Easier to debug (can test vLLM server separately)
- Less code to manage

## Testing

### Test with RunPod API:
```python
import requests

response = requests.post(
    "https://api.runpod.ai/v2/YOUR-ENDPOINT-ID/runsync",
    headers={"Authorization": "Bearer YOUR_RUNPOD_API_KEY"},
    json={
        "input": {
            "prompt": "Hello World",  # For completions
            "max_tokens": 50
        }
    }
)
```

### Test with OpenAI format:
```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_RUNPOD_API_KEY",
    base_url="https://api.runpod.ai/v2/YOUR-ENDPOINT-ID/openai/v1"
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## Files Changed

- ✅ `handler.py` - Completely rewritten
- ✅ `README.md` - Updated with OpenWebUI instructions
- ✅ `Dockerfile` - Added aiohttp dependency
- 🗑️ `handler_old.py` - Old version (backup)
- 🗑️ `start_openwebui_mode.py` - No longer needed

## Next Steps

1. **Build and push Docker image**:
   ```bash
   docker build -t your-username/vllm-worker .
   docker push your-username/vllm-worker
   ```

2. **Deploy to RunPod**:
   - Use your Docker image
   - Set environment variables (MODEL_NAME, HF_HOME, etc.)
   - Attach network volume

3. **Connect from OpenWebUI**:
   - Base URL: `https://api.runpod.ai/v2/YOUR-ENDPOINT-ID/openai/v1`
   - API Key: Your RunPod API key
   - Model: Same as MODEL_NAME env var

## Troubleshooting

If you see errors about missing `messages` field with `prompt` input:
- ✅ **FIXED** - Handler now properly detects completion vs chat requests
- Checks for `prompt` field and routes to `/v1/completions`
- Checks for `messages` field and routes to `/v1/chat/completions`

The handler will work seamlessly with OpenWebUI! 🎉
