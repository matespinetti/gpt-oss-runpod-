# RunPod Serverless vLLM Handler for GPT-OSS

This handler runs OpenAI's GPT-OSS models (120B or 20B) on RunPod Serverless using vLLM.

## Features

- ✅ Automatic model loading from HuggingFace cache or download
- ✅ OpenAI-compatible API endpoint
- ✅ Support for both `vllm serve` and legacy command syntax
- ✅ Real-time vLLM output streaming for debugging
- ✅ Graceful shutdown handling
- ✅ RunPod network volume support
- ✅ Configurable via environment variables

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

### 4. Test the Endpoint

```python
import requests

url = "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/runsync"
headers = {
    "Authorization": "Bearer {YOUR_API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "input": {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 256,
        "temperature": 1.0
    }
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

## Troubleshooting

### vLLM Command Issues

**This issue has been fixed!** The handler now defaults to using the legacy `python -m vllm.entrypoints.openai.api_server` syntax, which is compatible with the `vllm/vllm-openai:latest` Docker image.

If you still have issues:

1. **Check model path:**
   - Look for "Found local model at:" in logs
   - Verify HF_HOME is mounted correctly

2. **Run diagnostic script:**
   ```bash
   python test_vllm_command.py
   ```

### Model Not Loading

1. **Check HuggingFace token:** Required for gated models
2. **Verify GPU memory:** 120B model needs ~70GB+ VRAM
3. **Check network volume:** Model should be cached at `/runpod-volume/huggingface`

### Timeout During Startup

- Increase timeout in code (currently 5 minutes)
- Large models take time to load on first run
- Subsequent runs will be faster with cached model

## Model Requirements

| Model | VRAM Needed | Recommended GPU |
|-------|-------------|-----------------|
| `openai/gpt-oss-20b` | ~16GB | A100 40GB, L40S |
| `openai/gpt-oss-120b` | ~70GB | A100 80GB, H100 |

## API Request Format

```json
{
  "input": {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 256,
    "temperature": 1.0,
    "stream": false
  }
}
```

## Response Format

```json
{
  "id": "cmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "openai/gpt-oss-120b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 8,
    "total_tokens": 28
  }
}
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

## References

- [OpenAI GPT-OSS Cookbook](https://cookbook.openai.com/articles/gpt-oss/run-vllm)
- [RunPod Serverless Documentation](https://docs.runpod.io/serverless/workers/vllm/overview)
- [vLLM Documentation](https://docs.vllm.ai/)
