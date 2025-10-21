# Quick Start Guide

## Your Setup

Based on your configuration:
- Model is installed at: `/runpod-volume/huggingface/model-gpt-oss` (or similar)
- Using RunPod Serverless

## Recommended Environment Variables

Set these in your RunPod Serverless endpoint:

```bash
# Model configuration - set the exact path since you already have it downloaded
MODEL_PATH=/runpod-volume/huggingface/model-gpt-oss
MODEL_NAME=openai/gpt-oss-120b

# Storage
HF_HOME=/runpod-volume/huggingface

# GPU settings (adjust based on your GPU)
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=8192
DTYPE=bfloat16

# Server settings
VLLM_PORT=8000

# Command style - defaults to false (legacy syntax, more compatible)
# Only set to true if your vLLM version explicitly supports 'vllm serve'
# USE_VLLM_SERVE=false
```

## If You Get the Help Menu Error

**The error has been fixed!** The handler now defaults to `USE_VLLM_SERVE=false`, which uses the legacy `python -m vllm.entrypoints.openai.api_server` syntax that's compatible with the `vllm/vllm-openai:latest` Docker image.

### Additional Fixes (if still needed):

### Fix 1: Set Explicit Model Path
```bash
MODEL_PATH=/runpod-volume/huggingface/model-gpt-oss
```

Replace with your actual model directory path.

### Fix 2: Check Model Directory
Make sure your model directory contains:
- `config.json`
- `*.safetensors` or `pytorch_model.bin` files
- `tokenizer.json` and `tokenizer_config.json`

## Build and Deploy

1. **Build Docker image:**
   ```bash
   docker build --platform linux/amd64 -t your-username/gpt-oss-runpod .
   docker push your-username/gpt-oss-runpod
   ```

2. **Deploy to RunPod:**
   - Go to RunPod Serverless
   - Create new endpoint
   - Use your Docker image
   - Set environment variables above
   - Attach network volume to `/runpod-volume`
   - Select A100 80GB or H100 GPU

3. **Monitor logs:**
   - Watch for: `✓ Found model in custom directory: /runpod-volume/huggingface/model-gpt-oss`
   - Watch for: `✅ vLLM server is ready!`

## Expected Startup Flow

```
==================================================
Initializing vLLM Server for openai/gpt-oss-120b
==================================================
Searching for model...
HF_HOME: /runpod-volume/huggingface (exists: True)
✓ Found model in custom directory: /runpod-volume/huggingface/model-gpt-oss

Starting vLLM server on port 8000...
Command style: python -m vllm.entrypoints.openai.api_server
Model argument: /runpod-volume/huggingface/model-gpt-oss
Full command: python -m vllm.entrypoints.openai.api_server --model /runpod-volume/huggingface/model-gpt-oss --host 0.0.0.0 --port 8000 ...

Configuration:
  - GPU Memory Utilization: 0.95
  - Max Model Length: 8192
  - Data Type: bfloat16
  - Trust Remote Code: True

[vLLM] INFO: Loading model...
[vLLM] INFO: Model loaded successfully
✅ vLLM server is ready!

==================================================
Starting RunPod Serverless Handler
==================================================
```

## Test Request

Once deployed, test with:

```python
import requests

url = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

payload = {
    "input": {
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 100,
        "temperature": 1.0
    }
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

## Troubleshooting

### Model Not Found
- Check `MODEL_PATH` points to correct directory
- Verify network volume is mounted at `/runpod-volume`
- Look for "Found model in custom directory" in logs

### vLLM Shows Help Menu
- Set `USE_VLLM_SERVE=false`
- Check model path is correct
- Run diagnostic: `python test_vllm_command.py`

### Out of Memory
- Lower `GPU_MEMORY_UTILIZATION=0.90`
- Lower `MAX_MODEL_LEN=4096`
- Use larger GPU (A100 80GB or H100)

### Timeout
- Normal for first load (large model takes time)
- Subsequent runs will be faster
- Check logs for actual errors

## Support

See [README.md](README.md) for full documentation.
