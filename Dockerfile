FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update -y && apt-get install -y python3-pip
RUN ldconfig /usr/local/cuda-12.1/compat/

# Install dependencies
RUN pip3 install --no-cache-dir \
    runpod~=1.7.7 \
    huggingface-hub \
    hf-transfer \
    transformers>=4.55.0 \
    torch==2.6.0

# Install latest vLLM with FlashInfer
RUN pip3 install vllm==0.10.2 && \
    pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3

# Environment variables
ENV BASE_PATH="/runpod-volume" \
    HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub" \
    HF_HOME="/runpod-volume/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONPATH="/:/vllm-workspace"

# Copy official worker source
COPY src /src

# Start the handler
CMD ["python3", "/src/handler.py"]
