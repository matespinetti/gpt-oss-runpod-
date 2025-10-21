FROM vllm/vllm-openai:latest
RUN uv pip install --system --no-cache-dir runpod aiohttp requests
COPY handler.py /app/handler.py
WORKDIR /app
EXPOSE 8000
# Clear any existing entrypoint from base image
ENTRYPOINT []
CMD ["python3", "-u", "handler.py"]
