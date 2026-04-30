FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
RUN mkdir -p app/data model

# Download model at build time using HF token build arg
ARG HF_TOKEN
ARG HF_MODEL_REPO
RUN if [ -n "$HF_TOKEN" ] && [ -n "$HF_MODEL_REPO" ]; then \
    python -c "\
from huggingface_hub import hf_hub_download; \
import os; \
hf_hub_download(repo_id='${HF_MODEL_REPO}', filename='model.onnx', token='${HF_TOKEN}', local_dir='/app/model'); \
print('model.onnx downloaded'); \
"; fi

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV USE_MOCK=false

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
