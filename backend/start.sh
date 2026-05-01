#!/bin/bash
set -e

MODEL_FILE="/app/model/model.onnx"

if [ ! -f "$MODEL_FILE" ]; then
    echo "model.onnx not found — downloading from HuggingFace..."

    if [ -z "$HF_TOKEN" ] || [ -z "$HF_MODEL_REPO" ]; then
        echo "ERROR: HF_TOKEN and HF_MODEL_REPO must be set as environment variables in Railway."
        echo "Starting in mock mode (USE_MOCK=true) as fallback..."
        export USE_MOCK=true
    else
        python -c "
from huggingface_hub import hf_hub_download
import os
print('Downloading model.onnx ...')
hf_hub_download(
    repo_id=os.environ['HF_MODEL_REPO'],
    filename='model.onnx',
    token=os.environ['HF_TOKEN'],
    local_dir='/app/model'
)
print('model.onnx ready.')
"
    fi
fi

echo "Starting uvicorn on port ${PORT:-8000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
