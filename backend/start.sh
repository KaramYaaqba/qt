#!/bin/bash

MODEL_FILE="/app/model/model.onnx"

if [ ! -f "$MODEL_FILE" ]; then
    echo "model.onnx not found — downloading from HuggingFace..."

    if [ -z "$HF_TOKEN" ] || [ -z "$HF_MODEL_REPO" ]; then
        echo "ERROR: HF_TOKEN and HF_MODEL_REPO are not set."
        echo "Add them as Variables in Railway dashboard, then redeploy."
        exit 1
    fi

    python -c "
from huggingface_hub import hf_hub_download
import os, sys
print('Downloading model.onnx from', os.environ['HF_MODEL_REPO'], '...')
try:
    hf_hub_download(
        repo_id=os.environ['HF_MODEL_REPO'],
        filename='model.onnx',
        token=os.environ['HF_TOKEN'],
        local_dir='/app/model'
    )
    print('model.onnx ready.')
except Exception as e:
    print('Download failed:', e)
    sys.exit(1)
"
    if [ $? -ne 0 ]; then
        echo "ERROR: model download failed — check HF_TOKEN and HF_MODEL_REPO values."
        exit 1
    fi
fi

echo "Starting uvicorn on port ${PORT:-8000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
