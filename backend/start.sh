#!/bin/bash

MODEL_FILE="/app/model/model.onnx"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading model.onnx from HuggingFace..."
    python -c "
import os, sys
try:
    from huggingface_hub import hf_hub_download
    repo_id = os.environ.get('HF_MODEL_REPO', '')
    token   = os.environ.get('HF_TOKEN', '')
    if not repo_id or not token:
        print('HF_MODEL_REPO or HF_TOKEN not set — starting in mock mode')
        sys.exit(0)
    os.makedirs('/app/model', exist_ok=True)
    print('Downloading model.onnx...')
    hf_hub_download(repo_id=repo_id, filename='model.onnx', token=token, local_dir='/app/model')
    print('model.onnx ready')
except Exception as e:
    print(f'Model download failed: {e}')
    print('Starting in mock mode')
"
fi

echo "Starting uvicorn on port ${PORT:-8000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
