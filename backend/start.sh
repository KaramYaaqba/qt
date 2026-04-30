#!/bin/bash
set -e

MODEL_FILE="/app/model/model.onnx"
TOKENS_FILE="/app/model/tokens.txt"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading model from private HuggingFace repo..."
    python -c "
from huggingface_hub import hf_hub_download
import os, shutil

repo_id = os.environ['HF_MODEL_REPO']
token   = os.environ['HF_TOKEN']

os.makedirs('/app/model', exist_ok=True)
for filename in ['model.onnx', 'tokens.txt']:
    path = hf_hub_download(repo_id=repo_id, filename=filename, token=token, local_dir='/app/model')
    print(f'Downloaded {filename}')
print('Model ready')
"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
