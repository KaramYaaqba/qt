#!/bin/bash
set -e

# Download model from HuggingFace if not present
MODEL_DIR="/app/model"
MODEL_FILE="$MODEL_DIR/model.onnx"
TOKENS_FILE="$MODEL_DIR/tokens.txt"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading model from HuggingFace..."
    pip install huggingface_hub -q
    python -c "
from huggingface_hub import hf_hub_download
import shutil, os

repo_id = os.environ.get('HF_MODEL_REPO', '')
if not repo_id:
    print('HF_MODEL_REPO not set, starting with mock mode')
    exit(0)

os.makedirs('/app/model', exist_ok=True)
print(f'Downloading from {repo_id}...')
hf_hub_download(repo_id=repo_id, filename='model.onnx', local_dir='/app/model')
hf_hub_download(repo_id=repo_id, filename='tokens.txt', local_dir='/app/model')
print('Model downloaded successfully')
"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
