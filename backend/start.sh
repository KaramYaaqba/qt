#!/bin/bash
set -e

MODEL_FILE="/app/model/model.onnx"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Model not found, downloading from HuggingFace..."
    python -c "
from huggingface_hub import hf_hub_download
import os

repo_id = os.environ['HF_MODEL_REPO']
token   = os.environ['HF_TOKEN']
os.makedirs('/app/model', exist_ok=True)
for f in ['model.onnx', 'tokens.txt']:
    print(f'Downloading {f}...')
    hf_hub_download(repo_id=repo_id, filename=f, token=token, local_dir='/app/model')
    print(f'{f} done')
print('All files ready')
"
fi

echo "Starting server..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
