#!/usr/bin/env python3
"""
Export Trained Model to ONNX Format

Converts the trained NeMo Conformer-CTC model to ONNX format
for efficient CPU inference.

Usage:
    python export_onnx.py --nemo_path ./output/conformer_ctc_quran.nemo
"""
import argparse
from pathlib import Path

import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# NeMo imports
from nemo.collections.asr.models import EncDecCTCModel
from nemo.utils import logging


def export_to_onnx(
    nemo_path: str,
    output_dir: str = "./export",
    quantize: bool = True,
):
    """
    Export NeMo model to ONNX format.
    
    Args:
        nemo_path: Path to .nemo checkpoint
        output_dir: Directory to save ONNX files
        quantize: Whether to apply INT8 quantization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading model from: {nemo_path}")
    model = EncDecCTCModel.restore_from(nemo_path, map_location="cpu")
    model.eval()
    
    # Export paths
    onnx_path = output_path / "model.onnx"
    quantized_path = output_path / "model.int8.onnx"
    tokens_path = output_path / "tokens.txt"
    
    logging.info("Exporting to ONNX...")
    
    # NeMo provides built-in ONNX export
    model.export(
        str(onnx_path),
        onnx_opset_version=17,
        check_trace=True,
    )
    
    logging.info(f"Saved ONNX model to: {onnx_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logging.info("ONNX model validation passed")
    
    # Apply quantization
    if quantize:
        logging.info("Applying INT8 dynamic quantization...")
        quantize_dynamic(
            str(onnx_path),
            str(quantized_path),
            weight_type=QuantType.QInt8,
        )
        logging.info(f"Saved quantized model to: {quantized_path}")
        
        # Print size comparison
        original_size = onnx_path.stat().st_size / (1024 * 1024)
        quantized_size = quantized_path.stat().st_size / (1024 * 1024)
        logging.info(f"Original size: {original_size:.1f} MB")
        logging.info(f"Quantized size: {quantized_size:.1f} MB")
        logging.info(f"Compression ratio: {original_size / quantized_size:.2f}x")
    
    # Save vocabulary.
    # model.decoder.vocabulary contains only non-blank tokens; append <blank> once.
    vocab = model.decoder.vocabulary
    with open(tokens_path, "w", encoding="utf-8") as f:
        for idx, token in enumerate(vocab):
            f.write(f"{token} {idx}\n")
        f.write(f"<blank> {len(vocab)}\n")
    logging.info(f"Saved vocabulary to: {tokens_path} ({len(vocab) + 1} tokens incl. blank)")
    
    # Print deployment instructions
    logging.info("\n" + "=" * 50)
    logging.info("DEPLOYMENT INSTRUCTIONS")
    logging.info("=" * 50)
    logging.info(f"\n1. Copy these files to backend/model/:")
    if quantize:
        logging.info(f"   - {quantized_path} -> model.onnx")
    else:
        logging.info(f"   - {onnx_path} -> model.onnx")
    logging.info(f"   - {tokens_path} -> tokens.txt")
    logging.info(f"\n2. Set USE_MOCK=false in your environment")
    logging.info(f"\n3. Restart the backend server")
    logging.info("=" * 50)


def verify_onnx_inference(onnx_path: str, tokens_path: str):
    """
    Verify ONNX model can run inference.
    
    Args:
        onnx_path: Path to ONNX model
        tokens_path: Path to vocabulary
    """
    import numpy as np
    import onnxruntime as ort
    
    logging.info("Verifying ONNX inference...")
    
    # Load vocabulary
    vocab = {}
    with open(tokens_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                vocab[int(parts[1])] = parts[0]
    
    # Create session
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    # Inspect actual input shapes from the exported model before building dummies
    inputs = session.get_inputs()
    for inp in inputs:
        logging.info(f"Input: {inp.name}, shape: {inp.shape}, type: {inp.type}")

    # NeMo Conformer ONNX export uses (batch, features, time) = (1, 80, T)
    n_frames = 100
    dummy_mel    = np.random.default_rng(0).standard_normal((1, 80, n_frames)).astype(np.float32)
    dummy_length = np.array([n_frames], dtype=np.int64)

    outputs = session.run(
        None,
        {
            inputs[0].name: dummy_mel,
            inputs[1].name: dummy_length,
        }
    )
    
    logging.info(f"Output shape: {outputs[0].shape}")
    logging.info("ONNX inference verification passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--nemo_path", type=str, required=True,
                        help="Path to .nemo checkpoint")
    parser.add_argument("--output_dir", type=str, default="./export",
                        help="Directory to save ONNX files")
    parser.add_argument("--no_quantize", action="store_true",
                        help="Skip INT8 quantization")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX inference after export")
    args = parser.parse_args()
    
    export_to_onnx(
        args.nemo_path,
        args.output_dir,
        quantize=not args.no_quantize,
    )
    
    if args.verify:
        onnx_path = Path(args.output_dir) / ("model.onnx" if args.no_quantize else "model.int8.onnx")
        tokens_path = Path(args.output_dir) / "tokens.txt"
        verify_onnx_inference(str(onnx_path), str(tokens_path))
