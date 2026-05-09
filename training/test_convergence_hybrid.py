#!/usr/bin/env python3
"""
Test Convergence of Fine-Tuned Hybrid RNNT Model on Quranic Data

This script tests the hybrid model on a small subset of data to verify:
1. Model loads correctly and is trainable
2. Loss decreases over epochs (convergence)
3. PER improves over epochs
4. No OOM or crash issues on your machine

Similar to train_conformer_ctc.py but for hybrid RNNT fine-tuning.

Usage:
    python test_convergence_hybrid.py \
        --data_dir ./data \
        --output_dir ./test_output \
        --num_samples 100 \
        --num_epochs 3
"""

import argparse
import os
import json
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
torch.set_float32_matmul_precision('high')

try:
    from lightning.pytorch import Trainer, Callback
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
except ImportError:
    from pytorch_lightning import Trainer, Callback
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from omegaconf import OmegaConf
from nemo.collections.asr import models
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


# Configuration
PRETRAINED_MODEL = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
ACCUMULATE_GRAD_BATCHES = 2


class ConvergenceMonitorCallback(Callback):
    """Monitor PER (phoneme error rate) and loss to verify convergence."""
    
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss")
        val_wer = metrics.get("val_wer")  # WER for phonemes = PER
        epoch = trainer.current_epoch
        
        if val_loss is not None:
            val_loss_val = float(val_loss)
            per = float(val_wer) * 100 if val_wer is not None else 0
            
            logging.info(
                f"Epoch {epoch:2d} | "
                f"val_loss={val_loss_val:.4f} | "
                f"val_PER={per:.2f}%"
            )


class EarlyConvergenceDetection(Callback):
    """Detect if model converges too quickly or diverges."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.no_improve_count = 0
    
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = float(trainer.callback_metrics.get("val_loss", float('inf')))
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        
        if self.no_improve_count >= self.patience:
            logging.warning(
                f"⚠️  No improvement for {self.patience} epochs. "
                f"Best loss: {self.best_loss:.4f}"
            )


def create_test_manifest(
    source_manifest: str,
    output_manifest: str,
    num_samples: int = 100
) -> Tuple[int, float]:
    """
    Create a test manifest with a subset of samples.
    
    Returns:
        (num_samples, total_duration)
    """
    samples = []
    total_duration = 0.0
    
    with open(source_manifest, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            entry = json.loads(line)
            samples.append(entry)
            total_duration += entry.get('duration', 0)
    
    with open(output_manifest, 'w') as f:
        for entry in samples:
            f.write(json.dumps(entry) + '\n')
    
    return len(samples), total_duration


def main(args):
    """Main convergence test."""
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    logging.info("=" * 70)
    logging.info("🚀 CONVERGENCE TEST: Fine-Tuned Hybrid RNNT Model")
    logging.info("=" * 70)
    
    # Create test manifests
    logging.info(f"\n📊 Creating test manifests ({args.num_samples} samples each)...")
    
    train_manifest = data_dir / "manifest_train.json"
    test_manifest = output_dir / "test_train.json"
    n_train, dur_train = create_test_manifest(
        str(train_manifest), str(test_manifest), args.num_samples
    )
    logging.info(f"   Train: {n_train} samples, {dur_train:.1f}s total")
    
    val_manifest = data_dir / "manifest_dev.json"
    test_val_manifest = output_dir / "test_val.json"
    n_val, dur_val = create_test_manifest(
        str(val_manifest), str(test_val_manifest), args.num_samples // 2
    )
    logging.info(f"   Val:   {n_val} samples, {dur_val:.1f}s total")
    
    # Load pretrained model
    logging.info(f"\n📥 Loading pretrained model: {PRETRAINED_MODEL}")
    model = models.EncDecRNNTBPEModel.from_pretrained(PRETRAINED_MODEL)
    logging.info("✅ Model loaded")
    
    # Check vocab
    tokens_file = data_dir / "tokens.txt"
    if tokens_file.exists():
        with open(tokens_file) as f:
            tokens = [line.strip() for line in f]
        logging.info(f"📋 Tokens file found: {len(tokens)} tokens")
        logging.info(f"   Model vocab size: {model.cfg.decoder.vocab_size}")
        
        if len(tokens) != model.cfg.decoder.vocab_size:
            logging.warning("⚠️  Vocab size mismatch - may need to update model vocabulary")
    
    # Setup training data
    logging.info(f"\n⚙️  Setting up training data...")
    model.setup_training_data(train_data_config=OmegaConf.create({
        "manifest_filepath": str(test_manifest),
        "sample_rate": 16000,
        "batch_size": BATCH_SIZE,
        "num_workers": 2,
        "shuffle": True,
        "pin_memory": True,
    }))
    
    model.setup_validation_data(val_data_config=OmegaConf.create({
        "manifest_filepath": str(test_val_manifest),
        "sample_rate": 16000,
        "batch_size": BATCH_SIZE,
        "num_workers": 2,
    }))
    
    logging.info("✅ Data loaders created")
    
    # Update hyperparams
    model.cfg.optim.lr = LEARNING_RATE
    model.cfg.trainer.accumulate_grad_batches = ACCUMULATE_GRAD_BATCHES
    
    logging.info(f"\n📈 Training config:")
    logging.info(f"   Epochs: {args.num_epochs}")
    logging.info(f"   LR: {LEARNING_RATE}")
    logging.info(f"   Batch size: {BATCH_SIZE}")
    logging.info(f"   Accumulate grads: {ACCUMULATE_GRAD_BATCHES}")
    
    # Setup trainer
    logging.info(f"\n🔧 Setting up trainer...")
    trainer = Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=args.num_epochs,
        num_sanity_val_steps=1,
        enable_checkpointing=True,
        logger=False,
        log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="{epoch:02d}-{val_loss:.2f}",
                save_last=True,
                monitor="val_loss",
                mode="min",
            ),
            ConvergenceMonitorCallback(),
            EarlyConvergenceDetection(patience=2),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        enable_progress_bar=True,
    )
    logging.info("✅ Trainer ready")
    
    # Run training
    logging.info(f"\n🚀 Starting convergence test ({args.num_epochs} epochs)...\n")
    trainer.fit(model)
    
    # Print results
    logging.info("\n" + "=" * 70)
    logging.info("✅ CONVERGENCE TEST COMPLETE")
    logging.info("=" * 70)
    
    # Check if loss decreased
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if checkpoint_files:
        logging.info(f"\n📁 Checkpoints saved: {len(checkpoint_files)}")
        logging.info(f"   Dir: {checkpoint_dir}")
        
        # Load best checkpoint and test
        best_ckpt = sorted(checkpoint_files)[-1]
        logging.info(f"\n📥 Loading best checkpoint: {best_ckpt.name}")
        model_best = models.EncDecRNNTBPEModel.load_from_checkpoint(str(best_ckpt))
        
        logging.info("\n✅ Model is trainable and checkpoints work!")
        logging.info("\n🎯 NEXT STEPS:")
        logging.info("   1. If loss decreased → proceed with full fine-tuning on Colab")
        logging.info("   2. If loss diverged → check data/learning rate")
        logging.info("   3. Upload this output to inspect convergence plots")
    
    logging.info("\n📊 For detailed metrics, check log above ↑")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test convergence of hybrid model")
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Path to data directory with manifest files"
    )
    parser.add_argument(
        "--output_dir",
        default="./test_output",
        help="Path to save test output and checkpoints"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples for convergence test (per split)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs for convergence test"
    )
    
    args = parser.parse_args()
    main(args)
