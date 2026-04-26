#!/usr/bin/env python3
"""
Train Conformer-CTC Model for Quranic Phoneme Recognition

Uses NVIDIA NeMo to train a Conformer-CTC-Small model for
speech-to-phoneme transcription of Quranic recitation.

Usage:
    python train_conformer_ctc.py [--config config.yaml]
"""
import os
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

# NeMo imports
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModel
from nemo.utils import logging


def create_model_config(
    vocab_path: str,
    train_manifest: str,
    val_manifest: str,
    output_dir: str,
) -> OmegaConf:
    """
    Create model configuration for Conformer-CTC-Small.
    
    Args:
        vocab_path: Path to tokens.txt vocabulary file
        train_manifest: Path to training manifest
        val_manifest: Path to validation manifest
        output_dir: Directory for checkpoints and logs
        
    Returns:
        OmegaConf configuration
    """
    # Load vocabulary — exclude <blank>; NeMo's CTC loss adds the blank internally
    vocab = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] != "<blank>":
                vocab.append(parts[0])

    # num_classes passed to ConvASRDecoder must equal len(vocabulary) (non-blank tokens).
    # NeMo appends the blank class automatically, so blank must NOT be counted here.
    vocab_size = len(vocab)
    logging.info(f"Vocabulary size (non-blank): {vocab_size}")
    
    config = OmegaConf.create({
        "name": "Conformer-CTC-Quran",
        
        # Model architecture
        "model": {
            # Preprocessor - 80-dim log-mel features
            "preprocessor": {
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "sample_rate": 16000,
                "normalize": "per_feature",
                "window_size": 0.025,
                "window_stride": 0.01,
                "window": "hann",
                "features": 80,
                "n_fft": 512,
                "frame_splicing": 1,
                "dither": 0.00001,
                "pad_to": 0,
            },
            
            # Spec augmentation
            "spec_augment": {
                "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
                "freq_masks": 2,
                "freq_width": 27,
                "time_masks": 5,
                "time_width": 0.05,
            },
            
            # Encoder - Conformer Small
            "encoder": {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "feat_in": 80,
                "feat_out": -1,
                "n_layers": 16,
                "d_model": 256,
                "subsampling": "striding",
                "subsampling_factor": 4,
                "subsampling_conv_channels": 256,
                "ff_expansion_factor": 4,
                "self_attention_model": "rel_pos",
                "n_heads": 4,
                "att_context_size": [-1, -1],
                "xscaling": True,
                "untie_biases": True,
                "pos_emb_max_len": 5000,
                "conv_kernel_size": 31,
                "dropout": 0.1,
                "dropout_emb": 0.0,
                "dropout_att": 0.1,
            },
            
            # Decoder
            "decoder": {
                "_target_": "nemo.collections.asr.modules.ConvASRDecoder",
                "feat_in": 256,
                "num_classes": vocab_size,
                "vocabulary": vocab,
            },
            
            # Dataset configs
            "train_ds": {
                "manifest_filepath": train_manifest,
                "sample_rate": 16000,
                "batch_size": 16,
                "shuffle": True,
                "num_workers": 4,
                "pin_memory": True,
                "trim_silence": False,
                "max_duration": 30.0,
                "min_duration": 0.5,
            },
            
            "validation_ds": {
                "manifest_filepath": val_manifest,
                "sample_rate": 16000,
                "batch_size": 16,
                "shuffle": False,
                "num_workers": 4,
                "pin_memory": True,
                "trim_silence": False,
            },
            
            # Optimizer
            "optim": {
                "name": "adamw",
                "lr": 0.001,
                "betas": [0.9, 0.98],
                "weight_decay": 0.001,
                "sched": {
                    "name": "NoamAnnealing",
                    "warmup_steps": 1000,
                    "warmup_ratio": None,
                    "min_lr": 1e-6,
                },
            },
        },
        
        # Trainer config
        "trainer": {
            "devices": 1,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "max_epochs": 50,
            "accumulate_grad_batches": 1,
            "gradient_clip_val": 1.0,
            "precision": 16 if torch.cuda.is_available() else 32,
            "log_every_n_steps": 10,
            "val_check_interval": 1.0,
            "enable_checkpointing": True,
            "default_root_dir": output_dir,
        },
        
        # Experiment manager
        "exp_manager": {
            "exp_dir": output_dir,
            "name": "conformer_ctc_quran",
            "create_tensorboard_logger": True,
            "create_checkpoint_callback": True,
            "checkpoint_callback_params": {
                # val_wer is word-level and misleading for a phoneme model;
                # val_loss is the correct signal for checkpoint selection here.
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": 3,
                "always_save_nemo": True,
            },
            "resume_if_exists": True,
            "resume_ignore_no_checkpoint": True,
        },
    })
    
    return config


def train(
    data_dir: str = "./data",
    output_dir: str = "./output",
):
    """
    Train the Conformer-CTC model.
    
    Args:
        data_dir: Directory containing manifests and vocabulary
        output_dir: Directory for checkpoints and logs
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check for required files
    vocab_path = data_path / "tokens.txt"
    train_manifest = data_path / "manifest_train.json"
    val_manifest = data_path / "manifest_dev.json"
    
    for path in [vocab_path, train_manifest, val_manifest]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    
    logging.info("Creating model configuration...")
    config = create_model_config(
        vocab_path=str(vocab_path),
        train_manifest=str(train_manifest),
        val_manifest=str(val_manifest),
        output_dir=str(output_path),
    )
    
    logging.info("Initializing model...")
    model = EncDecCTCModel(cfg=config.model)
    
    # Setup experiment manager
    from nemo.utils.exp_manager import exp_manager
    
    trainer = pl.Trainer(**config.trainer)
    exp_manager(trainer, config.exp_manager)
    
    logging.info("Starting training...")
    trainer.fit(model)
    
    # Save final model
    final_model_path = output_path / "conformer_ctc_quran.nemo"
    model.save_to(str(final_model_path))
    logging.info(f"Saved final model to: {final_model_path}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conformer-CTC model")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory for checkpoints and logs")
    args = parser.parse_args()
    
    train(args.data_dir, args.output_dir)
