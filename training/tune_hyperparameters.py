#!/usr/bin/env python3
"""
Hyperparameter Tuning for Quran Conformer-CTC Model

Runs grid search over key hyperparameters to find optimal configuration.
"""
import argparse
from pathlib import Path
import itertools
import json

import torch
torch.set_float32_matmul_precision('high')

try:
    from lightning.pytorch import Trainer, Callback
    from lightning.pytorch.callbacks import LearningRateMonitor
except ImportError:
    from pytorch_lightning import Trainer, Callback
    from pytorch_lightning.callbacks import LearningRateMonitor

from nemo.collections.asr.models import EncDecCTCModel
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf

# Import functions from train_conformer_ctc.py
from train_conformer_ctc import (
    load_vocab, build_ctc_model, ThreeStageTrainingCallback,
    ValidationMetricsCallback, LR_STAGE1_DECODER, LR_STAGE2_DECODER,
    LR_STAGE3_DECODER, LR_STAGE2_ENCODER, LR_STAGE3_ENCODER,
    ENCODER_FREEZE_EPOCHS, ENCODER_PARTIAL_EPOCHS
)


def tune_hyperparameters(data_dir: str = "./data", output_dir: str = "./tune_output"):
    """
    Run hyperparameter grid search for best model configuration.
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab_path = data_path / "tokens.txt"
    train_manifest = data_path / "manifest_train.json"
    val_manifest = data_path / "manifest_dev.json"

    for path in [vocab_path, train_manifest, val_manifest]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    vocab = load_vocab(str(vocab_path))

    # Define hyperparameter search space
    search_space = {
        'batch_size': [16, 20],
        'lr_stage1': [3e-4, 5e-4],
        'weight_decay': [1e-3, 5e-4],
        'dropout': [0.1, 0.15],
        'freq_masks': [4, 6],
        'time_masks': [8, 12],
    }

    # Generate all combinations
    keys = search_space.keys()
    values = search_space.values()
    combinations = list(itertools.product(*values))

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        logging.info(f"Trial {i+1}/{len(combinations)}: {params}")

        # Build model with current hyperparameters
        model = build_ctc_model_tuned(vocab, str(train_manifest), str(val_manifest), params)

        # Quick training run (reduced epochs)
        trainer = Trainer(
            devices=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=5,  # Quick evaluation
            accumulate_grad_batches=2,
            gradient_clip_val=1.0,
            precision="32-true",
            log_every_n_steps=10,
            val_check_interval=1.0,
            enable_checkpointing=False,
            logger=False,
            callbacks=[
                ThreeStageTrainingCallback(),
                ValidationMetricsCallback(),
                LearningRateMonitor(logging_interval="epoch"),
            ],
        )

        trainer.fit(model, model._train_dl, model._validation_dl)

        # Record final validation metrics
        final_metrics = trainer.callback_metrics
        val_wer = final_metrics.get("val_wer", float('inf'))

        results.append({
            'trial': i+1,
            'params': params,
            'val_wer': float(val_wer),
        })

        logging.info(f"Trial {i+1} completed. Val WER: {val_wer:.4f}")

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Save results
    results_path = output_path / "tuning_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Find best configuration
    best_result = min(results, key=lambda x: x['val_wer'])
    logging.info(f"Best configuration: {best_result}")

    return best_result


def build_ctc_model_tuned(vocab, train_manifest, val_manifest, params):
    """Build CTC model with tuned hyperparameters."""
    use_gpu = torch.cuda.is_available()
    vocab_size = len(vocab)

    cfg = OmegaConf.create({
        "sample_rate": 16000,
        "labels": vocab,

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
            "preemph": 0.97,
            "mel_norm": "slaney",
            "log_zero_guard_type": "add",
            "log_zero_guard_value": 2e-24,
            "mag_power": 2.0,
        },

        "spec_augment": {
            "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
            "freq_masks": params['freq_masks'],
            "freq_width": 27,
            "time_masks": params['time_masks'],
            "time_width": 0.05,
        },

        "encoder": {
            "_target_": "nemo.collections.asr.modules.ConformerEncoder",
            "feat_in": 80,
            "feat_out": -1,
            "n_layers": 17,
            "d_model": 512,
            "subsampling": "dw_striding",
            "subsampling_factor": 8,
            "subsampling_conv_channels": 256,
            "ff_expansion_factor": 4,
            "self_attention_model": "rel_pos_local_attn",
            "n_heads": 8,
            "att_context_size": [256, 256],
            "xscaling": True,
            "untie_biases": True,
            "pos_emb_max_len": 5000,
            "conv_kernel_size": 9,
            "dropout": params['dropout'],
            "dropout_emb": 0.0,
            "dropout_att": params['dropout'],
        },

        "decoder": {
            "_target_": "nemo.collections.asr.modules.ConvASRDecoder",
            "feat_in": 512,
            "num_classes": vocab_size,
            "vocabulary": vocab,
        },

        "loss": {
            "_target_": "nemo.collections.asr.losses.CTCLoss",
            "num_classes": vocab_size,
            "zero_infinity": True,
            "reduction": "mean_batch",
        },

        "train_ds": {
            "manifest_filepath": train_manifest,
            "sample_rate": 16000,
            "batch_size": params['batch_size'],
            "shuffle": True,
            "num_workers": 4,  # Reduced for tuning
            "pin_memory": True,
            "trim_silence": False,
            "max_duration": 25.0,
            "min_duration": 0.5,
            "augmentor": {
                "speed": {
                    "prob": 0.7,
                    "sr": 16000,
                    "resample_type": "kaiser_fast",
                    "min_speed_rate": 0.85,
                    "max_speed_rate": 1.15,
                },
                "noise": {
                    "prob": 0.3,
                    "min_snr_db": 10,
                    "max_snr_db": 50,
                    "noise_type": "white",
                },
                "gain": {
                    "prob": 0.5,
                    "min_gain_dbfs": -10,
                    "max_gain_dbfs": 10,
                },
            },
        },

        "validation_ds": {
            "manifest_filepath": val_manifest,
            "sample_rate": 16000,
            "batch_size": params['batch_size'],
            "shuffle": False,
            "num_workers": 4,
            "pin_memory": True,
            "trim_silence": False,
        },

        "optim": {
            "name": "adamw",
            "lr": params['lr_stage1'],
            "betas": [0.9, 0.98],
            "weight_decay": params['weight_decay'],
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": 1000,
                "min_lr": 5e-7,
            },
        },
    })

    model = EncDecCTCModel(cfg=cfg)

    # Load pretrained weights (same as original)
    PRETRAINED_MODEL = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
    logging.info(f"Loading pretrained encoder from: {PRETRAINED_MODEL}")
    pretrained = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        PRETRAINED_MODEL, map_location="cpu"
    )

    pretrained_state = pretrained.encoder.state_dict()
    model_state = model.encoder.state_dict()
    transferred = 0
    for key, param in pretrained_state.items():
        if key in model_state and model_state[key].shape == param.shape:
            model_state[key].copy_(param)
            transferred += 1
    model.encoder.load_state_dict(model_state)
    logging.info(f"Transferred {transferred} encoder layers")

    try:
        model.preprocessor.load_state_dict(
            pretrained.preprocessor.state_dict(), strict=True
        )
        logging.info("Preprocessor weights transplanted")
    except Exception as e:
        logging.warning(f"Could not transplant preprocessor: {e}")

    del pretrained
    torch.cuda.empty_cache()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./tune_output")
    args = parser.parse_args()
    best_config = tune_hyperparameters(args.data_dir, args.output_dir)
    print(f"Best hyperparameters found: {best_config}")