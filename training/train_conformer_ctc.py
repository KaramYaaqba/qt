#!/usr/bin/env python3
"""
Fine-tune Arabic FastConformer for Quranic Phoneme Recognition

Loads the pretrained Arabic FastConformer encoder weights and attaches a
fresh CTC decoder with our Quranic phoneme vocabulary (40 tokens).
This avoids BPE tokenizer issues — CTC uses a simple character vocab.

Usage:
    python train_conformer_ctc.py [--data_dir ./data] [--output_dir ./output]
"""
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

# NeMo 2.x uses lightning (not pytorch_lightning) internally
try:
    from lightning.pytorch import Trainer, Callback
except ImportError:
    from pytorch_lightning import Trainer, Callback

from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCBPEModel
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


PRETRAINED_MODEL = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
ENCODER_FREEZE_EPOCHS = 10  # train decoder only for first N epochs


class EncoderUnfreezeCallback(Callback):
    """Freeze encoder at start, unfreeze after ENCODER_FREEZE_EPOCHS."""

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            pl_module.encoder.freeze()
            logging.info("Encoder frozen — training decoder only for first epochs")
        elif trainer.current_epoch == ENCODER_FREEZE_EPOCHS:
            pl_module.encoder.unfreeze()
            logging.info(f"Encoder unfrozen at epoch {trainer.current_epoch} — full fine-tuning")


def load_vocab(vocab_path: str) -> list[str]:
    """Load non-blank tokens from tokens.txt."""
    vocab = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] != "<blank>":
                vocab.append(parts[0])
    return vocab


def build_ctc_model(vocab: list[str], train_manifest: str,
                    val_manifest: str) -> EncDecCTCModel:
    """
    Build a Conformer-CTC model with Quranic phoneme vocabulary,
    then transplant the pretrained Arabic encoder weights into it.
    """
    use_gpu = torch.cuda.is_available()
    vocab_size = len(vocab)

    # Build fresh CTC model config matching FastConformer-Large architecture
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
            "features": 80,        # match pretrained model (80 mel bins) so weights transfer
            "n_fft": 512,
            "frame_splicing": 1,
            "dither": 0.00001,
            "pad_to": 0,
        },

        "spec_augment": {
            "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
            "freq_masks": 2,
            "freq_width": 27,
            "time_masks": 5,
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
            "att_context_size": [128, 128],
            "xscaling": True,
            "untie_biases": True,
            "pos_emb_max_len": 5000,
            "conv_kernel_size": 9,
            "dropout": 0.1,
            "dropout_emb": 0.0,
            "dropout_att": 0.1,
        },

        "decoder": {
            "_target_": "nemo.collections.asr.modules.ConvASRDecoder",
            "feat_in": 512,
            "num_classes": vocab_size,
            "vocabulary": vocab,
        },

        # Label smoothing discourages blank collapse by spreading probability
        # mass across all tokens, preventing the model from collapsing to blank
        "loss": {
            "_target_": "nemo.collections.asr.losses.CTCLoss",
            "num_classes": vocab_size,
            "zero_infinity": True,
            "reduction": "mean_batch",
        },

        "train_ds": {
            "manifest_filepath": train_manifest,
            "sample_rate": 16000,
            "batch_size": 16,
            "shuffle": True,
            "num_workers": 0,
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
            "num_workers": 0,
            "pin_memory": True,
            "trim_silence": False,
        },

        "optim": {
            "name": "adamw",
            "lr": 5e-4,  # higher LR to push decoder out of blank collapse faster
            "betas": [0.9, 0.98],
            "weight_decay": 1e-3,
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": 500,
                "min_lr": 1e-6,
            },
        },
    })

    logging.info("Initializing CTC model with Quranic phoneme vocab...")
    model = EncDecCTCModel(cfg=cfg)

    # Load pretrained Arabic model and transplant encoder weights
    logging.info(f"Loading pretrained encoder from: {PRETRAINED_MODEL}")
    pretrained = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        PRETRAINED_MODEL, map_location="cpu"
    )

    # Transplant compatible encoder weights key-by-key, skipping shape mismatches
    logging.info("Transplanting pretrained encoder weights (compatible layers only)...")
    pretrained_state = pretrained.encoder.state_dict()
    model_state = model.encoder.state_dict()
    transferred = skipped_keys = 0
    for key, param in pretrained_state.items():
        if key in model_state and model_state[key].shape == param.shape:
            model_state[key].copy_(param)
            transferred += 1
        else:
            skipped_keys += 1
    model.encoder.load_state_dict(model_state)
    logging.info(f"Transferred {transferred} layers, skipped {skipped_keys} (shape mismatch)")

    # Also transplant preprocessor weights if compatible
    try:
        model.preprocessor.load_state_dict(
            pretrained.preprocessor.state_dict(), strict=True
        )
        logging.info("Preprocessor weights transplanted successfully")
    except Exception as e:
        logging.warning(f"Could not transplant preprocessor weights: {e} — using random init")

    del pretrained
    torch.cuda.empty_cache() if use_gpu else None

    return model


def train(data_dir: str = "./data", output_dir: str = "./output"):
    data_path   = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab_path     = data_path / "tokens.txt"
    train_manifest = data_path / "manifest_train.json"
    val_manifest   = data_path / "manifest_dev.json"

    for path in [vocab_path, train_manifest, val_manifest]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    vocab = load_vocab(str(vocab_path))
    logging.info(f"Vocabulary size (non-blank): {len(vocab)}")

    model = build_ctc_model(
        vocab=vocab,
        train_manifest=str(train_manifest),
        val_manifest=str(val_manifest),
    )

    use_gpu = torch.cuda.is_available()

    trainer_cfg = OmegaConf.create({
        "devices":                 1,
        "accelerator":             "gpu" if use_gpu else "cpu",
        "max_epochs":              50,
        "accumulate_grad_batches": 4,
        "gradient_clip_val":       1.0,
        "precision":               "32-true",
        "log_every_n_steps":       5,
        "val_check_interval":      1.0,
        "enable_checkpointing":    False,  # exp_manager creates its own checkpointer
        "default_root_dir":        str(output_path),
        "logger":                  False,  # exp_manager creates its own logger
    })

    exp_manager_cfg = OmegaConf.create({
        "exp_dir":                    str(output_path),
        "name":                       "fastconformer_quran_6surahs",
        "create_tensorboard_logger":  True,
        "create_checkpoint_callback": True,
        "checkpoint_callback_params": {
            "monitor":          "val_loss",
            "mode":             "min",
            "save_top_k":       3,
            "always_save_nemo": True,
        },
        "resume_if_exists":            True,
        "resume_ignore_no_checkpoint": True,
    })

    trainer = Trainer(
        **OmegaConf.to_container(trainer_cfg, resolve=True),
        callbacks=[EncoderUnfreezeCallback()],
    )
    exp_manager(trainer, exp_manager_cfg)

    logging.info("Starting fine-tuning...")
    trainer.fit(model, model._train_dl, model._validation_dl)

    final_path = output_path / "fastconformer_quran_6surahs.nemo"
    model.save_to(str(final_path))
    logging.info(f"Saved final model: {final_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./data")
    parser.add_argument("--output_dir", default="./output")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)
