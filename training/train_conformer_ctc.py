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
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
torch.set_float32_matmul_precision('high')
from omegaconf import OmegaConf

# NeMo 2.x uses lightning (not pytorch_lightning) internally
try:
    from lightning.pytorch import Trainer, Callback
    from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
except ImportError:
    from pytorch_lightning import Trainer, Callback
    from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCBPEModel
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


PRETRAINED_MODEL = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
ENCODER_FREEZE_EPOCHS   = 8   # Stage 1: decoder only (more epochs to escape blank collapse)
ENCODER_PARTIAL_EPOCHS  = 15  # Stage 2: top 50% encoder + decoder
# Stage 3: epoch 15+ = full encoder + decoder with fresh LR reset

# Per-group learning rates
LR_STAGE1_DECODER  = 3e-3   # higher LR to break blank collapse fast in Stage 1
LR_STAGE2_DECODER  = 5e-4   # back off once encoder starts contributing
LR_STAGE2_ENCODER  = 5e-5
LR_STAGE3_DECODER  = 5e-5   # conservative fine-tuning — model already trained, avoid overshooting
LR_STAGE3_ENCODER  = 5e-6   # 10x lower than decoder


def _set_optimizer_lrs(trainer, encoder_lr, decoder_lr):
    """Set per-group learning rates in the optimizer."""
    optimizer = trainer.optimizers[0]
    for group in optimizer.param_groups:
        role = group.get("name", "")
        if role == "encoder":
            group["lr"] = encoder_lr
        else:
            group["lr"] = decoder_lr
    logging.info(f"  LR → encoder={encoder_lr}, decoder={decoder_lr}")


def _unfreeze_top_layers(encoder, layers, split):
    """Unfreeze layers[split:] and the encoder's final norm; return unfrozen params."""
    unfrozen = []
    for layer in layers[split:]:
        for p in layer.parameters():
            p.requires_grad = True
            unfrozen.append(p)
    if hasattr(encoder, "norm"):
        for p in encoder.norm.parameters():
            p.requires_grad = True
            unfrozen.append(p)
    return unfrozen


def _add_params_to_enc_group(optimizer, params):
    """Add params to the named 'encoder' group so they receive gradient updates."""
    enc_group = next((g for g in optimizer.param_groups if g.get("name") == "encoder"), None)
    if enc_group is None:
        return
    existing = {id(p) for p in enc_group["params"]}
    for p in params:
        if id(p) not in existing:
            enc_group["params"].append(p)
            existing.add(id(p))


class ThreeStageTrainingCallback(Callback):
    """
    3-stage progressive unfreezing:
      Stage 1 (epochs 0–7):   encoder fully frozen, decoder only (high LR, no scheduler)
      Stage 2 (epochs 8–14):  top 50% encoder layers unfrozen, scheduler re-enabled
      Stage 3 (epochs 15+):   full encoder unfrozen, warmup then full LRs
    """

    def on_train_start(self, trainer, pl_module):
        # NeMo creates a single param group by default, so we split it into two:
        # one for encoder params and one for everything else (decoder + preprocessor).
        optimizer = trainer.optimizers[0]
        if len(optimizer.param_groups) == 1:
            encoder_params = {id(p) for p in pl_module.encoder.parameters()}
            enc_group = {"params": [], "name": "encoder"}
            dec_group = {"params": [], "name": "decoder"}
            for p in optimizer.param_groups[0]["params"]:
                (enc_group if id(p) in encoder_params else dec_group)["params"].append(p)
            # Copy base hyperparams from the original group, then replace
            base = {k: v for k, v in optimizer.param_groups[0].items() if k != "params"}
            enc_group.update(base)
            dec_group.update(base)
            optimizer.param_groups.clear()
            optimizer.add_param_group(enc_group)
            optimizer.add_param_group(dec_group)
            logging.info(
                f"Split optimizer into 2 groups: "
                f"{len(enc_group['params'])} encoder params, "
                f"{len(dec_group['params'])} decoder params"
            )
        else:
            for group in optimizer.param_groups:
                if "name" not in group:
                    encoder_params = {id(p) for p in pl_module.encoder.parameters()}
                    is_enc = all(id(p) in encoder_params for p in group["params"])
                    group["name"] = "encoder" if is_enc else "decoder"
            logging.info("Optimizer param groups tagged: encoder / decoder")

    def _enter_stage1(self, trainer, pl_module):
        pl_module.encoder.freeze()
        # Bypass the warmup scheduler in Stage 1 — it would spend the first 500
        # steps ramping up from near-zero, starving the decoder of gradient signal
        # exactly when it needs to break out of blank collapse.
        for config in trainer.lr_scheduler_configs:
            config.frequency = 99999  # disable scheduler for Stage 1
        _set_optimizer_lrs(trainer, 0.0, LR_STAGE1_DECODER)
        logging.info(f"Stage 1: encoder frozen — decoder LR={LR_STAGE1_DECODER} (scheduler disabled)")

    def _enter_stage2(self, trainer, pl_module):
        layers = list(pl_module.encoder.layers)
        split = len(layers) // 2
        newly_unfrozen = _unfreeze_top_layers(pl_module.encoder, layers, split)
        _add_params_to_enc_group(trainer.optimizers[0], newly_unfrozen)
        # Re-enable the scheduler for Stage 2 now that the decoder is stable
        for config in trainer.lr_scheduler_configs:
            config.frequency = 1
        _set_optimizer_lrs(trainer, LR_STAGE2_ENCODER, LR_STAGE2_DECODER)
        # Ramp up spec augmentation now that the decoder can handle noisier inputs
        spec_aug = pl_module.spec_augmentation
        if spec_aug is not None:
            spec_aug.freq_masks  = 4
            spec_aug.freq_width  = 27
            spec_aug.time_masks  = 8
            spec_aug.time_width  = 0.05
            logging.info("Stage 2: spec augmentation increased")
        logging.info(
            f"Stage 2: unfroze top {len(layers) - split}/{len(layers)} encoder layers "
            f"(indices {split}–{len(layers)-1}), added {len(newly_unfrozen)} params to optimizer"
        )

    def _enter_stage3(self, trainer, pl_module):
        pl_module.encoder.unfreeze()

        # Kill the scheduler entirely — it's mid-cycle from Stage 2 and will
        # keep computing cosine(global_step/max_steps), overriding our manual
        # LRs every step. Setting frequency=99999 isn't enough because the
        # scheduler still fires once before the check, and base_lrs doesn't
        # reset the cosine position. Replacing last_epoch with a large value
        # freezes it at min_lr, so we just disable firing altogether.
        for config in trainer.lr_scheduler_configs:
            config.frequency = 99999

        # Set final LRs directly — no warmup needed, encoder LR is already
        # low enough (1e-5) that a ramp doesn't help and adds complexity.
        _set_optimizer_lrs(trainer, LR_STAGE3_ENCODER, LR_STAGE3_DECODER)

        logging.info(
            f"Stage 3: full encoder unfrozen — "
            f"encoder LR={LR_STAGE3_ENCODER}, decoder LR={LR_STAGE3_DECODER} (scheduler disabled)"
        )

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch == 0:
            self._enter_stage1(trainer, pl_module)
        elif epoch == ENCODER_FREEZE_EPOCHS:
            self._enter_stage2(trainer, pl_module)
        elif epoch == ENCODER_PARTIAL_EPOCHS:
            self._enter_stage3(trainer, pl_module)


class DifferentialLRCallback(Callback):
    """Set encoder LR lower than decoder LR from the start for resume training."""

    def on_train_start(self, trainer, pl_module):
        optimizer = trainer.optimizers[0]
        if len(optimizer.param_groups) == 1:
            encoder_params = {id(p) for p in pl_module.encoder.parameters()}
            enc_group = {"params": [], "name": "encoder"}
            dec_group = {"params": [], "name": "decoder"}
            for p in optimizer.param_groups[0]["params"]:
                (enc_group if id(p) in encoder_params else dec_group)["params"].append(p)
            base = {k: v for k, v in optimizer.param_groups[0].items() if k != "params"}
            enc_group.update(base)
            dec_group.update(base)
            optimizer.param_groups.clear()
            optimizer.add_param_group(enc_group)
            optimizer.add_param_group(dec_group)
        _set_optimizer_lrs(trainer, LR_STAGE3_ENCODER, LR_STAGE3_DECODER)
        logging.info(f"DifferentialLR: encoder={LR_STAGE3_ENCODER}, decoder={LR_STAGE3_DECODER}")


class ValidationMetricsCallback(Callback):
    """Log validation PER (phoneme error rate) alongside val_loss each epoch."""

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss")
        val_wer  = metrics.get("val_wer")  # NeMo logs this as WER; for phonemes it's PER
        parts = []
        if val_loss is not None:
            parts.append(f"val_loss={float(val_loss):.4f}")
        if val_wer is not None:
            parts.append(f"val_PER={float(val_wer)*100:.2f}%")
        if parts:
            logging.info("Validation — " + "  ".join(parts))


def load_vocab(vocab_path: str) -> list[str]:
    """Load non-blank tokens from tokens.txt.

    Token inventory notes:
      Q   — silence / non-speech marker emitted by the Quranic Phonemizer for
            pauses (e.g. waqf positions). Distinct from CTC <blank>.
      j̃ m̃ w̃ ñ — nasalized variants used by the phonemizer for specific
            assimilation rules (idgham bighunna).
    """
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
            "features": 80,
            "n_fft": 512,
            "frame_splicing": 1,
            "dither": 0.00001,     # training-only noise (disabled at eval automatically)
            "pad_to": 0,
            # Explicit NeMo defaults — match pretrained Arabic model exactly
            "preemph": 0.97,                # high-freq boost; critical for Arabic fricatives
            "mel_norm": "slaney",           # area-normalized mel filterbank
            "log_zero_guard_type": "add",
            "log_zero_guard_value": 2e-24,  # NeMo default (2**-24)
            "mag_power": 2.0,               # power spectrum
        },

        "spec_augment": {
            "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
            # Mild augmentation during Stage 1 — heavy masking on top of a frozen
            # encoder corrupts the stable pretrained features and prevents the
            # decoder from learning the CTC alignment at all (blank collapse).
            # ThreeStageTrainingCallback increases these in Stage 2+.
            "freq_masks": 2,
            "freq_width": 15,
            "time_masks": 3,
            "time_width": 0.03,
        },

        "encoder": {
            "_target_": "nemo.collections.asr.modules.ConformerEncoder",
            "feat_in": 80,
            "feat_out": -1,
            "n_layers": 17,
            "d_model": 512,
            "subsampling": "dw_striding",
            "subsampling_factor": 4,
            "subsampling_conv_channels": 256,
            "ff_expansion_factor": 4,
            "self_attention_model": "rel_pos_local_attn",
            "n_heads": 8,
            "att_context_size": [256, 256],
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
            "batch_size": 8,  # reduced for Stage 3 full-encoder memory budget
            "shuffle": True,
            "num_workers": 8,
            "pin_memory": True,
            "trim_silence": False,
            "max_duration": 25.0,  # reduced from 30.0 for more samples per epoch
            "min_duration": 0.5,
            "augmentor": {
                # Speed perturbation changes phoneme duration — disabled because
                # distinguishing short vs. long vowels (e.g. /i/ vs /i:/) is a
                # core Tajweed grading signal, not noise to be augmented away.
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
            "batch_size": 8,
            "shuffle": False,
            "num_workers": 8,
            "pin_memory": True,
            "trim_silence": False,
        },

        "optim": {
            "name": "adamw",
            "lr": LR_STAGE1_DECODER,
            "betas": [0.9, 0.98],
            "weight_decay": 1e-4,
            "sched": {
                "name": "CosineAnnealing",
                # Warmup and max_steps only active in Stage 2+ (scheduler disabled in Stage 1).
                # ~275 steps/epoch × 100 epochs; warmup over ~1 epoch of Stage 2.
                "warmup_steps": 275,
                "max_steps": 28000,
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


def train(data_dir: str = "./data", output_dir: str = "./output", max_epochs: int = 100,
          resume_weights: str = None):
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

    if resume_weights:
        # Load weights only from a .nemo file — fresh optimizer, fresh scheduler.
        # Use this to continue from a prior run without inheriting broken LR state.
        logging.info(f"Loading weights from: {resume_weights}")
        model = EncDecCTCModel.restore_from(resume_weights, map_location="cpu")
        model.setup_training_data(OmegaConf.create({
            "manifest_filepath": str(train_manifest),
            "sample_rate": 16000,
            "batch_size": 8,
            "shuffle": True,
            "num_workers": 8,
            "pin_memory": True,
            "trim_silence": False,
            "max_duration": 25.0,
            "min_duration": 0.5,
        }))
        model.setup_validation_data(OmegaConf.create({
            "manifest_filepath": str(val_manifest),
            "sample_rate": 16000,
            "batch_size": 8,
            "shuffle": False,
            "num_workers": 8,
            "pin_memory": True,
            "trim_silence": False,
        }))
        # Single LR for now — ThreeStageCallback is skipped so we rely on
        # the optimizer to handle encoder vs decoder via param groups set manually
        # in on_train_start if needed. Use decoder LR as the base; encoder will
        # be set lower by _set_optimizer_lrs after the first forward pass.
        model.setup_optimization(OmegaConf.create({
            "name": "adamw",
            "lr": LR_STAGE3_DECODER,  # 5e-4 for decoder
            "betas": [0.9, 0.98],
            "weight_decay": 5e-4,
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": 200,
                "max_steps": 28000,
                "min_lr": 5e-7,
            },
        }))
        # Manually freeze nothing — train full model end-to-end
        # but set encoder to a lower LR after optimizer is built
        logging.info(f"Resume mode: training end-to-end, encoder LR={LR_STAGE3_ENCODER}, decoder LR={LR_STAGE3_DECODER}")
    else:
        model = build_ctc_model(
            vocab=vocab,
            train_manifest=str(train_manifest),
            val_manifest=str(val_manifest),
        )

    use_gpu = torch.cuda.is_available()

    trainer_cfg = OmegaConf.create({
        "devices":                 1,
        "accelerator":             "gpu" if use_gpu else "cpu",
        "max_epochs":              max_epochs,
        "accumulate_grad_batches": 5,  # batch 8 × 5 = effective 40, same as before
        "gradient_clip_val":       1.0,
        "precision":               "bf16-mixed",  # 2x faster on 4090, stable training
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
        "create_early_stopping_callback": False,
        # When resuming from weights only, do NOT let exp_manager pick up the
        # existing .ckpt — it would restore the old optimizer state (2 param
        # groups) which conflicts with the freshly built 1-group optimizer.
        "resume_if_exists":            not bool(resume_weights),
        "resume_ignore_no_checkpoint": True,
    })

    callbacks = [
        ValidationMetricsCallback(),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=15,
            min_delta=0.1,
            verbose=True,
        ),
    ]
    if resume_weights:
        callbacks.insert(0, DifferentialLRCallback())
    else:
        callbacks.insert(0, ThreeStageTrainingCallback())

    trainer = Trainer(
        **OmegaConf.to_container(trainer_cfg, resolve=True),
        callbacks=callbacks,
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
    parser.add_argument("--data_dir",       default="./data")
    parser.add_argument("--output_dir",     default="./output")
    parser.add_argument("--max_epochs",     default=100, type=int)
    parser.add_argument("--resume_weights", default=None,
                        help="Path to .nemo file to load weights from (fresh optimizer)")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.max_epochs, args.resume_weights)
