#!/usr/bin/env python3
"""
Fine-tune Arabic FastConformer for Quranic Phoneme Recognition

Starts from nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0 and fine-tunes
on 6 surahs of Juz' Amma using non-Arabic speaker recordings (RetaSy dataset).

Usage:
    python train_conformer_ctc.py [--data_dir ./data] [--output_dir ./output]
"""
import argparse
import os
import tempfile
from pathlib import Path

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


PRETRAINED_MODEL = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"


def load_vocab(vocab_path: str) -> list[str]:
    """Load non-blank tokens from tokens.txt."""
    vocab = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] != "<blank>":
                vocab.append(parts[0])
    return vocab


def build_tokenizer_dir(vocab: list[str], output_path: Path) -> Path:
    """
    Build a BPE SentencePiece tokenizer where every Quranic phoneme is a
    single token. We use user_defined_symbols so SPM never splits them.
    """
    import sentencepiece as spm

    tokenizer_dir = output_path / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # Corpus: each phoneme repeated many times so SPM has enough data
    corpus_path = tokenizer_dir / "corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for _ in range(200):
            for token in vocab:
                f.write(token + "\n")

    model_prefix = str(tokenizer_dir / "tokenizer")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=128,             # larger than needed — SPM requires headroom
        character_coverage=1.0,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=list(vocab),  # each phoneme becomes one token
    )

    logging.info(f"Built tokenizer at: {tokenizer_dir}")
    return tokenizer_dir


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

    # Build character-level tokenizer for Quranic phonemes
    logging.info("Building Quranic phoneme tokenizer...")
    tokenizer_dir = build_tokenizer_dir(vocab, output_path)

    # Load pretrained Arabic model
    logging.info(f"Loading pretrained model: {PRETRAINED_MODEL}")
    model = EncDecHybridRNNTCTCBPEModel.from_pretrained(PRETRAINED_MODEL)

    # Swap tokenizer to Quranic phoneme vocabulary
    logging.info("Swapping tokenizer to Quranic phoneme vocabulary...")
    model.change_vocabulary(
        new_tokenizer_dir=str(tokenizer_dir),
        new_tokenizer_type="char",
    )

    # Update data configs
    model.cfg.train_ds = OmegaConf.create({
        "manifest_filepath": str(train_manifest),
        "sample_rate":       16000,
        "batch_size":        8,
        "shuffle":           True,
        "num_workers":       2,
        "pin_memory":        True,
        "trim_silence":      False,
        "max_duration":      30.0,
        "min_duration":      0.5,
    })
    model.cfg.validation_ds = OmegaConf.create({
        "manifest_filepath": str(val_manifest),
        "sample_rate":       16000,
        "batch_size":        8,
        "shuffle":           False,
        "num_workers":       2,
        "pin_memory":        True,
        "trim_silence":      False,
    })
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)

    # Fine-tuning optimizer — lower LR than scratch training
    model.cfg.optim = OmegaConf.create({
        "name":         "adamw",
        "lr":           1e-4,
        "betas":        [0.9, 0.98],
        "weight_decay": 1e-3,
        "sched": {
            "name":          "CosineAnnealing",
            "warmup_steps":  100,
            "min_lr":        1e-6,
        },
    })
    model.setup_optimization(model.cfg.optim)

    use_gpu = torch.cuda.is_available()

    trainer_cfg = OmegaConf.create({
        "devices":                 1,
        "accelerator":             "gpu" if use_gpu else "cpu",
        "max_epochs":              30,
        "accumulate_grad_batches": 4,
        "gradient_clip_val":       1.0,
        "precision":               "16-mixed" if use_gpu else "32-true",
        "log_every_n_steps":       5,
        "val_check_interval":      1.0,
        "enable_checkpointing":    True,
        "default_root_dir":        str(output_path),
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

    trainer = pl.Trainer(**OmegaConf.to_container(trainer_cfg, resolve=True))
    exp_manager(trainer, exp_manager_cfg)

    logging.info("Starting fine-tuning...")
    trainer.fit(model)

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
