"""Load the yaml train config."""

from __future__ import annotations

from pathlib import Path

import yaml

from cesc.paths import ROOT

REQUIRED = [
    "model_name",
    "seed",
    "max_length",
    "batch_size",
    "eval_batch_size",
    "grad_accum",
    "epochs",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "baseline_head_epochs",
    "baseline_head_lr",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "fp16",
    "max_train_samples",
    "max_eval_samples",
    "train_file",
    "val_file",
    "test_file",
    "output_dir",
    "adapter_dir",
    "logging_steps",
    "run_name",
]


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    missing = [key for key in REQUIRED if key not in cfg]
    if missing:
        raise SystemExit("Config is missing keys: " + ", ".join(missing))
    if not isinstance(cfg["target_modules"], list) or not cfg["target_modules"]:
        raise SystemExit("target_modules must be a non empty list")
    cfg["_config_path"] = str(config_path)
    return cfg


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def optional_limit(value):
    if value is None or value == "":
        return None
    return int(value)
