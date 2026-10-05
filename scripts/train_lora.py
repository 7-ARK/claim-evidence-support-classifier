"""Train the LoRA classifier from a yaml config."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cesc.env import prepare_runtime

prepare_runtime()

import torch

from cesc.config import load_config, optional_limit, resolve_path
from cesc.data import limit_rows, read_jsonl, tokenize_rows
from cesc.labels import label_maps
from cesc.modeling import (
    BestAdapterCallback,
    attach_lora,
    build_training_arguments,
    load_base_model,
    load_tokenizer,
    make_trainer,
    parameter_counts,
    save_adapter,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA classification adapter.")
    parser.add_argument("--config", required=True, help="Path to the yaml config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    set_seed(int(cfg["seed"]))

    train_rows = limit_rows(
        read_jsonl(resolve_path(cfg["train_file"])),
        optional_limit(cfg["max_train_samples"]),
        int(cfg["seed"]),
    )
    val_rows = limit_rows(
        read_jsonl(resolve_path(cfg["val_file"])),
        optional_limit(cfg["max_eval_samples"]),
        int(cfg["seed"]),
    )
    print(f"Train rows: {len(train_rows)}  Validation rows: {len(val_rows)}", flush=True)

    tokenizer = load_tokenizer(cfg["model_name"], int(cfg["max_length"]))
    label2id, _id2label = label_maps()
    train_ds = tokenize_rows(tokenizer, train_rows, label2id, int(cfg["max_length"]))
    val_ds = tokenize_rows(tokenizer, val_rows, label2id, int(cfg["max_length"]))

    model = load_base_model(cfg["model_name"])
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model, heads = attach_lora(model, cfg)
    trainable, total = parameter_counts(model)
    print(f"Trainable parameters: {trainable} of {total}", flush=True)
    print("Head modules kept trainable: " + ", ".join(heads), flush=True)

    output_dir = resolve_path(cfg["output_dir"])
    trainer_dir = output_dir / "trainer_work"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = resolve_path(cfg["adapter_dir"])
    meta = {
        "model_name": cfg["model_name"],
        "max_length": int(cfg["max_length"]),
        "run_name": cfg["run_name"],
        "seed": int(cfg["seed"]),
        "epochs": int(cfg["epochs"]),
        "learning_rate": float(cfg["learning_rate"]),
        "lora_r": int(cfg["lora_r"]),
        "lora_alpha": int(cfg["lora_alpha"]),
        "lora_dropout": float(cfg["lora_dropout"]),
        "target_modules": list(cfg["target_modules"]),
        "modules_to_save": heads,
        "max_train_samples": optional_limit(cfg["max_train_samples"]),
        "train_rows_used": len(train_rows),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "best_val_macro_f1": None,
    }
    callback = BestAdapterCallback(adapter_dir, meta)
    args_obj = build_training_arguments(
        cfg,
        trainer_dir,
        epochs=int(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        num_examples=len(train_rows),
    )
    trainer = make_trainer(model, args_obj, train_ds, val_ds, tokenizer, [callback.bind(model)])
    trainer.train()
    if not callback.saved:
        save_adapter(model, adapter_dir, meta)
    print(f"Adapter saved to {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
