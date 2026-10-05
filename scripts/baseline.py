"""Majority class baseline and a frozen backbone head baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cesc.env import prepare_runtime

prepare_runtime()

import torch
from transformers import TrainerCallback

from cesc.config import load_config, optional_limit, resolve_path
from cesc.data import limit_rows, read_jsonl, tokenize_rows
from cesc.labels import LABELS, label_maps
from cesc.metrics import classification_scores
from cesc.modeling import (
    build_training_arguments,
    freeze_backbone,
    load_base_model,
    load_tokenizer,
    make_trainer,
    parameter_counts,
    predict_rows,
    set_seed,
)


def majority_label(rows: list[dict]) -> str:
    counts = Counter(row["label"] for row in rows)
    best = max(counts[label] for label in LABELS)
    for label in LABELS:
        if counts[label] == best:
            return label
    raise SystemExit("Could not choose a majority label.")


def trainable_state(model):
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def restore_trainable_state(model, state) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value.to(params[name].device))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the majority and frozen head baselines.")
    parser.add_argument("--config", required=True, help="Path to the yaml config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    set_seed(int(cfg["seed"]))

    full_train = read_jsonl(resolve_path(cfg["train_file"]))
    test_rows = read_jsonl(resolve_path(cfg["test_file"]))
    chosen = majority_label(full_train)
    majority_ids = [LABELS.index(chosen)] * len(test_rows)
    true_ids = [LABELS.index(row["label"]) for row in test_rows]
    majority_scores = classification_scores(true_ids, majority_ids)
    majority_note = (
        "Majority class predicts the most common training label for every test row. "
        "On a tie, the earliest label in the fixed label order is used. "
        f"This run predicts {chosen}. The classes are balanced, so the tie rule applies. "
        "The majority count uses the full training file."
    )

    train_rows = limit_rows(full_train, optional_limit(cfg["max_train_samples"]), int(cfg["seed"]))
    val_rows = limit_rows(
        read_jsonl(resolve_path(cfg["val_file"])),
        optional_limit(cfg["max_eval_samples"]),
        int(cfg["seed"]),
    )
    tokenizer = load_tokenizer(cfg["model_name"], int(cfg["max_length"]))
    label2id, _id2label = label_maps()
    model = load_base_model(cfg["model_name"])
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    heads = freeze_backbone(model)
    trainable, total = parameter_counts(model)
    print(f"Frozen head trainable parameters: {trainable} of {total}", flush=True)
    print("Trainable modules: " + ", ".join(heads), flush=True)

    holder = {"state": None, "score": None}

    class KeepBestHead(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return
            score = metrics.get("eval_macro_f1", metrics.get("macro_f1"))
            if score is None:
                return
            print(f"Frozen head validation macro F1: {float(score):.4f}", flush=True)
            if holder["score"] is None or float(score) >= holder["score"]:
                holder["score"] = float(score)
                holder["state"] = trainable_state(model)

    output_dir = resolve_path(cfg["output_dir"])
    trainer_dir = output_dir / "baseline_trainer_work"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    train_args = build_training_arguments(
        cfg,
        trainer_dir,
        epochs=int(cfg["baseline_head_epochs"]),
        learning_rate=float(cfg["baseline_head_lr"]),
        num_examples=len(train_rows),
    )
    trainer = make_trainer(
        model,
        train_args,
        tokenize_rows(tokenizer, train_rows, label2id, int(cfg["max_length"])),
        tokenize_rows(tokenizer, val_rows, label2id, int(cfg["max_length"])),
        tokenizer,
        [KeepBestHead()],
    )
    trainer.train()
    if holder["state"] is not None:
        restore_trainable_state(model, holder["state"])

    pred_ids, _probs = predict_rows(
        model, tokenizer, test_rows, int(cfg["max_length"]), int(cfg["eval_batch_size"])
    )
    frozen_scores = classification_scores(true_ids, pred_ids)
    frozen_note = (
        "Frozen head trains "
        + ", ".join(heads)
        + " and keeps the rest of the backbone frozen. "
        f"Train rows used: {len(train_rows)}. Epochs: {int(cfg['baseline_head_epochs'])}."
    )
    payload = {
        "run_name": cfg["run_name"],
        "majority": majority_scores,
        "majority_note": majority_note,
        "majority_label": chosen,
        "frozen_head": frozen_scores,
        "frozen_note": frozen_note,
        "frozen_trainable_parameters": trainable,
        "frozen_total_parameters": total,
        "frozen_modules": heads,
        "train_rows_used": len(train_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "baseline_metrics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Baseline metrics written to {path}", flush=True)
    print(f"Majority macro F1: {majority_scores['macro_f1']:.4f}", flush=True)
    print(f"Frozen head macro F1: {frozen_scores['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
