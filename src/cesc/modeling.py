"""Model setup, training, and prediction.

LoRA is attached to the query and value projections. The classification head stays trainable.
This is not QLoRA: the base weights are not quantized.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import softmax

from cesc.labels import LABELS, label_maps
from cesc.metrics import compute_metrics


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_name: str, max_length: int):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        print("Fast tokenizer failed. Loading the slow tokenizer.", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.model_max_length = int(max_length)
    return tokenizer


def load_base_model(model_name: str):
    from transformers import AutoModelForSequenceClassification

    label2id, id2label = label_maps()
    # The hub file is float16. Adam on those weights turns non finite on CPU,
    # and mixed precision should be left to the trainer when CUDA is present.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        dtype=torch.float32,
    )
    if getattr(model.config, "problem_type", None) in (None, ""):
        model.config.problem_type = "single_label_classification"
    return model


def head_module_names(model) -> list[str]:
    found = []
    param_names = [name for name, _param in model.named_parameters()]
    for candidate in ("classifier", "pooler"):
        if any(name == candidate or name.startswith(candidate + ".") for name in param_names):
            found.append(candidate)
    if "classifier" not in found:
        raise SystemExit("Could not find a classifier head on this model.")
    return found


def leaf_regex(names: list[str]) -> str:
    inner = "|".join(re.escape(name) for name in names)
    return rf"(^|.*\.)({inner})$"


def attach_lora(model, cfg: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    heads = head_module_names(model)
    pattern = leaf_regex(list(cfg["target_modules"]))
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=pattern,
        bias="none",
        modules_to_save=list(heads),
    )
    model = get_peft_model(model, lora_config)
    leaves = lora_leaf_names(model)
    expected = set(cfg["target_modules"])
    if set(leaves) != expected:
        raise SystemExit(
            "LoRA target modules did not match. Expected "
            + ", ".join(sorted(expected))
            + ". Found "
            + ", ".join(sorted(set(leaves)))
            + "."
        )
    return model, heads


def lora_leaf_names(model) -> list[str]:
    leaves = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            leaves.append(name.split(".")[-1])
    return leaves


def freeze_backbone(model) -> list[str]:
    heads = head_module_names(model)
    for name, param in model.named_parameters():
        param.requires_grad = any(name == head or name.startswith(head + ".") for head in heads)
    return heads


def parameter_counts(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def estimate_warmup_steps(num_examples: int, batch_size: int, grad_accum: int, epochs: int, warmup_ratio: float) -> int:
    import math

    steps_per_epoch = max(1, math.ceil(num_examples / max(1, batch_size)))
    updates_per_epoch = max(1, math.ceil(steps_per_epoch / max(1, grad_accum)))
    total_updates = updates_per_epoch * max(1, epochs)
    return int(round(total_updates * float(warmup_ratio)))


def build_training_arguments(cfg: dict, output_dir: Path, epochs: int, learning_rate: float, num_examples: int):
    from transformers import TrainingArguments

    use_fp16 = bool(cfg["fp16"]) and torch.cuda.is_available()
    if cfg["fp16"] and not torch.cuda.is_available():
        print("CUDA is not available, so fp16 is off for this run.", flush=True)
    warmup_steps = estimate_warmup_steps(
        num_examples,
        int(cfg["batch_size"]),
        int(cfg["grad_accum"]),
        epochs,
        float(cfg["warmup_ratio"]),
    )
    print(f"Warmup steps: {warmup_steps}", flush=True)
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": int(cfg["batch_size"]),
        "per_device_eval_batch_size": int(cfg["eval_batch_size"]),
        "gradient_accumulation_steps": int(cfg["grad_accum"]),
        "learning_rate": float(learning_rate),
        "weight_decay": float(cfg["weight_decay"]),
        "warmup_steps": warmup_steps,
        "fp16": use_fp16,
        "max_grad_norm": 1.0,
        "eval_strategy": "epoch",
        "save_strategy": "no",
        "logging_steps": int(cfg["logging_steps"]),
        "seed": int(cfg["seed"]),
        "data_seed": int(cfg["seed"]),
        "report_to": "none",
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": bool(torch.cuda.is_available()),
        "remove_unused_columns": True,
        "optim": "adamw_torch",
        "lr_scheduler_type": "linear",
        "load_best_model_at_end": False,
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    skipped = [key for key in list(kwargs) if key not in params]
    for key in skipped:
        kwargs.pop(key)
    if skipped:
        print("Skipping unsupported trainer options: " + ", ".join(skipped), flush=True)
    return TrainingArguments(**kwargs)


def make_trainer(model, args, train_ds, val_ds, tokenizer, callbacks):
    from transformers import DataCollatorWithPadding, Trainer

    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


class BestAdapterCallback:
    """Saves the adapter when validation macro F1 improves."""

    def __init__(self, adapter_dir: Path, meta: dict):
        self.adapter_dir = Path(adapter_dir)
        self.meta = meta
        self.best = None
        self.saved = False

    def bind(self, model):
        from transformers import TrainerCallback

        outer = self

        class _Callback(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if not metrics:
                    return
                score = metrics.get("eval_macro_f1", metrics.get("macro_f1"))
                if score is None:
                    print("Eval metrics had no macro F1: " + ", ".join(sorted(metrics)), flush=True)
                    return
                print(f"Validation macro F1: {float(score):.4f}", flush=True)
                if score != score:
                    print("Validation macro F1 is not finite. Skipping the save.", flush=True)
                    return
                if not weights_are_finite(model):
                    print("Weights are not finite. Skipping the save.", flush=True)
                    return
                if outer.best is None or float(score) >= outer.best:
                    outer.best = float(score)
                    outer.meta["best_val_macro_f1"] = outer.best
                    save_adapter(model, outer.adapter_dir, outer.meta)
                    outer.saved = True

        return _Callback()


def weights_are_finite(model) -> bool:
    return all(bool(torch.isfinite(param).all().item()) for param in model.parameters())


def write_adapter_note(adapter_dir: Path) -> None:
    from cesc.prose import assert_no_dash

    note = (
        "# Adapter folder\n\n"
        "train_lora.py writes the PEFT adapter here.\n\n"
        "adapter_config.json describes the LoRA setup.\n"
        "adapter_model.safetensors holds the adapter weights.\n"
        "label_map.json maps ids to the three labels.\n"
        "train_meta.json records the run that produced the weights, including the base model id.\n\n"
        "If train_meta.json says the run name is smoke, the weights only prove that training finished. "
        "Run the full train config to replace them.\n"
    )
    assert_no_dash(note, "adapter note")
    (Path(adapter_dir) / "README.md").write_text(note, encoding="utf-8")


def save_adapter(model, adapter_dir: Path, meta: dict) -> None:
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    label2id, id2label = label_maps()
    payload = {"id2label": {str(index): name for index, name in id2label.items()}, "label2id": label2id}
    (adapter_dir / "label_map.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (adapter_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    write_adapter_note(adapter_dir)


def load_adapter(adapter_dir: Path):
    from peft import PeftModel

    adapter_dir = Path(adapter_dir)
    meta = json.loads((adapter_dir / "train_meta.json").read_text(encoding="utf-8"))
    label_payload = json.loads((adapter_dir / "label_map.json").read_text(encoding="utf-8"))
    id2label = {int(key): value for key, value in label_payload["id2label"].items()}
    label2id = {value: key for key, value in id2label.items()}
    from transformers import AutoModelForSequenceClassification

    base = AutoModelForSequenceClassification.from_pretrained(
        meta["model_name"],
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, meta, label2id


def predict_rows(model, tokenizer, rows: list[dict], max_length: int, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions = []
    probabilities = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["claim"] for row in batch],
            [row["evidence"] for row in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            probs = softmax(logits, dim=-1)
        predictions.extend(probs.argmax(dim=-1).cpu().tolist())
        probabilities.extend(probs.cpu().tolist())
    return predictions, probabilities
