"""Read jsonl rows and build tokenized datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path

from cesc.labels import LABELS


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def limit_rows(rows: list[dict], max_samples, seed: int) -> list[dict]:
    if max_samples is None or max_samples >= len(rows):
        return rows
    by_label = {label: [] for label in LABELS}
    for index, row in enumerate(rows):
        by_label[row["label"]].append(index)
    rng = random.Random(seed)
    base = max_samples // len(LABELS)
    extra = max_samples % len(LABELS)
    chosen = []
    for offset, label in enumerate(LABELS):
        indices = by_label[label][:]
        rng.shuffle(indices)
        take = base + (1 if offset < extra else 0)
        if take > len(indices):
            raise SystemExit(f"Not enough rows labeled {label} to build the subset")
        chosen.extend(indices[:take])
    rng.shuffle(chosen)
    return [rows[index] for index in chosen]


def tokenize_rows(tokenizer, rows: list[dict], label2id: dict, max_length: int):
    from datasets import Dataset

    dataset = Dataset.from_list(rows)

    def preprocess(batch):
        encoded = tokenizer(
            batch["claim"],
            batch["evidence"],
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = [label2id[label] for label in batch["label"]]
        return encoded

    keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    extra = [name for name in tokenized.column_names if name not in keep]
    if extra:
        tokenized = tokenized.remove_columns(extra)
    return tokenized
