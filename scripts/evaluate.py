"""Score a saved LoRA adapter on the test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cesc.env import prepare_runtime

prepare_runtime()

import peft
import torch
import transformers

from cesc.config import load_config, resolve_path
from cesc.data import read_jsonl
from cesc.labels import LABELS
from cesc.metrics import classification_scores
from cesc.modeling import load_adapter, predict_rows
from cesc.report import render_error_analysis, render_report, write_confusion_csv, write_confusion_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved LoRA adapter.")
    parser.add_argument("--config", required=True, help="Path to the yaml config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    adapter_dir = resolve_path(cfg["adapter_dir"])
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit("No adapter found. Run train_lora.py with this config first.")

    model, meta, _label2id = load_adapter(adapter_dir)
    test_rows = read_jsonl(resolve_path(cfg["test_file"]))
    max_length = int(meta.get("max_length", cfg["max_length"]))
    pred_ids, probs = predict_rows(model, load_tokenizer_for_eval(meta, max_length), test_rows, max_length, int(cfg["eval_batch_size"]))
    true_ids = [LABELS.index(row["label"]) for row in test_rows]
    scores = classification_scores(true_ids, pred_ids)

    output_dir = resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "baseline_metrics.json"
    majority = None
    frozen = None
    majority_note = ""
    frozen_note = ""
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        majority = baseline.get("majority")
        frozen = baseline.get("frozen_head")
        majority_note = baseline.get("majority_note", "")
        frozen_note = baseline.get("frozen_note", "")

    smoke = cfg["run_name"] == "smoke" or meta.get("max_train_samples") not in (None, "")
    payload = {
        "run_name": cfg["run_name"],
        "scores": scores,
        "num_test": len(test_rows),
        "train_rows_used": meta.get("train_rows_used"),
        "epochs": meta.get("epochs"),
        "max_length": max_length,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "majority": majority,
        "frozen": frozen,
        "majority_note": majority_note,
        "frozen_note": frozen_note,
        "smoke": smoke,
        "best_val_macro_f1": meta.get("best_val_macro_f1"),
    }
    report = render_report(payload)
    records = []
    for row, pred_id, prob in zip(test_rows, pred_ids, probs):
        records.append(
            {
                "claim": row["claim"],
                "evidence": row["evidence"],
                "true_label": row["label"],
                "predicted_label": LABELS[pred_id],
                "predicted_probability": float(prob[pred_id]),
            }
        )
    analysis = render_error_analysis(payload, records)
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    (output_dir / "error_analysis.md").write_text(analysis, encoding="utf-8")
    (output_dir / "errors.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    write_confusion_csv(output_dir / "confusion_matrix.csv", scores["confusion_matrix"], LABELS)
    try:
        write_confusion_png(output_dir / "confusion_matrix.png", scores["confusion_matrix"], LABELS)
    except Exception:
        print("Could not write the confusion matrix image.", flush=True)
    print(report, flush=True)


def load_tokenizer_for_eval(meta, max_length):
    from cesc.modeling import load_tokenizer

    return load_tokenizer(meta["model_name"], max_length)


if __name__ == "__main__":
    main()
