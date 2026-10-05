"""Score one claim and evidence pair with a saved LoRA adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cesc.env import prepare_runtime

prepare_runtime()

from cesc.labels import LABELS
from cesc.modeling import load_adapter, load_tokenizer, predict_rows
from cesc.paths import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a label for one claim and one evidence snippet.")
    parser.add_argument("--adapter", default="adapters/lora", help="Folder with the saved LoRA adapter.")
    parser.add_argument("--claim", required=True, help="The job requirement claim.")
    parser.add_argument("--evidence", required=True, help="The evidence snippet.")
    args = parser.parse_args()
    if not args.claim.strip() or not args.evidence.strip():
        raise SystemExit("Claim and evidence must both be non empty.")

    adapter_dir = Path(args.adapter)
    if not adapter_dir.is_absolute():
        adapter_dir = ROOT / adapter_dir
    if not (adapter_dir / "train_meta.json").exists():
        raise SystemExit("No adapter metadata found. Train an adapter before inference.")

    model, meta, _label2id = load_adapter(adapter_dir)
    max_length = int(meta["max_length"])
    tokenizer = load_tokenizer(meta["model_name"], max_length)
    rows = [{"claim": args.claim.strip(), "evidence": args.evidence.strip(), "label": LABELS[0]}]
    pred_ids, probs = predict_rows(model, tokenizer, rows, max_length, batch_size=1)
    pred = LABELS[pred_ids[0]]
    print(f"Run name recorded with the adapter: {meta.get('run_name')}")
    print(f"Prediction: {pred}")
    for index, label in enumerate(LABELS):
        print(f"{label}: {probs[0][index]:.4f}")


if __name__ == "__main__":
    main()
