"""Build train, validation, and test jsonl files.

The generator is seeded. The split is stratified by label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cesc.generate import generate_rows, public_row, stratified_split
from cesc.labels import LABELS
from cesc.paths import ROOT
from cesc.prose import assert_no_dash


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = public_row(row)
            assert_no_dash(payload["claim"], "claim")
            assert_no_dash(payload["evidence"], "evidence")
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def label_counts(rows: list[dict]) -> dict:
    counts = Counter(row["label"] for row in rows)
    return {label: int(counts[label]) for label in LABELS}


def pattern_counts(rows: list[dict]) -> dict:
    counts = Counter(row["pattern"] for row in rows)
    return dict(sorted(counts.items()))


def write_preview(path: Path, rows: list[dict]) -> None:
    seen = set()
    blocks = []
    for row in rows:
        if row["pattern"] in seen:
            continue
        seen.add(row["pattern"])
        blocks.append(
            "\n".join(
                [
                    f"Pattern: {row['pattern']}",
                    f"Label: {row['label']}",
                    f"Claim: {row['claim']}",
                    f"Evidence: {row['evidence']}",
                ]
            )
        )
    text = "\n\n".join(blocks) + "\n"
    assert_no_dash(text, "preview")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the claim and evidence splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--total", type=int, default=1200, help="Number of examples to build.")
    parser.add_argument("--output-dir", default="data", help="Folder for the jsonl files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = generate_rows(seed=args.seed, total=args.total)
    train, val, test = stratified_split(rows, seed=args.seed)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)
    write_jsonl(output_dir / "test.jsonl", test)
    write_preview(output_dir / "preview.txt", rows)

    by_claim = defaultdict(set)
    for row in rows:
        by_claim[row["claim"]].add(row["label"])
    multi = sum(1 for labels in by_claim.values() if len(labels) > 1)

    stats = {
        "seed": args.seed,
        "total": len(rows),
        "labels": label_counts(rows),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "split_labels": {
            "train": label_counts(train),
            "val": label_counts(val),
            "test": label_counts(test),
        },
        "patterns": pattern_counts(rows),
        "split_patterns": {
            "train": pattern_counts(train),
            "val": pattern_counts(val),
            "test": pattern_counts(test),
        },
        "unique_claims": len(by_claim),
        "claims_with_more_than_one_label": multi,
        "sha256": {
            "train": sha256(output_dir / "train.jsonl"),
            "val": sha256(output_dir / "val.jsonl"),
            "test": sha256(output_dir / "test.jsonl"),
        },
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    for split_name, split_rows in ("train", train), ("val", val), ("test", test):
        counts = label_counts(split_rows)
        if len(set(counts.values())) != 1:
            raise SystemExit(f"{split_name} is not label balanced: {counts}")
    keys = [(row["claim"], row["evidence"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("Duplicate claim and evidence pairs were written.")
    print(json.dumps({"total": stats["total"], "splits": stats["splits"], "labels": stats["labels"]}, indent=2))


if __name__ == "__main__":
    main()
