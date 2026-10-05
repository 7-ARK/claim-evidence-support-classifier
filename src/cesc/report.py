"""Text reports and the error analysis writeup."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from cesc.labels import LABELS
from cesc.prose import assert_no_dash

REASONS = {
    ("Supported", "Partially Supported"): (
        "The evidence covers the claim, but the model called it only partial."
    ),
    ("Supported", "Not Evidenced"): (
        "The evidence covers the claim, but the model found no support."
    ),
    ("Partially Supported", "Supported"): (
        "The evidence covers only part of the claim, or it is weaker than a full match, "
        "but the model called it full support."
    ),
    ("Partially Supported", "Not Evidenced"): (
        "The evidence is related but incomplete. The model treated that weak overlap as no support."
    ),
    ("Not Evidenced", "Supported"): (
        "The evidence does not support the claim, but the model called it full support. "
        "A related tool or a shared phrase can pull the score that way."
    ),
    ("Not Evidenced", "Partially Supported"): (
        "The evidence does not support the claim, but the model saw a weak relation and called it partial."
    ),
}


def fmt_score(value) -> str:
    if value is None:
        return "missing"
    return f"{float(value):.4f}"


def _score_row(name: str, scores: dict | None) -> list[str]:
    if not scores:
        return [name, "missing", "missing", "missing", "missing"]
    per = scores["per_class"]
    return [name, fmt_score(scores["macro_f1"])] + [fmt_score(per[label]["f1"]) for label in LABELS]


def comparison_block(majority, frozen, lora) -> str:
    headers = ["Method", "Macro F1", "Supported F1", "Partially Supported F1", "Not Evidenced F1"]
    rows = [
        _score_row("Majority class", majority),
        _score_row("Frozen head", frozen),
        _score_row("LoRA", lora),
    ]
    widths = [len(header) for header in headers]
    for values in rows:
        for index, value in enumerate(values):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    lines = ["Comparison on the test set", format_row(headers)]
    lines.extend(format_row(values) for values in rows)
    if majority is None or frozen is None:
        lines.append(
            "Baseline numbers are missing. Run baseline.py with the same config before reading this as a comparison."
        )
    text = "\n".join(lines)
    assert_no_dash(text, "comparison")
    return text


def render_report(payload: dict) -> str:
    scores = payload["scores"]
    lines = [
        f"Run name: {payload['run_name']}",
        "Method: LoRA",
        f"Test rows: {payload['num_test']}",
        f"Train rows used: {payload['train_rows_used']}",
        f"Epochs: {payload['epochs']}",
        f"Max length: {payload['max_length']}",
        f"Libraries: torch {payload['torch_version']}, transformers {payload['transformers_version']}, peft {payload['peft_version']}",
        "",
        f"Macro F1: {fmt_score(scores['macro_f1'])}",
        f"Accuracy: {fmt_score(scores['accuracy'])}",
    ]
    for label in LABELS:
        per = scores["per_class"][label]
        lines.append(
            f"{label} F1: {fmt_score(per['f1'])}  precision: {fmt_score(per['precision'])}  recall: {fmt_score(per['recall'])}  support: {per['support']}"
        )
    lines.append("")
    lines.append(
        "Confusion matrix rows are true labels. Columns are predicted labels in this order: "
        + ", ".join(LABELS)
        + "."
    )
    matrix = scores["confusion_matrix"]
    for index, label in enumerate(LABELS):
        nums = "  ".join(str(int(value)) for value in matrix[index])
        lines.append(f"{label}: {nums}")
    lines.append("")
    lines.append(comparison_block(payload.get("majority"), payload.get("frozen"), scores))
    if payload.get("majority_note"):
        lines.append(payload["majority_note"])
    if payload.get("frozen_note"):
        lines.append(payload["frozen_note"])
    lines.append("")
    if payload.get("smoke"):
        lines.append(
            "This is a smoke run. It checks that training and scoring finish. Do not read the scores as the quality of a full train."
        )
    else:
        lines.append(
            "The test split was not used to pick the adapter. Validation macro F1 after each epoch decides which adapter is kept."
        )
    text = "\n".join(lines) + "\n"
    assert_no_dash(text, "report")
    return text


def pick_errors(records: list[dict], limit: int = 9) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        if record["true_label"] != record["predicted_label"]:
            groups[(record["true_label"], record["predicted_label"])].append(record)
    picked = []
    keys = sorted(groups)
    while len(picked) < limit and any(groups[key] for key in keys):
        for key in keys:
            if groups[key] and len(picked) < limit:
                picked.append(groups[key].pop(0))
    return picked


def render_error_analysis(payload: dict, records: list[dict]) -> str:
    wrong = [record for record in records if record["true_label"] != record["predicted_label"]]
    picked = pick_errors(records)
    counts = defaultdict(int)
    for record in wrong:
        counts[(record["true_label"], record["predicted_label"])] += 1
    lines = [
        "# Error analysis",
        "",
        f"Run name: {payload['run_name']}",
        f"Test rows: {payload['num_test']}",
        f"Missed rows: {len(wrong)}",
        "",
    ]
    predicted_labels = {record["predicted_label"] for record in records}
    collapsed_label = next(iter(predicted_labels)) if len(predicted_labels) == 1 else None
    if payload.get("smoke"):
        epoch_word = "epoch" if str(payload.get("epochs")) == "1" else "epochs"
        lines.append(
            "This file quotes test rows the smoke run got wrong. "
            f"The smoke run uses {payload.get('train_rows_used')} training rows and {payload.get('epochs')} {epoch_word}. "
            "Read the misses as pipeline output, not as a study of a converged model."
        )
    else:
        lines.append(
            "This file quotes test rows the saved adapter got wrong. The quotes are copied from the test file."
        )
    if collapsed_label is not None:
        lines.append("")
        lines.append(
            f"Every test row was predicted as {collapsed_label}. "
            "The probabilities sit near one third, so the head has not separated the classes. "
            "The quotes below are misses from that collapse."
        )
    lines.append("")
    lines.append("Misses by true label and predicted label:")
    if not counts:
        lines.append("The model missed nothing on this test run.")
    for true_label in LABELS:
        for pred_label in LABELS:
            if true_label == pred_label:
                continue
            key = (true_label, pred_label)
            if key in counts:
                lines.append(f"{true_label} predicted as {pred_label}: {counts[key]}")
    lines.append("")
    if not picked:
        lines.append("There are no failure examples to quote.")
    for index, record in enumerate(picked, start=1):
        pair = (record["true_label"], record["predicted_label"])
        if collapsed_label is not None:
            reason = (
                f"This row is one miss from a model that predicted {collapsed_label} for every test row. "
                "The probability is only a little above one third, which fits a head that has not learned the labels."
            )
        else:
            reason = REASONS.get(pair, "The predicted label does not match the assigned label.")
        lines.extend(
            [
                f"## Miss {index}",
                "",
                f"True label: {record['true_label']}",
                f"Predicted label: {record['predicted_label']}",
                f"Predicted probability: {record['predicted_probability']:.4f}",
                f"Claim: {record['claim']}",
                f"Evidence: {record['evidence']}",
                "",
                reason,
                "",
            ]
        )
    lines.append("## How to read the misses")
    lines.append("")
    lines.append(
        "Supported versus Partially Supported is the finer cut: the tool may match while the years, the second requirement, or the setting does not. Not Evidenced includes near misses such as PostgreSQL versus MySQL, plus rows where the claimed tool is named and then rejected."
    )
    lines.append("")
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    assert_no_dash(text, "error analysis")
    return text


def write_confusion_csv(path: Path, matrix, labels) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        header = ["true_label"] + [f"pred_{name}" for name in labels]
        handle.write(",".join(header) + "\n")
        for index, name in enumerate(labels):
            values = ",".join(str(int(value)) for value in matrix[index])
            handle.write(f"{name},{values}\n")


def write_confusion_png(path: Path, matrix, labels) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.2, 6.2))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_yticklabels(labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion matrix")
    for row_index in range(len(labels)):
        for col_index in range(len(labels)):
            axis.text(col_index, row_index, str(int(matrix[row_index][col_index])), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
