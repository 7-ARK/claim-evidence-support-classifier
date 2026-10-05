"""Scores for the three way classifier."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from cesc.labels import LABELS


def classification_scores(y_true, y_pred) -> dict:
    label_ids = list(range(len(LABELS)))
    macro = f1_score(y_true, y_pred, average="macro", labels=label_ids, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, labels=label_ids, zero_division=0)
    precision, recall, _f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    per_class = {}
    for index, name in enumerate(LABELS):
        per_class[name] = {
            "f1": float(per_f1[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "support": int(support[index]),
        }
    return {
        "macro_f1": float(macro),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
        "labels": list(LABELS),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(np.asarray(logits), axis=-1)
    labels = np.asarray(labels)
    scores = classification_scores(labels, preds)
    return {"macro_f1": scores["macro_f1"], "accuracy": scores["accuracy"]}
