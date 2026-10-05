"""Label names and id maps.

The order here is the order used by the model head, the metrics, and the reports.
"""

LABELS = ["Supported", "Partially Supported", "Not Evidenced"]


def label_maps():
    label2id = {name: i for i, name in enumerate(LABELS)}
    id2label = {i: name for name, i in label2id.items()}
    return label2id, id2label
