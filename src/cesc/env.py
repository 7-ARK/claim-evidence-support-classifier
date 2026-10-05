"""Process settings that should be in place before model libraries load."""

import os

from cesc.paths import ROOT


def prepare_runtime() -> None:
    cache = ROOT / ".cache"
    os.environ.setdefault("HF_HOME", str(cache / "hf"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
