"""Silence the model-loading progress bars, at the source.

`Loading weights: 100%|…` is tqdm from sentence-transformers, printed to stderr
while the embedder or the reranker is constructed. It overwrites a line in a chat
that is otherwise plain text, and in a non-tty it is pure noise — every run of a
script produced two of these lines regardless of anything the caller asked for.

Not suppressed by setting an environment variable: that has to happen before the
transformers libraries read it, and the import happens deep inside the model
constructor. Doing it at import time here covers both loaders.
"""
from __future__ import annotations

import os


def quiet_loading() -> None:
    """Turn off download/weight progress bars. Idempotent and cheap."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # noqa: BLE001 - purely cosmetic, older/absent hub
        pass
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:  # noqa: BLE001 - purely cosmetic
        pass
