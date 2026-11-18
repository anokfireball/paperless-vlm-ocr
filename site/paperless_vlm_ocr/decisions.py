"""Append-only JSONL decision log; never raises."""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("paperless.parsing.vlm_ocr")


def append(path: Path, record: dict) -> None:
    """Append one JSON line to ``path``, warning instead of failing."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        log.warning("vlm_ocr: cannot write decision log %s", path, exc_info=True)
