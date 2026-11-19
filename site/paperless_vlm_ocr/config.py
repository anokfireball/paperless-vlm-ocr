"""Runtime configuration, read from the environment at call time."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Literal

_PREFIX = "PAPERLESS_VLM_OCR_"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(_PREFIX + name, default).strip()


def _env_bool(name: str) -> bool:
    return _env(name).lower() in ("true", "1")


def _default_decision_log() -> Path:
    from django.conf import settings

    return Path(settings.DATA_DIR) / "vlm-ocr" / "decisions.jsonl"


@dataclass(frozen=True)
class VlmOcrConfig:
    enabled: bool = False
    endpoint: str = ""
    api_key: str = field(default="", repr=False)
    model: str = ""
    gate_mode: Literal["shadow", "enforce"] = "shadow"
    threshold: float = 0.9
    min_tokens: int = 15
    timeout_s: float = 120.0
    doc_budget_s: float = 900.0
    concurrency: int = 4
    max_pages: int = 50
    image_px: int = 1800
    decision_log: Path = Path("decisions.jsonl")

    @classmethod
    def from_env(cls) -> VlmOcrConfig:
        mode = _env("GATE_MODE", "shadow").lower()
        log = _env("DECISION_LOG")
        return cls(
            enabled=_env_bool("ENABLED"),
            endpoint=_env("ENDPOINT"),
            api_key=_env("API_KEY"),
            model=_env("MODEL"),
            gate_mode="enforce" if mode == "enforce" else "shadow",
            threshold=float(_env("GATE_THRESHOLD", "0.9")),
            min_tokens=int(_env("GATE_MIN_TOKENS", "15")),
            timeout_s=float(_env("TIMEOUT_S", "120")),
            doc_budget_s=float(_env("DOC_BUDGET_S", "900")),
            concurrency=int(_env("CONCURRENCY", "4")),
            max_pages=int(_env("MAX_PAGES", "50")),
            image_px=int(_env("IMAGE_PX", "1800")),
            decision_log=Path(log) if log else _default_decision_log(),
        )
