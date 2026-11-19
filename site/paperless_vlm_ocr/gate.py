"""Text-layer gate: does a born-digital layer agree with a Tesseract read?"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

if TYPE_CHECKING:
    from paperless_vlm_ocr.config import VlmOcrConfig

_WORD = re.compile(r"[^\W_]+")

# A layer the reference read barely confirms is fabricated noise, however much
# of the reference it happens to contain by sheer volume.
MIN_SUPPORT = 0.2


def tokenize(text: str) -> list[str]:
    """Casefolded alphanumeric tokens.

    Runs of three or more single-character tokens separated by exactly one
    character are joined (letter-spaced text such as "K a r l" becomes
    "karl"); a wider gap ends the run. All other boundaries are kept, so
    wrongly split words ("Ko nto inha ber") stay split.
    """
    norm = unicodedata.normalize("NFKC", text).casefold()
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= 3:
            out.append("".join(run))
        else:
            out.extend(run)
        run.clear()

    prev_end = -1
    for m in _WORD.finditer(norm):
        tok = m.group()
        if len(tok) == 1:
            if run and m.start() - prev_end != 1:
                flush()
            run.append(tok)
        else:
            flush()
            out.append(tok)
        prev_end = m.end()
    flush()
    return out


def agreement(layer: list[str], ref: list[str]) -> tuple[float, float]:
    """Return (coverage, support): shared tokens over ref, and over layer."""
    common = sum((Counter(layer) & Counter(ref)).values())
    coverage = common / len(ref) if ref else 0.0
    support = common / len(layer) if layer else 0.0
    return coverage, support


def passes(coverage: float, support: float, threshold: float) -> bool:
    """Keep a layer that the reference read confirms and that covers it.

    A support floor rejects a huge, mostly fabricated layer that covers the
    reference only by volume. Above the floor, coverage is the primary signal:
    split, letter-spaced, garbled or missing layer text lowers it; low support
    alone mostly means Tesseract missed light or small print, so support only
    counts within 0.1 below threshold.
    """
    return support >= MIN_SUPPORT and (
        coverage >= threshold
        or (coverage >= threshold - 0.1 and support >= threshold)
    )


@dataclass
class PageVerdict:
    kind: Literal["empty_layer", "gated"]
    decision: Literal["keep", "vlm", "would_vlm", "no_signal"]
    coverage: float | None
    support: float | None
    layer_tokens: int
    ref_tokens: int | None


def classify(layer_text: str, ref_text: str | None, cfg: VlmOcrConfig) -> PageVerdict:
    """Decide what to do with one page. ``ref_text`` None means not computed."""
    layer = tokenize(layer_text)
    if len(layer) < cfg.min_tokens:
        return PageVerdict("empty_layer", "vlm", None, None, len(layer), None)
    ref = tokenize(ref_text or "")
    if len(ref) < cfg.min_tokens:
        return PageVerdict("gated", "no_signal", None, None, len(layer), len(ref))
    coverage, support = agreement(layer, ref)
    if passes(coverage, support, cfg.threshold):
        decision = "keep"
    elif cfg.gate_mode == "enforce":
        decision = "vlm"
    else:
        decision = "would_vlm"
    return PageVerdict("gated", decision, coverage, support, len(layer), len(ref))
