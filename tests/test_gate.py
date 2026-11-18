"""Unit tests for the text-layer gate."""

from __future__ import annotations

from pathlib import Path

from paperless_vlm_ocr.config import VlmOcrConfig
from paperless_vlm_ocr.gate import agreement
from paperless_vlm_ocr.gate import classify
from paperless_vlm_ocr.gate import passes
from paperless_vlm_ocr.gate import tokenize


def _cfg(**kwargs) -> VlmOcrConfig:
    kwargs.setdefault("decision_log", Path("/dev/null"))
    return VlmOcrConfig(**kwargs)


def test_tokenize_joins_letter_spaced_runs() -> None:
    assert tokenize("K a r l  K o l l e r") == ["karl", "koller"]


def test_tokenize_keeps_split_words_split() -> None:
    assert tokenize("Ko nto inha ber") == ["ko", "nto", "inha", "ber"]


def test_agreement_identical() -> None:
    text = tokenize("Kontoinhaber Fabian Karl Koller")
    assert agreement(text, text) == (1.0, 1.0)


def test_agreement_broken_layer() -> None:
    layer = tokenize("Kontoinhaber Fabian Karl Koller")
    ref = tokenize("Ko nto inha ber Fa bia n Ka rl Ko ller")
    coverage, support = agreement(layer, ref)
    assert coverage < 0.3
    assert support < 0.3


def test_agreement_empty_layer_is_zero() -> None:
    assert agreement([], tokenize("Kontoinhaber")) == (0.0, 0.0)


def test_classify_shadow_reports_would_vlm() -> None:
    layer = " ".join(f"layer{i}" for i in range(20))
    ref = " ".join(f"ref{i}" for i in range(20))
    verdict = classify(layer, ref, _cfg(gate_mode="shadow"))
    assert verdict.kind == "gated"
    assert verdict.decision == "would_vlm"
    assert verdict.layer_tokens == 20
    assert verdict.ref_tokens == 20


def test_classify_enforce_routes_to_vlm() -> None:
    layer = " ".join(f"layer{i}" for i in range(20))
    ref = " ".join(f"ref{i}" for i in range(20))
    verdict = classify(layer, ref, _cfg(gate_mode="enforce"))
    assert verdict.kind == "gated"
    assert verdict.decision == "vlm"


def test_classify_keeps_agreeing_layer() -> None:
    layer = " ".join(f"wort{i}" for i in range(20))
    verdict = classify(layer, layer, _cfg(gate_mode="enforce"))
    assert verdict.decision == "keep"


def test_classify_short_reference_has_no_signal() -> None:
    layer = " ".join(f"wort{i}" for i in range(20))
    ref = "eins zwei drei vier fuenf"
    verdict = classify(layer, ref, _cfg(gate_mode="enforce"))
    assert verdict.kind == "gated"
    assert verdict.decision == "no_signal"
    assert verdict.coverage is None
    assert verdict.ref_tokens == 5


def test_classify_short_layer_is_empty() -> None:
    verdict = classify("eins zwei drei", None, _cfg(gate_mode="enforce"))
    assert verdict.kind == "empty_layer"
    assert verdict.decision == "vlm"
    assert verdict.layer_tokens == 3
    assert verdict.ref_tokens is None
    assert verdict.coverage is None


def test_passes_keeps_layer_tesseract_under_reads() -> None:
    # light-grey footer Tesseract misses: full coverage, low support (doc 2700)
    assert passes(0.989, 0.472, 0.9)


def test_passes_rejects_split_words_with_high_coverage() -> None:
    # partially split words (docs 1880 p4, 694): coverage below threshold
    assert not passes(0.875, 0.768, 0.9)
    assert not passes(0.822, 0.807, 0.9)


def test_passes_confirmed_layer_near_threshold() -> None:
    # coverage slightly low but every layer word confirmed (doc 1028 p2)
    assert passes(0.847, 0.908, 0.9)
    assert not passes(0.79, 0.99, 0.9)
