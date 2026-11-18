"""Validation helpers of the VLM output: clean and reject_reason."""

from __future__ import annotations

from paperless_vlm_ocr.vlm import VlmResult
from paperless_vlm_ocr.vlm import clean
from paperless_vlm_ocr.vlm import reject_reason

FALLBACK = (
    "Kontoinhaber Fabian Karl Koller\n"
    "IBAN DE89 3704 0044 0532 0130 00\n"
    "Buchungstag 01.11.2026\n"
    "Verwendungszweck Miete November\n"
    "Betrag -1.250,00 EUR\n"
    "Neuer Kontostand 4.321,00 EUR\n"
)


def result(text: str, finish_reason: str | None = "stop") -> VlmResult:
    return VlmResult(
        text=text,
        finish_reason=finish_reason,
        input_tokens=0,
        output_tokens=0,
        seconds=0.0,
    )


def test_reject_truncated() -> None:
    assert reject_reason(result(FALLBACK, finish_reason="length"), FALLBACK) == "truncated"


def test_reject_empty() -> None:
    assert reject_reason(result("   \n  ", finish_reason="stop"), FALLBACK) == "empty"


def test_reject_repetition() -> None:
    text = "\n".join(["Betrag -1.250,00 EUR"] * 8)
    assert reject_reason(result(text), FALLBACK) == "repetition"


def test_reject_runaway() -> None:
    fallback = "Buchungstag 01.11.2026 Verwendungszweck Miete " * 10
    text = "x" * (4 * len(fallback) + 2001)
    assert reject_reason(result(text), fallback) == "runaway"


def test_reject_none_for_normal_transcription() -> None:
    text = (
        "Kontoinhaber Fabian Karl Koller\n"
        "IBAN DE89 3704 0044 0532 0130 00\n"
        "Buchungstag 01.11.2026\n"
        "Verwendungszweck Miete November\n"
        "Betrag -1.250,00 EUR\n"
        "Neuer Kontostand 4.321,00 EUR\n"
    )
    assert len(text) > 20
    assert reject_reason(result(text), FALLBACK) is None


def test_clean_strips_text_fence() -> None:
    raw = "```text\nKontoinhaber Fabian Karl Koller\nIBAN DE89 3704 0044 0532 0130 00\n```"
    assert clean(raw) == "Kontoinhaber Fabian Karl Koller\nIBAN DE89 3704 0044 0532 0130 00"
