"""Validation of VLM output: loop detection, acceptance rules and cleanup."""

from __future__ import annotations

from paperless_vlm_ocr.vlm import VlmResult
from paperless_vlm_ocr.vlm import clean
from paperless_vlm_ocr.vlm import combine_results
from paperless_vlm_ocr.vlm import evaluate
from paperless_vlm_ocr.vlm import find_loop
from paperless_vlm_ocr.vlm import reject_reason


def result(text: str, finish_reason: str | None = "stop") -> VlmResult:
    return VlmResult(
        text=text,
        finish_reason=finish_reason,
        input_tokens=0,
        output_tokens=0,
        seconds=0.0,
    )


def plain(length: int) -> str:
    """Non-periodic filler text of exactly ``length`` characters."""
    text = " ".join(f"wort{index}" for index in range(length))[:length]
    if text.endswith(" "):
        text = text[:-1] + "x"
    return text


FALLBACK = plain(300)


# ----------------------------------------------------------------------
# Loop detector: real degenerate shapes, at the measured offsets
# ----------------------------------------------------------------------


def test_find_loop_rule_inside_one_line() -> None:
    head = "Rechnung Bäckerei\nSumme 4,20 EUR\n"
    assert find_loop(head + "- " * 200) == len(head)


def test_find_loop_unit_1_1_1() -> None:
    head = "Prozess: "
    assert find_loop(head + "1:" * 200) == len(head)


def test_find_loop_unit_5p0b2b() -> None:
    head = "Prozess-Typ: "
    assert find_loop(head + "5p0b2b" * 50) == len(head)


def test_find_loop_unit_qds() -> None:
    head = "Projekt: C:"
    assert find_loop(head + "\\qds\\" * 60) == len(head)


def test_find_loop_repeated_base64_lines() -> None:
    line = "M2YwNjM2M2YwNjM2M2YwNjM2M2YwNjM2M2YwNjM2\n"
    head = "TSE Prüfwert:"
    text = head + "\n" + line * 10
    # the newline that ends the head is the first character of the run
    assert find_loop(text) == len(head)


def test_find_loop_repeated_table_rows() -> None:
    row = "| 01.11.2026 | Miete November | -1.250,00 EUR |\n"
    head = "Kontoauszug"
    assert find_loop(head + "\n" + row * 12) == len(head)


def test_find_loop_needs_two_hundred_characters() -> None:
    assert find_loop("x" * 199) is None
    assert find_loop("x" * 200) == 0


def test_find_loop_ignores_a_few_repeats() -> None:
    # 7 identical 40-character lines are a table, not a loop
    assert find_loop(("A" * 39 + "\n") * 7) is None


def test_find_loop_ignores_normal_text() -> None:
    text = (
        "Rechnung Nr. 2026-0042\n"
        "Datum: 03.11.2026\n"
        "Kunde: Beispiel GmbH\n"
        "Pos | Artikel | Menge | Einzelpreis | Summe\n"
        "1 | Schraube M4 | 10 | 0,15 | 1,50\n"
        "2 | Dübel 8 mm | 20 | 0,10 | 2,00\n"
        "3 | Kabelbinder | 5 | 0,20 | 1,00\n"
        "Zwischensumme | | | | 4,50\n"
        "Pos | Artikel | Menge | Einzelpreis | Summe\n"
        "1 x Artikel 1,00\n"
        "2 x Artikel 2,00\n"
        "1 x Artikel 1,00\n"
        "Pos | Artikel | Menge | Einzelpreis | Summe\n"
        + "-" * 40
        + "\n"
        "Zahlbar innerhalb von 14 Tagen ohne Abzug.\n"
        "Vielen Dank für Ihren Einkauf.\n"
    )
    assert len(text) > 200
    assert find_loop(text) is None


def test_find_loop_ignores_a_long_statement() -> None:
    lines = ["Kontoauszug Beispiel GmbH", "Buchungstag | Verwendungszweck | Betrag"]
    for page in range(6):
        lines.append(f"Seite {page + 1}")
        lines.append("Buchungstag | Verwendungszweck | Betrag")
        for row in range(20):
            lines.append(
                f"{row + 1:02d}.11.2026 | Rechnung {page * 100 + row} | "
                f"-{row + 1},{(page * 7 + row) % 100:02d} EUR"
            )
        lines.append("-" * 40)
        lines.append("Hinweise zum Kontoauszug")
        lines.append(
            "Dieser Auszug wird maschinell erstellt und ist ohne Unterschrift gültig."
        )
    text = "\n".join(lines)
    assert len(text) > 5000
    assert find_loop(text) is None


# ----------------------------------------------------------------------
# Acceptance: cut the prefix, otherwise reject
# ----------------------------------------------------------------------


def test_evaluate_accepts_a_prefix_at_the_ratio() -> None:
    prefix = plain(180)
    text = prefix + "\n" + "1:" * 200
    kept, reason, loop_at = evaluate(result(text), FALLBACK)
    assert kept == prefix
    assert reason == "loop_cut"
    assert loop_at == len(prefix) + 1


def test_evaluate_rejects_a_prefix_below_the_ratio() -> None:
    prefix = plain(179)
    text = prefix + "\n" + "1:" * 200
    kept, reason, loop_at = evaluate(result(text), FALLBACK)
    assert kept is None
    assert reason == "repetition"
    assert loop_at == len(prefix) + 1


def test_evaluate_rejects_a_loop_without_a_usable_prefix() -> None:
    text = "- " * 300
    assert evaluate(result(text), FALLBACK) == (None, "repetition", 0)


def test_evaluate_cuts_a_line_aligned_loop_after_the_previous_line() -> None:
    head = plain(200)
    row = "| 01.11.2026 | Miete November | -1.250,00 EUR |"
    text = head + "\n" + (row + "\n") * 12
    kept, reason, loop_at = evaluate(result(text), FALLBACK)
    assert kept == head
    assert reason == "loop_cut"
    assert loop_at == len(head)


def test_evaluate_ignores_the_ratio_when_the_fallback_is_short() -> None:
    text = plain(250) + "\n" + "1:" * 200
    assert evaluate(result(text), "kurz")[1] == "repetition"


def test_evaluate_loop_wins_over_a_length_finish() -> None:
    prefix = plain(180)
    text = prefix + "\n" + "1:" * 200
    kept, reason, _ = evaluate(result(text, finish_reason="length"), FALLBACK)
    assert kept == prefix
    assert reason == "loop_cut"


def test_evaluate_scales_the_prefix_threshold_by_share() -> None:
    prefix = plain(150)
    text = prefix + "\n" + "1:" * 200
    fallback = plain(600)
    assert evaluate(result(text), fallback, share=1 / 3)[0] == prefix
    assert evaluate(result(text), fallback)[1] == "repetition"


def test_evaluate_scales_the_runaway_threshold_by_share() -> None:
    fallback = plain(600)
    text = plain(861)
    assert evaluate(result(text), fallback, share=1 / 3)[1] == "runaway"
    assert evaluate(result(text), fallback)[0] == text


def test_combine_results_sums_tokens_and_attempts() -> None:
    first = VlmResult(
        text="erste",
        finish_reason="length",
        input_tokens=100,
        output_tokens=20,
        seconds=1.0,
        attempts=1,
        response_id="r1",
        request_id="q1",
        image_px="100x200",
        payload_bytes=10,
    )
    second = VlmResult(
        text="zweite",
        finish_reason="stop",
        input_tokens=120,
        output_tokens=30,
        seconds=2.0,
        attempts=2,
        response_id="r2",
        request_id="q2",
        image_px="120x220",
        payload_bytes=12,
    )
    merged = combine_results([first, second])
    assert (merged.input_tokens, merged.output_tokens) == (220, 50)
    assert merged.attempts == 3
    assert merged.seconds == 3.0
    assert merged.payload_bytes == 22
    assert merged.text == "zweite"
    assert merged.finish_reason == "stop"
    assert merged.response_id == "r2" and merged.request_id == "q2"
    assert merged.image_px == "120x220"


def test_evaluate_rejects_truncated() -> None:
    assert evaluate(result(plain(300), finish_reason="length"), FALLBACK) == (
        None,
        "truncated",
        None,
    )


def test_evaluate_rejects_empty() -> None:
    assert evaluate(result("   \n  "), FALLBACK) == (None, "empty", None)


def test_evaluate_accepts_empty_when_the_fallback_is_short() -> None:
    assert evaluate(result(""), "kurz") == ("", None, None)


def test_evaluate_runaway_threshold() -> None:
    fallback = plain(200)
    assert evaluate(result(plain(860)), fallback) == (plain(860), None, None)
    assert evaluate(result(plain(861)), fallback) == (None, "runaway", None)


def test_evaluate_runaway_needs_a_long_fallback() -> None:
    assert evaluate(result(plain(2000)), plain(199)) == (plain(2000), None, None)


def test_evaluate_accepts_a_normal_transcription() -> None:
    text = (
        "Kontoinhaber Beispiel GmbH\n"
        "Buchungstag 01.11.2026\n"
        "Verwendungszweck Miete November\n"
        "Betrag -1.250,00 EUR\n"
    )
    assert evaluate(result(text), FALLBACK) == (text.strip(), None, None)


# ----------------------------------------------------------------------
# Backwards-compatible reject_reason and cleanup
# ----------------------------------------------------------------------


def test_reject_reason_keeps_its_semantics() -> None:
    assert reject_reason(result(plain(300), finish_reason="length"), FALLBACK) == (
        "truncated"
    )
    assert reject_reason(result("   \n  "), FALLBACK) == "empty"
    assert reject_reason(result("Betrag\n" * 40), FALLBACK) == "repetition"
    assert reject_reason(result(plain(300)), FALLBACK) is None


def test_reject_reason_accepts_a_cut_prefix() -> None:
    text = plain(180) + "\n" + "1:" * 200
    assert reject_reason(result(text), FALLBACK) is None


def test_clean_strips_text_fence() -> None:
    raw = "```text\nKontoinhaber Beispiel GmbH\nBetrag -1.250,00 EUR\n```"
    assert clean(raw) == "Kontoinhaber Beispiel GmbH\nBetrag -1.250,00 EUR"
