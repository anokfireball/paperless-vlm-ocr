"""Log line formatting: exact field names, order and redaction guarantees."""

from __future__ import annotations

import logging
import struct
from pathlib import Path

from paperless_vlm_ocr.decisions import append as append_decision
from paperless_vlm_ocr.logfmt import endpoint_host
from paperless_vlm_ocr.logfmt import format_page_line
from paperless_vlm_ocr.logfmt import format_start_line
from paperless_vlm_ocr.logfmt import format_summary_line
from paperless_vlm_ocr.logfmt import png_size
from paperless_vlm_ocr.logfmt import sha_prefix
from paperless_vlm_ocr.vlm import VlmResult
from paperless_vlm_ocr.vlm import format_llm_failure_line
from paperless_vlm_ocr.vlm import format_llm_line
from paperless_vlm_ocr.vlm import format_llm_rejection_line

SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"
SHA12 = "a1b2c3d4e5f6"
HOST = "api.example.test"
MODEL = "vision-model"


def _png(width: int, height: int) -> bytes:
    ihdr = struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00"
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr
    return b"\x89PNG\r\n\x1a\n" + chunk + b"\x00\x00\x00\x00"


def _result(**kwargs) -> VlmResult:
    kwargs.setdefault("text", "Kontoinhaber Beispiel GmbH\nBetrag -1.250,00 EUR")
    kwargs.setdefault("finish_reason", "stop")
    kwargs.setdefault("input_tokens", 1700)
    kwargs.setdefault("output_tokens", 420)
    kwargs.setdefault("seconds", 12.34)
    kwargs.setdefault("attempts", 2)
    kwargs.setdefault("response_id", "chatcmpl-abc")
    kwargs.setdefault("request_id", "req-123")
    kwargs.setdefault("image_px", "1800x2546")
    kwargs.setdefault("payload_bytes", 731003)
    return VlmResult(**kwargs)


def test_sha_prefix() -> None:
    assert sha_prefix(SHA) == SHA12
    assert sha_prefix(None) == "-"


def test_png_size_reads_ihdr() -> None:
    assert png_size(_png(1800, 2546)) == "1800x2546"


def test_png_size_rejects_non_png() -> None:
    assert png_size(b"") == "-"
    assert png_size(b"%PDF-1.7 not a png image") == "-"
    assert png_size(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32) == "-"


def test_endpoint_host_strips_userinfo_path_and_query() -> None:
    assert endpoint_host("https://user:secret@api.example.test/v1?k=1") == HOST
    assert endpoint_host("https://api.example.test/v1") == HOST
    assert endpoint_host("https://api.example.test:8443/v1") == HOST
    assert endpoint_host("") == "-"
    assert endpoint_host("not a url") == "-"


def test_start_line() -> None:
    line = format_start_line(
        file="statement.pdf",
        sha256=SHA,
        mime="application/pdf",
        pages=5,
        archive=True,
        gate_mode="shadow",
        threshold=0.9,
        model=MODEL,
    )
    assert line == (
        f"vlm_ocr start file=statement.pdf sha={SHA12} mime=application/pdf "
        "pages=5 archive=yes gate_mode=shadow threshold=0.90 "
        f"model={MODEL}"
    )


def test_page_line_keep() -> None:
    line = format_page_line(
        page=2,
        pages=5,
        sha256=SHA,
        kind="gated",
        decision="keep",
        coverage=0.987,
        support=0.991,
        threshold=0.9,
        layer_tokens=120,
        ref_tokens=118,
        source="layer",
        reason=None,
    )
    assert line == (
        f"vlm_ocr page 2/5 sha={SHA12} kind=gated decision=keep "
        "coverage=0.987 support=0.991 threshold=0.90 layer_tokens=120 "
        "ref_tokens=118 source=layer reason=- tiles=1 loop_at=-"
    )


def test_page_line_vlm_with_fallback_agreement() -> None:
    line = format_page_line(
        page=1,
        pages=3,
        sha256=SHA,
        kind="empty_layer",
        decision="vlm",
        coverage=None,
        support=None,
        threshold=0.9,
        layer_tokens=0,
        ref_tokens=None,
        source="vlm",
        reason=None,
        vlm_called=True,
        vlm_vs_fallback_coverage=0.9414,
        vlm_vs_fallback_support=0.9327,
    )
    assert line == (
        f"vlm_ocr page 1/3 sha={SHA12} kind=empty_layer decision=vlm "
        "coverage=- support=- threshold=0.90 layer_tokens=0 ref_tokens=- "
        "source=vlm reason=- vlm_vs_fallback_coverage=0.941 "
        "vlm_vs_fallback_support=0.933 tiles=1 loop_at=-"
    )


def test_page_line_tesseract_fallback_with_reason() -> None:
    line = format_page_line(
        page=4,
        pages=5,
        sha256=SHA,
        kind="gated",
        decision="vlm",
        coverage=0.312,
        support=0.298,
        threshold=0.9,
        layer_tokens=87,
        ref_tokens=95,
        source="tesseract",
        reason="RuntimeError",
    )
    assert line == (
        f"vlm_ocr page 4/5 sha={SHA12} kind=gated decision=vlm "
        "coverage=0.312 support=0.298 threshold=0.90 layer_tokens=87 "
        "ref_tokens=95 source=tesseract reason=RuntimeError tiles=1 loop_at=-"
    )


def test_page_line_loop_cut() -> None:
    line = format_page_line(
        page=3,
        pages=3,
        sha256=SHA,
        kind="empty_layer",
        decision="vlm",
        coverage=None,
        support=None,
        threshold=0.9,
        layer_tokens=0,
        ref_tokens=None,
        source="vlm",
        reason="loop_cut",
        vlm_called=True,
        vlm_vs_fallback_coverage=0.812,
        vlm_vs_fallback_support=0.774,
        tiles=3,
        loop_at=533,
    )
    assert line == (
        f"vlm_ocr page 3/3 sha={SHA12} kind=empty_layer decision=vlm "
        "coverage=- support=- threshold=0.90 layer_tokens=0 ref_tokens=- "
        "source=vlm reason=loop_cut vlm_vs_fallback_coverage=0.812 "
        "vlm_vs_fallback_support=0.774 tiles=3 loop_at=533"
    )


def test_summary_line() -> None:
    line = format_summary_line(
        file="statement.pdf",
        sha256=SHA,
        pages=3,
        keep=1,
        vlm=1,
        fallback=1,
        would_vlm=0,
        no_signal=0,
        gate_mode="shadow",
        threshold=0.9,
        coverage_min=0.312,
        coverage_mean=0.6495,
        support_min=0.298,
        support_mean=0.6445,
        sources="LVT",
        content="assembled",
        tokens_in=1700,
        tokens_out=420,
        seconds=18.0,
    )
    assert line == (
        f"vlm_ocr summary file=statement.pdf sha={SHA12} pages=3 keep=1 vlm=1 "
        "fallback=1 would_vlm=0 no_signal=0 gate_mode=shadow threshold=0.90 "
        "coverage_min=0.312 coverage_mean=0.649 support_min=0.298 "
        "support_mean=0.644 sources=LVT content=assembled tokens_in=1700 "
        "tokens_out=420 seconds=18.0 budget_exhausted=no"
    )


def test_summary_line_reports_budget_exhaustion() -> None:
    line = format_summary_line(
        file="statement.pdf",
        sha256=SHA,
        pages=3,
        keep=1,
        vlm=0,
        fallback=2,
        would_vlm=0,
        no_signal=0,
        gate_mode="shadow",
        threshold=0.9,
        coverage_min=0.312,
        coverage_mean=0.6495,
        support_min=0.298,
        support_mean=0.6445,
        sources="LTT",
        content="assembled",
        tokens_in=1700,
        tokens_out=420,
        seconds=18.0,
        budget_exhausted=True,
    )
    assert line.endswith("seconds=18.0 budget_exhausted=yes")


def test_llm_success_line() -> None:
    line = format_llm_line(
        page=1,
        sha256=SHA,
        model=MODEL,
        host=HOST,
        result=_result(),
    )
    assert line == (
        f"vlm_ocr llm page=1 sha={SHA12} model={MODEL} host={HOST} "
        "attempts=2 finish_reason=stop tokens_in=1700 tokens_out=420 "
        "seconds=12.3 image=1800x2546 bytes=731003 chars=47 "
        "response_id=chatcmpl-abc request_id=req-123 "
        "tiles=1 reason=- loop_at=-"
    )
    assert "Kontoinhaber" not in line


def test_llm_success_line_with_loop_cut() -> None:
    line = format_llm_line(
        page=1,
        sha256=SHA,
        model=MODEL,
        host=HOST,
        result=_result(),
        tiles=3,
        reason="loop_cut",
        loop_at=533,
    )
    assert line.endswith("tiles=3 reason=loop_cut loop_at=533")


def test_llm_failure_line() -> None:
    line = format_llm_failure_line(
        page=3,
        sha256=SHA,
        model=MODEL,
        host=HOST,
        error="APIStatusError",
        status=429,
        request_id="req-999",
        seconds=3.47,
    )
    assert line == (
        f"vlm_ocr llm page=3 sha={SHA12} model={MODEL} host={HOST} failed "
        "error=APIStatusError status=429 request_id=req-999 seconds=3.5 tiles=1"
    )


def test_llm_failure_line_without_status_or_request_id() -> None:
    line = format_llm_failure_line(
        page=3,
        sha256=SHA,
        model=MODEL,
        host=HOST,
        error="APIConnectionError",
        status=None,
        request_id=None,
        seconds=None,
    )
    assert line == (
        f"vlm_ocr llm page=3 sha={SHA12} model={MODEL} host={HOST} failed "
        "error=APIConnectionError status=- request_id=- seconds=- tiles=1"
    )


def test_llm_rejection_line() -> None:
    line = format_llm_rejection_line(
        page=2,
        sha256=SHA,
        model=MODEL,
        host=HOST,
        reason="repetition",
        result=_result(text="Betrag -1.250,00 EUR\n" * 8, finish_reason="stop"),
        fallback_chars=812,
    )
    assert line == (
        f"vlm_ocr llm page=2 sha={SHA12} model={MODEL} host={HOST} "
        "attempts=2 finish_reason=stop tokens_in=1700 tokens_out=420 "
        "seconds=12.3 image=1800x2546 bytes=731003 chars=167 "
        "response_id=chatcmpl-abc request_id=req-123 "
        "rejected reason=repetition fallback_chars=812 tiles=1 loop_at=-"
    )


def test_llm_lines_never_contain_secrets() -> None:
    secret = "sk-live-should-never-appear"
    lines = [
        format_llm_line(
            page=1, sha256=SHA, model=MODEL, host=HOST, result=_result()
        ),
        format_llm_failure_line(
            page=1,
            sha256=SHA,
            model=MODEL,
            host=HOST,
            error="APIStatusError",
            status=500,
            request_id=None,
            seconds=1.0,
        ),
        format_llm_rejection_line(
            page=1,
            sha256=SHA,
            model=MODEL,
            host=HOST,
            reason="empty",
            result=_result(text="", output_tokens=4),
            fallback_chars=0,
        ),
    ]
    for line in lines:
        assert secret not in line
        assert "https://" not in line
        assert "user:secret" not in line


def test_decision_log_warns_instead_of_raising(
    tmp_path: Path,
    caplog,
) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="paperless.parsing.vlm_ocr"):
        append_decision(blocker / "decisions.jsonl", {"page": 1})
    assert "cannot write decision log" in caplog.text
