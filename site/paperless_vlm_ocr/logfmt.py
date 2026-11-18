"""Log line formatting for the plugin; pure functions, no I/O and no Django."""

from __future__ import annotations

import struct
from urllib.parse import urlsplit

_SHA_CHARS = 12
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def sha_prefix(sha256: str | None) -> str:
    """Short checksum used in every line, so a line maps to a document."""
    return sha256[:_SHA_CHARS] if sha256 else "-"


def opt_float(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def opt_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def opt_text(value: str | None) -> str:
    return value if value else "-"


def endpoint_host(endpoint: str) -> str:
    """Hostname of the endpoint base URL: no userinfo, port, path or query."""
    try:
        host = urlsplit(endpoint).hostname
    except ValueError:
        return "-"
    return host or "-"


def png_size(png: bytes) -> str:
    """``WxH`` from a PNG IHDR chunk, ``-`` for anything else."""
    if (
        len(png) < 24
        or not png.startswith(_PNG_SIGNATURE)
        or png[12:16] != b"IHDR"
    ):
        return "-"
    width, height = struct.unpack(">II", png[16:24])
    return f"{width}x{height}"


def format_start_line(
    *,
    file: str,
    sha256: str | None,
    mime: str,
    pages: int,
    archive: bool,
    gate_mode: str,
    threshold: float,
    model: str,
) -> str:
    return (
        f"vlm_ocr start file={file} sha={sha_prefix(sha256)} mime={mime} "
        f"pages={pages} archive={'yes' if archive else 'no'} "
        f"gate_mode={gate_mode} threshold={threshold:.2f} "
        f"model={opt_text(model)}"
    )


def format_page_line(
    *,
    page: int,
    pages: int,
    sha256: str | None,
    kind: str,
    decision: str,
    coverage: float | None,
    support: float | None,
    threshold: float,
    layer_tokens: int,
    ref_tokens: int | None,
    source: str,
    reason: str | None,
    vlm_called: bool = False,
    vlm_vs_fallback_coverage: float | None = None,
    vlm_vs_fallback_support: float | None = None,
) -> str:
    line = (
        f"vlm_ocr page {page}/{pages} sha={sha_prefix(sha256)} "
        f"kind={kind} decision={decision} "
        f"coverage={opt_float(coverage)} support={opt_float(support)} "
        f"threshold={threshold:.2f} layer_tokens={layer_tokens} "
        f"ref_tokens={opt_int(ref_tokens)} source={source} "
        f"reason={opt_text(reason)}"
    )
    if vlm_called:
        line += (
            " vlm_vs_fallback_coverage="
            f"{opt_float(vlm_vs_fallback_coverage)}"
            " vlm_vs_fallback_support="
            f"{opt_float(vlm_vs_fallback_support)}"
        )
    return line


def format_summary_line(
    *,
    file: str,
    sha256: str | None,
    pages: int,
    keep: int,
    vlm: int,
    fallback: int,
    would_vlm: int,
    no_signal: int,
    gate_mode: str,
    threshold: float,
    coverage_min: float | None,
    coverage_mean: float | None,
    support_min: float | None,
    support_mean: float | None,
    sources: str,
    content: str,
    tokens_in: int,
    tokens_out: int,
    seconds: float,
) -> str:
    return (
        f"vlm_ocr summary file={file} sha={sha_prefix(sha256)} pages={pages} "
        f"keep={keep} vlm={vlm} fallback={fallback} "
        f"would_vlm={would_vlm} no_signal={no_signal} "
        f"gate_mode={gate_mode} threshold={threshold:.2f} "
        f"coverage_min={opt_float(coverage_min)} "
        f"coverage_mean={opt_float(coverage_mean)} "
        f"support_min={opt_float(support_min)} "
        f"support_mean={opt_float(support_mean)} "
        f"sources={sources or '-'} content={content} "
        f"tokens_in={tokens_in} tokens_out={tokens_out} "
        f"seconds={seconds:.1f}"
    )
