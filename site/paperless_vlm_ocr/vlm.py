"""Vision-model transcription over an OpenAI-compatible endpoint."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from paperless_vlm_ocr.logfmt import opt_float
from paperless_vlm_ocr.logfmt import opt_int
from paperless_vlm_ocr.logfmt import opt_text
from paperless_vlm_ocr.logfmt import png_size
from paperless_vlm_ocr.logfmt import sha_prefix

if TYPE_CHECKING:
    from paperless_vlm_ocr.config import VlmOcrConfig

PROMPT = (
    "Transcribe all text on this page exactly as written, in natural reading order. "
    "Output plain text only. Do not translate, correct, summarize, or add anything. "
    "Keep line breaks. For tables, output one row per line with cells separated by ' | '. "
    "Write [?] for illegible words."
)
USER_TEXT = "Transcribe this page."


@dataclass
class VlmResult:
    text: str
    finish_reason: str | None
    input_tokens: int
    output_tokens: int
    seconds: float
    attempts: int = 0
    response_id: str | None = None
    request_id: str | None = None
    image_px: str = "-"
    payload_bytes: int = 0


class VlmClient:
    """Thin wrapper around one chat-completions endpoint."""

    def __init__(self, cfg: VlmOcrConfig) -> None:
        import openai

        self._cfg = cfg
        self._client = openai.OpenAI(
            base_url=cfg.endpoint,
            api_key=cfg.api_key,
            timeout=cfg.timeout_s,
            max_retries=2,
        )

    def transcribe(self, png_bytes: bytes) -> VlmResult:
        data = base64.b64encode(png_bytes).decode("ascii")
        started = time.monotonic()
        raw = self._client.chat.completions.with_raw_response.create(
            model=self._cfg.model,
            temperature=0,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_TEXT},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + data},
                        },
                    ],
                },
            ],
            extra_body={"reasoning_effort": "none"},
        )
        response = raw.parse()
        seconds = time.monotonic() - started
        choice = response.choices[0]
        message = choice.message
        usage = response.usage
        return VlmResult(
            text=message.content or "",
            finish_reason=choice.finish_reason,
            input_tokens=usage.prompt_tokens if usage is not None else 0,
            output_tokens=usage.completion_tokens if usage is not None else 0,
            seconds=seconds,
            attempts=getattr(raw, "retries_taken", 0) + 1,
            response_id=response.id,
            request_id=raw.headers.get("x-request-id"),
            image_px=png_size(png_bytes),
            payload_bytes=len(png_bytes),
        )


def _llm_prefix(
    page: int,
    sha256: str | None,
    model: str,
    host: str,
) -> str:
    return (
        f"vlm_ocr llm page={page} sha={sha_prefix(sha256)} "
        f"model={opt_text(model)} host={opt_text(host)}"
    )


def _call_fields(result: VlmResult) -> str:
    """Call details shared by the accepted and the rejected line."""
    return (
        f"attempts={result.attempts} "
        f"finish_reason={opt_text(result.finish_reason)} "
        f"tokens_in={result.input_tokens} tokens_out={result.output_tokens} "
        f"seconds={result.seconds:.1f} image={result.image_px} "
        f"bytes={result.payload_bytes} chars={len(clean(result.text))} "
        f"response_id={opt_text(result.response_id)} "
        f"request_id={opt_text(result.request_id)}"
    )


def format_llm_line(
    *,
    page: int,
    sha256: str | None,
    model: str,
    host: str,
    result: VlmResult,
) -> str:
    """One accepted model call; never contains the transcript or the key."""
    return f"{_llm_prefix(page, sha256, model, host)} {_call_fields(result)}"


def format_llm_failure_line(
    *,
    page: int,
    sha256: str | None,
    model: str,
    host: str,
    error: str,
    status: int | None,
    request_id: str | None,
    seconds: float | None,
) -> str:
    """One failed model call: exception class, status and request id only."""
    return (
        f"{_llm_prefix(page, sha256, model, host)} failed "
        f"error={error} status={opt_int(status)} "
        f"request_id={opt_text(request_id)} "
        f"seconds={opt_float(seconds, 1)}"
    )


def format_llm_rejection_line(
    *,
    page: int,
    sha256: str | None,
    model: str,
    host: str,
    reason: str,
    result: VlmResult,
    fallback_chars: int,
) -> str:
    """A paid-for model call whose output was rejected in favour of the fallback."""
    return (
        f"{_llm_prefix(page, sha256, model, host)} "
        f"{_call_fields(result)} "
        f"rejected reason={reason} "
        f"fallback_chars={fallback_chars}"
    )


def clean(text: str) -> str:
    """Strip one leading and one trailing Markdown code fence."""
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _has_repetition(text: str) -> bool:
    last = ""
    run = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or len(line) < 4:
            last = ""
            run = 0
            continue
        if line == last:
            run += 1
        else:
            last = line
            run = 1
        if run >= 8:
            return True
    return False


def reject_reason(result: VlmResult, fallback_text: str) -> str | None:
    """Return why a VLM transcription is unusable, or None if it is fine."""
    text = clean(result.text)
    if result.finish_reason == "length":
        return "truncated"
    if len(text) < 20 and len(fallback_text) >= 50:
        return "empty"
    if _has_repetition(text):
        return "repetition"
    if len(fallback_text) >= 200 and len(text) > 4 * len(fallback_text) + 2000:
        return "runaway"
    return None
