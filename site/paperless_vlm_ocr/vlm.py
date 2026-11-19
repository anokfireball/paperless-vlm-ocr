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

# Loop detector: a unit of 1-120 characters repeated at least 8 times and at
# least 200 characters long is degenerate generation, not content.
MAX_PERIOD = 120
MIN_REPEATS = 8
MIN_LOOP_CHARS = 200
# A cut prefix is kept when it covers this much of the expected page text.
LOOP_CUT_RATIO = 0.6
# Runaway guard: text far longer than the fallback is an expansion.
RUNAWAY_FACTOR = 2.8
RUNAWAY_SLACK = 300


def deadline_from(
    t0: float,
    doc_budget_s: float,
    time_limit: float,
    margin: float = 120.0,
) -> float:
    """Absolute monotonic deadline for one parse.

    The document budget may not run past the Celery hard limit minus a margin
    that is left for post-processing after the parser returns.
    """
    return t0 + min(doc_budget_s, time_limit - margin)


def call_timeout(remaining: float, timeout_s: float) -> float:
    """Per-request timeout: never more than the remaining document budget."""
    return max(0.0, min(timeout_s, remaining))


def may_retry(remaining: float, timeout_s: float) -> bool:
    """Start one more paid call only when a full timeout still fits."""
    return remaining >= timeout_s


def transport_retries(remaining: float, timeout_s: float) -> int:
    """Transport retries whose worst case (attempts * timeout) fits the budget."""
    if remaining < 2 * timeout_s:
        return 0
    if remaining < 3 * timeout_s:
        return 1
    return 2


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

    def transcribe(
        self,
        png_bytes: bytes,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> VlmResult:
        data = base64.b64encode(png_bytes).decode("ascii")
        client = self._client
        options: dict[str, float | int] = {}
        if timeout is not None:
            options["timeout"] = timeout
        if max_retries is not None:
            options["max_retries"] = max_retries
        if options:
            client = client.with_options(**options)
        started = time.monotonic()
        raw = client.chat.completions.with_raw_response.create(
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
    tiles: int = 1,
    reason: str | None = None,
    loop_at: int | None = None,
) -> str:
    """One accepted model call; never contains the transcript or the key."""
    return (
        f"{_llm_prefix(page, sha256, model, host)} {_call_fields(result)} "
        f"tiles={tiles} reason={opt_text(reason)} loop_at={opt_int(loop_at)}"
    )


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
    tiles: int = 1,
) -> str:
    """One failed model call: exception class, status and request id only."""
    return (
        f"{_llm_prefix(page, sha256, model, host)} failed "
        f"error={error} status={opt_int(status)} "
        f"request_id={opt_text(request_id)} "
        f"seconds={opt_float(seconds, 1)} tiles={tiles}"
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
    tiles: int = 1,
    loop_at: int | None = None,
) -> str:
    """A paid-for model call whose output was rejected in favour of the fallback."""
    return (
        f"{_llm_prefix(page, sha256, model, host)} "
        f"{_call_fields(result)} "
        f"rejected reason={reason} "
        f"fallback_chars={fallback_chars} "
        f"tiles={tiles} loop_at={opt_int(loop_at)}"
    )


def clean(text: str) -> str:
    """Strip one leading and one trailing Markdown code fence."""
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def find_loop(text: str) -> int | None:
    """Earliest offset of a degenerate periodic run, or None.

    A run repeats one unit of 1-120 characters at least 8 times and is at
    least 200 characters long. Consecutive matches of ``text[i]`` with
    ``text[i + period]`` form a run, so a loop is found inside one long line
    as well as across lines; the scan is O(len(text) * 120).
    """
    n = len(text)
    if n < MIN_LOOP_CHARS:
        return None
    earliest: int | None = None
    for period in range(1, MAX_PERIOD + 1):
        # matches needed for MIN_REPEATS units and MIN_LOOP_CHARS characters
        needed = max((MIN_REPEATS - 1) * period, MIN_LOOP_CHARS - period)
        limit = n - period
        run_start = -1
        for i in range(limit):
            if text[i] == text[i + period]:
                if run_start < 0:
                    run_start = i
            elif run_start >= 0:
                if i - run_start >= needed and (
                    earliest is None or run_start < earliest
                ):
                    earliest = run_start
                run_start = -1
        if run_start >= 0 and limit - run_start >= needed:
            if earliest is None or run_start < earliest:
                earliest = run_start
    return earliest


def _loop_prefix(text: str, offset: int) -> str:
    """Text before the line that contains the loop start."""
    line_start = text.rfind("\n", 0, offset) + 1
    if offset < len(text) and text[offset] == "\n":
        # The run starts on the newline that ends a line: that line is
        # content, the loop begins with the next one.
        line_start = offset + 1
    return text[:line_start].strip()


def evaluate(
    result: VlmResult,
    fallback_text: str,
    *,
    share: float = 1.0,
) -> tuple[str | None, str | None, int | None]:
    """Decide what to keep from one model call.

    Returns ``(text, reason, loop_at)``: the accepted transcription or None,
    the final reason (``loop_cut`` when a prefix was accepted, a rejection
    reason, or None) and the loop offset inside this call's text.

    ``share`` is the fraction of the page's text this call is expected to
    produce: 1.0 for a whole page, ``1 / tiles`` for one tile. It scales the
    prefix and runaway thresholds only.
    """
    text = clean(result.text)
    loop_at = find_loop(text)
    if loop_at is not None:
        prefix = _loop_prefix(text, loop_at)
        if (
            len(fallback_text) >= MIN_LOOP_CHARS
            and len(prefix) >= LOOP_CUT_RATIO * share * len(fallback_text)
        ):
            return prefix, "loop_cut", loop_at
        return None, "repetition", loop_at
    if result.finish_reason == "length":
        return None, "truncated", None
    if len(text) < 20 and len(fallback_text) >= 50:
        return None, "empty", None
    if (
        len(fallback_text) >= 200
        and len(text)
        > RUNAWAY_FACTOR * share * len(fallback_text) + RUNAWAY_SLACK
    ):
        return None, "runaway", None
    return text, None, None


def combine_results(results: list[VlmResult]) -> VlmResult:
    """Merge the calls made for one page or tile into one record.

    Tokens, seconds, payload bytes and transport attempts are summed; finish
    reason, image size and upstream ids come from the last call.
    """
    last = results[-1]
    return VlmResult(
        text=last.text,
        finish_reason=last.finish_reason,
        input_tokens=sum(item.input_tokens for item in results),
        output_tokens=sum(item.output_tokens for item in results),
        seconds=sum(item.seconds for item in results),
        attempts=sum(item.attempts for item in results),
        response_id=last.response_id,
        request_id=last.request_id,
        image_px=last.image_px,
        payload_bytes=sum(item.payload_bytes for item in results),
    )


def reject_reason(result: VlmResult, fallback_text: str) -> str | None:
    """Return why a VLM transcription is unusable, or None if it is fine."""
    text, reason, _ = evaluate(result, fallback_text)
    return None if text is not None else reason
