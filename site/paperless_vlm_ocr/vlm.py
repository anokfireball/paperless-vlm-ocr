"""Vision-model transcription over an OpenAI-compatible endpoint."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
        response = self._client.chat.completions.create(
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
