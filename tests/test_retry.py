"""Retry policy, output cap and per-call timeout."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from paperless_vlm_ocr.config import VlmOcrConfig
from paperless_vlm_ocr.vlm import MAX_TOKENS
from paperless_vlm_ocr.vlm import VlmClient
from paperless_vlm_ocr.vlm import VlmResult
from paperless_vlm_ocr.vlm import retryable_error
from paperless_vlm_ocr.vlm import transcribe_with_retry

FALLBACK = "wort " * 200


class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class APIStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def result(
    text: str = "Kontoinhaber Beispiel GmbH\nBetrag -1.250,00 EUR",
    finish_reason: str = "stop",
) -> VlmResult:
    return VlmResult(
        text=text,
        finish_reason=finish_reason,
        input_tokens=100,
        output_tokens=20,
        seconds=1.0,
        attempts=1,
        image_px="100x200",
        payload_bytes=10,
    )


def _run(answers, remaining_values, *, timeout_s: float = 240.0):
    calls: list[float] = []
    pending = list(answers)

    def transcribe(timeout: float) -> VlmResult:
        calls.append(timeout)
        answer = pending.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    left = iter(remaining_values)
    outcome = transcribe_with_retry(
        transcribe,
        FALLBACK,
        share=1.0,
        timeout_s=timeout_s,
        remaining=lambda: next(left),
    )
    return outcome, calls


def test_retryable_error_policy() -> None:
    assert not retryable_error(APITimeoutError())
    assert retryable_error(APIConnectionError())
    assert retryable_error(APIStatusError(429))
    assert retryable_error(APIStatusError(503))
    assert not retryable_error(APIStatusError(400))
    assert not retryable_error(RuntimeError())


def test_timeout_is_not_retried() -> None:
    with pytest.raises(APITimeoutError) as caught:
        _run([APITimeoutError()], [900.0, 900.0])
    assert caught.value.__class__.__name__ == "APITimeoutError"
    assert caught.value.args == ()


def test_timeout_makes_exactly_one_call() -> None:
    calls: list[float] = []

    def transcribe(timeout: float) -> VlmResult:
        calls.append(timeout)
        raise APITimeoutError("timed out")

    with pytest.raises(APITimeoutError):
        transcribe_with_retry(
            transcribe,
            FALLBACK,
            share=1.0,
            timeout_s=240.0,
            remaining=lambda: 900.0,
        )
    assert calls == [240.0]


def test_status_503_is_retried_once() -> None:
    outcome, calls = _run([APIStatusError(503), result()], [900.0, 900.0])
    assert calls == [240.0, 240.0]
    assert outcome[1] == result().text
    assert outcome[2] is None


def test_connection_error_is_retried_once() -> None:
    outcome, calls = _run([APIConnectionError(), result()], [900.0, 900.0])
    assert calls == [240.0, 240.0]
    assert outcome[1] == result().text
    assert outcome[2] is None


def test_status_400_is_not_retried() -> None:
    with pytest.raises(APIStatusError):
        _run([APIStatusError(400), result()], [900.0, 900.0])


def test_truncated_result_is_retried_once() -> None:
    outcome, calls = _run(
        [result("Kurz", finish_reason="length"), result()],
        [900.0, 900.0],
    )
    assert calls == [240.0, 240.0]
    assert outcome[1] == result().text
    assert outcome[2] is None


def test_budget_exhausted_skips_the_call() -> None:
    outcome, calls = _run([result()], [0.0])
    assert outcome == (None, None, "budget", None)
    assert calls == []


def test_budget_blocks_the_retry() -> None:
    outcome, calls = _run(
        [result("Kurz", finish_reason="length"), result()],
        [900.0, 100.0],
    )
    assert calls == [240.0]
    assert outcome[1] is None
    assert outcome[2] == "truncated"


def test_budget_caps_the_call_timeout() -> None:
    outcome, calls = _run([result()], [100.0])
    assert calls == [100.0]
    assert outcome[1] == result().text


class _FakeRaw:
    headers = {"x-request-id": "req-1"}

    def parse(self):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Beispiel GmbH"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5),
            id="chatcmpl-1",
        )


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeRaw()


class _FakeClient:
    def __init__(self) -> None:
        self.options = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                with_raw_response=_FakeCompletions(),
            )
        )

    def with_options(self, **kwargs):
        self.options = kwargs
        return self


def _cfg(**kwargs) -> VlmOcrConfig:
    kwargs.setdefault("endpoint", "http://127.0.0.1:9/v1")
    kwargs.setdefault("api_key", "k")
    kwargs.setdefault("model", "m")
    kwargs.setdefault("timeout_s", 240.0)
    kwargs.setdefault("decision_log", Path("/dev/null"))
    return VlmOcrConfig(**kwargs)


def test_client_has_no_transport_retries() -> None:
    client = VlmClient(_cfg())
    assert client._client.max_retries == 0


def test_request_uses_max_tokens() -> None:
    client = VlmClient(_cfg())
    fake = _FakeClient()
    client._client = fake
    outcome = client.transcribe(b"png")
    kwargs = fake.chat.completions.with_raw_response.kwargs
    assert kwargs["max_tokens"] == MAX_TOKENS
    assert kwargs["model"] == "m"
    assert kwargs["temperature"] == 0
    assert outcome.text == "Beispiel GmbH"
    assert outcome.input_tokens == 11


def test_transcribe_applies_the_timeout_option() -> None:
    client = VlmClient(_cfg())
    fake = _FakeClient()
    client._client = fake
    client.transcribe(b"png", timeout=5.0)
    assert fake.options == {"timeout": 5.0}


def test_timeout_default_is_240(monkeypatch) -> None:
    assert VlmOcrConfig().timeout_s == 240.0
    monkeypatch.setenv("PAPERLESS_VLM_OCR_DECISION_LOG", "/dev/null")
    monkeypatch.delenv("PAPERLESS_VLM_OCR_TIMEOUT_S", raising=False)
    assert VlmOcrConfig.from_env().timeout_s == 240.0
    monkeypatch.setenv("PAPERLESS_VLM_OCR_TIMEOUT_S", "300")
    assert VlmOcrConfig.from_env().timeout_s == 300.0
