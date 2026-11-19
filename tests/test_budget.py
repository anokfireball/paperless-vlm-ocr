"""Per-document budget helpers: timeouts, retries and exhaustion."""

from __future__ import annotations

from paperless_vlm_ocr.vlm import call_timeout
from paperless_vlm_ocr.vlm import deadline_from
from paperless_vlm_ocr.vlm import may_retry
from paperless_vlm_ocr.vlm import transport_retries


def test_deadline_uses_the_document_budget_when_the_limit_is_large() -> None:
    assert deadline_from(100.0, 900.0, 1800.0) == 1000.0
    assert deadline_from(100.0, 900.0, 5000.0) == 1000.0


def test_deadline_is_capped_by_the_task_limit_minus_the_margin() -> None:
    assert deadline_from(100.0, 900.0, 600.0) == 580.0
    assert deadline_from(100.0, 900.0, 1000.0) == 980.0
    assert deadline_from(100.0, 900.0, 100.0) == 80.0


def test_slow_parent_parse_leaves_no_budget() -> None:
    t0 = 1000.0
    now = t0 + 1700.0
    deadline = deadline_from(t0, 900.0, 1800.0)
    remaining = deadline - now
    assert remaining <= 0
    assert call_timeout(remaining, 120.0) == 0.0
    assert not may_retry(remaining, 120.0)


def test_call_timeout_is_capped_by_the_remaining_budget() -> None:
    assert call_timeout(500.0, 120.0) == 120.0
    assert call_timeout(120.0, 120.0) == 120.0
    assert call_timeout(30.0, 120.0) == 30.0


def test_call_timeout_never_goes_negative() -> None:
    assert call_timeout(0.0, 120.0) == 0.0
    assert call_timeout(-5.0, 120.0) == 0.0


def test_may_retry_needs_a_full_timeout() -> None:
    assert may_retry(500.0, 120.0)
    assert may_retry(120.0, 120.0)
    assert not may_retry(119.9, 120.0)
    assert not may_retry(0.0, 120.0)


def test_transport_retries_fit_the_remaining_budget() -> None:
    # worst case (retries + 1) * timeout must not exceed the remaining budget
    assert transport_retries(239.9, 120.0) == 0
    assert transport_retries(240.0, 120.0) == 1
    assert transport_retries(359.9, 120.0) == 1
    assert transport_retries(360.0, 120.0) == 2
    assert transport_retries(900.0, 120.0) == 2
