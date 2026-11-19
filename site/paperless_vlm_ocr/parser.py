"""Paperless-ngx parser: per-page text-layer gate plus VLM transcription."""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Literal

from django.conf import settings
from paperless.parsers.tesseract import RasterisedDocumentParser
from paperless.parsers.utils import post_process_text
from paperless.version import __version__ as paperless_version

from paperless_vlm_ocr import SUPPORTED_PAPERLESS
from paperless_vlm_ocr import __version__
from paperless_vlm_ocr.assemble import needs_reassembly
from paperless_vlm_ocr.config import VlmOcrConfig
from paperless_vlm_ocr.decisions import append as append_decision
from paperless_vlm_ocr.gate import agreement
from paperless_vlm_ocr.gate import classify
from paperless_vlm_ocr.gate import passes
from paperless_vlm_ocr.gate import tokenize
from paperless_vlm_ocr.logfmt import endpoint_host
from paperless_vlm_ocr.logfmt import format_page_line
from paperless_vlm_ocr.logfmt import format_start_line
from paperless_vlm_ocr.logfmt import format_summary_line
from paperless_vlm_ocr.logfmt import sha_prefix
from paperless_vlm_ocr.pages import image_to_png
from paperless_vlm_ocr.pages import join_tiles
from paperless_vlm_ocr.pages import layer_text
from paperless_vlm_ocr.pages import page_count
from paperless_vlm_ocr.pages import render_page_tiles
from paperless_vlm_ocr.pages import render_png
from paperless_vlm_ocr.pages import tesseract_text
from paperless_vlm_ocr.vlm import VlmClient
from paperless_vlm_ocr.vlm import VlmResult
from paperless_vlm_ocr.vlm import call_timeout
from paperless_vlm_ocr.vlm import clean
from paperless_vlm_ocr.vlm import combine_results
from paperless_vlm_ocr.vlm import deadline_from
from paperless_vlm_ocr.vlm import evaluate
from paperless_vlm_ocr.vlm import format_llm_failure_line
from paperless_vlm_ocr.vlm import format_llm_line
from paperless_vlm_ocr.vlm import format_llm_rejection_line
from paperless_vlm_ocr.vlm import may_retry
from paperless_vlm_ocr.vlm import transport_retries

LOGGER_NAME = "paperless.parsing.vlm_ocr"
logger = logging.getLogger(LOGGER_NAME)

Source = Literal["layer", "vlm", "tesseract"]

_REJECT_REASONS = frozenset(
    {"truncated", "empty", "repetition", "runaway", "budget"}
)
_RETRY_REASONS = frozenset({"truncated", "repetition", "empty"})

_warned: set[str] = set()


def _warn_once(key: str, message: str, *args: object) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message, *args)


@dataclass
class _Page:
    """Mutable per-page state carried through the parse phases."""

    number: int
    layer: str = ""
    layer_tokens: int = 0
    ref: str | None = None
    ref_tokens: int | None = None
    ref_attempted: bool = False
    ref_error: bool = False
    ref_skipped: bool = False
    kind: str = "gated"
    decision: str = "keep"
    coverage: float | None = None
    support: float | None = None
    text: str = ""
    source: Source = "layer"
    fallback_reason: str | None = None
    vlm_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    vlm_called: bool = False
    finish_reason: str | None = None
    attempts: int | None = None
    response_id: str | None = None
    request_id: str | None = None
    payload_bytes: int | None = None
    image_px: str | None = None
    vlm_chars: int | None = None
    fallback_chars: int | None = None
    vlm_vs_fallback_coverage: float | None = None
    vlm_vs_fallback_support: float | None = None
    error_status: int | None = None
    tiles: int = 1
    loop_at: int | None = None
    reason: str | None = None


@dataclass
class _Call:
    """Outcome of one page's transcription attempt."""

    fallback: str = ""
    result: VlmResult | None = None
    text: str | None = None
    reason: str | None = None
    loop_at: int | None = None
    tiles: int = 1
    error: str | None = None
    called: bool = False
    status: int | None = None
    request_id: str | None = None
    seconds: float | None = None


class VlmOcrParser(RasterisedDocumentParser):
    """Tesseract parser whose text layer is gated and repaired by a VLM."""

    name = "Paperless VLM OCR"
    version = __version__
    author = "anokfireball"
    url = "https://github.com/anokfireball/homelab-as-code"
    uses_remote_service = True

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    @classmethod
    def score(
        cls,
        mime_type: str,
        filename: str,
        path: Path | None = None,
    ) -> int | None:
        try:
            cfg = VlmOcrConfig.from_env()
        except Exception as exc:
            _warn_once(
                "config-unreadable",
                "vlm_ocr disabled: cannot read configuration (%s)",
                type(exc).__name__,
            )
            return None
        if not cfg.enabled:
            return None
        if not cfg.endpoint or not cfg.api_key or not cfg.model:
            _warn_once(
                "config-incomplete",
                "vlm_ocr disabled: PAPERLESS_VLM_OCR_ENDPOINT, "
                "PAPERLESS_VLM_OCR_API_KEY and PAPERLESS_VLM_OCR_MODEL "
                "must all be set",
            )
            return None
        if paperless_version[:2] not in SUPPORTED_PAPERLESS:
            _warn_once(
                "paperless-version",
                "vlm_ocr disabled: Paperless %s is not one of the supported "
                "versions %s",
                ".".join(str(part) for part in paperless_version),
                sorted(SUPPORTED_PAPERLESS),
            )
            return None
        if mime_type not in cls.supported_mime_types():
            return None
        return 15

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, logging_group: object | None = None) -> None:
        super().__init__(logging_group)
        # ``self.log`` stays the parent's Tesseract logger on purpose: its own
        # messages must keep their historical logger and format.
        self.cfg = VlmOcrConfig.from_env()

    def parse(
        self,
        document_path: Path,
        mime_type: str,
        *,
        produce_archive: bool = True,
    ) -> None:
        document_path = Path(document_path)
        t0 = time.monotonic()
        super().parse(document_path, mime_type, produce_archive=produce_archive)
        parent_text = self.text
        try:
            self._process(document_path, mime_type, t0)
        except Exception:
            self.text = parent_text
            logger.exception(
                "vlm_ocr failed, keeping Tesseract/layer result",
            )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _process(self, document_path: Path, mime_type: str, t0: float) -> None:
        cfg = self.cfg
        started = time.monotonic()
        time_limit = getattr(settings, "CELERY_TASK_TIME_LIMIT", None) or 1800
        deadline = deadline_from(t0, cfg.doc_budget_s, float(time_limit))
        name = document_path.name
        sha256 = hashlib.sha256(document_path.read_bytes()).hexdigest()

        orig_pdf = document_path if mime_type == "application/pdf" else None
        render_pdf = (
            self.archive_path if self.archive_path is not None else orig_pdf
        )
        pages_total = page_count(render_pdf) if render_pdf is not None else 1
        if pages_total > cfg.max_pages:
            logger.warning(
                "vlm_ocr skipped file=%s sha=%s pages=%d max_pages=%d",
                name,
                sha_prefix(sha256),
                pages_total,
                cfg.max_pages,
            )
            return
        logger.info(
            format_start_line(
                file=name,
                sha256=sha256,
                mime=mime_type,
                pages=pages_total,
                archive=self.archive_path is not None,
                gate_mode=cfg.gate_mode,
                threshold=cfg.threshold,
                model=cfg.model,
            ),
        )

        pages = [_Page(number=number) for number in range(1, pages_total + 1)]

        self._read_layers(pages, orig_pdf)
        self._reference_ocr(pages, orig_pdf, deadline)
        self._classify(pages)
        self._transcribe(pages, document_path, render_pdf, sha256, deadline)
        self._log_pages(pages, pages_total, sha256)
        assembled = self._apply_text(pages)
        self._write_decisions(pages, pages_total, mime_type, sha256, name)

        elapsed = time.monotonic() - started
        self._log_summary(pages, elapsed, name, sha256, assembled)

    # ------------------------------------------------------------------
    # Phase A: layer text, reference OCR, gate verdict
    # ------------------------------------------------------------------

    def _read_layers(
        self,
        pages: list[_Page],
        orig_pdf: Path | None,
    ) -> None:
        if orig_pdf is None:
            return
        for page in pages:
            try:
                page.layer = layer_text(orig_pdf, page.number)
            except Exception as exc:
                logger.warning(
                    "vlm_ocr page %d: cannot read text layer (%s)",
                    page.number,
                    type(exc).__name__,
                )
                page.layer = ""
            page.layer_tokens = len(tokenize(page.layer))
            page.text = page.layer

    def _reference_ocr(
        self,
        pages: list[_Page],
        orig_pdf: Path | None,
        deadline: float,
    ) -> None:
        if orig_pdf is None:
            return
        targets = [
            page for page in pages if page.layer_tokens >= self.cfg.min_tokens
        ]
        if not targets:
            return
        if deadline - time.monotonic() <= 0:
            for page in targets:
                page.ref_skipped = True
            return
        workers = max(1, int(getattr(settings, "THREADS_PER_WORKER", 1)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._reference_text, orig_pdf, page.number): page
                for page in targets
            }
            for future in as_completed(futures):
                page = futures[future]
                page.ref_attempted = True
                try:
                    page.ref = future.result()
                except Exception as exc:
                    page.ref_error = True
                    logger.warning(
                        "vlm_ocr page %d: reference OCR failed (%s)",
                        page.number,
                        type(exc).__name__,
                    )

    def _reference_text(self, pdf: Path, number: int) -> str:
        png = render_png(
            pdf,
            number,
            self.tempdir / f"vlmocr-ref-{number}.png",
            scale_to=None,
            gray=True,
        )
        return tesseract_text(png, self.settings.language)

    def _classify(self, pages: list[_Page]) -> None:
        for page in pages:
            if page.ref_skipped:
                page.kind = "gated"
                page.decision = "keep"
                page.fallback_reason = "budget"
                page.reason = "budget"
                continue
            if page.ref_error:
                page.kind = "gated"
                page.decision = "keep"
                page.fallback_reason = "ref_error"
                continue
            ref_text = page.ref if page.ref_attempted else None
            verdict = classify(page.layer, ref_text, self.cfg)
            page.kind = verdict.kind
            page.decision = verdict.decision
            page.coverage = verdict.coverage
            page.support = verdict.support
            page.layer_tokens = verdict.layer_tokens
            page.ref_tokens = verdict.ref_tokens

    # ------------------------------------------------------------------
    # Phase B: VLM transcription
    # ------------------------------------------------------------------

    def _needs_vlm(self, page: _Page) -> bool:
        if page.kind == "empty_layer":
            return True
        if self.cfg.gate_mode != "enforce":
            return False
        return page.decision in ("vlm", "no_signal")

    def _transcribe(
        self,
        pages: list[_Page],
        document_path: Path,
        render_pdf: Path | None,
        sha256: str,
        deadline: float,
    ) -> None:
        targets = [page for page in pages if self._needs_vlm(page)]
        if not targets:
            return

        try:
            client: VlmClient | None = VlmClient(self.cfg)
        except Exception as exc:
            client = None
            client_error: str | None = type(exc).__name__
            logger.warning(
                "vlm_ocr: cannot create VLM client (%s)",
                client_error,
            )
        if client is None:
            for page in targets:
                fallback = self._fallback_text(page, document_path, render_pdf)
                self._apply_vlm(page, _Call(fallback=fallback, error=client_error))
            return

        workers = max(1, self.cfg.concurrency)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self._vlm_page,
                    client,
                    document_path,
                    render_pdf,
                    page,
                    deadline,
                ): page
                for page in targets
            }
            for future in as_completed(futures):
                page = futures[future]
                try:
                    call = future.result()
                except Exception as exc:
                    call = _Call(error=type(exc).__name__)
                self._apply_vlm(page, call)
                self._log_llm(page, call, sha256)

    def _vlm_page(
        self,
        client: VlmClient,
        document_path: Path,
        render_pdf: Path | None,
        page: _Page,
        deadline: float,
    ) -> _Call:
        fallback = self._fallback_text(page, document_path, render_pdf)
        started = time.monotonic()
        if deadline - started <= 0:
            return _Call(fallback=fallback, reason="budget")
        try:
            tiles = self._render_vlm_tiles(
                document_path,
                render_pdf,
                page.number,
            )
        except Exception as exc:
            return _Call(fallback=fallback, error=type(exc).__name__)
        results: list[VlmResult] = []
        texts: list[str] = []
        reason: str | None = None
        loop_at: int | None = None
        for png in tiles:
            try:
                result, text, tile_reason, tile_loop = self._transcribe_tile(
                    client,
                    png,
                    fallback,
                    share=1.0 / len(tiles),
                    deadline=deadline,
                )
            except Exception as exc:
                return _Call(
                    fallback=fallback,
                    error=type(exc).__name__,
                    called=True,
                    status=getattr(exc, "status_code", None),
                    request_id=getattr(exc, "request_id", None),
                    seconds=time.monotonic() - started,
                    tiles=len(tiles),
                )
            if result is None:
                return _Call(
                    fallback=fallback,
                    result=combine_results(results) if results else None,
                    reason=tile_reason,
                    loop_at=loop_at,
                    tiles=len(tiles),
                    called=bool(results),
                    seconds=time.monotonic() - started,
                )
            results.append(result)
            if tile_loop is not None and loop_at is None:
                loop_at = tile_loop
            if text is None:
                reason = tile_reason
                break
            if tile_reason is not None and reason is None:
                reason = tile_reason
            texts.append(text)
        combined = combine_results(results)
        accepted = reason not in _REJECT_REASONS
        if accepted:
            combined.text = join_tiles(texts)
        return _Call(
            fallback=fallback,
            result=combined,
            text=combined.text if accepted else None,
            reason=reason,
            loop_at=loop_at,
            tiles=len(tiles),
            called=True,
            seconds=time.monotonic() - started,
        )

    def _transcribe_tile(
        self,
        client: VlmClient,
        png: Path,
        fallback: str,
        *,
        share: float,
        deadline: float,
    ) -> tuple[VlmResult | None, str | None, str | None, int | None]:
        """One tile: one call, plus one retry on a retryable rejection.

        A call is only started when the document budget can still cover it;
        otherwise the tile reports ``budget`` and the page falls back. Per-call
        timeouts and transport retries are capped by the same budget.
        Transport retries stay inside the client; the retry here is only for a
        paid-for answer that the evaluator rejected. Returns the merged call
        record, the accepted text (or None), the reason and the loop offset.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, None, "budget", None
        timeout_s = self.cfg.timeout_s
        data = png.read_bytes()
        result = client.transcribe(
            data,
            timeout=call_timeout(remaining, timeout_s),
            max_retries=transport_retries(remaining, timeout_s),
        )
        text, reason, loop_at = evaluate(result, fallback, share=share)
        results = [result]
        if text is None and reason in _RETRY_REASONS:
            remaining = deadline - time.monotonic()
            if may_retry(remaining, timeout_s):
                result = client.transcribe(
                    data,
                    timeout=call_timeout(remaining, timeout_s),
                    max_retries=transport_retries(remaining, timeout_s),
                )
                text, reason, loop_at = evaluate(result, fallback, share=share)
                results.append(result)
        return combine_results(results), text, reason, loop_at

    def _render_vlm_tiles(
        self,
        document_path: Path,
        render_pdf: Path | None,
        number: int,
    ) -> list[Path]:
        if render_pdf is not None:
            source = render_png(
                render_pdf,
                number,
                self.tempdir / f"vlmocr-page-{number}.png",
                scale_to=None,
                gray=False,
            )
        else:
            source = document_path
        return render_page_tiles(
            source,
            self.tempdir,
            number,
            image_px=self.cfg.image_px,
        )

    def _fallback_text(
        self,
        page: _Page,
        document_path: Path,
        render_pdf: Path | None,
    ) -> str:
        try:
            if page.kind == "empty_layer":
                if self.archive_path is not None:
                    return layer_text(self.archive_path, page.number)
                return self._tesseract_page(
                    document_path,
                    render_pdf,
                    page.number,
                )
            if page.decision == "no_signal":
                return page.layer
            return page.ref or ""
        except Exception as exc:
            logger.warning(
                "vlm_ocr page %d: fallback text failed (%s)",
                page.number,
                type(exc).__name__,
            )
            return ""

    def _tesseract_page(
        self,
        document_path: Path,
        render_pdf: Path | None,
        number: int,
    ) -> str:
        out = self.tempdir / f"vlmocr-fallback-{number}.png"
        if render_pdf is not None:
            png = render_png(
                render_pdf,
                number,
                out,
                scale_to=None,
                gray=True,
            )
        else:
            png = image_to_png(document_path, out, self.cfg.image_px)
        return tesseract_text(png, self.settings.language)

    def _apply_vlm(self, page: _Page, call: _Call) -> None:
        result = call.result
        page.fallback_chars = len(call.fallback)
        page.tiles = call.tiles
        page.loop_at = call.loop_at
        page.reason = call.reason
        text = ""
        if result is not None:
            page.vlm_called = True
            page.vlm_seconds = result.seconds
            page.input_tokens = result.input_tokens
            page.output_tokens = result.output_tokens
            page.finish_reason = result.finish_reason
            page.attempts = result.attempts
            page.response_id = result.response_id
            page.request_id = result.request_id
            page.payload_bytes = result.payload_bytes
            page.image_px = result.image_px
            if call.text is not None:
                text = call.text
                page.vlm_chars = len(text)
                fallback_tokens = tokenize(call.fallback)
                if fallback_tokens:
                    (
                        page.vlm_vs_fallback_coverage,
                        page.vlm_vs_fallback_support,
                    ) = agreement(tokenize(text), fallback_tokens)
            else:
                page.vlm_chars = len(clean(result.text))

        if result is None:
            page.error_status = call.status
            page.request_id = call.request_id
            page.vlm_seconds = call.seconds
            page.reason = call.reason or call.error
        if result is None or call.text is None:
            page.text = call.fallback
            page.fallback_reason = call.reason or call.error
            page.source = (
                "layer" if page.decision == "no_signal" else "tesseract"
            )
            return

        if page.decision == "no_signal":
            coverage, support = agreement(
                tokenize(page.layer),
                tokenize(text),
            )
            page.coverage = coverage
            page.support = support
            if passes(coverage, support, self.cfg.threshold):
                page.text = page.layer
                page.source = "layer"
                return
        page.text = text
        page.source = "vlm"

    def _log_llm(self, page: _Page, call: _Call, sha256: str) -> None:
        cfg = self.cfg
        host = endpoint_host(cfg.endpoint)
        if call.result is not None:
            if page.fallback_reason in _REJECT_REASONS:
                logger.warning(
                    format_llm_rejection_line(
                        page=page.number,
                        sha256=sha256,
                        model=cfg.model,
                        host=host,
                        reason=page.fallback_reason or "-",
                        result=call.result,
                        fallback_chars=len(call.fallback),
                        tiles=call.tiles,
                        loop_at=call.loop_at,
                    ),
                )
            else:
                logger.info(
                    format_llm_line(
                        page=page.number,
                        sha256=sha256,
                        model=cfg.model,
                        host=host,
                        result=call.result,
                        tiles=call.tiles,
                        reason=page.reason,
                        loop_at=call.loop_at,
                    ),
                )
        elif call.called:
            logger.warning(
                format_llm_failure_line(
                    page=page.number,
                    sha256=sha256,
                    model=cfg.model,
                    host=host,
                    error=call.error or "-",
                    status=call.status,
                    request_id=call.request_id,
                    seconds=call.seconds,
                    tiles=call.tiles,
                ),
            )

    # ------------------------------------------------------------------
    # Result assembly, decision log, summary
    # ------------------------------------------------------------------

    def _apply_text(self, pages: list[_Page]) -> bool:
        if not needs_reassembly(page.source for page in pages):
            return False
        self.text = post_process_text(
            "\n\n".join(page.text for page in pages),
        ) or ""
        return True

    def _write_decisions(
        self,
        pages: list[_Page],
        pages_total: int,
        mime_type: str,
        sha256: str,
        file_name: str,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        gate_mode = self.cfg.gate_mode
        for page in pages:
            append_decision(
                self.cfg.decision_log,
                {
                    "ts": ts,
                    "sha256": sha256,
                    "mime": mime_type,
                    "file": file_name,
                    "page": page.number,
                    "pages": pages_total,
                    "kind": page.kind,
                    "decision": page.decision,
                    "coverage": page.coverage,
                    "support": page.support,
                    "threshold": self.cfg.threshold,
                    "layer_tokens": page.layer_tokens,
                    "ref_tokens": page.ref_tokens,
                    "source": page.source,
                    "fallback_reason": page.fallback_reason,
                    "vlm_seconds": page.vlm_seconds,
                    "input_tokens": page.input_tokens,
                    "output_tokens": page.output_tokens,
                    "model": self.cfg.model,
                    "finish_reason": page.finish_reason,
                    "attempts": page.attempts,
                    "response_id": page.response_id,
                    "request_id": page.request_id,
                    "payload_bytes": page.payload_bytes,
                    "image_px": page.image_px,
                    "vlm_chars": page.vlm_chars,
                    "fallback_chars": page.fallback_chars,
                    "layer_chars": len(page.layer),
                    "vlm_vs_fallback_coverage": page.vlm_vs_fallback_coverage,
                    "vlm_vs_fallback_support": page.vlm_vs_fallback_support,
                    "error_status": page.error_status,
                    "gate_mode": gate_mode,
                    "plugin_version": __version__,
                    "tiles": page.tiles,
                    "loop_at": page.loop_at,
                    "reason": page.reason,
                },
            )

    def _log_pages(
        self,
        pages: list[_Page],
        pages_total: int,
        sha256: str,
    ) -> None:
        for page in pages:
            logger.info(
                format_page_line(
                    page=page.number,
                    pages=pages_total,
                    sha256=sha256,
                    kind=page.kind,
                    decision=page.decision,
                    coverage=page.coverage,
                    support=page.support,
                    threshold=self.cfg.threshold,
                    layer_tokens=page.layer_tokens,
                    ref_tokens=page.ref_tokens,
                    source=page.source,
                    reason=page.reason,
                    vlm_called=page.vlm_called,
                    vlm_vs_fallback_coverage=page.vlm_vs_fallback_coverage,
                    vlm_vs_fallback_support=page.vlm_vs_fallback_support,
                    tiles=page.tiles,
                    loop_at=page.loop_at,
                ),
            )

    def _log_summary(
        self,
        pages: list[_Page],
        elapsed: float,
        file_name: str,
        sha256: str,
        assembled: bool,
    ) -> None:
        keep = sum(1 for page in pages if page.source == "layer")
        vlm = sum(1 for page in pages if page.source == "vlm")
        fallback = sum(1 for page in pages if page.source == "tesseract")
        would_vlm = sum(1 for page in pages if page.decision == "would_vlm")
        no_signal = sum(1 for page in pages if page.decision == "no_signal")
        tokens_in = sum(page.input_tokens or 0 for page in pages)
        tokens_out = sum(page.output_tokens or 0 for page in pages)

        gated = [
            page
            for page in pages
            if page.kind == "gated" and page.coverage is not None
        ]
        coverages = [page.coverage for page in gated if page.coverage is not None]
        supports = [page.support for page in gated if page.support is not None]
        sources = "".join(
            {"layer": "L", "vlm": "V", "tesseract": "T"}[page.source]
            for page in pages
        )
        logger.info(
            format_summary_line(
                file=file_name,
                sha256=sha256,
                pages=len(pages),
                keep=keep,
                vlm=vlm,
                fallback=fallback,
                would_vlm=would_vlm,
                no_signal=no_signal,
                gate_mode=self.cfg.gate_mode,
                threshold=self.cfg.threshold,
                coverage_min=min(coverages) if coverages else None,
                coverage_mean=(
                    sum(coverages) / len(coverages) if coverages else None
                ),
                support_min=min(supports) if supports else None,
                support_mean=sum(supports) / len(supports) if supports else None,
                sources=sources,
                content="assembled" if assembled else "parent",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                seconds=elapsed,
                budget_exhausted=any(
                    page.reason == "budget" for page in pages
                ),
            ),
        )
