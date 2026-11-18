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
from paperless_vlm_ocr.config import VlmOcrConfig
from paperless_vlm_ocr.decisions import append as append_decision
from paperless_vlm_ocr.gate import agreement
from paperless_vlm_ocr.gate import passes
from paperless_vlm_ocr.gate import classify
from paperless_vlm_ocr.gate import tokenize
from paperless_vlm_ocr.pages import image_to_png
from paperless_vlm_ocr.pages import layer_text
from paperless_vlm_ocr.pages import page_count
from paperless_vlm_ocr.pages import render_png
from paperless_vlm_ocr.pages import tesseract_text
from paperless_vlm_ocr.vlm import VlmClient
from paperless_vlm_ocr.vlm import VlmResult
from paperless_vlm_ocr.vlm import clean
from paperless_vlm_ocr.vlm import reject_reason

LOGGER_NAME = "paperless.parsing.vlm_ocr"
logger = logging.getLogger(LOGGER_NAME)

Source = Literal["layer", "vlm", "tesseract"]

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
        self.log = logging.getLogger(LOGGER_NAME)
        self.cfg = VlmOcrConfig.from_env()

    def parse(
        self,
        document_path: Path,
        mime_type: str,
        *,
        produce_archive: bool = True,
    ) -> None:
        document_path = Path(document_path)
        super().parse(document_path, mime_type, produce_archive=produce_archive)
        parent_text = self.text
        try:
            self._process(document_path, mime_type)
        except Exception:
            self.text = parent_text
            self.log.exception(
                "vlm_ocr failed, keeping Tesseract/layer result",
            )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _process(self, document_path: Path, mime_type: str) -> None:
        cfg = self.cfg
        started = time.monotonic()

        orig_pdf = document_path if mime_type == "application/pdf" else None
        render_pdf = (
            self.archive_path if self.archive_path is not None else orig_pdf
        )
        pages_total = page_count(render_pdf) if render_pdf is not None else 1
        if pages_total > cfg.max_pages:
            self.log.warning(
                "vlm_ocr skipped: %d pages > max_pages=%d",
                pages_total,
                cfg.max_pages,
            )
            return

        sha256 = hashlib.sha256(document_path.read_bytes()).hexdigest()
        pages = [_Page(number=number) for number in range(1, pages_total + 1)]

        self._read_layers(pages, orig_pdf)
        self._reference_ocr(pages, orig_pdf)
        self._classify(pages)
        self._transcribe(pages, document_path, render_pdf)
        self._apply_text(pages)
        self._write_decisions(pages, pages_total, mime_type, sha256)

        elapsed = time.monotonic() - started
        self._log_summary(pages, elapsed)

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
                self.log.warning(
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
    ) -> None:
        if orig_pdf is None:
            return
        targets = [
            page for page in pages if page.layer_tokens >= self.cfg.min_tokens
        ]
        if not targets:
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
                    self.log.warning(
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
    ) -> None:
        targets = [page for page in pages if self._needs_vlm(page)]
        if not targets:
            return

        try:
            client: VlmClient | None = VlmClient(self.cfg)
        except Exception as exc:
            client = None
            error: str | None = type(exc).__name__
            self.log.warning(
                "vlm_ocr: cannot create VLM client (%s)",
                error,
            )
        if client is None:
            for page in targets:
                fallback = self._fallback_text(page, document_path, render_pdf)
                self._apply_vlm(page, None, fallback, error)
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
                ): page
                for page in targets
            }
            for future in as_completed(futures):
                page = futures[future]
                try:
                    result, fallback, failure = future.result()
                except Exception as exc:
                    result = None
                    fallback = ""
                    failure = type(exc).__name__
                self._apply_vlm(page, result, fallback, failure)

    def _vlm_page(
        self,
        client: VlmClient,
        document_path: Path,
        render_pdf: Path | None,
        page: _Page,
    ) -> tuple[VlmResult | None, str, str | None]:
        fallback = self._fallback_text(page, document_path, render_pdf)
        try:
            png = self._render_vlm_page(document_path, render_pdf, page.number)
            result = client.transcribe(png.read_bytes())
        except Exception as exc:
            return None, fallback, type(exc).__name__
        return result, fallback, None

    def _render_vlm_page(
        self,
        document_path: Path,
        render_pdf: Path | None,
        number: int,
    ) -> Path:
        out = self.tempdir / f"vlmocr-vlm-{number}.png"
        if render_pdf is not None:
            return render_png(
                render_pdf,
                number,
                out,
                scale_to=self.cfg.image_px,
                gray=False,
            )
        return image_to_png(document_path, out, self.cfg.image_px)

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
            self.log.warning(
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

    def _apply_vlm(
        self,
        page: _Page,
        result: VlmResult | None,
        fallback: str,
        error: str | None,
    ) -> None:
        if result is not None:
            page.vlm_seconds = result.seconds
            page.input_tokens = result.input_tokens
            page.output_tokens = result.output_tokens

        reason = error if result is None else reject_reason(result, fallback)
        if result is None or reason is not None:
            page.text = fallback
            page.fallback_reason = reason
            page.source = (
                "layer" if page.decision == "no_signal" else "tesseract"
            )
            return

        text = clean(result.text)
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

    # ------------------------------------------------------------------
    # Result assembly, decision log, summary
    # ------------------------------------------------------------------

    def _apply_text(self, pages: list[_Page]) -> None:
        if not any(page.source == "vlm" for page in pages):
            return
        self.text = post_process_text(
            "\n\n".join(page.text for page in pages),
        ) or ""

    def _write_decisions(
        self,
        pages: list[_Page],
        pages_total: int,
        mime_type: str,
        sha256: str,
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
                    "page": page.number,
                    "pages": pages_total,
                    "kind": page.kind,
                    "decision": page.decision,
                    "coverage": page.coverage,
                    "support": page.support,
                    "layer_tokens": page.layer_tokens,
                    "ref_tokens": page.ref_tokens,
                    "source": page.source,
                    "fallback_reason": page.fallback_reason,
                    "vlm_seconds": page.vlm_seconds,
                    "input_tokens": page.input_tokens,
                    "output_tokens": page.output_tokens,
                    "gate_mode": gate_mode,
                    "plugin_version": __version__,
                },
            )

    def _log_summary(self, pages: list[_Page], elapsed: float) -> None:
        keep = sum(1 for page in pages if page.source == "layer")
        vlm = sum(1 for page in pages if page.source == "vlm")
        fallback = sum(1 for page in pages if page.source == "tesseract")
        would_vlm = sum(1 for page in pages if page.decision == "would_vlm")
        no_signal = sum(1 for page in pages if page.decision == "no_signal")
        tokens_in = sum(page.input_tokens or 0 for page in pages)
        tokens_out = sum(page.output_tokens or 0 for page in pages)
        self.log.info(
            "vlm_ocr pages=%d keep=%d vlm=%d fallback=%d would_vlm=%d "
            "no_signal=%d tokens_in=%d tokens_out=%d seconds=%.1f",
            len(pages),
            keep,
            vlm,
            fallback,
            would_vlm,
            no_signal,
            tokens_in,
            tokens_out,
            elapsed,
        )
