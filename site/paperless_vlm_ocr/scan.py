"""Offline text-layer gate scan: one decision record per PDF page.

No Django and no VLM here: the gate is always evaluated in enforce mode so the
report shows which pages would be routed to the vision model.

Usage:
    python3 -m paperless_vlm_ocr.scan --lang deu+eng --jobs 8 \
        --threshold 0.9 --min-tokens 15 --out results.jsonl PDF...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

from paperless_vlm_ocr.config import VlmOcrConfig
from paperless_vlm_ocr.gate import classify
from paperless_vlm_ocr.gate import tokenize
from paperless_vlm_ocr.pages import layer_text
from paperless_vlm_ocr.pages import page_count
from paperless_vlm_ocr.pages import render_png
from paperless_vlm_ocr.pages import tesseract_text

_PROGRESS_EVERY = 500


def _error(file: str, page: int | None, exc: BaseException) -> dict:
    return {
        "file": file,
        "page": page,
        "decision": "error",
        "error": f"{type(exc).__name__}: {exc}",
    }


def scan_page(file: str, page: int, lang: str, threshold: float, min_tokens: int) -> dict:
    """Classify one page: born-digital layer against a Tesseract read of it."""
    cfg = VlmOcrConfig(
        gate_mode="enforce",
        threshold=threshold,
        min_tokens=min_tokens,
        decision_log=Path(os.devnull),
    )
    try:
        path = Path(file)
        layer = layer_text(path, page)
        ref: str | None = None
        if len(tokenize(layer)) >= cfg.min_tokens:
            with tempfile.TemporaryDirectory() as tmp:
                png = render_png(path, page, Path(tmp) / "page.png", scale_to=None, gray=True)
                ref = tesseract_text(png, lang)
    except Exception as exc:  # a page we cannot read is reported, never fatal
        return _error(file, page, exc)

    verdict = classify(layer, ref, cfg)
    decision = "empty_layer" if verdict.kind == "empty_layer" else verdict.decision
    return {
        "file": file,
        "page": page,
        "layer_tokens": verdict.layer_tokens,
        "ref_tokens": verdict.ref_tokens,
        "coverage": verdict.coverage,
        "support": verdict.support,
        "decision": decision,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Text-layer gate scan for PDF pages.")
    parser.add_argument("--lang", default="deu+eng", help="Tesseract language(s)")
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--threshold", type=float, default=0.9, help="agreement threshold")
    parser.add_argument("--min-tokens", type=int, default=15, help="minimum token count")
    parser.add_argument("--out", default="results.jsonl", help="output JSONL path")
    parser.add_argument("pdfs", nargs="+", help="PDF files")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    jobs = max(1, args.jobs)
    pages_done = 0

    with out_path.open("w", encoding="utf-8") as out:
        def emit(record: dict) -> None:
            nonlocal pages_done
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            pages_done += 1
            if pages_done % _PROGRESS_EVERY == 0:
                print(f"scanned {pages_done} pages", file=sys.stderr, flush=True)

        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {}
            for pdf in args.pdfs:
                file = str(pdf)
                try:
                    count = page_count(Path(pdf))
                except Exception as exc:
                    emit(_error(file, None, exc))
                    continue
                for page in range(1, count + 1):
                    future = pool.submit(
                        scan_page,
                        file,
                        page,
                        args.lang,
                        args.threshold,
                        args.min_tokens,
                    )
                    futures[future] = (file, page)
            for future in as_completed(futures):
                file, page = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # worker died (MemoryError, crash, ...)
                    record = _error(file, page, exc)
                emit(record)

    print(f"scanned {pages_done} pages into {out_path}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
