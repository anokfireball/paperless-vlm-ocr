"""Smoke parse inside the Paperless image; CI-only, needs Django and the OCR tools.

Run from /usr/src/paperless/src with the plugin tree on PYTHONPATH after
``manage.py migrate``. It proves the parent-parser contract on the Paperless
version under test without a real vision model: the plugin is selected, a
one-page image-only PDF is parsed through it, the page is routed to the model
(the endpoint refuses connections), and the Tesseract fallback text and the
archive survive.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "paperless.settings")

import django

django.setup()

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from paperless.parsers.registry import get_parser_registry
from paperless_vlm_ocr.parser import VlmOcrParser

PDF = Path("/tmp/smoke-parse.pdf")
DECISIONS = Path(os.environ["PAPERLESS_VLM_OCR_DECISION_LOG"])
LINES = ("PAPERLESS VLM OCR", "SMOKE TEST PAGE", "KONTOINHABER", "IBAN DE89 3704 0044")


def write_scan(path: Path) -> None:
    """One A4 page of large text saved as an image-only PDF: no text layer."""
    page = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=72)
    top = 220
    for line in LINES:
        draw.text((110, top), line, fill="black", font=font)
        top += 180
    page.save(path, "PDF", resolution=150)


def main() -> None:
    write_scan(PDF)

    selected = get_parser_registry().get_parser_for_file(
        "application/pdf",
        "x.pdf",
        None,
        allow_remote=True,
    )
    assert selected is VlmOcrParser, selected

    with VlmOcrParser() as parser:
        parser.parse(PDF, "application/pdf", produce_archive=True)
        text = parser.get_text()
        archive = parser.get_archive_path()
        assert text.strip(), "parser produced no text"
        assert archive is not None and Path(archive).is_file(), archive

    records = [
        json.loads(line)
        for line in DECISIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1, records
    record = records[0]
    assert record["page"] == 1, record
    assert record["kind"] == "empty_layer", record
    assert record["source"] == "tesseract", record
    assert record["fallback_reason"], record

    print(
        "smoke parse ok: one page, "
        f"{len(text)} chars, archive present, "
        f"source={record['source']} fallback={record['fallback_reason']}"
    )


main()
