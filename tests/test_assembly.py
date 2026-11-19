"""Document text assembly policy: layer-only documents keep the parent text."""

from __future__ import annotations

from paperless_vlm_ocr.assemble import needs_reassembly


def test_layer_only_document_keeps_the_parent_text() -> None:
    assert not needs_reassembly(["layer", "layer", "layer"])


def test_tesseract_page_forces_reassembly() -> None:
    assert needs_reassembly(["layer", "tesseract"])


def test_vlm_page_forces_reassembly() -> None:
    assert needs_reassembly(["vlm"])


def test_empty_document_keeps_the_parent_text() -> None:
    assert not needs_reassembly([])
