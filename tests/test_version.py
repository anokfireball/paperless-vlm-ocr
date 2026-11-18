"""Version bookkeeping: one verified Paperless release, three plugin version places."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from paperless_vlm_ocr import SUPPORTED_PAPERLESS
from paperless_vlm_ocr import VERIFIED_PAPERLESS
from paperless_vlm_ocr import __version__

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
INIT = SITE / "paperless_vlm_ocr" / "__init__.py"
ANNOTATION = "# renovate: datasource=docker depName=ghcr.io/paperless-ngx/paperless-ngx"


def require_repository() -> None:
    """Layout tests read the checkout; inside the Paperless image only site/ is mounted."""
    if not INIT.is_file():
        pytest.skip("version layout test needs the repository checkout")


def test_version_places_agree() -> None:
    require_repository()
    dist_dirs = sorted(SITE.glob("paperless_vlm_ocr-*.dist-info"))
    expected = f"paperless_vlm_ocr-{__version__}.dist-info"
    assert [path.name for path in dist_dirs] == [expected]

    metadata = (dist_dirs[0] / "METADATA").read_text(encoding="utf-8")
    assert re.findall(r"^Version: (.+)$", metadata, re.MULTILINE) == [__version__]


def test_annotation_sits_above_verified_paperless() -> None:
    require_repository()
    lines = INIT.read_text(encoding="utf-8").splitlines()
    anchors = [
        index
        for index, line in enumerate(lines)
        if line.startswith("VERIFIED_PAPERLESS = ")
    ]
    assert len(anchors) == 1, anchors
    assert lines[anchors[0] - 1] == ANNOTATION
    assert re.fullmatch(r'VERIFIED_PAPERLESS = "\d+\.\d+\.\d+"', lines[anchors[0]])


def test_verified_paperless_is_supported() -> None:
    parts = tuple(int(part) for part in VERIFIED_PAPERLESS.split("."))
    assert len(parts) == 3
    assert parts[:2] in SUPPORTED_PAPERLESS


def test_supported_paperless_lists_minors() -> None:
    assert SUPPORTED_PAPERLESS
    for entry in SUPPORTED_PAPERLESS:
        assert isinstance(entry, tuple) and len(entry) == 2
        assert all(isinstance(part, int) for part in entry)
