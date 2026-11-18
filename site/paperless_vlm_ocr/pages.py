"""Poppler/Tesseract/Pillow helpers for single pages."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_TIMEOUT_S = 180


def _post_process(text: str | None) -> str | None:
    """Same normalisation as paperless.parsers.utils.post_process_text."""
    if not text:
        return None
    collapsed_spaces = re.sub(r"([^\S\r\n]+)", " ", text)
    no_leading_whitespace = re.sub(r"([\n\r]+)([^\S\n\r]+)", "\\1", collapsed_spaces)
    no_trailing_whitespace = re.sub(r"([^\S\n\r]+)(?=\n?\Z)", "", no_leading_whitespace)
    result = no_trailing_whitespace.strip().replace("\0", " ")
    return result or None


def _run(cmd: list[str], env: dict[str, str] | None = None) -> bytes:
    return subprocess.run(
        cmd,
        capture_output=True,
        timeout=_TIMEOUT_S,
        check=True,
        env=env,
    ).stdout


def page_count(pdf: Path) -> int:
    out = _run(["pdfinfo", str(pdf)]).decode("utf-8", "replace")
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"pdfinfo reported no page count for {pdf}")


def layer_text(pdf: Path, n: int) -> str:
    out = _run(
        ["pdftotext", "-q", "-layout", "-enc", "UTF-8", "-f", str(n), "-l", str(n), str(pdf), "-"],
    )
    return _post_process(out.decode("utf-8", "replace")) or ""


def render_png(pdf: Path, n: int, out: Path, *, scale_to: int | None, gray: bool) -> Path:
    stem = out.with_suffix("")
    cmd = ["pdftoppm", "-r", "300", "-png", "-singlefile", "-f", str(n), "-l", str(n)]
    if gray:
        cmd.append("-gray")
    if scale_to:
        cmd += ["-scale-to", str(scale_to)]
    cmd += [str(pdf), str(stem)]
    _run(cmd)
    return stem.with_suffix(".png")


def tesseract_text(png: Path, lang: str) -> str:
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    out = _run(["tesseract", str(png), "stdout", "-l", lang], env=env)
    return out.decode("utf-8", "replace")


def image_to_png(image: Path, out: Path, long_side: int) -> Path:
    from PIL import Image
    from PIL import ImageOps

    with Image.open(image) as im:
        im.seek(0)
        img = ImageOps.exif_transpose(im).convert("RGB")
    if max(img.size) > long_side:
        scale = long_side / max(img.size)
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(size, Image.Resampling.LANCZOS)
    img.save(out, format="PNG")
    return out
