"""Poppler/Tesseract/Pillow helpers for single pages."""

from __future__ import annotations

import math
import os
import re
import subprocess
from collections.abc import Callable
from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

_TIMEOUT_S = 180

# Tall-page tiling: split above this height/width ratio into ~2:1 tiles. Seams
# go inside a blank run of at least _MIN_GAP_PX rows; the previous tile keeps
# the first complete line below the seam, so one whole line is shared.
_TILE_RATIO = 2.5
_TILE_ASPECT = 2.0
_CUT_SEARCH = 0.06
_BLANK_INK = 12
_MIN_GAP_PX = 8
_MAX_RENDER_PX = 10000

# Seam dedup: at most two lines per seam, fuzzy match threshold.
_MAX_SEAM_LINES = 2
_FUZZY_RATIO = 0.85


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


def load_image(path: Path) -> Image.Image:
    """RGB page raster with EXIF orientation applied."""
    from PIL import Image
    from PIL import ImageOps

    with Image.open(path) as im:
        im.seek(0)
        return ImageOps.exif_transpose(im).convert("RGB")


def image_to_png(image: Path, out: Path, long_side: int) -> Path:
    from PIL import Image

    img = load_image(image)
    if max(img.size) > long_side:
        scale = long_side / max(img.size)
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(size, Image.Resampling.LANCZOS)
    img.save(out, format="PNG")
    return out


def tile_count(width: int, height: int) -> int:
    """Tiles for a page: 1 unless height/width > 2.5, else ceil((h/w)/2)."""
    if width <= 0 or height <= 0:
        return 1
    if height <= _TILE_RATIO * width:
        return 1
    return max(1, math.ceil(height / width / _TILE_ASPECT))


def least_ink_row(brightness: Sequence[int], top: int) -> int:
    """Row with the most light pixels in one boundary band.

    ``brightness`` holds one mean grey value (0-255, higher is brighter) per
    row starting at row ``top``. Ties keep the first row.
    """
    best = 0
    for index in range(1, len(brightness)):
        if brightness[index] > brightness[best]:
            best = index
    return top + best


def row_ink(img: Image.Image) -> list[int]:
    """Mean ink per row (0-255, higher is darker)."""
    from PIL import Image as PilImage

    gray = img.convert("L")
    rows = gray.resize((1, gray.height), PilImage.Resampling.BOX)
    return [255 - value for value in rows.tobytes()]


def _runs(
    values: Sequence[int],
    is_member: Callable[[int], bool],
) -> list[tuple[int, int]]:
    """Maximal inclusive ``[start, end]`` runs accepted by ``is_member``."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if is_member(value):
            if start is None:
                start = index
        elif start is not None:
            out.append((start, index - 1))
            start = None
    if start is not None:
        out.append((start, len(values) - 1))
    return out


def tile_seams(img: Image.Image, tiles: int) -> list[int]:
    """Cut rows inside a blank run near each nominal tile boundary.

    The run must be at least ``_MIN_GAP_PX`` rows and the cut is the row nearest
    the nominal boundary. Without a blank run in the search band the least-inked
    row is used; ``split_tiles`` still shares one complete line across the seam.
    """
    ink = row_ink(img)
    gaps = [
        (start, end)
        for start, end in _runs(ink, lambda value: value <= _BLANK_INK)
        if end - start + 1 >= _MIN_GAP_PX
    ]
    reach = max(1, round(img.height / tiles * _CUT_SEARCH))
    rows: list[int] = []
    for index in range(1, tiles):
        nominal = round(index * img.height / tiles)
        near = [
            gap
            for gap in gaps
            if gap[1] >= nominal - reach and gap[0] <= nominal + reach
        ]
        if near:
            gap = min(
                near,
                key=lambda item: abs(
                    min(max(nominal, item[0]), item[1]) - nominal
                ),
            )
            rows.append(min(max(nominal, gap[0]), gap[1]))
        else:
            lo = max(1, nominal - reach)
            hi = min(img.height - 1, nominal + reach)
            rows.append(
                least_ink_row(
                    [255 - value for value in ink[lo : hi + 1]],
                    lo,
                )
            )
    return rows


def tile_bounds(img: Image.Image, cuts: Sequence[int]) -> list[tuple[int, int]]:
    """Row bounds of the tiles: the next tile starts on the cut row and the
    previous tile keeps the first complete ink run below it.

    Exactly one whole text line is therefore shared across each seam.
    """
    ink = row_ink(img)
    lines = _runs(ink, lambda value: value > _BLANK_INK)
    edges = [0, *cuts, img.height]
    bounds: list[tuple[int, int]] = []
    for index in range(len(cuts) + 1):
        top = edges[index]
        bottom = edges[index + 1]
        if index < len(cuts):
            below = [line for line in lines if line[0] > bottom]
            if below:
                bottom = below[0][1] + 1
        bounds.append((top, bottom))
    return bounds


def split_tiles(img: Image.Image, cuts: Sequence[int]) -> list[Image.Image]:
    """Cut ``img`` at ``cuts`` sharing one complete line across each seam."""
    return [
        img.crop((0, top, img.width, bottom))
        for top, bottom in tile_bounds(img, cuts)
    ]


def _trim_blank(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _normalise_line(line: str) -> str:
    return " ".join(line.split())


def _first_nonempty(lines: Sequence[str], limit: int) -> list[int]:
    out: list[int] = []
    for index, line in enumerate(lines):
        if line.strip():
            out.append(index)
            if len(out) >= limit:
                break
    return out


def _last_nonempty(lines: Sequence[str], limit: int) -> list[int]:
    out: list[int] = []
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip():
            out.append(index)
            if len(out) >= limit:
                break
    return out


def _similar(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if left.startswith(right) or right.startswith(left):
        # one tile read only part of the line
        return True
    return SequenceMatcher(None, left, right).ratio() >= _FUZZY_RATIO


def join_tiles(texts: Sequence[str]) -> str:
    """Join tile transcriptions, dropping the line(s) shared across each seam.

    The trailing up-to-two non-empty lines of the text so far are matched, in
    order, against the leading up-to-two non-empty lines of the next tile:
    exact after whitespace normalisation, or fuzzy (ratio >= 0.85). The matched
    leading lines are dropped; the previous tile's version wins unless the next
    tile has the strictly longer line and the previous one is its prefix.
    """
    joined: list[str] = []
    for text in texts:
        lines = _trim_blank(text.splitlines())
        if not lines:
            continue
        if joined:
            tail = _last_nonempty(joined, _MAX_SEAM_LINES)  # newest first
            head = _first_nonempty(lines, _MAX_SEAM_LINES)  # first first
            tail_lines = [_normalise_line(joined[index]) for index in tail]
            head_lines = [_normalise_line(lines[index]) for index in head]
            # compare the tail suffix with the head prefix, longest first
            tail_oldest = list(reversed(tail_lines))
            size = 0
            for count in range(
                min(len(tail_oldest), len(head_lines), _MAX_SEAM_LINES),
                0,
                -1,
            ):
                pairs = zip(tail_oldest[-count:], head_lines[:count])
                if all(_similar(left, right) for left, right in pairs):
                    size = count
                    break
            if size:
                tail_indices = list(reversed(tail))  # oldest first
                for offset in range(size):
                    left = tail_oldest[-size + offset]
                    right = head_lines[offset]
                    if len(right) > len(left) and right.startswith(left):
                        joined[tail_indices[-size + offset]] = lines[head[offset]]
                drop = set(head[:size])
                lines = [
                    line
                    for index, line in enumerate(lines)
                    if index not in drop
                ]
                lines = _trim_blank(lines)
        if lines:
            joined.extend(lines)
    return "\n".join(joined).strip()


def render_page_tiles(
    source: Path,
    out_dir: Path,
    number: int,
    *,
    image_px: int,
) -> list[Path]:
    """Rasterise one page once and cut tall pages into seam-aligned tiles.

    The page is scaled so that each tile's long side is about ``image_px``
    (full page height ~= tiles * image_px, capped at 10000 px). A page with
    height/width <= 2.5 stays one image. The returned list is never empty.
    """
    from PIL import Image

    img = load_image(source)
    tiles = tile_count(img.width, img.height)
    target = image_px if tiles == 1 else min(tiles * image_px, _MAX_RENDER_PX)
    if max(img.size) > target:
        scale = target / max(img.size)
        img = img.resize(
            (
                max(1, round(img.width * scale)),
                max(1, round(img.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    if tiles == 1:
        out = out_dir / f"vlmocr-vlm-{number}.png"
        img.save(out, format="PNG")
        return [out]
    paths: list[Path] = []
    for index, tile in enumerate(split_tiles(img, tile_seams(img, tiles))):
        out = out_dir / f"vlmocr-vlm-{number}-t{index}.png"
        tile.save(out, format="PNG")
        paths.append(out)
    return paths
