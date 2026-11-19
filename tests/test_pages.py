"""Tiling helpers: tile count, seams, shared lines and boundary dedup."""

from __future__ import annotations

import pytest

from paperless_vlm_ocr.pages import join_tiles
from paperless_vlm_ocr.pages import least_ink_row
from paperless_vlm_ocr.pages import tile_count


def test_tile_count_keeps_normal_pages_whole() -> None:
    assert tile_count(1800, 2546) == 1  # A4, ratio 1.41
    assert tile_count(1000, 2500) == 1  # exactly 2.5


def test_tile_count_splits_tall_pages() -> None:
    assert tile_count(1000, 2501) == 2
    assert tile_count(1334, 7442) == 3  # receipt strip, ratio 5.58
    assert tile_count(1000, 6000) == 3  # exactly three 2:1 tiles
    assert tile_count(1000, 7000) == 4


def test_tile_count_handles_empty_dimensions() -> None:
    assert tile_count(0, 0) == 1
    assert tile_count(-1, 500) == 1
    assert tile_count(500, 0) == 1


def test_least_ink_row_picks_the_brightest_row() -> None:
    assert least_ink_row([10, 200, 30, 210, 50], 100) == 103
    assert least_ink_row([9, 9, 9], 7) == 7


def _line_runs(ink: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for y, value in enumerate(ink):
        if value > 12:
            if start is None:
                start = y
        elif start is not None:
            runs.append((start, y - 1))
            start = None
    if start is not None:
        runs.append((start, len(ink) - 1))
    return runs


def _seam_page(width: int = 200, height: int = 1200):
    """60 px text lines every 80 px, with 100 px blank zones around 400 and 800."""
    from PIL import Image
    from PIL import ImageDraw

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    for top in range(0, height, 80):
        if top + 59 >= 380 and top <= 420:
            continue
        if top + 59 >= 780 and top <= 820:
            continue
        draw.rectangle((10, top, width - 10, top + 59), fill=0)
    return img


def _tight_gap_page(width: int = 200, height: int = 1200):
    """Varying text bars with exactly an 8 px and a 30 px blank run at the seams."""
    from PIL import Image
    from PIL import ImageDraw

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    heights = [20, 35, 60, 45, 25, 55]
    y = 0
    index = 0
    while y < height:
        bar = heights[index % len(heights)]
        draw.rectangle(
            (10, y, width - 10, min(y + bar - 1, height - 1)),
            fill=0,
        )
        y += bar + 20
        index += 1
    draw.rectangle((0, 388, width, 395), fill=0)
    draw.rectangle((0, 396, width, 403), fill=255)
    draw.rectangle((0, 404, width, 411), fill=0)
    draw.rectangle((0, 777, width, 784), fill=0)
    draw.rectangle((0, 785, width, 814), fill=255)
    draw.rectangle((0, 815, width, 822), fill=0)
    return img


def test_tile_seams_accept_tight_gaps() -> None:
    pytest.importorskip("PIL.Image")
    from paperless_vlm_ocr.pages import row_ink
    from paperless_vlm_ocr.pages import tile_seams

    img = _tight_gap_page()
    seams = tile_seams(img, 3)
    ink = row_ink(img)
    assert seams == [400, 800]
    for row, length in zip(seams, (8, 30)):
        assert ink[row] == 0
        start = row
        while start > 0 and ink[start - 1] <= 12:
            start -= 1
        end = row
        while end + 1 < len(ink) and ink[end + 1] <= 12:
            end += 1
        assert end - start + 1 == length


def test_tile_seams_cut_inside_blank_runs() -> None:
    pytest.importorskip("PIL.Image")
    from paperless_vlm_ocr.pages import row_ink
    from paperless_vlm_ocr.pages import tile_seams

    img = _seam_page()
    seams = tile_seams(img, 3)
    ink = row_ink(img)
    assert seams == [400, 800]
    assert all(ink[row] == 0 for row in seams)


def test_tiles_share_exactly_one_complete_line() -> None:
    pytest.importorskip("PIL.Image")
    from paperless_vlm_ocr.pages import row_ink
    from paperless_vlm_ocr.pages import tile_bounds

    img = _seam_page()
    bounds = tile_bounds(img, [400, 800])
    lines = _line_runs(row_ink(img))
    assert bounds == [(0, 540), (400, 940), (800, 1200)]
    for line in lines:
        assert any(
            top <= line[0] and line[1] < bottom for top, bottom in bounds
        ), line
    for index in range(len(bounds) - 1):
        shared = [
            line
            for line in lines
            if bounds[index][0] <= line[0]
            and line[1] < bounds[index][1]
            and bounds[index + 1][0] <= line[0]
            and line[1] < bounds[index + 1][1]
        ]
        assert len(shared) == 1, (index, shared)


def test_tile_bounds_share_the_next_line_after_a_dense_cut() -> None:
    pytest.importorskip("PIL.Image")
    from paperless_vlm_ocr.pages import row_ink
    from paperless_vlm_ocr.pages import tile_bounds

    img = _seam_page()
    bounds = tile_bounds(img, [350, 800])
    lines = _line_runs(row_ink(img))
    # the cut is inside the 320-379 line, so that line is clipped; the next
    # complete line below it (480-539) is shared
    assert bounds[0] == (0, 540)
    shared = [
        line
        for line in lines
        if bounds[0][0] <= line[0]
        and line[1] < bounds[0][1]
        and bounds[1][0] <= line[0]
        and line[1] < bounds[1][1]
    ]
    assert shared == [(480, 539)]


def test_tile_seams_fall_back_to_the_least_ink_row() -> None:
    pytest.importorskip("PIL.Image")
    from PIL import Image
    from PIL import ImageDraw

    from paperless_vlm_ocr.pages import row_ink
    from paperless_vlm_ocr.pages import tile_seams

    img = Image.new("L", (200, 1200), 240)  # ink 15: never blank
    draw = ImageDraw.Draw(img)
    for y in range(50, 1200, 100):
        draw.rectangle((0, y, 199, y + 29), fill=100)
    seams = tile_seams(img, 3)
    ink = row_ink(img)
    reach = round(img.height / 3 * 0.06)
    assert len(seams) == 2
    for index, row in enumerate(seams, 1):
        nominal = round(index * img.height / 3)
        lo = max(1, nominal - reach)
        hi = min(img.height - 1, nominal + reach)
        assert ink[row] == min(ink[lo : hi + 1])


def test_join_tiles_drops_a_duplicated_boundary_line() -> None:
    first = "HORNBACH\nSumme EUR 173,20\nDanke"
    second = "Danke\nRückgabe bis 30.03.2025"
    assert join_tiles([first, second]) == (
        "HORNBACH\nSumme EUR 173,20\nDanke\nRückgabe bis 30.03.2025"
    )


def test_join_tiles_normalises_whitespace_when_comparing() -> None:
    first = "Summe EUR 173,20\nDanke   sehr"
    second = "Danke sehr\nRückgabe"
    assert join_tiles([first, second]) == (
        "Summe EUR 173,20\nDanke   sehr\nRückgabe"
    )


def test_join_tiles_drops_a_fuzzy_duplicate() -> None:
    first = "A\nK-U-N-D-E-N-B-E-L-E-G-"
    second = "K-U-N-D-E-N-B-E-L-E-G\nB"
    assert join_tiles([first, second]) == (
        "A\nK-U-N-D-E-N-B-E-L-E-G-\nB"
    )


def test_join_tiles_keeps_the_longer_prefix_variant() -> None:
    first = "A\nKARTENZAHLUNG"
    second = "KARTENZAHLUNG 30,17 EUR\nB"
    assert join_tiles([first, second]) == (
        "A\nKARTENZAHLUNG 30,17 EUR\nB"
    )


def test_join_tiles_drops_two_shared_lines() -> None:
    assert join_tiles(["X\nA\nB", "A\nB\nC"]) == "X\nA\nB\nC"


def test_join_tiles_never_drops_more_than_two_lines() -> None:
    # a three-line repeat is not guessed away: at most two lines may be dropped
    first = "X\nA\nB\nC"
    second = "A\nB\nC\nD"
    assert join_tiles([first, second]) == "X\nA\nB\nC\nA\nB\nC\nD"


def test_join_tiles_keeps_different_boundary_lines() -> None:
    assert join_tiles(["A\nEnde 1", "Anfang 2\nB"]) == "A\nEnde 1\nAnfang 2\nB"


def test_join_tiles_skips_empty_tiles() -> None:
    assert join_tiles([]) == ""
    assert join_tiles(["", "  \n", "A"]) == "A"


def test_join_tiles_drops_a_duplicate_that_empties_the_tile() -> None:
    assert join_tiles(["A\nB", "B"]) == "A\nB"
