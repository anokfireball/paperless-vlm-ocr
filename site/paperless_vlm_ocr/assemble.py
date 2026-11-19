"""Document text assembly policy: pure, no Django, no I/O."""

from __future__ import annotations

from collections.abc import Iterable


def needs_reassembly(sources: Iterable[str]) -> bool:
    """True iff any page's text came from the model or from Tesseract.

    A document whose pages all kept their born-digital layer must keep the
    parent parser's text bit-identical; any replaced or recovered page means
    the plugin owns the assembled text.
    """
    return any(source in ("vlm", "tesseract") for source in sources)
