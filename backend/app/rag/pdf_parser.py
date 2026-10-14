"""Project-owned PDF parser for RAG-Anything.

RAG-Anything 1.4.1 drives MinerU through the legacy ``mineru -p <pdf> -o <dir> -m auto``
command line. MinerU 4.x ships a different, document-library oriented CLI
(``mineru parse <path>``) that rejects those flags and requires a separately
started local server plus downloaded models, so every PDF parse fails with
``MineruExecutionError`` (return code 2) before any content is produced.

This module implements the same RAG-Anything content-list contract on top of
PyMuPDF, which is already a project dependency, and recovers the parts of the
corpus that have no text layer with the bundled offline RapidOCR models:

* ``text``   - layout-aware text blocks in reading order, minus running headers
               and footers, and minus text already captured inside a table.
* ``table``  - ruled tables recovered as row lists, captioned by the text above.
* ``image``  - embedded figures, captioned by nearby text. Full-page scans are
               skipped because they duplicate the page text they sit behind.

Nothing here calls a network service or an LLM. Registering it makes
``RAG_PARSER=pymupdf`` work end to end through ``process_document_complete``.
"""
from __future__ import annotations

import logging
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

PARSER_NAME = "pymupdf"

# A page holding less usable text than this is treated as scanned and sent to OCR.
OCR_MIN_PAGE_CHARS = 200
OCR_DPI = 200
# Repeated margin text (running headers/footers) is dropped from the body text.
RUNNING_TEXT_MIN_PAGES = 4
RUNNING_TEXT_PAGE_SHARE = 0.4
MARGIN_BAND = 0.09
# Figure filters: smaller than this in either axis is an icon, not a figure.
MIN_FIGURE_PIXELS = 150
FULL_PAGE_FIGURE_SHARE = 0.75
# Tables below these bounds are ruled fragments, not real tables.
MIN_TABLE_ROWS = 2
MIN_TABLE_CELLS = 4
MIN_TABLE_FILL = 0.4
MAX_CAPTION_CHARS = 240
CAPTION_GAP = 44

_OCR_ENGINE = None
_OCR_LOCK = threading.Lock()


def _ocr_engine():
    """Build the bundled PP-OCRv4 engine once; it never downloads at call time."""
    global _OCR_ENGINE
    with _OCR_LOCK:
        if _OCR_ENGINE is None:
            from rapidocr_onnxruntime import RapidOCR

            _OCR_ENGINE = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1,
                                    use_cls=False, det_limit_side_len=960, det_limit_type="max")
    return _OCR_ENGINE


def _clean(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_running_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _ocr_blocks(page) -> list[dict[str, Any]]:
    """Read an image-only page, returning positioned text blocks.

    OCR lines keep their rendered positions so a figure on a scanned page can be
    captioned by the label beside it, exactly as on a page with a text layer.
    """
    engine = _ocr_engine()
    with tempfile.TemporaryDirectory(prefix="raganything_ocr_") as folder:
        image_path = Path(folder) / "page.png"
        page.get_pixmap(dpi=OCR_DPI).save(image_path)
        result, _ = engine(str(image_path))
    if not result:
        return []
    scale = 72.0 / OCR_DPI
    lines = []
    for entry in result:
        try:
            polygon, text, score = entry[0], entry[1], entry[2]
        except (TypeError, IndexError):
            continue
        if float(score) < 0.5:
            continue
        text = _clean(str(text))
        if not text:
            continue
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        lines.append({
            "bbox": (min(xs) * scale, min(ys) * scale, max(xs) * scale, max(ys) * scale),
            "text": text,
            "top": round(min(ys) * scale, 1),
            "left": round(min(xs) * scale, 1),
        })
    lines.sort(key=lambda line: (line["top"], line["left"]))
    return [{"bbox": line["bbox"], "text": line["text"]} for line in lines]


def _running_texts(pages: Iterable[Any]) -> set[str]:
    """Collect margin lines that repeat across the document."""
    counts: dict[str, int] = {}
    total = 0
    for page in pages:
        total += 1
        rect = page.rect
        band = rect.height * MARGIN_BAND
        seen: set[str] = set()
        for block in page.get_text("blocks"):
            if not (block[1] <= rect.y0 + band or block[3] >= rect.y1 - band):
                continue
            for line in str(block[4]).splitlines():
                key = _normalize_running_text(line)
                if len(key) >= 3:
                    seen.add(key)
        for key in seen:
            counts[key] = counts.get(key, 0) + 1
    if not total:
        return set()
    threshold = max(RUNNING_TEXT_MIN_PAGES, int(total * RUNNING_TEXT_PAGE_SHARE))
    return {key for key, count in counts.items() if count >= threshold}


def _is_running_only(block: dict, running: set[str]) -> bool:
    """True when a margin block contains nothing but repeated running text."""
    lines = [_normalize_running_text(line) for line in block["text"].splitlines()]
    lines = [line for line in lines if line]
    return bool(lines) and all(line in running for line in lines)


def _table_markdown(rows: list[list[Any]]) -> str:
    rendered = [
        "| " + " | ".join(re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row) + " |"
        for row in rows
    ]
    if not rendered:
        return ""
    width = max(len(row) for row in rows)
    rendered.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(rendered)


def _overlaps(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> bool:
    return (min(inner[2], outer[2]) - max(inner[0], outer[0]) > 2
            and min(inner[3], outer[3]) - max(inner[1], outer[1]) > 2)


def _caption_quality(text: str) -> bool:
    """Reject rule artefacts and other glyph noise that OCR picks up from lines."""
    letters = sum(character.isalnum() for character in text)
    visible = sum(not character.isspace() for character in text)
    return bool(visible) and letters / visible >= 0.4 and len(text) >= 3


def _caption(blocks: list[dict], bbox) -> tuple[str, str]:
    """Caption above/below and label beside a figure or table.

    Handbooks lay out side-by-side figures with the label to the right, so all
    four edges are searched and the closest plausible text wins.
    """
    x0, y0, x1, y1 = bbox
    best: tuple[float, str] | None = None

    def consider(gap: float, text: str) -> None:
        nonlocal best
        text = text.strip()
        if gap < 0 or gap > CAPTION_GAP or len(text) > MAX_CAPTION_CHARS:
            return
        if not _caption_quality(text):
            return
        if best is None or gap < best[0]:
            best = (gap, text)

    for block in blocks:
        bx0, by0, bx1, by1 = block["bbox"]
        if bx0 >= bx1 or by0 >= by1:
            continue
        text = block["text"]
        if min(x1, bx1) - max(x0, bx0) > 0:
            consider(by1 - y0, text)
            consider(y0 - by1, text)
        if min(y1, by1) - max(y0, by0) > 0:
            consider(bx0 - x1, text)
            consider(x0 - bx1, text)
    return best[1] if best else ""


def _save_figure(document, page, xref: int, destination: Path) -> Path | None:
    import pymupdf

    target = destination / f"p{page.number + 1:04d}_x{xref}.png"
    try:
        pixmap = pymupdf.Pixmap(document, xref)
        if pixmap.n - pixmap.alpha >= 4:
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
        destination.mkdir(parents=True, exist_ok=True)
        pixmap.save(target)
        return target
    except Exception:
        logger.debug("figure_xref=%s could not be extracted", xref, exc_info=True)
        return None


def _page_has_ink(page) -> bool:
    """A page with no images and no vector drawing is blank; OCR cannot help."""
    if page.get_images(full=True):
        return True
    return bool(page.get_drawings())


def _page_tables(page) -> list[dict[str, Any]]:
    try:
        detections = page.find_tables().tables
    except Exception:
        logger.debug("page=%s table detection failed", page.number + 1, exc_info=True)
        return []
    tables = []
    for detection in detections:
        try:
            rows = [list(row) for row in detection.extract()]
        except Exception:
            continue
        cells = sum(len(row) for row in rows)
        filled = sum(1 for row in rows for cell in row if str(cell or "").strip())
        if len(rows) < MIN_TABLE_ROWS or filled < MIN_TABLE_CELLS:
            continue
        if cells and filled < MIN_TABLE_FILL * cells:
            # A ruled grid around photographs yields rows of empty cells; that is
            # page furniture, not tabular data, and it would only cost a model call.
            logger.debug("page=%s dropped table with %s/%s filled cells",
                         page.number + 1, filled, cells)
            continue
        tables.append({"bbox": tuple(detection.bbox), "rows": rows})
    return tables


def _page_figures(page, min_pixels: int) -> list[dict[str, Any]]:
    import pymupdf

    page_area = abs(page.rect.get_area()) or 1.0
    figures = []
    for info in page.get_image_info(xrefs=True):
        width, height = info.get("width", 0), info.get("height", 0)
        if width < min_pixels or height < min_pixels:
            continue
        bbox = tuple(info["bbox"])
        # A page scan repeats the page text; indexing it twice would only add an
        # image of text the page already contributes.
        if abs(pymupdf.Rect(bbox).get_area()) / page_area >= FULL_PAGE_FIGURE_SHARE:
            continue
        xref = int(info.get("xref") or 0)
        if xref:
            figures.append({"bbox": bbox, "xref": xref})
    return figures


def parse_pdf(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    ocr_min_page_chars: int = OCR_MIN_PAGE_CHARS,
    min_figure_pixels: int = MIN_FIGURE_PIXELS,
    extract_figures: bool = True,
    extract_tables: bool = True,
) -> list[dict[str, Any]]:
    """Return a RAG-Anything content list for ``pdf_path``."""
    import pymupdf

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    figure_dir = Path(output_dir) / "images" if output_dir is not None else None

    content: list[dict[str, Any]] = []
    with pymupdf.open(pdf_path) as document:
        running = _running_texts(document)
        emitted_xrefs: set[int] = set()
        for page in document:
            page_number = page.number
            blocks = [
                {"bbox": tuple(block[:4]), "text": _clean(str(block[4]))}
                for block in page.get_text("blocks")
                if str(block[4]).strip()
            ]

            tables = _page_tables(page) if extract_tables else []
            figures = _page_figures(page, min_figure_pixels) if extract_figures else []
            captured = [table["bbox"] for table in tables]
            body = [block for block in blocks
                    if not any(_overlaps(block["bbox"], bbox) for bbox in captured)]
            page_text = "\n\n".join(
                block["text"] for block in body
                if not (
                    (block["bbox"][1] <= page.rect.height * MARGIN_BAND
                     or block["bbox"][3] >= page.rect.height * (1 - MARGIN_BAND))
                    and _is_running_only(block, running)
                )
            ).strip()
            if len(page_text) < ocr_min_page_chars and _page_has_ink(page):
                # Thin or absent text layer over real page ink: this page is a scan,
                # so read it with OCR rather than dropping it from the corpus. OCR
                # lines join ``body`` so figures on the scan can be captioned too.
                scanned = _ocr_blocks(page)
                page_text = "\n\n".join(part for part in (page_text,
                                                          "\n\n".join(block["text"] for block in scanned))
                                        if part)
                body.extend(block for block in scanned
                            if not any(_overlaps(block["bbox"], bbox) for bbox in captured))

            ordered: list[tuple[float, int, dict[str, Any]]] = []
            if page_text:
                ordered.append((page.rect.y0, 0,
                                {"type": "text", "text": page_text, "page_idx": page_number}))
            for table in tables:
                above = _caption(body, table["bbox"])
                ordered.append((table["bbox"][1], 1, {
                    "type": "table",
                    "table_data": table["rows"],
                    "table_body": _table_markdown(table["rows"]),
                    "table_caption": [above] if above else [],
                    "page_idx": page_number,
                }))
            for figure in figures:
                xref = figure["xref"]
                if xref in emitted_xrefs or figure_dir is None:
                    continue
                path = _save_figure(document, page, xref, figure_dir)
                if path is None:
                    continue
                emitted_xrefs.add(xref)
                ordered.append((figure["bbox"][1], 1, {
                    "type": "image",
                    "img_path": str(path.resolve()),
                    "image_caption": [above] if (above := _caption(body, figure["bbox"])) else [],
                    "page_idx": page_number,
                }))
            ordered.sort(key=lambda item: (item[0], item[1]))
            content.extend(block for _, _, block in ordered)
            logger.debug("raganything_pdf_page page=%s text=%s tables=%s figures=%s",
                         page_number + 1, bool(page_text), len(tables), len(figures))

    if not content:
        raise ValueError(f"No text, table or figure content could be extracted from {pdf_path}")
    return content


_PARSER_CLASS = None


def _relevant_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only the options this parser understands.

    RAG-Anything forwards its own parsing options (``parse_method``, MinerU
    backend/source selectors) straight through, so unknown keys are dropped
    rather than crashing the call.
    """
    allowed = {"ocr_min_page_chars", "min_figure_pixels", "extract_figures", "extract_tables"}
    return {key: value for key, value in kwargs.items() if key in allowed}


class _ParserBase:
    """Shared option handling for the ``raganything`` parser entry points."""

    def parse_pdf(self, pdf_path, output_dir=None, method: str = "auto", **kwargs):
        return parse_pdf(pdf_path, output_dir, **_relevant_kwargs(kwargs))


def build_parser_class():
    """Return the ``raganything``-compatible parser class, building it on demand.

    ``raganything.parser.register_parser`` insists on a real subclass of its base
    ``Parser``. Importing that base here, rather than at module scope, keeps the
    parsing functions above usable (and testable) without RAG-Anything installed.
    """
    global _PARSER_CLASS
    if _PARSER_CLASS is None:
        from raganything.parser import Parser

        class PyMuPDFLayoutParser(_ParserBase, Parser):
            """``raganything`` parser protocol implemented over PyMuPDF."""

            name = PARSER_NAME

            def check_installation(self) -> bool:
                try:
                    import pymupdf  # noqa: F401

                    return True
                except ImportError:
                    return False

            def parse_image(self, image_path, output_dir=None, **kwargs):
                raise NotImplementedError("pymupdf parser handles PDF documents only")

            def parse_office_doc(self, doc_path, output_dir=None, **kwargs):
                raise NotImplementedError("pymupdf parser handles PDF documents only")

            def parse_document(self, file_path, method: str = "auto", output_dir=None, **kwargs):
                path = Path(file_path)
                if path.suffix.lower() == ".pdf":
                    return self.parse_pdf(path, output_dir, method=method, **kwargs)
                raise NotImplementedError(
                    f"pymupdf parser does not handle {path.suffix or 'extensionless'} files"
                )

        _PARSER_CLASS = PyMuPDFLayoutParser
    return _PARSER_CLASS