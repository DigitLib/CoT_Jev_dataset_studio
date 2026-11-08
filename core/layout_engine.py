"""
Layout analysis engine supporting deep vision layout models (Surya)
and fast local vector layout analysis (PyMuPDF Layout).
Segments documents into semantic blocks (Text, Section-header, Table, Reference, Acknowledgment, etc.)
and renders visual bounding box previews.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import logging
import json
import re

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Standard semantic block types
LABEL_TEXT = "text"
LABEL_SECTION_HEADER = "section-header"
LABEL_TITLE = "title"
LABEL_LIST_ITEM = "list-item"
LABEL_TABLE = "table"
LABEL_FOOTNOTE = "footnote"
LABEL_PAGE_HEADER = "page-header"
LABEL_PAGE_FOOTER = "page-footer"
LABEL_REFERENCE = "reference"
LABEL_ACKNOWLEDGMENT = "acknowledgment"
LABEL_FIGURE = "figure"
LABEL_CAPTION = "caption"

# Standard visual noise labels to exclude from chunking by default
DEFAULT_EXCLUDED_LABELS: Set[str] = {
    LABEL_PAGE_HEADER,
    LABEL_PAGE_FOOTER,
    LABEL_FOOTNOTE,
    LABEL_REFERENCE,
    LABEL_ACKNOWLEDGMENT,
    LABEL_FIGURE,
    LABEL_CAPTION,
}

# Standard sections to skip entirely once encountered
DEFAULT_EXCLUDED_SECTIONS: Set[str] = {
    "acknowledgements",
    "acknowledgments",
    "funding",
    "authors",
    "details",
    "conflicts of interest",
    "references",
    "bibliography",
    "literature cited",
    "works cited",
    "table of contents",
    "contents",
    "preface",
    "index",
}

# Color palette for bounding box visualization
LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    LABEL_TEXT: (34, 139, 34),             # Forest Green
    LABEL_LIST_ITEM: (46, 139, 87),        # Sea Green
    LABEL_SECTION_HEADER: (30, 144, 255),  # Dodger Blue
    LABEL_TITLE: (75, 0, 130),             # Indigo
    LABEL_TABLE: (218, 165, 32),           # Goldenrod
    LABEL_FOOTNOTE: (220, 20, 60),         # Crimson (excluded)
    LABEL_PAGE_HEADER: (178, 34, 34),      # Firebrick (excluded)
    LABEL_PAGE_FOOTER: (178, 34, 34),      # Firebrick (excluded)
    LABEL_REFERENCE: (139, 0, 0),          # Dark Red (excluded)
    LABEL_ACKNOWLEDGMENT: (255, 69, 0),    # Red-Orange (excluded)
    LABEL_FIGURE: (128, 128, 128),         # Gray (excluded)
    LABEL_CAPTION: (112, 128, 144),        # Slate Gray (excluded)
}

CITATION_REGEX = re.compile(
    r"[“\"][^”\"]+[”\"];\s*[A-Z]|"  # "Title"; Author e.g. Frankfurt, "Title"; Dworkin
    r"\b(?:pages?|pp?\.?)\s*\d+|"   # p. 12, pp. 21-46, page 4
    r"[A-Z][a-z]+,\s*\d+\s*[–\-]\s*\d+|"  # Oshana, 21–46
    r",\s*\d+\s*[–\-]\s*\d+\.?(?:\s|$)|"  # , 21–46.
    r"\b(?:et\s+al\.|ibid\.|op\.\s*cit\.|passim)\b|"
    r"\b(?:isbn|issn|doi)\b|"
    r"\b(?:university\s+press|press|routledge|springer|cambridge|oxford|mit\s+press|harvard)\b,\s*\d{4}|"
    r"^[A-Z][a-z]+,\s*[“\"][^”\"]+[”\"]|"
    r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\x27s|\u2019s)\s+(?:account|view|theory|work|paper|book|analysis),\s*which\b|"
    r"^\s*(?:see|cf\.|compare)\s+(?:also\s+)?(?:\d+\s+)?[A-Z][a-z]+.*?(?:(?:19|20)\d{2}|[“\"][^”\"]+[”\"]|sec\.\s*\d+)",
    re.IGNORECASE,
)

FOOTNOTE_START_REGEX = re.compile(
    r"^\s*(?:\d{1,3}|[ivxlcdm]{1,4}|[∗†‡§¶#])\s+[A-Z“\"'\(]|"
    r"^\s*(?:see|cf\.|compare)\s+(?:also\s+)?(?:\d+\s+)?[A-Z][a-z]+.*?(?:(?:19|20)\d{2}|[“\"][^”\"]+[”\"]|sec\.\s*\d+)",
    re.IGNORECASE,
)


def is_footnote_or_citation_block(
    text: str,
    y0: Optional[float] = None,
    min_footnote_y0: Optional[float] = None,
    page_height: Optional[float] = None,
) -> bool:
    """
    Detect whether a document block is an academic footnote, citation entry,
    or bibliographical debris based on syntactic pattern and vertical layout position.
    """
    text_clean = text.strip()
    if not text_clean:
        return False
    is_fn_start = bool(FOOTNOTE_START_REGEX.search(text_clean))
    has_cit = bool(CITATION_REGEX.search(text_clean))

    in_fn_zone = False
    if y0 is not None:
        if min_footnote_y0 is not None and y0 >= min_footnote_y0 - 25:
            in_fn_zone = True
        elif page_height and y0 >= 0.65 * page_height:
            in_fn_zone = True

    # Vertical position protection: Body text located in the upper 65% of the page
    # cannot be classified as a footnote unless containing explicit formal publishing metadata.
    if y0 is not None and page_height and not in_fn_zone:
        if any(k in text_clean.lower() for k in ("isbn", "issn", "doi.org/10.", "available as open access", "all rights reserved")):
            return True
        return False

    if has_cit:
        return True
    if in_fn_zone and is_fn_start:
        return True
    if is_fn_start and (has_cit or len(text_clean.split()) < 30):
        return True
    return False


_NOISE_OUTLINE_RE = re.compile(
    r"^(?:(?:chapter|part|section)?\s*[0-9ivxlc]+[.:)]?\s+)?"
    r"(?:cover|front\s*cover|back\s*cover|half[\s-]?title(?:\s+page)?|title\s+page|copyright(?:\s+page)?|"
    r"front\s*matter|back\s*matter|(?:table\s+of\s+)?contents|brief\s+contents|"
    r"acknowledge?ments?|dedication|references|bibliography|works\s+cited|index|"
    r"(?:notes\s+on\s+)?contributors|about\s+the\s+authors?|list\s+of\s+(?:figures|tables|abbreviations))"
    r"\s*[.:]?\s*$",
    re.IGNORECASE)


def is_noise_outline_title(title: str) -> bool:
    """True for outline entries that are front/back matter as a whole ("Contents", "3 References", "Index")."""
    return bool(_NOISE_OUTLINE_RE.match((title or "").strip()))


@dataclass
class LayoutBlock:
    block_id: str
    page_number: int
    label: str  # e.g. "text", "section-header", "table", "reference", etc.
    bbox: List[float]  # [x0, y0, x1, y1] in PDF coordinate space
    text: str
    confidence: float = 1.0
    header_level: int = 0
    reading_order: int = 0

    def is_noise(self, excluded_labels: Optional[Set[str]] = None) -> bool:
        ex = excluded_labels if excluded_labels is not None else DEFAULT_EXCLUDED_LABELS
        return self.label.lower() in ex


class BaseLayoutEngine(ABC):
    """Abstract interface for document layout analysis engines."""

    @abstractmethod
    def extract_blocks(
        self,
        file_path: str,
        pages: Optional[List[int]] = None,
        excluded_labels: Optional[Set[str]] = None,
        excluded_sections: Optional[Set[str]] = None,
    ) -> List[LayoutBlock]:
        """Extract structured layout blocks from a document file."""
        pass

    @abstractmethod
    def render_page_overlay(
        self,
        file_path: str,
        page_number: int,
        excluded_labels: Optional[Set[str]] = None,
        dpi: int = 150,
    ) -> Image.Image:
        """Render page as image with color-coded bounding boxes overlaid."""
        pass


class PyMuPDFLayoutEngine(BaseLayoutEngine):
    """
    Native layout analysis engine using PyMuPDF and pymupdf4llm.
    High throughput, runs 100% locally on CPU without neural model overhead.
    Accurately classifies section headers, body paragraphs, tables, lists, headers, footers, and footnotes.
    """

    def __init__(self):
        # A missing / too old pymupdf4llm must not stop the app: extract_blocks() raises, and DocumentParser
        # falls back to plain PyMuPDF block parsing for that file (with a warning).
        try:
            import pymupdf4llm
            self.pymupdf4llm = pymupdf4llm
        except ImportError:
            self.pymupdf4llm = None

    def _require_llm4(self):
        if self.pymupdf4llm is None:
            raise ImportError("pymupdf4llm is not installed (pip install pymupdf4llm); layout analysis unavailable")
        if not hasattr(self.pymupdf4llm, "to_json"):
            raise ImportError(f"pymupdf4llm {getattr(self.pymupdf4llm, '__version__', '?')} has no to_json(); "
                              "upgrade with: pip install -U pymupdf4llm")

    def extract_blocks(
        self,
        file_path: str,
        pages: Optional[List[int]] = None,
        excluded_labels: Optional[Set[str]] = None,
        excluded_sections: Optional[Set[str]] = None,
    ) -> List[LayoutBlock]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ex_labels = {l.lower() for l in (excluded_labels if excluded_labels is not None else DEFAULT_EXCLUDED_LABELS)}
        ex_secs = {s.lower() for s in (excluded_sections if excluded_sections is not None else DEFAULT_EXCLUDED_SECTIONS)}

        self._require_llm4()
        import pymupdf
        doc_stem = path.stem.lower().replace("_", " ").replace("-", " ")
        with pymupdf.open(str(path)) as doc:            # closed again: an open handle locks the file on Windows
            n_pages = len(doc)
            page_heights = [float(p.rect.height) for p in doc]
            try:
                toc = doc.get_toc()
            except Exception as e:
                logger.debug(f"TOC extraction notice: {e}")
                toc = []

        # Frontmatter / backmatter pages from the PDF outline. Whole titles only: a chapter called
        # "Discovering ..." or "Indexical reference" must not be dropped because it contains "cover" / "index".
        toc_excluded_pages: Set[int] = set()
        for i, (lvl, title, p_start) in enumerate(toc or []):
            if is_noise_outline_title(title):
                p_end = toc[i + 1][2] - 1 if i + 1 < len(toc) else n_pages
                for p in range(p_start, p_end + 1):
                    toc_excluded_pages.add(p)

        # Parse with pymupdf4llm JSON breakdown
        j_str = self.pymupdf4llm.to_json(str(path), pages=pages)
        parsed = json.loads(j_str)

        clean_blocks: List[LayoutBlock] = []
        skipping_section = False
        reading_idx = 0

        for page in parsed.get("pages", []):
            p_num = page.get("page_number", 1)

            # Skip whole page if explicitly mapped as frontmatter/cover/acknowledgments in TOC
            if p_num in toc_excluded_pages:
                continue

            page_height = page_heights[p_num - 1] if 1 <= p_num <= n_pages else None

            # Identify footnote zone on this page
            fn_y0s = [
                float(b.get("y0", 0.0))
                for b in page.get("boxes", [])
                if "footnote" in (b.get("boxclass") or "").lower() and b.get("y0") is not None
            ]
            min_footnote_y0 = min(fn_y0s) if fn_y0s else None

            for b in page.get("boxes", []):
                raw_class = (b.get("boxclass") or "text").lower()
                lines = b.get("textlines") or []
                text = " ".join(
                    s.get("text", "")
                    for tl in lines
                    for s in (tl.get("spans") or [])
                ).strip()

                if not text:
                    # Check if table block without textlines
                    if b.get("table") and isinstance(b["table"], dict):
                        text = self._format_table_dict(b["table"])
                        raw_class = LABEL_TABLE
                    else:
                        continue

                y0 = float(b.get("y0", 0.0)) if b.get("y0") is not None else None

                # Normalize label with footnote & academic citation detection
                norm_label = self._normalize_label(
                    raw_class, text, y0=y0, min_footnote_y0=min_footnote_y0, page_height=page_height
                )

                # Reclassify book/doc title on preliminary pages
                if norm_label == LABEL_SECTION_HEADER and p_num <= 6:
                    sec_clean = text.lower().strip(": .#0123456789")
                    if sec_clean == doc_stem or doc_stem in sec_clean or sec_clean in doc_stem:
                        norm_label = LABEL_TITLE

                # Detect Section-header boundaries and skip blacklisted sections
                if norm_label == LABEL_SECTION_HEADER:
                    sec_clean = text.lower().strip(": .#0123456789")
                    if any(sec_clean == es or sec_clean.startswith(es) for es in ex_secs):
                        skipping_section = True
                        continue
                    else:
                        skipping_section = False

                if skipping_section:
                    continue

                # Drop inline acknowledgment text blocks
                t_low = text.lower()
                if t_low.startswith(("acknowledgements:", "acknowledgments:", "funding:")):
                    continue

                if norm_label in ex_labels:
                    continue

                bbox = [
                    float(b.get("x0", 0.0)),
                    float(b.get("y0", 0.0)),
                    float(b.get("x1", 0.0)),
                    float(b.get("y1", 0.0)),
                ]

                clean_blocks.append(
                    LayoutBlock(
                        block_id=f"p{p_num}_b{reading_idx}",
                        page_number=p_num,
                        label=norm_label,
                        bbox=bbox,
                        text=text,
                        confidence=1.0,
                        header_level=int(b.get("header_level", 0)),
                        reading_order=reading_idx,
                    )
                )
                reading_idx += 1

        return clean_blocks

    def _normalize_label(
        self,
        raw_class: str,
        text: str,
        y0: Optional[float] = None,
        min_footnote_y0: Optional[float] = None,
        page_height: Optional[float] = None,
    ) -> str:
        rc = raw_class.lower()
        if any(k in rc for k in ("image", "figure", "picture", "graphic", "drawing", "illustration", "diagram")):
            return LABEL_FIGURE
        if "header" in rc and "section" not in rc:
            return LABEL_PAGE_HEADER
        if "footer" in rc:
            return LABEL_PAGE_FOOTER
        if "footnote" in rc:
            return LABEL_FOOTNOTE

        # Academic citation and footnote debris filtering on layout detection
        if is_footnote_or_citation_block(text, y0=y0, min_footnote_y0=min_footnote_y0, page_height=page_height):
            return LABEL_FOOTNOTE

        if "section-header" in rc or "heading" in rc:
            return LABEL_SECTION_HEADER
        if "title" in rc:
            return LABEL_TITLE
        if "list" in rc:
            return LABEL_LIST_ITEM
        if "table" in rc:
            return LABEL_TABLE

        t_low = text.lower().strip()

        # Author biography signals (frontmatter / contributor blurbs)
        author_bio_signals = (
            "is an associate professor", "is an assistant professor", "is a professor",
            "is a visiting scholar", "is a postdoctoral", "is a researcher", "is an instructor",
            "division of the humanities", "center for law, society", "department of philosophy at",
            "received their phd", "ph.d. in", "contributor biographies", "about the authors",
            "about the author", "notes on contributors"
        )
        if any(sig in t_low for sig in author_bio_signals) and len(text.split()) < 120:
            return LABEL_ACKNOWLEDGMENT

        # Publisher / Open Access imprint signals
        pub_open_access_signals = (
            "available as open access", "open access on cambridge core", "cambridge core",
            "oxford academic", "springerlink", "downloaded from https://", "ip address:",
            "university printing house", "first published 20", "all rights reserved",
            "information on this title:"
        )
        if any(sig in t_low for sig in pub_open_access_signals) and len(text.split()) < 80:
            return LABEL_PAGE_FOOTER

        # Section-like text inspections
        if t_low.startswith(("references", "bibliography", "works cited")) and len(text.split()) < 5:
            return LABEL_SECTION_HEADER
        if t_low.startswith(("acknowledgements", "acknowledgments")):
            return LABEL_ACKNOWLEDGMENT

        return LABEL_TEXT

    def _format_table_dict(self, tbl: Dict[str, Any]) -> str:
        rows = tbl.get("rows", [])
        if not rows:
            return ""
        lines = []
        for r in rows:
            cells = [str(c).strip() for c in r]
            lines.append(" | ".join(cells))
        return "\n".join(lines)

    def render_page_overlay(
        self,
        file_path: str,
        page_number: int,
        excluded_labels: Optional[Set[str]] = None,
        dpi: int = 150,
    ) -> Image.Image:
        self._require_llm4()
        import pymupdf
        with pymupdf.open(file_path) as doc:
            if page_number < 1 or page_number > len(doc):
                raise IndexError(f"Page number {page_number} out of range (1..{len(doc)})")
            page_idx = page_number - 1
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("RGBA")
            page_w, page_h = float(page.rect.width), float(page.rect.height)

        # Get raw layout boxes for this page
        j_str = self.pymupdf4llm.to_json(file_path, pages=[page_idx])
        parsed = json.loads(j_str)

        scale_x = pix.width / page_w
        scale_y = pix.height / page_h

        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        ex_labels = {l.lower() for l in (excluded_labels if excluded_labels is not None else DEFAULT_EXCLUDED_LABELS)}

        for p_data in parsed.get("pages", []):
            fn_y0s = [
                float(b.get("y0", 0.0))
                for b in p_data.get("boxes", [])
                if "footnote" in (b.get("boxclass") or "").lower() and b.get("y0") is not None
            ]
            min_footnote_y0 = min(fn_y0s) if fn_y0s else None

            for b in p_data.get("boxes", []):
                raw_class = (b.get("boxclass") or "text").lower()
                lines = b.get("textlines") or []
                text = " ".join(
                    s.get("text", "")
                    for tl in lines
                    for s in (tl.get("spans") or [])
                ).strip()
                y0_raw = float(b.get("y0", 0.0)) if b.get("y0") is not None else None
                norm_label = self._normalize_label(
                    raw_class, text, y0=y0_raw, min_footnote_y0=min_footnote_y0, page_height=page_h
                )

                x0 = b.get("x0", 0.0) * scale_x
                y0 = b.get("y0", 0.0) * scale_y
                x1 = b.get("x1", 0.0) * scale_x
                y1 = b.get("y1", 0.0) * scale_y

                color = LABEL_COLORS.get(norm_label, (100, 100, 100))
                is_excluded = norm_label in ex_labels

                if is_excluded:
                    # Draw semi-transparent reddish/strikethrough box
                    draw.rectangle([x0, y0, x1, y1], outline=(220, 20, 60, 200), width=2)
                    draw.rectangle([x0, y0, x1, y1], fill=(220, 20, 60, 40))
                    # Strikethrough diagonal
                    draw.line([x0, y0, x1, y1], fill=(220, 20, 60, 160), width=1)
                else:
                    # Draw solid colored box with soft tint
                    draw.rectangle([x0, y0, x1, y1], outline=(*color, 240), width=2)
                    draw.rectangle([x0, y0, x1, y1], fill=(*color, 35))

                # Label tag banner
                tag_text = f"{norm_label.upper()}"
                if is_excluded:
                    tag_text += " [EXCLUDED]"
                draw.rectangle([x0, max(0, y0 - 16), x0 + len(tag_text) * 7 + 8, y0], fill=(*color, 220))
                draw.text((x0 + 4, max(0, y0 - 15)), tag_text, fill=(255, 255, 255, 255))

        # Alpha composite overlay
        final_img = Image.alpha_composite(img, overlay).convert("RGB")
        return final_img


class SuryaLayoutEngine(BaseLayoutEngine):
    """
    Vision-based layout analysis model powered by Surya LayoutPredictor.
    Applies deep computer vision to detect text regions, columns, headers, and bibliography.
    """

    def __init__(self):
        try:
            from surya.layout import LayoutPredictor
            self.predictor = LayoutPredictor()
            self._is_available = True
        except ImportError:
            logger.warning(
                "surya-ocr not installed. SuryaLayoutEngine will delegate to PyMuPDFLayoutEngine fallback."
            )
            self._is_available = False
            self._fallback = PyMuPDFLayoutEngine()

    def extract_blocks(
        self,
        file_path: str,
        pages: Optional[List[int]] = None,
        excluded_labels: Optional[Set[str]] = None,
        excluded_sections: Optional[Set[str]] = None,
    ) -> List[LayoutBlock]:
        if not self._is_available:
            return self._fallback.extract_blocks(file_path, pages, excluded_labels, excluded_sections)

        import pymupdf
        doc = pymupdf.open(file_path)
        ex_labels = {l.lower() for l in (excluded_labels if excluded_labels is not None else DEFAULT_EXCLUDED_LABELS)}
        ex_secs = {s.lower() for s in (excluded_sections if excluded_sections is not None else DEFAULT_EXCLUDED_SECTIONS)}

        target_pages = pages if pages is not None else list(range(len(doc)))
        clean_blocks: List[LayoutBlock] = []
        skipping_section = False
        reading_idx = 0

        for page_idx in target_pages:
            if page_idx >= len(doc):
                continue
            page = doc[page_idx]
            p_num = page_idx + 1

            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            layout_results = self.predictor([img])[0]

            scale_x = page.rect.width / pix.width
            scale_y = page.rect.height / pix.height

            for box in layout_results.bboxes:
                label = box.label.lower()
                bbox = box.bbox  # [x1, y1, x2, y2]
                pdf_rect = pymupdf.Rect(
                    bbox[0] * scale_x, bbox[1] * scale_y,
                    bbox[2] * scale_x, bbox[3] * scale_y
                )
                text = page.get_text("text", clip=pdf_rect).strip()
                if not text:
                    continue

                if label == "section-header":
                    sec_clean = text.lower().strip(": .#0123456789")
                    if any(sec_clean == es or sec_clean.startswith(es) for es in ex_secs):
                        skipping_section = True
                        continue
                    else:
                        skipping_section = False

                if label not in ("section-header", "title", "table"):
                    if is_footnote_or_citation_block(text, y0=pdf_rect.y0, page_height=page.rect.height):
                        label = LABEL_FOOTNOTE

                if skipping_section or label in ex_labels:
                    continue

                clean_blocks.append(
                    LayoutBlock(
                        block_id=f"p{p_num}_b{reading_idx}",
                        page_number=p_num,
                        label=label,
                        bbox=[pdf_rect.x0, pdf_rect.y0, pdf_rect.x1, pdf_rect.y1],
                        text=text,
                        confidence=getattr(box, "confidence", 1.0),
                        reading_order=reading_idx,
                    )
                )
                reading_idx += 1

        doc.close()
        return clean_blocks

    def render_page_overlay(
        self,
        file_path: str,
        page_number: int,
        excluded_labels: Optional[Set[str]] = None,
        dpi: int = 150,
    ) -> Image.Image:
        if not self._is_available:
            return self._fallback.render_page_overlay(file_path, page_number, excluded_labels, dpi)

        import pymupdf
        with pymupdf.open(file_path) as doc:
            page = doc[page_number - 1]
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("RGBA")

        layout_results = self.predictor([img.convert("RGB")])[0]

        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        ex_labels = {l.lower() for l in (excluded_labels if excluded_labels is not None else DEFAULT_EXCLUDED_LABELS)}

        for box in layout_results.bboxes:
            label = box.label.lower()
            x0, y0, x1, y1 = box.bbox
            color = LABEL_COLORS.get(label, (100, 100, 100))
            is_excluded = label in ex_labels

            if is_excluded:
                draw.rectangle([x0, y0, x1, y1], outline=(220, 20, 60, 200), width=2)
                draw.rectangle([x0, y0, x1, y1], fill=(220, 20, 60, 40))
                draw.line([x0, y0, x1, y1], fill=(220, 20, 60, 160), width=1)
            else:
                draw.rectangle([x0, y0, x1, y1], outline=(*color, 240), width=2)
                draw.rectangle([x0, y0, x1, y1], fill=(*color, 35))

            tag_text = label.upper()
            if is_excluded:
                tag_text += " [EXCLUDED]"
            draw.rectangle([x0, max(0, y0 - 16), x0 + len(tag_text) * 7 + 8, y0], fill=(*color, 220))
            draw.text((x0 + 4, max(0, y0 - 15)), tag_text, fill=(255, 255, 255, 255))

        return Image.alpha_composite(img, overlay).convert("RGB")


def get_layout_engine(backend: str = "auto") -> BaseLayoutEngine:
    """Factory function for layout analysis engines."""
    if backend == "surya":
        try:
            import surya
            return SuryaLayoutEngine()
        except ImportError:
            logger.info("Surya not installed; falling back to PyMuPDFLayoutEngine.")
            return PyMuPDFLayoutEngine()
    return PyMuPDFLayoutEngine()

