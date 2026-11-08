"""
Document extraction module supporting PDF, DOCX, TXT, and Markdown.
Extracts text with page-level layout awareness, headings, and character spans.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import logging
import warnings
import re

# Suppress PyMuPDF legacy fitz deprecation notice
warnings.filterwarnings("ignore", category=FutureWarning, module="fitz")

logger = logging.getLogger(__name__)

from .layout_engine import (
    BaseLayoutEngine,
    PyMuPDFLayoutEngine,
    LayoutBlock,
    get_layout_engine,
    DEFAULT_EXCLUDED_LABELS,
    DEFAULT_EXCLUDED_SECTIONS,
    LABEL_SECTION_HEADER,
    LABEL_TEXT,
    LABEL_TABLE,
    LABEL_FOOTNOTE,
    LABEL_PAGE_HEADER,
    LABEL_PAGE_FOOTER,
    LABEL_REFERENCE,
    LABEL_ACKNOWLEDGMENT,
    LABEL_FIGURE,
    LABEL_CAPTION,
    LABEL_LIST_ITEM,
    LABEL_TITLE,
    is_footnote_or_citation_block,
)

# ---------------------------------------------------------------------------
# Unicode ligature decomposition table
# PDF typesetting often bakes fi/fl/ffi/ffl ligatures as single code points.
# Normalizing them ensures clean downstream search, tokenization, and CoT.
# ---------------------------------------------------------------------------
_LIGATURE_TABLE = str.maketrans({
    '\uFB00': 'ff',   # ﬀ  LATIN SMALL LIGATURE FF
    '\uFB01': 'fi',   # ﬁ  LATIN SMALL LIGATURE FI
    '\uFB02': 'fl',   # ﬂ  LATIN SMALL LIGATURE FL
    '\uFB03': 'ffi',  # ﬃ  LATIN SMALL LIGATURE FFI
    '\uFB04': 'ffl',  # ﬄ  LATIN SMALL LIGATURE FFL
    '\uFB05': 'st',   # ﬅ  LATIN SMALL LIGATURE LONG ST
    '\uFB06': 'st',   # ﬆ  LATIN SMALL LIGATURE ST
})


def _normalize_ligatures(text: str) -> str:
    """Decompose Unicode ligature characters to standard ASCII equivalents and clean invisible glyphs."""
    if not text:
        return ""
    t = text.translate(_LIGATURE_TABLE)
    t = t.replace('\xad', '').replace('\u00ad', '').replace('\u200b', '')
    return t


@dataclass
class DocumentBlock:
    block_type: str  # "paragraph", "heading", "table", "formula", "footnote", "frontmatter", "figure_content", "figure_caption", "references"
    text: str
    bbox: Optional[List[float]] = None
    header_level: int = 0
    reading_order: int = 0
    confidence: float = 1.0

@dataclass
class DocumentPage:
    page_number: int
    text: str
    blocks: List[DocumentBlock] = field(default_factory=list)

def is_frontmatter_block(text: str) -> bool:
    """
    Detect author names, affiliation lists, correspondence emails, and repository links
    typically found on paper title pages.
    """
    t = text.strip()
    t_lower = t.lower()

    if t_lower.startswith((
        "date:", "correspondence:", "contact:", "author for correspondence:",
        "code & data:", "code and data:", "model:", "github:", "huggingface:"
    )):
        return True
    if len(t.split()) < 15 and ("github.com/" in t_lower or "huggingface.co/" in t_lower):
        return True
    if re.search(r'\barxiv:\d+\.\d+', t_lower):
        return True

    author_symbols = ('∗', '†', '‡', '♠', '♡', '♣', '♢', '□', '♭', '\u2217', '\u2020', '\u2021', '\u2660', '\u2661', '\u2663', '\u2662', '\u25a1', '\u266d')
    has_author_sym = any(s in t for s in author_symbols)
    has_role_tag = any(r in t_lower for r in ("joint first author", "equal contribution", "corresponding author"))

    affil_words = (
        "university", "institute", "laboratory", "alliance for research", "alliance",
        "department", "school of", "faculty of", "college", "fair at meta",
        "meta ai", "google research", "deepmind", "microsoft research", "stanford",
        "allen institute", "massachusetts institute of technology", "mit", "berkeley",
        "singapore", "harvard", "oxford", "cambridge", "cmu"
    )
    has_affil = any(a in t_lower for a in affil_words)

    verbs = (
        " is ", " are ", " was ", " were ", " have ", " has ", " had ",
        " we ", " this paper ", " in this work ", " our ", " present ", " show ",
        " propose ", " investigate ", " demonstrate ", " develop ", " achieve ",
        " introduce ", " evaluate ", " explore ", " analyze ", " argue "
    )
    padded = f" {t_lower} "
    has_substantive_verb = any(v in padded for v in verbs)

    if has_role_tag:
        return True
    if has_author_sym and not has_substantive_verb:
        return True
    if has_affil and not has_substantive_verb:
        return True
    if has_author_sym and has_affil and len(t.split()) < 80:
        return True
    return False


def is_figure_debris(text: str) -> bool:
    """
    Detect OCR text boxes, axis ticks, chart labels, or diagram annotations inside vector figures.
    """
    t = text.strip()
    t_lower = t.lower()

    # Axis tick marks or sequence of numbers (e.g. "0 200 400 600", "64\n256\n10242048")
    nums = re.findall(r'\b\d+(?:\.\d+)?\b', t)
    words = re.findall(r'\b[a-zA-Z]{2,}\b', t)
    if len(nums) >= 3 and len(words) <= 2:
        return True
    if re.match(r'^\d+(?:\.\d+)?$', t):
        return True

    # Subfigure tags: (a) ..., (b) ..., (c) ...
    if re.match(r'^\([a-d]\)\s+[A-Za-z]', t) and len(t.split()) < 10:
        return True

    # Figure query token labels: e.g. "Varied-Length Query (167 tokens)", "Simple Query (13 tokens)"
    if re.search(r'\(\d+\s*tokens?\)', t_lower) and len(t.split()) < 10:
        return True

    # Diagram box labels
    diagram_labels = (
        "positive document", "hard negative document", "initial documents",
        "seed (positive)", "seed documents", "documents", "real-world web search",
        "query lengths", "difficulty measure"
    )
    if t_lower in diagram_labels:
        return True

    # Diagram syntax brackets: e.g. "{query, positive document}"
    if re.match(r'^\{[^}]+\}$', t):
        return True

    # Metric table rows in plots: e.g. "BM25 Error\n53.0\n37.3", "GRIT-7B Error\n42.3"
    if any(k in t_lower for k in ("error", "ndcg", "bm25", "grit")) and len(nums) >= 2 and len(words) <= 5:
        return True

    return False


def is_toc_block(text: str) -> bool:
    """
    Detect Table of Contents (TOC) entries, leader dot chains, or section outline indexes.
    """
    t = text.strip()
    if not t:
        return False
    # Leader dots followed or preceded by page numbers: e.g. ". . . . . 16" or "...... 25"
    if re.search(r'(?:\.\s*){5,}\s*\d*', t):
        return True
    # Table of contents section titles
    if t.lower() in ("contents", "table of contents", "appendix contents", "outline"):
        return True
    lines = [line.strip() for line in t.split("\n") if line.strip()]
    if not lines:
        return False
    toc_lines = 0
    for l in lines:
        if re.search(r'(?:\.\s*){5,}', l):
            toc_lines += 1
        elif re.match(r'^(?:[A-Z]|\d+)(?:\.\d+)*\s+.*?(?:\.|\s)+\d+$', l):
            toc_lines += 1
        elif re.match(r'^\d+\s+[A-Z](?:\.\d+)*\s+', l) and re.search(r'\d+$', l):
            toc_lines += 1
        elif re.search(r'\b(?:[A-Z]|\d+)\.\d+\s+[A-Za-z\s-]+\s+\d+\b', l):
            toc_lines += 1
    if toc_lines >= max(1, (len(lines) + 1) // 2):
        return True
    return False


def is_likely_heading(text: str) -> bool:
    t = text.strip()
    if not t or len(t) < 3 or len(t) > 120:
        return False
    t_lower = t.lower()

    # Reject frontmatter, figure diagram artifacts, and TOC lines
    if is_frontmatter_block(t) or is_figure_debris(t) or is_toc_block(t):
        return False
    if t_lower.startswith((
        "date:", "correspondence:", "contact:", "author for correspondence:",
        "code:", "code & data:", "code and data:", "model:", "github:", "huggingface:",
        "figure", "fig.", "table"
    )):
        return False
    if re.search(r'\(\d+\s*tokens?\)', t_lower):
        return False

    # Exclude copyright notices
    if "©" in t or "all rights reserved" in t_lower:
        return False
    # Exclude Table of contents lines with trailing page numbers
    if re.search(r'\n\s*\d+$', t):
        return False
    if t.lower() in ("contents", "table of contents"):
        return False
    # Exclude citation entries with publication years in parentheses
    if re.search(r'\(\d{4}[a-z]?\)', t):
        return False
    # Exclude page numbers, standalone digits or citation numbers
    if re.match(r'^\d+$', t) or re.match(r'^\[\s*\d+\s*\]$', t):
        return False
    # Exclude journal headers or running author headers / footers
    if any(j in t_lower for j in ("applied acoustics", "sciencedirect", "elsevier", "doi.org", "received", "available online")):
        return False
    if "et al" in t_lower or re.search(r'\bet\s+al\.?\b', t_lower):
        return False
    # Exclude table headers or labels
    if re.match(r'^Table\s+\d+', t, re.IGNORECASE):
        return False
    # Exclude author lines
    if any(k in t for k in ("@", "*")) and len(t.split()) < 10:
        return False

    # Exclude citation markers
    if any(k in t_lower for k in (", ed.", ", eds.", "eds.,", "university press", "pp. ", "vol. ", "doi:", "isbn", "issn")):
        return False

    # Exclude publisher addresses, copyright, and metadata
    pub_signals = (
        "road", "street", "avenue", "floor", "plaza", "cb2 8ea", "singapore",
        "australia", "new delhi", "cambridge university", "oxford university",
        "great clarendon", "united kingdom", "cataloging", "cataloguing",
        "library of congress", "control number",
        "issn", "isbn", "copyright", "rights reserved", "author for correspondence",
        "elements in ", "published online", "british library", "about the series",
        "series editor", "all rights reserved", "edited by "
    )
    if any(sig in t_lower for sig in pub_signals):
        return False

    # Real numbered section: e.g. "1 Introduction", "1. Introduction", "2.1. Terminology", "4.4.2. Social value"
    if re.match(r'^(?:\d+\.?|\d+\.\d+\.?|\d+\.\d+\.\d+\.?|\(\d+\))\s+[A-Z]', t):
        return True
    if re.match(r'^[A-Z]\.\s+[A-Z]', t) and not any(k in t_lower for k in (",", "press", "university", "journal", "in ")):
        return True

    # Standard academic headings
    if t.upper() in ("ABSTRACT", "A B S T R A C T", "INTRODUCTION", "METHODOLOGY", "RESULTS", "DISCUSSION", "CONCLUSION", "REFERENCES", "BIBLIOGRAPHY"):
        return True

    # If single-line and short, ensure it doesn't end with a preposition or conjunction
    if "\n" not in t and len(t) < 70 and not t.endswith("."):
        last_word = t.split()[-1].lower().rstrip(".,:;")
        if last_word in ("in", "of", "and", "to", "for", "with", "that", "their", "as", "or", "by", "on", "from", "at", "is", "are"):
            return False
        # Cannot start with a lowercase letter or quote mark
        if t[0].islower() or t[0] in ("‘", "“", "'", '"', "(", "["):
            return False
        # If very short (1 or 2 words), must be standard heading or numbered
        if len(t.split()) < 3:
            return False
        # Check if words look title-cased
        words = [w for w in t.split() if w.isalpha()]
        capitalized = sum(1 for w in words if w[0].isupper())
        if capitalized >= len(words) * 0.6:
            if any(k in t_lower for k in ("university", "press", "college", "edited by")):
                return False
            return True
    return False


class DocumentParser:
    def __init__(
        self,
        layout_engine: Optional[BaseLayoutEngine] = None,
        excluded_labels: Optional[Set[str]] = None,
        excluded_sections: Optional[Set[str]] = None,
    ):
        self.layout_engine = layout_engine or get_layout_engine()
        self.excluded_labels = excluded_labels
        self.excluded_sections = excluded_sections

    def parse_file(
        self,
        file_path: str,
        excluded_labels: Optional[Set[str]] = None,
        excluded_sections: Optional[Set[str]] = None,
    ) -> List[DocumentPage]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ex_labels = excluded_labels if excluded_labels is not None else self.excluded_labels
        ex_secs = excluded_sections if excluded_sections is not None else self.excluded_sections

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf(path, excluded_labels=ex_labels, excluded_sections=ex_secs)
        elif suffix == ".docx":
            return self._parse_docx(path)
        elif suffix == ".doc":
            raise ValueError(f"'{path.name}' is a legacy Word .doc file, which cannot be read here. "
                             "Save it as .docx or PDF and upload that instead.")
        elif suffix in (".txt", ".md", ".jsonl"):
            return self._parse_plain(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _parse_pdf(
        self,
        path: Path,
        excluded_labels: Optional[Set[str]] = None,
        excluded_sections: Optional[Set[str]] = None,
    ) -> List[DocumentPage]:
        pages_dict: Dict[int, List[DocumentBlock]] = {}
        max_page = 1

        try:
            # 1. Primary Layout Analysis Engine
            layout_blocks = self.layout_engine.extract_blocks(
                str(path),
                excluded_labels=excluded_labels,
                excluded_sections=excluded_sections,
            )

            # Map layout blocks to DocumentBlocks grouped by page
            for lb in layout_blocks:
                p_num = lb.page_number
                if p_num > max_page:
                    max_page = p_num

                # Map semantic layout labels to standard DocumentBlock types
                if lb.label == LABEL_SECTION_HEADER:
                    b_type = "heading"
                elif lb.label == LABEL_TITLE:
                    b_type = "heading" if p_num > 1 else "paragraph"
                elif lb.label == LABEL_TABLE:
                    b_type = "table"
                elif lb.label in (LABEL_PAGE_HEADER, LABEL_PAGE_FOOTER, LABEL_FOOTNOTE, LABEL_REFERENCE, LABEL_ACKNOWLEDGMENT, LABEL_FIGURE, LABEL_CAPTION):
                    b_type = lb.label
                elif is_footnote_or_citation_block(lb.text, y0=lb.bbox[1] if lb.bbox else None):
                    b_type = "footnote"
                else:
                    b_type = "paragraph"

                doc_block = DocumentBlock(
                    block_type=b_type,
                    text=_normalize_ligatures(lb.text).strip(),
                    bbox=lb.bbox,
                    header_level=lb.header_level,
                    reading_order=lb.reading_order,
                    confidence=lb.confidence,
                )
                pages_dict.setdefault(p_num, []).append(doc_block)

            # Ensure all pages exist even if blank/fully excluded
            import pymupdf
            with pymupdf.open(str(path)) as doc:
                total_doc_pages = len(doc)

            result_pages: List[DocumentPage] = []
            for p_num in range(1, total_doc_pages + 1):
                p_blocks = pages_dict.get(p_num, [])
                page_text = "\n\n".join(b.text for b in p_blocks)
                result_pages.append(
                    DocumentPage(
                        page_number=p_num,
                        text=page_text,
                        blocks=p_blocks
                    )
                )
            return result_pages

        except Exception as e:
            logger.warning(f"Layout engine extraction failed ({e}). Falling back to native PyMuPDF block parser.")

        # Fallback to PyMuPDF raw block parsing
        pages = []
        try:
            import pymupdf
            doc = pymupdf.open(str(path))
            in_references = False
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                rect = page.rect
                page_text = _normalize_ligatures(page.get_text("text"))
                raw_blocks = page.get_text("blocks")
                blocks = []
                for b in raw_blocks:
                    x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                    t_strip = _normalize_ligatures(text).strip()
                    if not t_strip or t_strip == '\x00':
                        continue
                    if y0 <= 48 or y1 >= rect.height - 35:
                        continue
                    if is_footnote_or_citation_block(t_strip, y0=y0, page_height=rect.height):
                        continue
                    b_type = "heading" if is_likely_heading(t_strip) else "paragraph"
                    blocks.append(DocumentBlock(block_type=b_type, text=t_strip, bbox=[x0, y0, x1, y1]))
                pages.append(DocumentPage(page_number=page_idx + 1, text=page_text, blocks=blocks))
            doc.close()
            return pages
        except Exception:
            pass

        # Fallback to pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                for idx, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    blocks = [DocumentBlock(block_type="paragraph", text=p.strip()) for p in text.split("\n\n") if p.strip()]
                    pages.append(DocumentPage(page_number=idx + 1, text=text, blocks=blocks))
            return pages
        except ImportError:
            raise ImportError("PyMuPDF or pdfplumber is required to parse PDF documents.")

    def _parse_docx(self, path: Path) -> List[DocumentPage]:
        try:
            import docx
            doc = docx.Document(str(path))
            text_lines = []
            blocks = []
            # paragraphs and tables in document order (doc.paragraphs alone silently drops every table)
            from docx.table import Table
            from docx.text.paragraph import Paragraph
            for child in doc.element.body.iterchildren():
                tag = child.tag.rsplit("}", 1)[-1]
                if tag == "p":
                    p = Paragraph(child, doc)
                    text = p.text.strip()
                    if text:
                        text_lines.append(text)
                        style = (p.style.name if p.style is not None else "") or ""
                        b_type = "heading" if style.startswith(("Heading", "Title")) else "paragraph"
                        blocks.append(DocumentBlock(block_type=b_type, text=text))
                elif tag == "tbl":
                    rows = []
                    for row in Table(child, doc).rows:
                        cells = []
                        for c in row.cells:
                            ct = " ".join(c.text.split())
                            if not cells or cells[-1] != ct:      # merged cells repeat their text
                                cells.append(ct)
                        if any(cells):
                            rows.append(" | ".join(cells))
                    if rows:
                        text = "\n".join(rows)
                        text_lines.append(text)
                        blocks.append(DocumentBlock(block_type="table", text=text))
            # DOCX has no fixed pages; everything is reported as page 1
            return [DocumentPage(page_number=1, text="\n\n".join(text_lines), blocks=blocks)]
        except ImportError:
            raise ImportError(
                "python-docx is required to parse DOCX documents. "
                "Please install it via: pip install python-docx"
            )
        except Exception as e:
            logger.error(f"Failed to read docx with python-docx: {e}")
            raise ValueError(f"Could not parse DOCX file '{path.name}': {e}")

    def _parse_plain(self, path: Path) -> List[DocumentPage]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        blocks = [DocumentBlock(block_type="paragraph", text=p) for p in paragraphs]
        return [DocumentPage(page_number=1, text=content, blocks=blocks)]
