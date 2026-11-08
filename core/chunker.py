"""
Document chunking module with provenance tracking.
Segments documents into paragraph seeds, sections, and semantic knowledge units.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Tuple
import uuid
import hashlib
import re
import logging
import numpy as np

from .document_parser import DocumentPage, is_likely_heading, is_toc_block
from .layout_engine import is_footnote_or_citation_block

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    page_number: int
    section_title: str
    text: str
    chunk_type: str  # "paragraph", "section", "theorem_def", "table", "semantic"
    token_estimate: int = 0
    embedding: Optional[List[float]] = None
    bbox: Optional[List[float]] = None
    layout_label: Optional[str] = None

    def __post_init__(self):
        if not self.token_estimate:
            # Quick estimation: ~1.3 tokens per word
            self.token_estimate = int(len(self.text.split()) * 1.3)


def clean_paragraph_text(
    text: str,
    strip_citations: bool = True,
    strip_table_pointers: bool = True,
) -> str:
    """
    Clean up PDF line breaks, word hyphenation (e.g. acous-\\ntics -> acoustics, soft hyphens \\xad),
    academic bracketed citations (e.g. [19], [19,28]), and table/figure pointer sentences.
    """
    if not text:
        return ""

    # 1. Soft hyphens (\xad or \u00ad) and line-wrap hyphens
    t = re.sub(r'(\w+)[\xad\-]\s*\n\s*(\w+)', r'\1\2', text)
    t = t.replace('\xad', '').replace('\u00ad', '').replace('\u200b', '')

    # Normalize Chinese/Unicode full-width ellipsis (…… or …) to standard ellipsis
    t = re.sub(r'[…\u2026]+', '... ', t)

    # 2. Line breaks within paragraph to spaces
    t = re.sub(r'(?<!\n)\n(?!\n)', ' ', t)
    # Clean dangling hyphen at end of words/phrases before spaces or punctuation
    t = re.sub(r'\b(\w+)\s*-\s+(?=[A-Z])', r'\1 - ', t)
    t = re.sub(r'[ \t]+', ' ', t)

    # 3. Strip academic bracketed citations & participant/expert tags
    if strip_citations:
        # Numeric literature citations: [19], [19,28], [7, 25], [1-3]
        t = re.sub(r'\[\s*\d+(?:\s*[-–,;\s]\s*\d+)*\s*\]', '', t)
        # Author-year citations: [Till 2014], [Smith et al., 2020]
        t = re.sub(r'\[\s*[A-Z][a-zA-Z\s,.\-–\d]+(?:19|20)\d{2}[a-z]?\s*\]', '', t)
        # Qualitative participant/expert tags: [Expert #1], [Participant 3], [Interviewee #9]
        t = re.sub(r'\[\s*(?:Expert|Participant|Interviewee|Author|P|Ref|No)\s*#?\s*\d+\s*\]', '', t)
        # Footnote superscript numbers attached or spaced after punctuation:
        # e.g. "authenticity.19 Each" -> "authenticity. Each", "themselves. 56 Brentano" -> "themselves. Brentano"
        # Negative lookahead (?!\.) protects numbered lists like ". 1. The next"
        t = re.sub(r'([.?!,”"\'\)])\s*\d{1,3}(?!\.)\s+(?=[A-Z“"\'\(])', r'\1 ', t)

        # Footnotes attached directly to closing quotes before punctuation:
        # e.g. '“qualitative character”13.' -> '“qualitative character”.'
        t = re.sub(r'([”"\'\)])\s*\d{1,3}(?=[.,;:?!])', r'\1', t)

        # Trailing footnote numbers at paragraph or chunk end:
        # e.g. "...study it. 14." -> "...study it.", "...study it. 14" -> "...study it."
        t = re.sub(r'([.?!,”"\'\)])\s*\d{1,3}\.?\s*$', r'\1', t)

        # Unicode superscript digits (¹, ², ³, ⁴-⁹, ⁰)
        t = re.sub(r'[\u00B9\u00B2\u00B3\u2070-\u2079]+', '', t)

    # 4. Remove space before punctuation marks introduced by removed citations or line wraps
    t = re.sub(r'\s+([,.;:?!])', r'\1', t)

    # 5. Split into sentences and remove table / figure pointer sentences and clauses
    if strip_table_pointers:
        # Strip trailing table/figure clauses: e.g. ", as reported in Table 5." -> "."
        t = re.sub(
            r',?\s*(?:as|which\s+is|which\s+are)\s+(?:reported|shown|illustrated|summarized|presented|listed|given|depicted|detailed|seen|described|outlined)\s+in\s+(?:table|figure|fig\.|appendix|scheme)\s+\d+(?:\s*[.?!]|\s*$)',
            '.',
            t,
            flags=re.IGNORECASE
        )
        # Strip trailing without comma: " reported in Table 5." -> "."
        t = re.sub(
            r'\s+(?:reported|shown|illustrated|summarized|presented|listed|given|depicted|detailed|seen)\s+in\s+(?:table|figure|fig\.|appendix|scheme)\s+\d+(?:\s*[.?!]|\s*$)',
            '.',
            t,
            flags=re.IGNORECASE
        )
        # Strip mid-sentence table clauses: ", as shown in Table 3," or ", as reported in Table 1,"
        t = re.sub(
            r',\s*(?:as|which\s+is|which\s+are)\s+(?:shown|reported|illustrated|listed|summarized|detailed|seen|presented|given|depicted)\s+in\s+(?:table|figure|fig\.|appendix|scheme)\s+\d+,?\s*',
            ' ',
            t,
            flags=re.IGNORECASE
        )
        # Strip inline parentheticals: "(as shown in Table 1)" or "(Table 5)" or "(see Figure 2)"
        t = re.sub(
            r'\s*\((?:(?:as|see|refer\s+to)\s+)?(?:table|figure|fig\.|appendix|scheme)\s+\d+[^)]*\)',
            '',
            t,
            flags=re.IGNORECASE
        )
        # Strip table summary parentheticals: "(summarizing in chronological order the definitions...)"
        t = re.sub(
            r'\s*\((?:summarizing|comparing|listing|showing|illustrating|presenting)[^)]*(?:table|literature|terms|chronological)[^)]*\)',
            '',
            t,
            flags=re.IGNORECASE
        )
        # Generalize remaining substantive mentions: "in Table 1" -> "in the literature"
        t = re.sub(
            r'\b(?:in|from)\s+(?:table|figure|fig\.)\s+\d+\b',
            'in the literature',
            t,
            flags=re.IGNORECASE
        )
        # Strip leading table clauses: "As reported in Table 5, the results..." -> "The results..."
        t = re.sub(
            r'(?:^|[.?!]\s+)(?:as\s+)?(?:reported|shown|illustrated|summarized|presented|listed|given|depicted|detailed|seen)\s+in\s+(?:table|figure|fig\.|appendix|scheme)\s+\d+,\s*([a-zA-Z])',
            lambda m: (m.group(0)[:m.group(0).find(m.group(1))] if m.group(0).startswith(('.', '?', '!')) else '') + m.group(1).upper(),
            t,
            flags=re.IGNORECASE
        )

        # Normalize any spacing around punctuation and duplicate periods
        t = re.sub(r'\s+([,.;:?!])', r'\1', t)
        t = re.sub(r'\.{2,}', '.', t)

        sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z0-9"\'‘“])', t)
        cleaned_sentences = []

        table_pointer_pat = re.compile(
            r'^(?:(?:an?\s+)?(?:overview|summary|comparison|list|breakdown|details|results)?\s*(?:of\s+[^.?!]+)?\s*(?:is|are|can\s+be|were)?\s*(?:illustrated|shown|summarized|presented|listed|given|depicted|reported|seen)\s+in\s+(?:table|figure|fig\.|appendix|scheme)\s+\d+.*?[.?!]?)$',
            re.IGNORECASE
        )
        table_intro_pat = re.compile(
            r'^(?:as\s+[^,]+,\s*)?(?:the\s+)?(?:table|figure|fig\.|appendix|scheme)\s+\d+\s+(?:shows|illustrates|summarizes|presents|lists|depicts|gives|provides|details).*?[.?!]?$',
            re.IGNORECASE
        )
        see_table_pat = re.compile(
            r'^(?:see|refer\s+to|as\s+(?:shown|illustrated|listed|presented|noted)\s+in)\s+(?:table|figure|fig\.|appendix|scheme)\s+\d+.*?[.?!]?$',
            re.IGNORECASE
        )
        details_in_table_pat = re.compile(
            r'^(?:further\s+)?(?:details|results|information)\s+(?:can\s+be\s+found|are\s+(?:given|provided|shown))\s+in\s+(?:table|figure|fig\.|appendix)\s+\d+.*?[.?!]?$',
            re.IGNORECASE
        )

        for s in sentences:
            s_strip = s.strip()
            if not s_strip:
                continue
            if (table_pointer_pat.match(s_strip) or
                see_table_pat.match(s_strip) or
                table_intro_pat.match(s_strip) or
                details_in_table_pat.match(s_strip)):
                continue
            cleaned_sentences.append(s_strip)

        t = ' '.join(cleaned_sentences).strip()

    return t.strip()


def split_into_paragraphs(raw_text: str) -> List[str]:
    """
    Splits a text block if it contains multiple distinct paragraphs (separated by \n\n)
    or distinct standalone quote blocks (e.g. 'Quote 1...' [Expert #1] \n 'Quote 2...' [Expert #9]).
    Never splits mid-sentence line wraps before inline quotation marks.
    """
    blocks = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    result = []
    for blk in blocks:
        lines = blk.split("\n")
        if len(lines) <= 1:
            result.append(blk)
            continue

        accumulated = []
        current_chunk_lines = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if this line starts with an opening quote AND the previous line was a completed quote/sentence
            if current_chunk_lines and re.match(r'^[‘“"\']', line_str):
                prev_line = current_chunk_lines[-1].strip()
                is_prev_complete = (
                    re.search(r'\[(?:Expert|Participant|Interviewee)\s*#?\d+\]\.?$', prev_line, re.IGNORECASE) or
                    re.search(r'[.?!][’”"\'\)]?$', prev_line)
                )
                ends_with_connector = re.search(r'[:,]\s*$', prev_line) or prev_line.split()[-1].lower() in (
                    "the", "a", "an", "of", "this", "that", "in", "to", "for", "and", "or", "as", "like", "by", "from", "with", "into"
                )

                if is_prev_complete and not ends_with_connector:
                    accumulated.append("\n".join(current_chunk_lines))
                    current_chunk_lines = [line_str]
                    continue

            current_chunk_lines.append(line_str)

        if current_chunk_lines:
            accumulated.append("\n".join(current_chunk_lines))

        result.extend(accumulated)

    return result


ACADEMIC_ABBREVIATIONS = [
    r'e\.g\.', r'i\.e\.', r'et\s+al\.', r'cf\.', r'vol\.', r'pp\.', r'p\.', r'no\.',
    r'fig\.', r'eq\.', r'approx\.', r'ca\.', r'vs\.', r'dept\.', r'univ\.',
    r'dr\.', r'prof\.', r'mr\.', r'mrs\.', r'ms\.', r'ed\.', r'eds\.'
]


def split_into_sentences(text: str) -> List[str]:
    """
    Split a paragraph into atomic sentences while protecting academic abbreviations,
    author initials, decimals, and ellipses.
    """
    if not text or not text.strip():
        return []
    t = text.strip()
    # Mask ellipses
    t = re.sub(r'\.{3,}', '__ELLIPSIS__', t)
    # Mask decimals: e.g. 3.14, 1.0
    t = re.sub(r'(?<=\d)\.(?=\d)', '__DECIMAL__', t)
    # Mask author initials: e.g. A. Sandgren, K. Tanaka, J. L. Mackie
    t = re.sub(r'\b([A-Z])\.(?=\s*(?:[A-Z]|and\b|\(|\d{4}|[,;]))', r'\1__INITIAL__', t)
    # Mask known academic abbreviations
    for abbr in ACADEMIC_ABBREVIATIONS:
        t = re.sub(r'\b' + abbr, lambda m: m.group(0).replace('.', '__DOT__'), t, flags=re.IGNORECASE)

    # Split on sentence terminals followed by whitespace and an uppercase letter or quote
    raw_sents = re.split(r'(?<=[.?!])\s+(?=[A-Z"\'“‘])', t)

    # Unmask
    results: List[str] = []
    for s in raw_sents:
        s = s.replace('__ELLIPSIS__', '...')
        s = s.replace('__DECIMAL__', '.')
        s = s.replace('__INITIAL__', '.')
        s = s.replace('__DOT__', '.')
        s = s.strip()
        if s:
            results.append(s)
    return results


def _fast_lexical_vectors(sentences: List[str]) -> List[np.ndarray]:
    """
    Zero-dependency deterministic lexical vectorizer with word unigrams and character 3-grams.
    Provides robust fallback cosine similarity calculation when external embedding models are unavailable.
    """
    if not sentences:
        return []
    doc_tokens = []
    vocab: Dict[str, int] = {}
    for s in sentences:
        cleaned = re.sub(r'[^a-z0-9\s]', ' ', s.lower())
        words = cleaned.split()
        features = list(words)
        for w in words:
            if len(w) >= 3:
                for k in range(len(w) - 2):
                    features.append(w[k:k+3])
        doc_tokens.append(features)
        for f in features:
            if f not in vocab:
                vocab[f] = len(vocab)

    dim = len(vocab)
    vectors: List[np.ndarray] = []
    for terms in doc_tokens:
        vec = np.zeros(dim, dtype=np.float32)
        for term in terms:
            vec[vocab[term]] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vectors.append(vec)
    return vectors


def normalize_paragraph_text(text: str) -> str:
    """Convenience alias using standard cleaning filters."""
    return clean_paragraph_text(text, strip_citations=True, strip_table_pointers=True)


def is_terminal_punct(s: str) -> bool:
    """Check if a string ends with sentence-terminal punctuation (accounting for abbreviations)."""
    s_clean = s.rstrip()
    if not s_clean:
        return True
    last_char = s_clean[-1]
    if last_char in ('.', '!', '?', '"', '”', '’', "'", ':', ';'):
        words = s_clean.split()
        if words and words[-1].lower() in ("e.g.", "i.e.", "al.", "ca.", "etc.", "vs.", "fig.", "ref.", "no."):
            return False
        return True
    return False


def is_boilerplate(text: str) -> bool:
    """
    Detect if chunk text is non-substantive publication metadata, copyright disclaimers,
    journal publishing dates, pure affiliations, raw keyword lists, cataloging records,
    Table of Contents, or cover fragments.
    """
    t_lower = text.lower().strip()

    # 1. Serial / Book publication identifiers (ISSN, ISBN)
    if re.search(r'\b(?:issn|isbn)s?\b', t_lower):
        return True

    # 2. Copyright / Licensing boilerplate
    if any(k in t_lower for k in ("creativecommons.org", "cc by-nc-nd", "all rights reserved", "cambridge core", "available as open access", "open access on", "downloaded from https:", "ip address:")):
        return True
    if "©" in text and len(text.split()) < 50:
        return True

    # Author biographies and academic contributor blurbs
    if any(k in t_lower for k in ("is an associate professor", "is an assistant professor", "is a professor", "is a postdoctoral", "visiting scholar", "postdoctoral instructor", "senior advisor to the")):
        if any(k in t_lower for k in ("department of", "university", "school", "division of", "institute", "center for", "sciences")):
            return True

    # 3. Editorial timeline block (e.g. Received 29 Sept, Accepted 17 Nov)
    if any(k in t_lower for k in ("received in revised form", "accepted for publication", "available online")):
        if len(text.split()) < 70 or "elsevier" in t_lower:
            return True

    # 4. Pure keyword list
    if t_lower.startswith("keywords:") and len(text.split()) < 35:
        return True

    # 5. Author affiliation lines without substantive paragraph text
    affil_orgs = ("university college london", "the bartlett", "faculty of architecture", "department of", "institute for sustainable heritage", "institute for environmental design")
    affil_locs = ("woburn place", "london, uk", "wc1h", "zagreb, croatia", "zagreb", "email:", "@", "postcode")
    if any(org in t_lower for org in affil_orgs) and any(loc in t_lower for loc in affil_locs):
        if len(text.split()) < 100:
            return True

    # 6. Table captions and table cell matrix data
    if re.match(r'^table\s+\d+', t_lower) and len(text.split()) < 35:
        return True
    if t_lower.count(" x ") >= 2 or "\nx\n" in t_lower or (t_lower.count("/") >= 4 and len(text.split()) < 40):
        return True
    if "real sound" in t_lower and "human sound" in t_lower and "x" in t_lower:
        return True

    # 7. Reference section entries
    if t_lower.startswith("http://") or t_lower.startswith("https://") or "unesco-framework" in t_lower or "doi.org/10." in t_lower:
        return True

    # 8. Library of Congress / British Library CIP cataloging data
    cip_signals = (
        "cataloging-in-publication", "cataloguing-in-publication",
        "cataloging in publication", "library of congress catalog",
        "british library catalogue", "british library cataloguing",
        "library of congress cataloging", "library of congress control number",
        "unmoved causes of receptivity", "catalogue record for this publication"
    )
    if any(sig in t_lower for sig in cip_signals):
        return True
    if "library of congress" in t_lower and any(k in t_lower for k in ("names:", "title:", "description:", "identifiers:", "subjects:")):
        return True
    if re.search(r'[\uf700-\uf7ff]', text):
        return True

    # 9. Publisher metadata, addresses, and imprint blurbs
    pub_signals = (
        "shaftesbury road", "cambridge cb2", "liberty plaza", "williamstown road",
        "great clarendon street", "oxford university press", "ox2 6dp",
        "cambridge university press & assessment", "cambridge university press",
        "in copyright. subject to statutory", "in copyright", "statutory exceptions",
        "collective licensing", "relevant collective licensing agreements",
        "first published online:", "author for correspondence:",
        "has no responsibility for the persistence or accuracy of urls",
        "visioncrest commercial", "first published 20", "www.cambridge.org",
        "information on this title:", "university’s mission to contribute",
        "university's mission to contribute", "about the series", "about the author",
        "doi: 10.", "doi.org/10.", "edited by ", "series editor",
        "elements in philosophy", "elements in the", "permission to reproduce",
        "third-party content", "reproduced under license", "written permission of",
        "specializing in philosophy of", "printed in the united kingdom",
        "printed in the united states", "printed and bound", "cover image:", "cover design:"
    )
    if any(sig in t_lower for sig in pub_signals):
        return True

    # 10. Table of contents lines and leader dot outlines
    if re.match(r'^(?:table of contents|contents)\b', t_lower):
        return True
    if t_lower.startswith("contents") or t_lower.startswith("table of contents"):
        return True
    if re.search(r'\bcontents\s+(?:preface|acknowledgments?|introduction|\d+)\b', t_lower):
        return True
    if re.search(r'\b(?:preface|introduction)\s+\d+\b', t_lower) and re.search(r'\b(?:references|bibliography|index)\s+\d+\b', t_lower):
        return True
    if is_toc_block(text):
        return True

    # 11. Author / title cover fragment (e.g. "Koji Tanaka and Alexander Sandgren The Many Faces of Impossibility")
    if len(text.split()) < 30 and ("alexander sandgren" in t_lower or "koji tanaka" in t_lower or "many faces of impossibility" in t_lower):
        if not any(v in t_lower for v in ("argue", "propose", "show", "demonstrate", "examine", "explore", "consider", "develop", "discuss")):
            return True

    # 12. Paper title page frontmatter (Date, Correspondence, Repositories, Author lists)
    if t_lower.startswith((
        "date:", "correspondence:", "contact:", "author for correspondence:",
        "code & data:", "code and data:", "model:", "github:", "huggingface:"
    )):
        return True
    if len(text.split()) < 15 and ("github.com/" in t_lower or "huggingface.co/" in t_lower):
        return True
    if re.search(r'\barxiv:\d+\.\d+', t_lower):
        return True

    # Author footnotes, symbols, and affiliations without substantive verbs
    author_symbols = ('∗', '†', '‡', '♠', '♡', '♣', '♢', '□', '♭', '\u2217', '\u2020', '\u2021', '\u2660', '\u2661', '\u2663', '\u2662', '\u25a1', '\u266d')
    has_author_sym = any(s in text for s in author_symbols)
    has_role_tag = any(r in t_lower for r in ("joint first author", "equal contribution", "corresponding author"))
    affil_words = (
        "university", "institute", "laboratory", "alliance for research", "alliance",
        "department", "school of", "faculty of", "college", "fair at meta",
        "meta ai", "google research", "deepmind", "microsoft research", "stanford",
        "allen institute", "massachusetts institute of technology", "mit", "berkeley",
        "singapore", "harvard", "oxford", "cambridge", "cmu"
    )
    has_affil = any(a in t_lower for a in affil_words)
    substantive_verbs = (
        " is ", " are ", " was ", " were ", " have ", " has ", " had ",
        " we ", " this paper ", " in this work ", " our ", " present ", " show ",
        " propose ", " investigate ", " demonstrate ", " develop ", " achieve ",
        " introduce ", " evaluate ", " explore ", " analyze ", " argue "
    )
    padded = f" {t_lower} "
    has_verb = any(v in padded for v in substantive_verbs)
    if has_role_tag or (has_author_sym and not has_verb) or (has_affil and not has_verb) or (has_author_sym and has_affil and len(text.split()) < 80):
        return True

    # 13. Figure diagram artifacts, plot ticks, subfigure tags, metric tables
    nums = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
    if len(nums) >= 3 and len(words) <= 2:
        return True
    if re.match(r'^\d+(?:\.\d+)?$', text):
        return True
    if re.match(r'^\([a-d]\)\s+[A-Za-z]', text) and len(text.split()) < 10:
        return True
    if re.search(r'\(\d+\s*tokens?\)', t_lower) and len(text.split()) < 10:
        return True
    diagram_labels = (
        "positive document", "hard negative document", "initial documents",
        "seed (positive)", "seed documents", "documents", "real-world web search",
        "query lengths", "difficulty measure"
    )
    if t_lower in diagram_labels:
        return True
    if re.match(r'^\{[^}]+\}$', text):
        return True
    if any(k in t_lower for k in ("error", "ndcg", "bm25", "grit")) and len(nums) >= 2 and len(words) <= 5:
        return True

    # 14. Academic footnote citations and standalone bibliographical debris
    if is_footnote_or_citation_block(text):
        return True

    # 15. Draft watermarks, reviewer callouts, pre-publication notices
    draft_signals = (
        "comments welcome", "treasured, longed-for", "draft comments", "first draft",
        "pre-print draft", "preprint", "do not cite", "do not distribute", "do not quote",
        "work in progress", "working paper", "for cambridge elements", "cambridge elements series"
    )
    if any(sig in t_lower for sig in draft_signals) and len(text.split()) < 40:
        return True

    return False


class DocumentChunker:
    """
    Document chunker supporting semantic Paragraph-level chunking,
    embedding-based Semantic Boundary chunking, and character-window buffer chunking,
    with automated academic citation and table pointer removal.
    """
    def __init__(
        self,
        min_chunk_chars: int = 80,
        max_chunk_chars: int = 1500,
        strategy: str = "paragraph",  # "paragraph", "semantic", "fixed_size"
        strip_citations: bool = True,
        strip_table_pointers: bool = True,
        percentile_threshold: float = 85.0,
        min_tokens: int = 50,
        max_tokens: int = 400,
        embed_client: Optional[Any] = None,
        embed_model: Optional[str] = None,
    ):
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.strategy = strategy
        self.strip_citations = strip_citations
        self.strip_table_pointers = strip_table_pointers
        self.percentile_threshold = percentile_threshold
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.embed_client = embed_client
        self.embed_model = embed_model

    def _compute_sentence_embeddings(
        self,
        sentences: List[str],
        client: Optional[Any] = None,
        model: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Embed sentences using client.embed() if available, otherwise fall back
        to deterministic lexical vectors.
        """
        if not sentences:
            return []

        active_client = client or self.embed_client
        active_model = model or self.embed_model

        if active_client is not None and hasattr(active_client, "embed"):
            try:
                batch_size = 64
                all_vecs: List[np.ndarray] = []
                for i in range(0, len(sentences), batch_size):
                    batch = sentences[i:i + batch_size]
                    if active_model:
                        res = active_client.embed(batch, model=active_model)
                    else:
                        res = active_client.embed(batch)
                    for item in res:
                        all_vecs.append(np.array(item, dtype=np.float32))
                return all_vecs
            except Exception as e:
                logger.warning(
                    f"External embedding generation failed ({e}). "
                    f"Falling back to local lexical vectors for semantic segmentation."
                )

        return _fast_lexical_vectors(sentences)

    def chunk_pages(
        self,
        doc_name: str,
        pages: List[DocumentPage],
        strategy: Optional[str] = None,
        percentile_threshold: Optional[float] = None,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        embed_client: Optional[Any] = None,
        embed_model: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Segment pages into chunks.
        - 'paragraph' (default): stitches fragmented blocks and cross-page sentences into natural paragraphs.
        - 'semantic': computes sentence-level embedding distances within each section to dynamically cut at topical shifts.
        - 'fixed_size': accumulates text blocks into a fixed-character buffer window.
        """
        active_strategy = strategy or self.strategy
        self._seen_ids: Dict[str, int] = {}     # stable IDs: duplicates within one run get a suffix
        chunks: List[Chunk] = []
        current_section = "Preliminaries"

        if active_strategy == "paragraph":
            # --- PARAGRAPH CHUNKS: one chunk per stitched paragraph ---
            for ev in self._stitched_units(pages):
                kind = ev[0]
                if kind == "raw_page":
                    _, page_num, section, text = ev
                    for rp in text.split("\n\n"):
                        clean_p = clean_paragraph_text(rp, strip_citations=self.strip_citations,
                                                       strip_table_pointers=self.strip_table_pointers)
                        if len(clean_p) >= self.min_chunk_chars and not is_boilerplate(clean_p):
                            chunks.append(self._create_chunk(doc_name, page_num, section, clean_p))
                elif kind == "table":
                    _, page_num, section, text, bbox = ev
                    if len(text) >= self.min_chunk_chars and not is_boilerplate(text):
                        chunks.append(self._create_chunk(doc_name, page_num, section, text, chunk_type="table",
                                                         bbox=bbox, layout_label="table"))
                elif kind == "para":
                    _, page_num, section, text = ev
                    for unit in split_into_paragraphs(text):
                        cleaned = clean_paragraph_text(unit, strip_citations=self.strip_citations,
                                                       strip_table_pointers=self.strip_table_pointers)
                        if len(cleaned) < self.min_chunk_chars or is_boilerplate(cleaned):
                            continue
                        if not is_terminal_punct(cleaned):
                            cleaned = cleaned.rstrip() + "."
                        chunks.append(self._create_chunk(doc_name, page_num, section, cleaned))
            return chunks

        elif active_strategy == "semantic":
            # --- SEMANTIC BOUNDARY CHUNKING (Embedding Distance) over the same stitched paragraphs ---
            active_p_thresh = percentile_threshold if percentile_threshold is not None else self.percentile_threshold
            active_min_tok = min_tokens if min_tokens is not None else self.min_tokens
            active_max_tok = max_tokens if max_tokens is not None else self.max_tokens
            active_embed_client = embed_client if embed_client is not None else self.embed_client
            active_embed_model = embed_model if embed_model is not None else self.embed_model

            sections: List[Tuple[str, List[Tuple[str, int]]]] = []
            cur_sec = "Preliminaries"
            cur_sents: List[Tuple[str, int]] = []

            def add_sentences(text: str, page_num: int):
                cleaned = clean_paragraph_text(text, strip_citations=self.strip_citations,
                                               strip_table_pointers=self.strip_table_pointers)
                if len(cleaned) < 30 or is_boilerplate(cleaned):
                    return
                for s in split_into_sentences(cleaned):
                    if len(s) >= 8:
                        cur_sents.append((s, page_num))

            for ev in self._stitched_units(pages):
                kind = ev[0]
                if kind == "section":
                    if cur_sents:
                        sections.append((cur_sec, cur_sents))
                        cur_sents = []
                    cur_sec = ev[1]
                elif kind == "raw_page":
                    for rp in ev[3].split("\n\n"):
                        add_sentences(rp, ev[1])
                elif kind == "table":
                    _, page_num, section, text, bbox = ev
                    if len(text) >= self.min_chunk_chars and not is_boilerplate(text):
                        chunks.append(self._create_chunk(doc_name, page_num, section, text, chunk_type="table",
                                                         bbox=bbox, layout_label="table"))
                elif kind == "para":
                    for unit in split_into_paragraphs(ev[3]):
                        add_sentences(unit, ev[1])
            if cur_sents:
                sections.append((cur_sec, cur_sents))

            for sec_title, sents in sections:
                if not sents:
                    continue

                if len(sents) == 1:
                    c_text = sents[0][0]
                    if len(c_text) >= self.min_chunk_chars and not is_boilerplate(c_text):
                        chunks.append(self._create_chunk(doc_name, sents[0][1], sec_title, c_text, chunk_type="semantic"))
                    continue

                s_texts = [s[0] for s in sents]
                embeddings = self._compute_sentence_embeddings(
                    s_texts,
                    client=active_embed_client,
                    model=active_embed_model
                )

                distances: List[float] = []
                for i in range(len(embeddings) - 1):
                    e1, e2 = embeddings[i], embeddings[i + 1]
                    norm1 = np.linalg.norm(e1)
                    norm2 = np.linalg.norm(e2)
                    if norm1 > 0 and norm2 > 0:
                        cos = float(np.dot(e1, e2) / (norm1 * norm2))
                    else:
                        cos = 0.0
                    distances.append(max(0.0, 1.0 - cos))

                threshold = float(np.percentile(distances, active_p_thresh))

                sec_chunks: List[Tuple[str, int, int]] = []
                curr_group = [sents[0]]
                curr_tokens = len(sents[0][0].split())

                for i in range(len(distances)):
                    dist = distances[i]
                    next_s = sents[i + 1]
                    next_tok = len(next_s[0].split())

                    should_split = (dist >= threshold and curr_tokens >= active_min_tok) or (curr_tokens + next_tok > active_max_tok)
                    if should_split:
                        chunk_text = " ".join(s[0] for s in curr_group)
                        sec_chunks.append((chunk_text, curr_group[0][1], curr_tokens))
                        curr_group = [next_s]
                        curr_tokens = next_tok
                    else:
                        curr_group.append(next_s)
                        curr_tokens += next_tok

                if curr_group:
                    if sec_chunks and curr_tokens < active_min_tok and (sec_chunks[-1][2] + curr_tokens <= active_max_tok):
                        prev_text, prev_page, prev_tok = sec_chunks[-1]
                        merged_text = prev_text + " " + " ".join(s[0] for s in curr_group)
                        sec_chunks[-1] = (merged_text, prev_page, prev_tok + curr_tokens)
                    else:
                        chunk_text = " ".join(s[0] for s in curr_group)
                        sec_chunks.append((chunk_text, curr_group[0][1], curr_tokens))

                for c_text, c_page, _ in sec_chunks:
                    c_clean = c_text.strip()
                    if len(c_clean) >= self.min_chunk_chars and not is_boilerplate(c_clean):
                        if not is_terminal_punct(c_clean):
                            c_clean = c_clean.rstrip() + "."
                        chunks.append(self._create_chunk(doc_name, c_page, sec_title, c_clean, chunk_type="semantic"))

            return chunks

        else:
            # --- FIXED SIZE CHUNKING (Buffer Window) ---
            buffer_text = []
            for page in pages:
                for b in page.blocks:
                    if b.block_type in ("heading", "footnote", "frontmatter", "figure_content", "figure_caption", "toc", "references", "bibliography"):
                        if b.block_type == "heading":
                            if buffer_text:
                                text_body = "\n\n".join(buffer_text).strip()
                                if len(text_body) >= self.min_chunk_chars and not is_boilerplate(text_body):
                                    chunks.append(self._create_chunk(doc_name, page.page_number, current_section, text_body))
                                buffer_text = []
                            current_section = b.text.strip()[:100]
                        continue

                    if current_section.lower() in ("references", "bibliography"):
                        continue

                    clean_text = clean_paragraph_text(
                        b.text,
                        strip_citations=self.strip_citations,
                        strip_table_pointers=self.strip_table_pointers
                    )
                    buffer_text.append(clean_text)
                    accumulated = "\n\n".join(buffer_text)
                    if len(accumulated) >= self.max_chunk_chars:
                        if not is_boilerplate(accumulated):
                            chunks.append(self._create_chunk(doc_name, page.page_number, current_section, accumulated))
                        buffer_text = []

                if buffer_text:
                    text_body = "\n\n".join(buffer_text).strip()
                    if len(text_body) >= self.min_chunk_chars and not is_boilerplate(text_body):
                        chunks.append(self._create_chunk(doc_name, page.page_number, current_section, text_body))
                    buffer_text = []
            return chunks

    _SKIP_BLOCK_TYPES = ("footnote", "frontmatter", "figure_content", "figure_caption", "figure", "caption", "toc",
                         "references", "bibliography", "page-header", "page-footer", "reference", "acknowledgment")
    _REF_HEADINGS = ("references", "bibliography", "works cited", "literature cited")
    _PUBLISHER_HINTS = ("road", "street", "cb2 8ea", "singapore", "copyright", "issn", "isbn", "library of congress",
                        "control number", "cataloging", "cataloguing", "great clarendon")

    def _stitched_units(self, pages: List[DocumentPage]):
        """
        Shared paragraph stitcher for the paragraph and semantic strategies. Yields, in reading order:
          ("section", title)                        the section changes
          ("para", start_page, section, raw_text)   a paragraph stitched across blocks / pages
          ("table", page, section, text, bbox)      a table block
          ("raw_page", page, section, text)         a page without layout blocks
        Skips noise blocks and everything inside a References / Bibliography section (until an appendix).
        Stitching: a block continues the previous one after a line-break hyphen, when the previous block has no
        terminal punctuation, when it starts lower-case or with continuation punctuation, or when it is very
        close below it on the same page; never across a large vertical gap.
        """
        section = "Preliminaries"
        para = None              # [start_page, section, text, bbox]
        in_refs = False

        def flush():
            nonlocal para
            out, para = para, None
            return [("para", out[0], out[1], out[2])] if out else []

        for page in pages:
            if not page.blocks:
                yield ("raw_page", page.page_number, section, page.text)
                continue
            for b in page.blocks:
                if b.block_type in self._SKIP_BLOCK_TYPES:
                    yield from flush()
                    continue
                if b.block_type == "table":
                    yield from flush()
                    if not in_refs:
                        yield ("table", page.page_number, section, b.text.strip(), b.bbox)
                    continue
                if b.block_type == "heading":
                    h_text = b.text.strip()
                    if h_text.lower() in self._REF_HEADINGS and page.page_number > 15 and not re.search(r'\d+$', h_text):
                        yield from flush()
                        in_refs, section = True, "References"
                        yield ("section", section)
                        continue
                    if in_refs and not re.match(r'^(?:appendix|annex)\b', h_text, re.IGNORECASE):
                        continue
                    yield from flush()
                    in_refs, section = False, h_text[:100]
                    yield ("section", section)
                    continue
                if in_refs:
                    continue
                raw_text = b.text.strip()
                if not raw_text:
                    continue

                # a block whose first line is a heading
                lines = raw_text.split("\n", 1)
                first_line = lines[0].strip()
                first_low = first_line.lower()
                is_pub = any(bad in first_low for bad in self._PUBLISHER_HINTS)
                if not is_pub and is_likely_heading(first_line) and len(lines) > 1:
                    yield from flush()
                    if first_low in self._REF_HEADINGS[:3] and page.page_number > 15 and not re.search(r'\d+$', raw_text):
                        in_refs, section = True, "References"
                        yield ("section", section)
                        continue
                    section = first_line[:100]
                    yield ("section", section)
                    raw_text = lines[1].strip()

                if section.lower() in ("references", "bibliography"):
                    continue

                if para is None:
                    para = [page.page_number, section, raw_text, list(b.bbox) if b.bbox else None]
                    continue
                prev_text, curr_text = para[2].rstrip(), raw_text.lstrip()
                same_page = page.page_number == para[0]
                large_gap = False
                if same_page and b.bbox and para[3]:
                    dy = b.bbox[1] - para[3][3]
                    large_gap = dy > 28 or b.bbox[1] < para[3][1] - 10
                combined = None
                if not large_gap:
                    if prev_text.endswith(("-", "\xad", "\u00ad")):
                        combined = prev_text[:-1] + curr_text
                    elif not is_terminal_punct(prev_text):
                        combined = prev_text + " " + curr_text
                    elif curr_text and (curr_text[0].islower() or curr_text[0] in (',', ';', ')', ']')):
                        combined = prev_text + " " + curr_text
                    elif b.bbox and para[3] and same_page:
                        if b.bbox[1] - para[3][3] < 14 and not prev_text.endswith(('.', '!', '?')):
                            combined = prev_text + " " + curr_text
                if combined is not None:
                    para[2] = combined
                    if b.bbox and para[3]:
                        para[3][3] = b.bbox[3]
                else:
                    yield from flush()
                    para = [page.page_number, section, raw_text, list(b.bbox) if b.bbox else None]
        yield from flush()

    def _stable_chunk_id(self, doc_name: str, page_num: int, section: str, text: str) -> str:
        """
        Content-derived ID: re-chunking the same document with the same settings gives the same IDs, so
        KG edges and saved samples keep pointing at their chunks. Identical text on the same page in the same
        section gets -2, -3 ... in reading order.
        """
        norm = re.sub(r"\s+", " ", text).strip()
        base = hashlib.sha1(f"{doc_name}\x1f{page_num}\x1f{section}\x1f{norm}".encode("utf-8")).hexdigest()[:12]
        seen = getattr(self, "_seen_ids", None)
        if seen is None:
            return base
        n = seen.get(base, 0) + 1
        seen[base] = n
        return base if n == 1 else f"{base}-{n}"

    def _create_chunk(
        self,
        doc_name: str,
        page_num: int,
        section: str,
        text: str,
        chunk_type: Optional[str] = None,
        bbox: Optional[List[float]] = None,
        layout_label: Optional[str] = None,
    ) -> Chunk:
        # Detect if chunk looks like a formal definition, theorem, or axiom
        is_def = any(
            text.lower().startswith(prefix)
            for prefix in ("definition", "theorem", "lemma", "proposition", "corollary", "axiom")
        )
        if chunk_type:
            c_type = "theorem_def" if is_def else chunk_type
        else:
            c_type = "theorem_def" if is_def else "paragraph"

        return Chunk(
            chunk_id=self._stable_chunk_id(doc_name, page_num, section, text),
            doc_name=doc_name,
            page_number=page_num,
            section_title=section,
            text=text,
            chunk_type=c_type,
            bbox=bbox,
            layout_label=layout_label,
        )

