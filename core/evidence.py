"""
evidence.py - check that a knowledge-graph edge is supported by the chunk it was extracted from.

The extraction prompt asks the LLM for an "exact supporting excerpt", but models paraphrase or invent.
ground_evidence() classifies each edge's evidence:

    exact        the quote occurs in the chunk (after normalising case, whitespace, quotes, dashes, hyphenation)
    fuzzy        >= 60% of the quote's word trigrams occur in the chunk (light paraphrase / OCR noise)
    located      no usable quote, but one sentence of the chunk mentions both subject and object;
                 that sentence becomes the evidence
    unsupported  none of the above: the edge is not grounded in this chunk

Only the relation's endpoints are checked against the text, not the predicate; "located" therefore means
"the two concepts co-occur in a sentence", which is weaker than a quote.
"""
import re
import unicodedata
from typing import Iterable, List, Optional, Set, Tuple

_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "to", "for", "with", "by", "as", "at", "from",
         "is", "are", "be", "its", "their", "this", "that", "these", "those"}
SUPPORTED = ("exact", "fuzzy", "located")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("­", "")
    t = re.sub(r"[‘’‚‛`´]", "'", t)
    t = re.sub(r"[“”„‟]", '"', t)
    t = re.sub(r"[‐-―−]", "-", t)
    t = re.sub(r"(\w)-\s+(\w)", r"\1\2", t)          # line-break hyphenation: "auto- nomy"
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize(text))


def _shingles(words: List[str], n: int = 3) -> Set[Tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", text or "") if len(s.strip()) > 10]


def _stem(w: str) -> str:
    return w[:5] if len(w) > 5 else w


def mentions(concept: str, text: str) -> bool:
    """Every content word of `concept` appears in `text` (prefix match, so plurals / inflections pass)."""
    words = [w for w in _words(concept) if w not in _STOP and len(w) >= 3] or _words(concept)
    if not words:
        return False
    stems = {_stem(w) for w in _words(text)}
    return all(_stem(w) in stems for w in words)


def quote_support(evidence: str, chunk_text: str) -> Optional[str]:
    """'exact' | 'fuzzy' | None."""
    ev, ch = normalize(evidence), normalize(chunk_text)
    if len(ev) >= 12 and ev.strip(" .\"'") in ch:
        return "exact"
    ev_w, ch_w = _words(evidence), _words(chunk_text)
    if len(ev_w) >= 4:
        es = _shingles(ev_w)
        if es and len(es & _shingles(ch_w)) / len(es) >= 0.6:
            return "fuzzy"
    return None


def ground_evidence(evidence: str, chunk_text: str, subject: str, obj: str) -> Tuple[str, str]:
    """(status, evidence_text) for an edge subject -> obj extracted from chunk_text."""
    if evidence and evidence.strip():
        st = quote_support(evidence, chunk_text)
        if st:
            return st, evidence.strip()
    for s in _sentences(chunk_text):
        if mentions(subject, s) and mentions(obj, s):
            return "located", s
    return "unsupported", (evidence or "").strip()


def summarize(statuses: Iterable[str]) -> dict:
    out = {k: 0 for k in SUPPORTED + ("unsupported",)}
    for s in statuses:
        out[s] = out.get(s, 0) + 1
    return out
