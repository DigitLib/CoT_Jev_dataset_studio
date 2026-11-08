"""
relation_logic.py - what a knowledge-graph relation means in propositional logic.

The KG stores relations as canonical predicates ("requires", "not requires", "excludes", "models" ...).
Only some of them license an inference. This table is the single place that decides which:

    entails   A --[p]--> B   gives the rule  A => B    (implies, requires, depends on, is a kind of, ...)
    excludes  A --[p]--> B   gives the rule  A => ~B   (lacks, excludes, precludes, contradicts, ...)
    none      no rule: contrast, modelling, authorship, enabling, negated entailment ("does not require")

"does not require" is NOT "A => ~B": it only denies the implication, so it yields no rule at all.
Edge direction is part of the meaning: a path that walks an edge backwards may not use it as A => B.
"""
import re
from typing import Optional

# negation inside a raw predicate phrase ("does not require", "cannot enable", "fails to imply")
NEGATION_RE = re.compile(
    r"\b(?:not|never|cannot|can\s+not|no\s+longer|fails?\s+to|unable\s+to|neither|nor)\b|n't\b",
    re.IGNORECASE)
# the same, at the end of the text before a verb (used by the heuristic sentence extractor)
NEGATION_TAIL_RE = re.compile(
    r"(?:\b(?:does|do|did|is|are|was|were|can|could|will|would|may|might|must|should|has|have|had)\s+)?"
    r"(?:not|never)\s*$|\b\w+n't\s*$|\bcannot\s*$|\bno\s+longer\s*$|\bfails?\s+to\s*$|\bunable\s+to\s*$",
    re.IGNORECASE)
# exclusion verbs: positive phrasing, negative meaning
EXCLUSION_RE = re.compile(
    r"\b(?:lacks?|lacking|excludes?|precludes?|prevents?|rules?\s+out|contradicts?|incompatible\s+with|"
    r"conflicts?\s+with|refutes?|negates?|is\s+inconsistent\s+with|inconsistent\s+with)\b",
    re.IGNORECASE)

ENTAILMENT_PREDICATES = {
    "implies", "requires", "depends on", "is a kind of", "proves", "is defined as",
    "causes",   # causal claims read as material implication; a defeasible reading would need another logic
}
EXCLUSION_PREDICATES = {"excludes"}


def split_negation(pred: str):
    """('not requires') -> (True, 'requires')."""
    p = (pred or "").strip().lower()
    if p.startswith("not "):
        return True, p[4:].strip()
    return False, p


# raw (non-canonical) phrasings, for edges added without canonicalization
ENTAILMENT_RE = re.compile(
    r"^(?:\w+\s+)?(?:impl(?:y|ies|ied)|entails?|leads?\s+to|results?\s+in|requires?|necessitates?|demands?|"
    r"depends?\s+on|rel(?:y|ies)\s+on|is\s+a\s+(?:kind|type|form)\s+of|(?:kind|type|form)s?\s+of|proves?|"
    r"is\s+defined\s+as|causes?|produces?|yields?)\b", re.IGNORECASE)


def relation_logic(pred: Optional[str]) -> str:
    """'entails' | 'excludes' | 'none' for a canonical predicate (or a plain relation phrase)."""
    raw = (pred or "").strip().lower()
    negated, base = split_negation(raw)
    if not negated and is_negated_phrase(raw):          # raw phrase such as "does not require"
        negated, base = True, re.sub(r"\s+", " ", NEGATION_RE.sub(" ", raw)).strip()
        base = re.sub(r"^(?:does|do|did|is|are|was|were|can|could|will|would)\s+", "", base)
    if base in ENTAILMENT_PREDICATES or ENTAILMENT_RE.match(base):
        return "none" if negated else "entails"
    if base in EXCLUSION_PREDICATES or EXCLUSION_RE.search(base):
        return "none" if negated else "excludes"   # "does not exclude" licenses nothing
    return "none"


def is_negated_phrase(raw: str) -> bool:
    return bool(NEGATION_RE.search(raw or ""))
