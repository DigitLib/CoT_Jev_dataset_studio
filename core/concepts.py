"""
concepts.py - what makes a knowledge-graph node a meaningful concept, and how to repair one that is not.

A KG node should name ONE reusable concept (an abstract notion such as "Substantive Independence", or a
named entity such as "COMPAS"), as a complete noun phrase. Extraction by small LLMs produces other things:

    truncated phrase   "Respect For Practical And"         (cut mid-phrase; older builds cut every name at 4 words)
    clause             "People Are Blameworthy", "Getting A Smoke Alarm"
    pronoun fragment   "Persons Subject To Them", "Understanding Of Where They"
    generic word       "Individual", "System", "Information", "Thing"
    variant spelling   "Autonomy Conception" / "Conception Of Autonomy"   (one concept, two nodes)

Everything here is deterministic and local (no NLP model needed): word classes, the evidence sentences an
edge was extracted from, and how often a concept recurs across chunks. Used by core/kg_builder.py both at
extraction time and by KnowledgeGraphBuilder.repair_concepts() on an existing graph. Shared by the ClearCoT
and Jev pipelines because both read the same graph.
"""
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------------------------- word classes
ARTICLES = {"the", "a", "an"}
DETERMINERS = ARTICLES | {"this", "that", "these", "those", "such", "some", "any", "each", "every", "another",
                          "his", "her", "their", "its", "our", "my", "your", "whose"}
PRONOUNS = {"i", "we", "us", "you", "he", "him", "she", "her", "they", "them", "it", "itself",
            "themselves", "himself", "herself", "his", "their", "theirs", "its", "our", "my", "your",
            "someone", "something", "anyone", "anything", "everyone", "everything", "one"}
PREPOSITIONS = {"of", "for", "to", "in", "on", "at", "by", "with", "from", "into", "onto", "over", "under",
                "about", "against", "between", "among", "through", "within", "without", "toward", "towards",
                "upon", "across", "via", "per", "than", "as", "regarding", "concerning", "including", "involving",
                "like", "unlike", "despite", "during", "beyond", "around", "behind", "near"}
CONJUNCTIONS = {"and", "or", "nor", "but", "yet", "so"}
# "will" and "might" are left out: in this literature they are nouns ("Democratic Will", "Will Theory")
AUXILIARIES = {"is", "are", "was", "were", "be", "been", "being", "am", "has", "have", "had", "do", "does",
               "did", "can", "could", "would", "shall", "should", "may", "must", "not"}
# last word looks like an adjective: a name ending in one after "and" / "of" was cut off
# ("Principle Of Informed Cognitive", "Predictive Analytics And Digital"), unless it is a known noun
ADJECTIVE_SUFFIX_RE = re.compile(r"(?:al|ive|ic|ous|ful|less|able|ible)$")
NOUNS_LOOKING_ADJECTIVAL = {
    "individual", "principal", "proposal", "approval", "rival", "animal", "trial", "arrival", "refusal", "denial",
    "removal", "survival", "criminal", "official", "professional", "capital", "potential", "signal", "material",
    "journal", "tribunal", "terminal", "interval", "manual", "ritual", "appraisal", "dismissal", "renewal",
    "objective", "initiative", "alternative", "incentive", "executive", "representative", "directive", "narrative",
    "perspective", "motive", "relative", "collective", "native", "detective", "detective", "archive", "explosive",
    "public", "logic", "republic", "topic", "critic", "rhetoric", "music", "clinic", "mechanic", "ethic", "epidemic",
    "panic", "traffic", "graphic", "fabric", "arithmetic", "metric", "basic", "classic", "academic", "cable",
    "table", "variable", "vegetable", "label", "fable", "responsible", "deliverable", "receivable", "bible",
    "ideal", "goal",
}
CLAUSE_MARKERS = {"that", "which", "who", "whom", "whose", "where", "when", "while", "because", "if",
                  "although", "though", "since", "unless", "whether", "how", "why", "what"}
# a phrase may not END with one of these: it was cut off mid-phrase
DANGLING = ARTICLES | DETERMINERS | PREPOSITIONS | CONJUNCTIONS | AUXILIARIES | CLAUSE_MARKERS | PRONOUNS \
    | {"more", "less", "most", "very", "also", "only", "just", "based", "make", "take", "get", "give", "do"}
# heads too vague to be a concept on their own (fine inside a phrase: "Moral Agent", "Data Subject")
GENERIC_HEADS = {"individual", "individuals", "person", "people", "persons", "system", "systems", "information",
                 "thing", "things", "way", "ways", "case", "cases", "issue", "issues", "fact", "facts", "idea",
                 "ideas", "aspect", "part", "kind", "sort", "type", "form", "other", "others", "stake", "stakes",
                 "user", "users", "agent", "matter", "point", "example", "concept", "notion", "question",
                 "problem", "situation", "condition", "factor", "element", "level", "role", "view", "account",
                 "claim", "reader", "author", "chapter", "section", "book", "paper", "both", "either", "neither"}
# modifiers that change the meaning of the head: "Non-domination" is not a kind of "Domination"
NEGATING_MODIFIERS = {"non", "not", "anti", "pseudo", "quasi", "false", "fake", "alleged", "apparent", "so-called",
                      "former", "lack", "absence", "un", "mis", "without", "no", "ex", "would-be"}

# 3rd-person relation verbs that are not also common plural nouns: inside a name they make it a clause
# ("Autonomy Demands Guidance Control" is a sentence, not a concept)
CLAUSE_VERBS = {"requires", "implies", "entails", "enables", "excludes", "undermines", "involves", "includes",
                "prevents", "allows", "provides", "creates", "demands", "presupposes", "contradicts", "ensures"}
# heads too vague for an "is a kind of" edge: claims the text makes about them are context-specific, so
# inheriting them ("Algorithmic Advertising Process" -> "Process" -> whatever "Process" requires) over-generalises
TOO_GENERAL_FOR_TAXONOMY = {"process", "event", "activity", "ability", "capacity", "constraint", "technology",
                            "preference", "belief", "reason", "expectation", "decision", "action", "approach",
                            "method", "model", "framework", "structure", "practice", "measure", "tool", "data",
                            "outcome", "effect", "result", "use", "goal", "interest", "need", "task", "relation",
                            "relationship", "state", "group", "set", "term", "feature", "mechanism", "property",
                            "theory", "processing", "phenomenon", "phenomena", "view", "position", "approach"}
# base-form verbs that open an action phrase ("Assess Consciousness", "Ensure Autonomy"): a goal or a task,
# not a kind of its object. Verbs that are also common nouns are only counted when no preposition follows
# ("Lack Of Oversight", "Measure Of Fit" are nouns); "test" is left out ("Test Data").
LEADING_VERBS = {"assess", "evaluate", "explain", "achieve", "attribute", "ascribe", "identify", "investigate",
                 "examine", "create", "build", "develop", "produce", "generate", "implement", "apply", "compare",
                 "consider", "describe", "distinguish", "ensure", "establish", "predict", "prevent", "promote",
                 "protect", "provide", "reduce", "understand", "determine", "decide", "make", "take", "get", "give",
                 "find", "keep", "see", "know", "obtain", "acquire", "detect", "discover", "perform", "integrate",
                 "combine", "enable", "allow", "avoid", "minimize", "maximize", "improve", "increase", "measure",
                 "address", "solve", "prove", "show", "demonstrate", "learn", "infer", "select", "exhibit",
                 "possess", "lack", "become"}
MODALS = {"may", "might", "could", "can", "would", "should", "must", "will", "shall"}
CONTRAST = {"rather", "instead", "but", "than", "whether", "if", "unless", "although", "though", "only"}
# words allowed BEFORE the inference cue of a relation phrase ("is the other key requirement for")
CUE_PREFIX_OK = {"is", "are", "was", "were", "be", "the", "a", "an", "also", "thus", "therefore", "hence", "then",
                 "often", "always", "necessarily", "other", "key", "main", "primary", "one", "first", "central",
                 "basic", "essentially", "fundamentally", "directly", "both"}

# words allowed AFTER the cue ("is defined as having", "requires that"); anything else means the phrase
# continues with its own content and the cue does not relate subject and object directly
CUE_SUFFIX_OK = {"the", "a", "an", "that", "to", "of", "for", "on", "in", "with", "as", "being", "having", "both",
                 "all", "any", "some", "by", "upon"}

# "Agent To Have Control", "Drivers To Take More Riders": an infinitive clause, not a concept
# ("Right To Explanation" / "Capacity To Be Responsible" stay fine: noun after "to", or "to be")
INFINITIVE_VERBS = {"have", "make", "take", "do", "get", "give", "go", "put", "see", "know", "use", "find", "keep",
                    "let", "become", "seem", "act", "decide", "determine", "understand", "exercise", "form", "ban"}

# nouns that take an infinitive complement: "Capacity To Act", "Right To Object", "Freedom To Ban" are concepts
COMPLEMENT_NOUNS = {"capacity", "ability", "right", "freedom", "power", "duty", "obligation", "need", "desire",
                    "willingness", "opportunity", "chance", "decision", "choice", "permission", "failure", "attempt",
                    "effort", "tendency", "disposition", "competence", "liberty", "license", "authority", "reason",
                    "incentive", "pressure", "claim", "entitlement", "license", "requirement", "capability"}

COMMON_ADJECTIVES = {"deep", "other", "underlying", "novel", "new", "old", "high", "low", "good", "bad", "many",
                     "few", "several", "various", "own", "same", "different", "key", "main", "major", "minor",
                     "certain", "wide", "broad", "full", "whole", "basic", "special", "particular", "specific",
                     "russian", "american", "european", "chinese", "british", "german", "french"}


def _adjectival(lw: str) -> bool:
    """Word that normally modifies a following noun: a phrase ending in one was cut off."""
    return lw in COMMON_ADJECTIVES or (ADJECTIVE_SUFFIX_RE.search(lw) is not None and lw not in NOUNS_LOOKING_ADJECTIVAL)


ISSUE_WEIGHTS = {
    "empty": 1.0, "truncated": 0.6, "clause": 0.6, "pronoun": 0.5, "generic": 0.55, "too_long": 0.35,
    "citation": 0.6, "numeric": 0.5, "verb_phrase": 0.45,
}


def words(name: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-]*", name or "")


def _low(ws: Sequence[str]) -> List[str]:
    return [w.lower().replace("’", "'") for w in ws]


def singular(w: str) -> str:
    """Light singular for matching keys only (not for display)."""
    low = w.lower()
    if len(low) <= 3 or low.endswith(("ss", "us", "is", "ics", "sis")) or low in {"data", "series", "species", "news", "means"}:
        return low
    if low.endswith("ies") and len(low) > 4:
        return low[:-3] + "y"
    if low.endswith(("sses", "xes", "ches", "shes", "zes")):
        return low[:-2]                         # processes -> process, approaches -> approach
    if low.endswith("s") and not low.endswith("'s"):
        return low[:-1]
    return low


def concept_key(name: str) -> str:
    """
    Order-insensitive identity of a concept: singular content words, sorted, possessives dropped.
    "Conception Of Autonomy" == "Autonomy Conception"; "Users' Capacity" == "User Capacity".
    """
    toks = []
    for w in _low(words(name)):
        w = re.sub(r"'s?$", "", w)
        if not w or w in ARTICLES or w in PREPOSITIONS or (w in CONJUNCTIONS and w != "or"):
            continue                            # "or" stays: "A Or B" is not "A And B"
        toks.append(singular(w))
    return " ".join(sorted(toks))


# ----------------------------------------------------------------------------------------- quality
def concept_issues(name: str, max_words: int = 6) -> List[str]:
    """Reasons why `name` is not a clean concept name ([] = looks fine). See ISSUE_WEIGHTS."""
    ws = words(name)
    if not ws:
        return ["empty"]
    low = _low(ws)
    issues = []
    # a single capital letter or roman numeral after a noun is a label ("Theorem A", "Type II"), not an article
    label_end = len(ws) > 1 and re.fullmatch(r"[A-Z]|[IVX]{1,4}", ws[-1]) is not None
    if label_end:
        low = low[:-1] + ["<label>"]
    cut_after_adjective = (len(low) >= 3 and _adjectival(low[-1])
                           and any(w in PREPOSITIONS | CONJUNCTIONS for w in low[1:-1]))
    dangling_possessive = ws[-1].endswith(("'", "’")) and len(ws) > 1
    if low[-1] in DANGLING or low[0] in (PREPOSITIONS | CONJUNCTIONS | AUXILIARIES) or cut_after_adjective \
            or dangling_possessive:
        issues.append("truncated")
    # a finite verb / auxiliary inside the phrase ("to be" excepted: "Capacity To Be Responsible"), a clause marker
    # after the head, or "Getting A ..." (gerund + determiner)
    aux_inside = [i for i, w in enumerate(low[1:], 1) if w in AUXILIARIES and not (w == "be" and low[i - 1] == "to")]
    to_verb = any(low[i] == "to" and low[i + 1] in INFINITIVE_VERBS
                  and singular(re.sub(r"'s?$", "", low[i - 1])) not in COMPLEMENT_NOUNS for i in range(1, len(low) - 1))
    if aux_inside or to_verb or any(w in CLAUSE_MARKERS for w in low[1:-1]) or low[0] in CLAUSE_MARKERS or \
            any(w in CLAUSE_VERBS for w in low[1:]) or \
            (len(low) > 1 and low[0].endswith("ing") and low[1] in DETERMINERS):
        issues.append("clause")
    if any(w in PRONOUNS and w not in {"one"} for w in low):
        issues.append("pronoun")
    elif low[0] == "own" and len(low) > 1:
        issues.append("truncated")               # "Own Consciousness": the possessor ("one's") was cut off
    if len(low) > 1 and low[0] in LEADING_VERBS and low[1] not in PREPOSITIONS | CONJUNCTIONS \
            and "clause" not in issues:
        issues.append("verb_phrase")
    content = [w for w in low if w not in ARTICLES | PREPOSITIONS | CONJUNCTIONS]
    if len(content) == 1 and content[0] in GENERIC_HEADS:
        issues.append("generic")
    if len(ws) > max_words:
        issues.append("too_long")
    if re.search(r"\b(?:\d{4}|pp?\.|et al|ibid)\b", name, re.I):
        issues.append("citation")
    if all(re.fullmatch(r"[\d.,\-]+", w) for w in ws):
        issues.append("numeric")
    return issues


def concept_quality(name: str, mentions: int = 1, issues: Optional[List[str]] = None) -> float:
    """
    0..1. Starts at 1, minus a penalty per issue; a concept that recurs in several chunks earns some back
    (recurrence is the strongest sign that a phrase names a concept of the text, not a one-off fragment).
    """
    issues = concept_issues(name) if issues is None else issues
    q = 1.0 - sum(ISSUE_WEIGHTS.get(i, 0.3) for i in issues)
    # recurrence supports a well-formed name; it cannot make a cut-off phrase, a clause or a pronoun fragment
    # meaningful (their words recur simply because they are common words of the text)
    if not set(issues) & {"truncated", "clause", "pronoun", "generic", "empty", "citation", "numeric", "verb_phrase"}:
        if mentions >= 3:
            q += 0.15
        elif mentions == 2:
            q += 0.05
    return round(max(0.0, min(1.0, q)), 3)


def is_named_entity(name: str, evidence_texts: Iterable[str] = ()) -> bool:
    """Acronym (COMPAS), mixed case (iPhone), or capitalised mid-sentence in the evidence (Cambridge Analytica)."""
    ws = words(name)
    if any(len(w) > 1 and (w.isupper() or any(c.isupper() for c in w[1:])) for w in ws):
        return True
    phrase = r"\b" + r"\W+".join(re.escape(w) for w in ws) + r"\b"
    for t in evidence_texts:
        for m in re.finditer(phrase, t or ""):
            if m.start() > 1 and not re.search(r"[.!?\"“:]\s*$", t[:m.start()]) and m.group(0)[:1].isupper():
                return True
    return False


# ----------------------------------------------------------------------------------------- naming
def canonical_concept(name: str, max_words: int = 6) -> str:
    """
    Clean a raw concept phrase WITHOUT cutting it mid-phrase (the old builder kept the first 4 words):
      * strip quotes, citations "(Smith 2019)", leading determiners / possessive pronouns ("their autonomy")
      * cut a trailing relative clause ("information that her file contained" -> "Information")
      * if still too long, drop trailing prepositional phrases from the right
      * drop dangling function words at the end
    Returns the phrase in its original casing (title-casing is the builder's job); "" if nothing is left.
    """
    s = (name or "").strip(" \t\n\r'\"`=:;,.").strip()
    s = re.sub(r"\(\s*[^)]*?\d{4}[a-z]?\s*[^)]*?\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(?:according\s+to|based\s+on|in\s+terms\s+of)\s+[^,]+,?\s*", "", s, flags=re.I)
    ws = s.split()
    while ws and ws[0].lower() in DETERMINERS:
        ws = ws[1:]
    low = _low(ws)
    for i, w in enumerate(low):
        if i >= 1 and w in CLAUSE_MARKERS:
            ws, low = ws[:i], low[:i]
            break
    while len(ws) > max_words:
        cut = max((i for i, w in enumerate(low) if w in PREPOSITIONS and i >= 1), default=None)
        if cut is None:
            return ""          # no structure to shorten without cutting a noun phrase in half
        ws, low = ws[:cut], low[:cut]
    while ws and low[-1] in DANGLING:
        ws, low = ws[:-1], low[:-1]
    return " ".join(ws)


def repair_from_evidence(name: str, texts: Iterable[str], max_extra: int = 4, max_words: int = 7) -> Optional[str]:
    """
    Recover the full phrase of a truncated concept from the sentences it was extracted from:
    "Respect For Practical And" + "... respect for practical and cognitive agency, which ..." ->
    "respect for practical and cognitive agency". The extension stops at punctuation, a clause marker or an
    auxiliary verb, and must end in a content word. None if no text continues the phrase cleanly.
    """
    ws = words(name)
    if not ws:
        return None
    pat = re.compile(r"\b" + r"[\W_]+".join(re.escape(w) for w in ws) + r"((?:[ \t\-]+[A-Za-z][\w'’\-]*){0," + str(max_extra) + "})",
                     re.IGNORECASE)
    best = None
    for t in texts:
        for m in pat.finditer(t or ""):
            ext = []
            for w in m.group(1).split():
                lw = w.lower().strip("-")
                if lw in CLAUSE_MARKERS or lw in AUXILIARIES:
                    break
                ext.append(w)
            base = m.group(0).split()[:len(ws)]
            for k in range(1, len(ext) + 1):          # shortest completion that closes the phrase
                last = ext[k - 1].lower().replace("’", "'")
                if last in DANGLING or re.search(r"'s?$", last) or _adjectival(last):
                    continue
                cand = " ".join(base + ext[:k])
                if len(words(cand)) <= max_words and not concept_issues(cand, max_words):
                    if best is None or len(cand) < len(best):
                        best = cand
                break
    return best


# ----------------------------------------------------------------------------------------- relations
# Phrasings whose subject and object must be SWAPPED to read as a canonical predicate:
#   "A is necessary for B" = "B requires A"
REVERSED_RELATIONS: Tuple[Tuple[str, str], ...] = (
    (r"\b(?:is|are|be)?\s*(?:a\s+)?(?:necessary|required|needed|essential|indispensable)\s+(?:for|to)\b", "requires"),
    (r"\b(?:is|are|be)?\s*(?:a\s+|the\s+)?(?:precondition|prerequisite|pre-?requisite|necessary\s+condition|requirement|condition)\s+(?:for|of)\b", "requires"),
    (r"^prerequisite$|^precondition$", "requires"),
    (r"\b(?:is|are|be)?\s*(?:a\s+)?(?:result|consequence|effect|product)\s+of\b|\b(?:is|are)\s+caused\s+by\b|\bresults?\s+from\b|\bstems?\s+from\b|\barises?\s+from\b", "causes"),
    (r"\b(?:is|are)\s+implied\s+by\b|\bfollows?\s+from\b", "implies"),
    (r"\b(?:is|are)\s+required\s+by\b|\b(?:is|are)\s+presupposed\s+by\b", "requires"),
    (r"\b(?:is|are)\s+excluded\s+by\b|\b(?:is|are)\s+precluded\s+by\b|\b(?:is|are)\s+prevented\s+by\b", "excludes"),
)
# Phrasings that read directly as a canonical predicate (added to kg_builder's verb table)
DIRECT_RELATIONS: Tuple[Tuple[str, str], ...] = (
    (r"\bpresuppos(?:e|es|ed|ing)\b|\bpresumes?\b", "requires"),
    (r"\b(?:is|are)\s+(?:based|grounded|founded|premised)\s+(?:on|in|upon)\b|^(?:based|grounded|founded|premised)\s+(?:on|in|upon)$|\brests?\s+(?:on|upon)\b|\bhinges?\s+on\b", "depends on"),
    (r"\b(?:is|are)\s+(?:an?\s+)?(?:instance|example|case|species|subtype|variety)\s+of\b", "is a kind of"),
    (r"\b(?:is|are)\s+(?:a\s+)?(?:sufficient\s+(?:condition\s+)?for)\b|\bguarantees?\b|\bensures?\b", "implies"),
    (r"\bviolates?\b|\bis\s+inconsistent\s+with\b|\bcannot\s+coexist\s+with\b|\brul(?:e|es|ed)\s+out\b", "excludes"),
)


def relation_from_statement(statement: str, subject: str, obj: str) -> Optional[str]:
    """The raw relation phrase between subject and object in a statement ("A is necessary for B." -> "is necessary for")."""
    st = statement or ""
    def find(name):
        ws = words(name)
        if not ws:
            return None
        return re.search(r"\b" + r"[\W_]+".join(re.escape(singular(w)[:max(3, len(singular(w)) - 1)]) + r"\w*" for w in ws), st, re.I)
    ms, mo = find(subject), find(obj)
    if not ms or not mo or ms.end() >= mo.start():
        return None
    mid = st[ms.end():mo.start()].strip(" ,;:-")
    mid = re.sub(r"\b(?:the|a|an)\s*$", "", mid, flags=re.I).strip()
    if not mid or len(mid.split()) > 8:
        return None
    return mid


def read_relation(raw: str) -> Tuple[Optional[str], bool]:
    """
    (canonical predicate, swap_subject_object) for phrasings the verb table does not cover, or (None, False).
    "is necessary for" -> ("requires", True); "is based on" -> ("depends on", False).
    """
    p = (raw or "").strip().lower()
    for pat, canon in REVERSED_RELATIONS:
        if re.search(pat, p):
            return canon, True
    for pat, canon in DIRECT_RELATIONS:
        if re.search(pat, p):
            return canon, False
    return None, False


def copula_relation(statement: str, obj: str) -> Optional[str]:
    """
    "X is a Y" / "X is a form of Y" -> "is a kind of". A bare copula ("X is Y") stays non-inferential: it is
    as often a description ("autonomy is valuable") as a classification.
    """
    st = (statement or "").lower()
    o = r"[\W_]+".join(re.escape(w.lower()) for w in words(obj))
    if not o:
        return None
    if len(words(obj)) == 1 and (obj.lower() in GENERIC_HEADS or singular(obj) in TOO_GENERAL_FOR_TAXONOMY):
        return None
    m = re.search(r"\b(?:is|are)\s+(?:an?|one)\s+(?:(?:kind|type|form|species|instance|variety)\s+of\s+(?:an?\s+)?)?(?:\w+\s+){0,2}" + o + r"\b", st)
    if m:
        nxt = re.match(r"\s*([a-z]+)", st[m.end():])
        if nxt and nxt.group(1) in PREPOSITIONS:
            return None          # "a source of", "an affront to": a relation, not a class
        return "is a kind of"
    return None


# ----------------------------------------------------------------------------------------- evidence gate
# An LLM's predicate label is not evidence of an inference: small models put "requires" / "causes" on edges
# whose evidence says nothing of the kind ("ChoicePoint causes Instant Result" <- "This is a common practice.").
# An inference predicate is kept only if its evidence states that family of relation, unhedged.
CUE_FAMILIES: Dict[str, str] = {
    "requires": r"\brequir\w*|\bdemand\w*|\bnecessit\w*|\bnecessary\b|\bneeded\b|\bneeds?\b|\bprerequisite\w*|"
                r"\bprecondition\w*|\bcondition\s+(?:of|for)\b|\bessential\s+(?:to|for)\b|\bpresuppos\w*|\bindispensable\b",
    "implies": r"\bimpl(?:y|ies|ied)\b|\bentail\w*|\bleads?\s+to\b|\bresults?\s+in\b|\bensur\w*|\bguarantee\w*|"
               r"\bsufficient\b|\bfollows?\s+from\b|\btherefore\b|\bthus\b|\bhence\b",
    "causes": r"\bcaus\w*|\bdue\s+to\b|\bresults?\s+(?:in|from)\b|\bleads?\s+to\b|\bproduc\w*|\bbecause\s+of\b|"
              r"\bgives?\s+rise\b|\btrigger\w*|\binduc\w*|\bstems?\s+from\b|\barises?\s+from\b|\bbrings?\s+about\b",
    "excludes": r"\bexclu\w*|\bincompatib\w*|\bcontradict\w*|\bpreclud\w*|\brul\w*\s+out\b|\bconflict\w*|"
                r"\binconsisten\w*|\bprevent\w*|\black\w*|\bviolat\w*|\bantithetical\b|\bcannot\s+coexist\b",
    "depends on": r"\bdepend\w*|\brel(?:y|ies|ied)\s+on\b|\bgrounded\b|\bbased\s+(?:on|upon)\b|\brests?\s+on\b|"
                  r"\bhinges?\s+on\b|\bturns?\s+on\b|\bfounded\s+on\b|\bpremised\b",
    "is a kind of": r"\b(?:kind|type|form|species|instance|example|variety|subset|case)\s+of\b|\bsuch\s+as\b|\bincluding\b|"
                    r"\b(?:is|are)\s+(?:an?|one)\s+\w+",
    "is defined as": r"\bdefin\w*|\bmeans\b|\bmeaning\b|\brefers?\s+to\b|\bis\s+called\b|\bdescrib\w*|\bi\.e\.|"
                     r"\bunderstood\s+as\b|\bconsists?\s+in\b|\bjust\s+is\b|\bthat\s+is\b",
    "proves": r"\bprov\w*|\bdemonstrat\w*|\bshows?\b",
}
HEDGES = MODALS | {"if", "whether", "suppose", "supposing", "unless", "perhaps", "possibly", "allegedly", "arguably",
                   "might", "not", "never", "no", "cannot", "seem", "seems", "appears"}


def inference_supported(predicate: str, evidence: str) -> Tuple[bool, str]:
    """
    (supported, reason) for an inference predicate. Supported when the evidence contains a cue of the same
    family (core.concepts.CUE_FAMILIES) with no hedge (modal, conditional, negation) in the 5 words before it
    and the sentence does not open with a conditional. Non-inference predicates are always "supported".
    """
    base = (predicate or "").strip().lower()
    pat = CUE_FAMILIES.get(base)
    if pat is None:
        return True, "not an inference predicate"
    ev = re.sub(r"(\w)- (\w)", r"\1\2", (evidence or "").replace("’", "'"))   # "Transform- ers"
    if not ev.strip():
        return False, "no evidence"
    found_hedged = False
    for sent in re.split(r"(?<=[.!?;])\s+", ev):
        low = sent.lower()
        opens_conditional = re.match(r"^\W*(?:if|whether|suppose|unless|when|even\s+if)\b", low) is not None
        for m in re.finditer(pat, low):
            before = re.findall(r"[a-z']+", low[:m.start()])[-5:]
            if opens_conditional or set(before) & HEDGES or any(w.endswith("n't") for w in before):
                found_hedged = True
                continue
            return True, f"cue '{m.group(0)}'"
    return False, ("hedged" if found_hedged else "no cue") + f" for '{base}' in evidence"


# cue phrasings whose grammatical subject is the relation's OBJECT ("B is necessary for A" = A requires B)
REVERSED_CUES: Dict[str, str] = {
    "requires": r"\bnecessary\b|\bneeded\b|\bprerequisite\w*|\bprecondition\w*|\bcondition\s+(?:of|for)\b|"
                r"\bessential\s+(?:to|for)\b|\bindispensable\b|\brequired\s+(?:for|by)\b|"
                r"\bdemanded\s+by\b|\bpresupposed\s+by\b|\bnecessitated\s+by\b",
    "causes": r"\bdue\s+to\b|\bresults?\s+from\b|\bbecause\s+of\b|\bstems?\s+from\b|\barises?\s+from\b|"
              r"\b(?:caused|produced|induced|triggered)\s+by\b",
    "implies": r"\bfollows?\s+from\b|\b(?:implied|entailed|guaranteed|ensured)\s+by\b",
    "depends on": r"\b(?:presupposed|relied\s+(?:up)?on)\s+by\b",
}
SYMMETRIC_PREDICATES = {"excludes"}


def _head_positions(concept: str, sentence_low: str) -> List[int]:
    """Character positions where the concept's head word (last content word, stem-matched) occurs."""
    content = [w for w in _low(words(concept)) if w not in ARTICLES | PREPOSITIONS | CONJUNCTIONS | DETERMINERS]
    if not content:
        return []
    head = re.sub(r"'s?$", "", content[-1])
    stem = singular(head)[:max(4, len(singular(head)) - 2)]
    return [m.start() for m in re.finditer(r"\b" + re.escape(stem) + r"\w*", sentence_low)]


# connective cues ("X. Therefore Y", "X, i.e. Y") have no grammatical subject of their own
CONNECTIVE_CUES = {"therefore", "thus", "hence", "i.e.", "that is", "such as", "including"}
# adverbs and fillers that are not part of a cue's subject
_LIGHT_WORDS = ARTICLES | {"also", "often", "then", "generally", "usually", "typically", "always", "necessarily",
                           "only", "just", "still", "directly", "strictly", "jointly", "together", "therefore",
                           "thus", "hence", "however", "indeed", "in", "fact", "turn", "particular", "moreover"}
_COPULAS = {"is", "are", "was", "were", "be", "been", "being"}
# words that open a new clause: a cue's subject never reaches back past one
_CLAUSE_OPENERS = {"that", "which", "who", "whom", "because", "although", "though", "while", "whereas", "but",
                   "if", "when", "since", "so", "unless", "whether", "where"}
# a demonstrative subject points back to a claim, not a concept ("this is implied by many theories")
_CLAUSE_ANAPHORS = {"this", "these", "those", "that"}
# "the proposal that X depends on Y", "the claim that ...": the clause is mentioned, not asserted
ATTITUDE_NOUNS = {"claim", "claims", "proposal", "proposals", "view", "views", "idea", "ideas", "hypothesis",
                  "hypotheses", "thesis", "suggestion", "possibility", "assumption", "assumptions", "notion",
                  "belief", "beliefs", "question", "worry", "objection", "conjecture", "speculation", "intuition",
                  "thought", "position", "argument", "contention", "allegation", "proposition"}
# nominal cues ("the entailment from X to Y", "a presupposition of X") have no verbal subject
_NOMINAL_CUE_RE = re.compile(r"(?:ment|ments|tion|tions|site|sites)$")


def _stem_of(word: str) -> str:
    w = singular(re.sub(r"'s?$", "", word))
    return w[:max(4, len(w) - 2)]


def _endpoint_matcher(subject: str, obj: str):
    """word -> True if it belongs to one of the endpoints (stem match, or the endpoint's acronym: IIT, GWT)."""
    stems, acronyms = set(), set()
    for c in (subject, obj):
        ws = _low(words(c))
        content = [x for x in ws if x not in ARTICLES | PREPOSITIONS | CONJUNCTIONS | DETERMINERS]
        stems |= {_stem_of(x) for x in content}
        if len(content) >= 3:          # two-letter "acronyms" are common words ("AI System" -> "as")
            acronyms.add("".join(x[0] for x in content))
        if len([x for x in ws if x not in ARTICLES]) >= 3:
            acronyms.add("".join(x[0] for x in ws if x not in ARTICLES))
    def match(w: str) -> bool:
        if w in acronyms:
            return True
        st = _stem_of(w)
        return any(st.startswith(e) or e.startswith(st) for e in stems if len(e) >= 3 and len(st) >= 3)
    return match


def _subject_phrase(low: str, cue_start: int) -> Tuple[List[str], List[str]]:
    """
    (subject words, parenthetical words) of the cue's subject noun phrase: back from the cue to the nearest
    clause boundary (punctuation or a clause opener such as "that"); if the span contains a copula ("Agency is
    the key requirement for", "recurrence is likely to be necessary"), only the part before the first copula
    is kept. Parentheticals are cut out of the phrase but returned, since they often name the subject
    ("the ability to track ... (known as metacognitive sensitivity) depends on").
    """
    raw_left = low[:cue_start]
    parens = re.findall(r"\(([^()]*)\)", raw_left[-200:])
    left = re.sub(r"\([^()]*\)", " ", raw_left)
    left = re.split(r"[,;:!?()\[\]\u2014\u2013]|\s-\s|\.\s", left)[-1]
    toks = re.findall(r"[a-z0-9][a-z0-9'\-]*", left)
    for i in range(len(toks) - 1, -1, -1):
        if toks[i] in _CLAUSE_OPENERS:
            toks = toks[i + 1:]
            break
    for i, t in enumerate(toks):
        if t in _COPULAS:
            toks = toks[:i]
            break
    np = [t for t in toks if t not in _LIGHT_WORDS and t not in AUXILIARIES and not t.endswith("ly")]
    paren_words = [w for p in parens for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", p)]
    return np, paren_words


def _clause_coverage(low: str, after: int, subject: str, obj: str) -> float:
    """Share of the 'that'-clause's content words that an endpoint names (best of the two endpoints)."""
    clause = re.split(r"[,;:.!?()]", low[after:], maxsplit=1)[0]
    cw = [w for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", clause)
          if w not in ARTICLES | PREPOSITIONS | CONJUNCTIONS | DETERMINERS | AUXILIARIES | {"that"}]
    if not cw:
        return 0.0
    best = 0.0
    for c in (subject, obj):
        m = _endpoint_matcher(c, "")
        best = max(best, sum(1 for w in cw if m(w)) / len(cw))
    return best


def _subject_check(low: str, m: "re.Match", base: str, subject: str, obj: str) -> Optional[str]:
    """Reason to reject the cue at m, or None if its grammatical subject may be one of the endpoints."""
    cue = m.group(0).strip()
    if cue in CONNECTIVE_CUES:
        return None
    # after "is a tool", "kind of" a "that" opens a relative clause about the class, not a claim
    tail = re.match(r"\s*,?\s*that\b", low[m.end():]) if base != "is a kind of" else None
    if tail and _clause_coverage(low, m.end() + tail.end(), subject, obj) < 0.5:
        # "theories imply that agency is necessary for consciousness": the cue relates a whole claim, which
        # neither endpoint names; "rectangularity entails that the interior angles sum to 360" is kept
        return f"cue '{cue}' introduces a clause that no endpoint names"
    # an article or demonstrative before the noun tells "the claim that" (mentioned) from "Hurley claims that"
    # and "her argument that" (asserted by an author, like any attributed theory claim)
    opener = re.search(r"\b(?:the|a|an|this|that|these|those|one|such|any|no)\s+"
                       r"(?:[a-z\-]+\s+)?([a-z]+)\s+(?:that|whether)\s+[^,;:]*$", low[:m.start()])
    if opener and opener.group(1) in ATTITUDE_NOUNS:
        return f"cue '{cue}' is inside 'the {opener.group(1)} that ...' (mentioned, not asserted)"
    np, paren_words = _subject_phrase(low, m.start())
    if not np or all(w in CONJUNCTIONS | PREPOSITIONS for w in np):
        return None                                    # cannot tell (sentence-initial cue, "..., and produces")
    match = _endpoint_matcher(subject, obj)
    if np[-1] in PRONOUNS - _CLAUSE_ANAPHORS or np[-1] in {"which", "who"}:
        # "it requires", "which requires": fine if an endpoint stands earlier in the sentence (the likely
        # antecedent); otherwise nothing in the evidence says the pronoun is that concept
        earlier = re.findall(r"[a-z0-9][a-z0-9'\-]*", low[:m.start()])
        if any(match(w) for w in earlier):
            return None
        return f"subject of '{cue}' is '{np[-1]}', and no endpoint precedes it in the sentence"
    if all(w in _CLAUSE_ANAPHORS | DETERMINERS for w in np):
        return f"subject of '{cue}' is '{' '.join(np)}' (a claim, not a concept)"
    if _NOMINAL_CUE_RE.search(cue.split()[0]) and not any(w in _COPULAS for w in re.findall(r"[a-z]+", low[max(0, m.start() - 40):m.start()])):
        return None                                    # "the entailment from X to Y": the side check decides
    if any(match(w) for w in np + paren_words):
        return None
    return f"subject of '{cue}' is '{' '.join(np[-4:])}', not an endpoint"


def check_inference(predicate: str, evidence: str, subject: str = "", obj: str = "") -> Dict[str, Any]:
    """
    Evidence check of one inference edge: {"ok", "reason", "direction"}.
      ok         the evidence states this family of relation (CUE_FAMILIES), unhedged, in a sentence that
                 mentions the subject or the object (by head word), and the cue's grammatical subject can be
                 an endpoint (not another noun, not "this"), and its argument is not a "that"-clause
      direction  "forward" (subject before the cue, object after; inverted for "necessary for", "due to" ...),
                 "reverse" (the LLM swapped the arguments: flip the edge), or None (cannot tell, or symmetric)
    Non-inference predicates: ok, reason "not an inference predicate".
    """
    base = (predicate or "").strip().lower()
    pat = CUE_FAMILIES.get(base)
    if pat is None:
        return {"ok": True, "reason": "not an inference predicate", "direction": None}
    ev = (evidence or "").replace("’", "'")
    if not ev.strip():
        return {"ok": False, "reason": "no evidence", "direction": None}
    hedged, off_topic, wrong_subject = False, False, None
    for sent in re.split(r"(?<=[.!?;])\s+", ev):
        low = sent.lower()
        opens_conditional = re.match(r"^\W*(?:if|whether|suppose|unless|when|even\s+if)\b", low) is not None
        sp, op = (_head_positions(subject, low) if subject else []), (_head_positions(obj, low) if obj else [])
        for m in re.finditer(pat, low):
            before = re.findall(r"[a-z']+", low[:m.start()])[-5:]
            if opens_conditional or set(before) & HEDGES or any(w.endswith("n't") for w in before):
                hedged = True
                continue
            if (subject or obj) and not sp and not op:
                off_topic = True
                continue
            bad_subject = _subject_check(low, m, base, subject, obj) if (subject or obj) else None
            if bad_subject:
                wrong_subject = bad_subject
                continue
            direction = None
            inverted = bool(REVERSED_CUES.get(base) and re.match(REVERSED_CUES[base], low[m.start():]))
            if base not in SYMMETRIC_PREDICATES and bool(sp) != bool(op):
                # one endpoint only: it must stand on its own side of the cue ("Cognitive agency requires ..."
                # cannot support "<something> requires Cognitive Agency")
                subj_side_before = not inverted
                if sp and not any((x < m.start()) == subj_side_before for x in sp):
                    off_topic = True
                    continue
                if op and not any((x > m.start()) == subj_side_before for x in op):
                    off_topic = True
                    continue
            if base not in SYMMETRIC_PREDICATES and sp and op:
                s_before, s_after = any(x < m.start() for x in sp), any(x > m.start() for x in sp)
                o_before, o_after = any(x < m.start() for x in op), any(x > m.start() for x in op)
                fwd = (s_before and o_after) if not inverted else (o_before and s_after)
                rev = (o_before and s_after) if not inverted else (s_before and o_after)
                direction = "forward" if fwd and not rev else ("reverse" if rev and not fwd else None)
            return {"ok": True, "reason": f"cue '{m.group(0)}'", "direction": direction}
    if wrong_subject and not hedged:
        return {"ok": False, "reason": wrong_subject, "direction": None}
    why = "hedged" if hedged else ("cue sentence is about other concepts" if off_topic else "no cue")
    return {"ok": False, "reason": f"{why} for '{base}' in evidence", "direction": None}


# ----------------------------------------------------------------------------------------- taxonomy
def specialization_of(specific: str, general: str) -> bool:
    """
    "Personal Autonomy" is a specialization of "Autonomy": the general phrase is the head (the last words) of
    the specific one, the extra words are modifiers (no preposition, no negating modifier such as "non-", "lack"),
    and the general phrase is not a generic word.
    """
    s, g = _low(words(specific)), _low(words(general))
    if not g or len(s) <= len(g) or s[-len(g):] != g:
        return False
    mods = s[:-len(g)]
    if any(m in PREPOSITIONS | CONJUNCTIONS | AUXILIARIES | DETERMINERS for m in s):
        return False
    if any(m in NEGATING_MODIFIERS or re.match(r"^(?:non|anti|pseudo|quasi|un|mis)-", m) for m in mods):
        return False
    # "Policies Using Technology", "Forming Justified Belief": a verb in the modifiers makes the first word the head
    if any(m.endswith("ing") for m in mods) or any(m in CLAUSE_MARKERS | CLAUSE_VERBS for m in mods):
        return False
    # "Other Relevant Concept", "Various Systems": determiner-like modifiers pick out members, not a subclass
    if mods and mods[0] in {"other", "another", "such", "various", "different", "same", "certain", "some", "many", "own",
                            "several", "these", "those", "both", "each", "every", "all", "any"}:
        return False
    if len(g) == 1 and (g[0] in GENERIC_HEADS or singular(g[0]) in TOO_GENERAL_FOR_TAXONOMY):
        return False
    if mods and mods[0] in LEADING_VERBS and (len(mods) == 1 or mods[1] not in PREPOSITIONS):
        return False                             # "Assess Consciousness" is an act, not a kind of Consciousness
    return True


def fix_legacy_singular(name: str) -> str:
    """Undo a bug of older builds that singularized 'Processes' to 'Processe' / 'Approaches' to 'Approache'."""
    keep = {"niche", "cache", "psyche", "cliche", "headache", "avalanche", "moustache", "quiche", "axe", "finesse",
            "largesse", "noblesse", "crevasse", "impasse", "lacrosse"}
    def fix(m):
        w = m.group(0)
        return w if w.lower() in keep else w[:-1]
    return re.sub(r"\b[A-Za-z]+(?:sse|xe|che|she)\b", fix, name)


def inference_cue(raw: str) -> Optional[Tuple[str, bool]]:
    """
    (canonical inference predicate, swap) if the relation phrase states an inference CATEGORICALLY, else None.
    Conservative on purpose, because a wrong inference edge creates false "verified" paths:
      * the cue must come first, after at most copulas / articles / light adverbs ("is the key requirement for")
      * no modal ("may be used to violate"), no contrast ("adds to rather than contradicts"), no pronoun
      * at most 7 words; negation means no inference ("does not require" licenses nothing)
    """
    p = re.sub(r"\s+", " ", (raw or "").strip().lower().replace("’", "'"))
    toks = re.findall(r"[a-z']+", p)
    if not toks or len(toks) > 7:
        return None
    if set(toks) & (MODALS | CONTRAST | PRONOUNS | {"not", "never", "no", "cannot"}) or "n't" in p:
        return None
    table = list(REVERSED_RELATIONS) + list(DIRECT_RELATIONS) + [
        (r"\brequires?\b|\bnecessitates?\b|\bdemands?\b", "requires"),
        (r"\bimpl(?:y|ies)\b|\bentails?\b|\bleads?\s+to\b|\bresults?\s+in\b", "implies"),
        (r"\bcauses?\b|\bproduces?\b|\binduces?\b", "causes"),
        (r"\bdepends?\s+on\b|\brel(?:y|ies)\s+on\b", "depends on"),
        (r"\b(?:an?\s+)?(?:kind|type|form)\s+of\b", "is a kind of"),
        (r"\bis\s+defined\s+as\b|\bdefined\s+as\b", "is defined as"),
        (r"\bexcludes?\b|\bprecludes?\b|\bcontradicts?\b|\bis\s+incompatible\s+with\b|\bincompatible\s+with\b", "excludes"),
    ]
    best = None                     # (start, end, canonical, swap) of the earliest cue
    for i, (pat, canon) in enumerate(table):
        m = re.search(pat, p)
        if m and (best is None or m.start() < best[0] or (m.start() == best[0] and m.end() > best[1])):
            best = (m.start(), m.end(), canon, i < len(REVERSED_RELATIONS))
    if best is None:
        return None
    start, end, canon, swap = best
    prefix = re.findall(r"[a-z']+", p[:start])
    if any(w not in CUE_PREFIX_OK for w in prefix):
        return None              # another verb comes first: "involves a kind of", "adds to ... contradicts"
    rest = re.findall(r"[a-z']+", p[end:])
    if any(w not in CUE_SUFFIX_OK for w in rest):
        return None              # the cue is part of a longer relation: "ensures causal [responsibility for]"
    return canon, swap


__all__ = [
    "concept_key", "concept_issues", "concept_quality", "is_named_entity", "canonical_concept",
    "repair_from_evidence", "relation_from_statement", "read_relation", "copula_relation", "specialization_of",
    "inference_cue", "fix_legacy_singular", "inference_supported", "check_inference", "CUE_FAMILIES",
    "GENERIC_HEADS", "ISSUE_WEIGHTS",
]
