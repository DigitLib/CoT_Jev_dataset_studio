"""
parser_cot.py - ClearCoT front end: natural-language claims and knowledge-graph paths -> sequents.

Only the ClearCoT reasoning pipeline uses this module. It shares nothing with the Jev front end
(nanogentzen/jev/parser_jev.py) except the engine syntax in nanogentzen/logic_syntax.py.
"""

import operator
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..kernel import (
    And,
    Atom,
    Formula,
    Imp,
    Not,
    Or,
    Sequent,
    Var,
)

from ..logic_syntax import FormulaParser, parse_symbolic_sequent  # noqa: F401


# =====================================================================
# 2. IDENTIFIER & PHRASE CLEANING
# =====================================================================

def clean_atom_name(text: str, prefix: str = "", max_words: int = 4) -> str:
    """
    Produce a clean, alphanumeric identifier for a logical atom.
    Matches the ATOM regex token: [a-zA-Z_][a-zA-Z0-9_]*
    """
    words = re.findall(r'[a-zA-Z0-9]+', text)
    if not words:
        words = ["Entity"]
    meaningful = [
        w for w in words
        if w.lower() not in {"the", "a", "an", "is", "are", "of", "and", "or", "in", "to", "that", "this", "it", "with", "for", "on", "as", "by"}
    ]
    selected = meaningful[:max_words] if meaningful else words[:max_words]
    joined = "_".join(w.capitalize() for w in selected)
    if prefix:
        joined = f"{prefix}_{joined}"
    if joined and not (joined[0].isalpha() or joined[0] == '_'):
        joined = f"A_{joined}"
    return joined or "Atom"


def clean_term(phrase: str) -> Formula:
    """Converts a phrase into a clean atomic Formula proposition."""
    phrase_clean = phrase.strip()
    if not phrase_clean:
        return Atom("X")

    # 1. Exact valid identifier
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", phrase_clean):
        return Atom(phrase_clean)

    # 2. Single letter: uppercase
    if len(phrase_clean) == 1 and phrase_clean.isalpha():
        return Atom(phrase_clean.upper())

    # 3. Clean multi-word phrase into snake_case identifier
    words = re.findall(r"[A-Za-z0-9]+", phrase_clean)
    if not words:
        return Atom("X")

    filtered = []
    fluff = {"a", "an", "the", "that", "this", "it", "there"}
    for w in words:
        w_low = w.lower()
        if len(words) > 3 and w_low in fluff:
            continue
        filtered.append(w_low)

    name = "_".join(filtered) if filtered else "_".join(w.lower() for w in words)
    if name and name[0].isdigit():
        name = "v_" + name
    return Atom(name if name else phrase_clean)


# =====================================================================
# 3. NATURAL LANGUAGE CONDITIONAL & SYLLOGISM COMPILER
# =====================================================================

def parse_nl_formula(phrase: str) -> Formula:
    """Recursively splits natural language sub-clauses on 'or', 'and', and 'not'."""
    phrase = phrase.strip()

    or_parts = re.split(r"\bor\b|\|", phrase, maxsplit=1, flags=re.IGNORECASE)
    if len(or_parts) == 2 and or_parts[0].strip() and or_parts[1].strip():
        return Or(parse_nl_formula(or_parts[0]), parse_nl_formula(or_parts[1]))

    and_parts = re.split(r"\band\b|&", phrase, maxsplit=1, flags=re.IGNORECASE)
    if len(and_parts) == 2 and and_parts[0].strip() and and_parts[1].strip():
        return And(parse_nl_formula(and_parts[0]), parse_nl_formula(and_parts[1]))

    if re.match(r"^(?:not\s+|~\s*|it\s+is\s+not\s+(?:the\s+case\s+that\s+)?)", phrase, re.IGNORECASE):
        core = re.sub(
            r"^(?:not\s+|~\s*|it\s+is\s+not\s+(?:the\s+case\s+that\s+)?)",
            "",
            phrase,
            flags=re.IGNORECASE,
        )
        return Not(parse_nl_formula(core))

    return clean_term(phrase)


def compile_to_formula(expr: str) -> Formula:
    """
    Robust semantic compiler converting symbolic expressions, business rules,
    or proposition identifiers into formal Formula AST nodes.
    """
    expr_clean = expr.strip()
    if not expr_clean:
        return Atom("X")

    # 1. Symbolic formula / sequent
    if any(tok in expr_clean for tok in ("=>", "->", "|-", "&", "|", "~")):
        seq = parse_symbolic_sequent(f"0 |- {expr_clean}" if "|-" not in expr_clean else expr_clean)
        if seq is not None:
            if seq.delta:
                return seq.delta[0]
            if seq.gamma:
                return seq.gamma[0]

    # 2. Natural language conditional ("If A then B", "A implies B")
    if re.search(r"^(?:if|assuming|suppose|given|whenever|when|provided)\b|\b(?:implies|leads to|means)\b", expr_clean, re.IGNORECASE):
        nl_res = parse_natural_language(expr_clean)
        if nl_res is not None:
            seq = nl_res[0] if isinstance(nl_res, tuple) else nl_res
            if seq.gamma:
                return seq.gamma[0]
            if seq.delta:
                return seq.delta[0]

    # 3. Clean atomic / identifier fallback
    return clean_term(expr_clean)


def parse_natural_language(text: str) -> Optional[Tuple[Sequent, str]]:
    """Compiles multi-sentence natural language text into a formal Sequent."""
    text_clean = text.strip()
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text_clean) if s.strip()]
    if not sentences:
        return None

    has_q = "?" in text_clean
    target_q = sentences[-1]
    premise_sentences = sentences[:-1] if (has_q or len(sentences) > 1) else sentences

    premises: List[Formula] = []
    cond_pattern = re.compile(
        r"^(?:if|assuming|suppose|supposing|given\s+that|given|whenever|when|provided\s+that|provided)\s+",
        re.IGNORECASE,
    )

    for s in premise_sentences:
        lower_s = s.lower().strip()
        if cond_pattern.match(lower_s):
            content = cond_pattern.sub("", s).strip()
            parts = re.split(r"\bthen\b|,", content, maxsplit=1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                premises.append(Imp(parse_nl_formula(parts[0]), parse_nl_formula(parts[1])))
                continue

        match_imp = re.split(r"\bimplies\b|\bleads to\b|\bmeans\b", s, maxsplit=1, flags=re.IGNORECASE)
        if len(match_imp) == 2 and match_imp[0].strip() and match_imp[1].strip():
            premises.append(Imp(parse_nl_formula(match_imp[0]), parse_nl_formula(match_imp[1])))
            continue

        premises.append(parse_nl_formula(s))

    q_lower = target_q.lower().strip()
    if " if " in q_lower:
        consequent_part, antecedent_part = re.split(r"\bif\b", target_q, maxsplit=1, flags=re.IGNORECASE)
        goal = Imp(parse_nl_formula(antecedent_part), parse_nl_formula(consequent_part))
    elif cond_pattern.match(q_lower):
        content = cond_pattern.sub("", target_q).strip()
        parts = re.split(r"\bthen\b|,", content, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            goal = Imp(parse_nl_formula(parts[0]), parse_nl_formula(parts[1]))
        else:
            goal = parse_nl_formula(target_q)
    else:
        core_q = re.sub(r"^(?:is|are|does|do|should|can|could|would)\s+", "", target_q, flags=re.IGNORECASE)
        goal = parse_nl_formula(core_q)

    seq = Sequent(tuple(premises), (goal,))
    desc = f"Extracted {len(premises)} premise(s) |- Goal: {goal.to_str()}"
    return seq, desc


# =====================================================================
# 4. ADAPTED DATASET-STUDIO TEXT & PATH SEQUENT COMPILERS
# =====================================================================

def text_to_sequent(
    premises: List[str],
    conclusion: str,
    include_deductive_bridge: bool = True
) -> Sequent:
    """
    Turn natural language claims into a sound symbolic Sequent in Gentzen LI.
    Features:
      1. Inspects premises for internal conditionality ('if/then', 'implies', 'not').
      2. If premises are factual sentences, assigns clean atom identifiers.
      3. If include_deductive_bridge=True, includes the deductive bridging rule
         (P1 & ... & Pk => C) to guarantee formal provability in Gentzen LI.
    """
    p_formulas: List[Formula] = []
    has_inherent_bridge = False
    cond_re = re.compile(r"\b(?:if\b|implies\b|leads\s+to\b|=>|->)", re.IGNORECASE)

    def prop_key(text: str) -> str:
        """Shared atom name for a proposition, so 'Acoustic heritage is cultural.' and the antecedent of
        'If acoustic heritage is cultural then ...' are the same atom."""
        words = [w for w in re.findall(r"[a-z0-9]+", text.replace("_", " ").lower()) if w not in ("the", "a", "an")]
        name = "_".join(words) or "atom"
        return f"n_{name}" if name[0].isdigit() else name

    compiled: List[Tuple[str, Optional[Formula]]] = []
    for p in premises:
        p_clean = p.strip()
        if not p_clean:
            continue
        comp = None
        if cond_re.search(p_clean):
            try:
                comp = compile_to_formula(p_clean)
            except Exception:
                comp = None
        compiled.append((p_clean, comp))
    aligned = any(c is not None for _, c in compiled)   # real conditionals present: one naming scheme for all

    if aligned:
        from ..kernel import rename_formula, extract_vars_set
        succedent = Atom(prop_key(conclusion))
        for p_clean, comp in compiled:
            if comp is not None:
                comp = rename_formula(comp, {v: prop_key(v) for v in extract_vars_set(comp)})
                p_formulas.append(comp)
                if isinstance(comp, Imp) and comp.right.to_str() == succedent.to_str():
                    has_inherent_bridge = True
            else:
                p_formulas.append(Atom(prop_key(p_clean)))
    else:
        c_name = clean_atom_name(conclusion, prefix="C")
        succedent = Atom(c_name)
        for idx, (p_clean, _) in enumerate(compiled, 1):
            p_formulas.append(Atom(clean_atom_name(p_clean, prefix=f"P{idx}")))

    if not p_formulas:
        return Sequent((succedent,), (succedent,))

    if not include_deductive_bridge or has_inherent_bridge:
        return Sequent(tuple(p_formulas), (succedent,))

    if len(p_formulas) == 1:
        bridge = Imp(p_formulas[0], succedent)
        return Sequent((p_formulas[0], bridge), (succedent,))

    # Multi-premise conjunction bridge
    conj = p_formulas[0]
    for p in p_formulas[1:]:
        conj = And(conj, p)
    bridge = Imp(conj, succedent)

    return Sequent(tuple(p_formulas + [bridge]), (succedent,))


_EDGE_RE = re.compile(r"^\((.*)\)\s*--\[(.*?)\]\s*(-->|<--)\s*\((.*)\)$")


def parse_path_edges(path_repr: Optional[str]) -> Optional[List[Dict[str, str]]]:
    """
    "(A) --[requires]--> (B) -> (B) --[models] <-- (C)" ->
    [{"u": "A", "predicate": "requires", "direction": "forward", "v": "B"},
     {"u": "B", "predicate": "models", "direction": "backward", "v": "C"}]
    "backward" means the KG edge is v -> u and the path walks it against its direction.
    """
    if not path_repr or "--[" not in path_repr:
        return None
    edges = []
    for part in path_repr.split(" -> "):
        m = _EDGE_RE.match(part.strip())
        if not m:
            return None
        u, pred, arrow, v = m.groups()
        edges.append({"u": u.strip(), "predicate": pred.strip(), "v": v.strip(),
                      "direction": "forward" if arrow == "-->" else "backward"})
    return edges


# KG concepts that are extraction debris, not propositions: a proof about them proves nothing.
# Seen as atoms of "verified" ClearCoT samples (Both, Claim, Reader, Other_Concept ...). Generic words only:
# domain nouns ("Teacher", "Impact") can be real concepts and are left to the KG cleanup.
DEGENERATE_CONCEPTS = {
    "both", "either", "neither", "claim", "claims", "reader", "readers", "author", "authors",
    "other", "others", "other concept", "concept", "thing", "things", "something", "it", "this",
    "that", "they", "we", "one", "way", "case", "fact", "idea", "issue", "example",
    "chapter", "section", "book", "paper", "page", "figure", "table", "entity", "item",
}


def is_degenerate_concept(name: Optional[str]) -> bool:
    """True for KG node labels that cannot stand for a proposition (pronouns, 'Both', 'Claim', 'Reader' ...)."""
    words = re.findall(r"[a-z0-9]+", (name or "").lower())
    if not words:
        return True
    return " ".join(words) in DEGENERATE_CONCEPTS


def path_missing_links(report: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    What a non-entailment KG path is missing to become an argument: one entry per hop that does not
    license a forward inference, with the rule that would be needed and why it is absent.
    """
    out: List[Dict[str, str]] = []
    for e in report.get("edges") or []:
        u, v = e.get("u", ""), e.get("v", "")
        if e.get("degenerate"):
            why = "degenerate concept (not a proposition)"
        elif e.get("direction") != "forward" and e.get("logic") != "excludes":
            why = "edge walked against its direction"
        elif e.get("logic") == "none":
            why = f"relation '{e.get('predicate')}' licenses no inference"
        else:
            continue
        out.append({"needed_rule": f"{clean_atom_name(u)} => {clean_atom_name(v)}",
                    "hop": f"({u}) --[{e.get('predicate')}]--> ({v})", "reason": why})
    return out


def path_logic_report(path_nodes: List[str], path_repr: Optional[str] = None,
                      edges: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    How a KG path can be read logically.
      kg_entailment_chain  every edge licenses an inference and is walked forwards; the last edge may be an
                           exclusion, walked in either direction (A => B, C excludes B  |-  A => ~C)
      kg_relational        some edge is backwards, carries no inference, follows an exclusion,
                           or touches a degenerate concept (is_degenerate_concept)
      kg_unlabelled        no relation labels available: the path cannot be read as an argument
    """
    from .relation_logic import relation_logic

    edges = edges if edges is not None else parse_path_edges(path_repr)
    if not path_nodes or len(path_nodes) < 2 or not edges or len(edges) != len(path_nodes) - 1:
        return {"logic_basis": "kg_unlabelled", "edges": edges or [], "broken_at": None}
    out, broken = [], None
    for i, e in enumerate(edges):
        kind = relation_logic(e.get("predicate"))
        degenerate = is_degenerate_concept(e.get("u")) or is_degenerate_concept(e.get("v"))
        out.append({**e, "logic": kind, **({"degenerate": True} if degenerate else {})})
        if broken is None:
            # exclusion is symmetric (A => ~B  is equivalent to  B => ~A, intuitionistically too), so an
            # "excludes" edge may be walked backwards; an entailment may not ("rain, therefore sprinkler")
            backward_ok = kind == "excludes"
            if kind == "none" or degenerate or (e.get("direction") != "forward" and not backward_ok):
                broken = i
            elif kind == "excludes" and i < len(edges) - 1:
                broken = i + 1            # nothing follows from ~B
    return {"logic_basis": "kg_entailment_chain" if broken is None else "kg_relational",
            "edges": out, "broken_at": broken}


def path_to_sequent(path_nodes: List[str], path_repr: Optional[str] = None,
                    edges: Optional[List[Dict[str, str]]] = None) -> Sequent:
    """
    Encode a KG path n0 -> n1 -> ... -> nk as  n0, rules |- goal, where each edge contributes only
    the rule its relation licenses (nanogentzen/relation_logic.py), in the edge's own direction:
        forward  entails   n_i => n_{i+1}        backward entails   n_{i+1} => n_i
        forward  excludes  n_i => ~n_{i+1}       backward excludes  n_{i+1} => ~n_i
        none               no rule
    The goal is what the chain would establish if every hop were a forward inference (n_k, or ~n_k when
    the last hop excludes). A path walked against an edge, or through a non-inferential relation, is
    therefore usually not provable, which is the correct verdict: it is not an argument.
    Without relation labels (no path_repr / edges) the result has no rules at all.
    """
    if not path_nodes:
        atom = Atom("Goal")
        return Sequent((atom,), (atom,))
    atoms = [Atom(clean_atom_name(n)) for n in path_nodes]
    if len(atoms) == 1:
        return Sequent((atoms[0],), (atoms[0],))

    rep = path_logic_report(path_nodes, path_repr, edges)
    rules: List[Formula] = []
    for i, e in enumerate(rep["edges"] if rep["logic_basis"] != "kg_unlabelled" else []):
        u, v = atoms[i], atoms[i + 1]
        fwd = e.get("direction") == "forward"
        if e["logic"] == "entails":
            rules.append(Imp(u, v) if fwd else Imp(v, u))
        elif e["logic"] == "excludes":
            rules.append(Imp(u, Not(v)) if fwd else Imp(v, Not(u)))
    last = rep["edges"][-1] if rep["edges"] else {}
    # an exclusion as last hop establishes ~n_k in either direction: forward n_{k-1} => ~n_k directly,
    # backward n_k => ~n_{k-1} together with n_{k-1} (contraposition, valid intuitionistically)
    goal: Formula = Not(atoms[-1]) if last.get("logic") == "excludes" else atoms[-1]
    return Sequent(tuple([atoms[0]] + rules), (goal,))
