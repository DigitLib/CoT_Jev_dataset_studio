# Vendored from the nanoGentzen v4 model folder by tools/vendor_v4_engine.py. Do not edit by hand.
"""
parser.py (nanoGentzen-Jev Unified Real-World Compiler)
Compiles symbolic logic, natural-language policy rules, and rubric-guided states
into formal Gentzen Intuitionistic Sequents (Gamma |- Delta).
"""

import operator
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .kernel import BOT, And, Bot, Formula, Imp, Not, Or, Sequent, Var


class FormulaParser:
    """Recursive descent parser for propositional formulas and Gentzen sequents."""

    VALID_VARS = ("P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "A", "B", "D", "E", "C")

    def __init__(self, text: str, auto_remap_vars: bool = False):
        text = (
            text.replace("⟶", "|-")
            .replace("⊢", "|-")
            .replace("<=>", "⇔")
            .replace("<->", "⇔")
            .replace("->", "=>")
            .replace("⇒", "=>")
            .replace("⊃", "=>")
            .replace("&&", "&")
            .replace("∧", "&")
            .replace("||", "|")
            .replace("∨", "|")
            .replace("¬", "~")
            .replace("!", "~")
        )

        if text.count("(") + text.count("[") + text.count("{") != text.count(")") + text.count("]") + text.count("}"):
            raise ValueError(f"unbalanced brackets in {text!r}")

        self.raw_text = text
        self.auto_remap_vars = auto_remap_vars
        self.var_map: Dict[str, str] = {}
        self.tokens = self._tokenize(text)
        self.pos = 0

    def _remap_var_name(self, name: str) -> str:
        if not self.auto_remap_vars:
            return name
        up = name.upper()
        if len(up) == 1 and up.isalpha():
            return up
        if name in self.var_map:
            return self.var_map[name]
        used = set(self.var_map.values())
        for candidate in self.VALID_VARS:
            if candidate not in used:
                self.var_map[name] = candidate
                return candidate
        return "P"

    def _tokenize(self, text: str) -> List[str]:
        token_spec = [
            ("TURNSTILE", r"\|-"),
            ("IFF", r"⇔|\biff\b"),
            ("IMP", r"=>|\bimplies\b|\btherefore\b"),
            ("LPAREN", r"[\(\[\{]"),
            ("RPAREN", r"[\)\]\}]"),
            ("ZERO_SYM", r"⊥|_\|_"),
            ("AND", r"&|\band\b"),
            ("OR", r"\||\bor\b"),
            ("NOT", r"~|\bnot\b"),
            ("COMMA", r",|;"),
            ("ZERO", r"\b0\b|⊥|_\|_|(?-i:\bFALSUM\b)"),
            ("NUM", r"\b[0-9]+\b"),
            ("VAR", r"[A-Za-z_][A-Za-z0-9_]*"),
            ("SKIP", r"\s+"),
        ]
        tok_regex = "|".join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_spec)
        tokens: List[str] = []
        pos = 0
        for mo in re.finditer(tok_regex, text, re.IGNORECASE):
            if text[pos:mo.start()].strip():
                raise ValueError(f"unrecognised text {text[pos:mo.start()].strip()!r} in {text!r}")
            pos = mo.end()
            kind = mo.lastgroup
            val = mo.group()
            if kind == "ZERO_SYM" or (kind == "ZERO" and val != "0"):
                tokens.append("⊥")              # explicit falsum symbol: never an empty-side marker
                continue
            if kind == "SKIP":
                continue
            if kind == "AND":
                tokens.append("&")
            elif kind == "OR":
                tokens.append("|")
            elif kind == "NOT":
                tokens.append("~")
            elif kind == "IMP":
                tokens.append("=>")
            elif kind == "IFF":
                tokens.append("<->")
            elif kind == "TURNSTILE":
                tokens.append("|-")
            elif kind == "LPAREN":
                tokens.append("(")
            elif kind == "RPAREN":
                tokens.append(")")
            elif kind == "COMMA":
                tokens.append(",")
            elif kind == "ZERO":
                tokens.append("0")
            elif kind == "NUM":
                tokens.append(self._remap_var_name(f"v_{val}"))
            elif kind == "VAR":
                tokens.append(self._remap_var_name(val))
            else:
                tokens.append(val)
        if text[pos:].strip():
            raise ValueError(f"unrecognised text {text[pos:].strip()!r} in {text!r}")
        return tokens

    def _peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self, expected: Optional[str] = None) -> str:
        if self.pos >= len(self.tokens):
            raise ValueError(f"Unexpected end of input, expected {expected}")
        tok = self.tokens[self.pos]
        if expected and tok != expected:
            raise ValueError(f"Expected '{expected}', got '{tok}' at position {self.pos}")
        self.pos += 1
        return tok

    def parse_sequent(self) -> Sequent:
        if "|-" not in self.tokens:
            goal_formula = self.parse_formula()
            return Sequent((), (goal_formula,))

        gamma: List[Formula] = []
        delta: List[Formula] = []

        if self._peek() and self._peek() != "|-":
            while True:
                if self._peek() == "0" and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1] == "|-" and not gamma:
                    self._consume()          # solitary '0' before the turnstile: empty gamma
                    break
                gamma.append(self.parse_formula())
                if self._peek() == ",":
                    self._consume(",")
                else:
                    break

        if self._peek() == "|-":
            self._consume("|-")

        if self._peek():
            if self._peek() == "0" and self.pos + 1 == len(self.tokens):
                self._consume()              # solitary '0' after the turnstile: goal is falsum
            else:
                while self._peek():
                    delta.append(self.parse_formula())
                    if self._peek() == ",":
                        self._consume(",")
                    else:
                        break

        return Sequent(tuple(gamma), tuple(delta))

    def parse_formula(self) -> Formula:
        return self._parse_iff()

    def _parse_iff(self) -> Formula:
        left = self._parse_imp()
        while self._peek() == "<->":
            self._consume("<->")
            right = self._parse_imp()
            left = And(Imp(left, right), Imp(right, left))
        return left

    def _parse_imp(self) -> Formula:
        left = self._parse_or()
        if self._peek() == "=>":
            self._consume("=>")
            right = self._parse_imp()
            return Imp(left, right)
        return left

    def _parse_or(self) -> Formula:
        node = self._parse_and()
        while self._peek() == "|":
            self._consume("|")
            right = self._parse_and()
            node = Or(node, right)
        return node

    def _parse_and(self) -> Formula:
        node = self._parse_not()
        while self._peek() == "&":
            self._consume("&")
            right = self._parse_not()
            node = And(node, right)
        return node

    def _parse_not(self) -> Formula:
        if self._peek() == "~":
            self._consume("~")
            return Not(self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> Formula:
        tok = self._peek()
        if tok in ("0", "⊥"):
            self._consume()
            return BOT
        if tok == "(":
            self._consume("(")
            expr = self.parse_formula()
            self._consume(")")
            return expr
        elif tok and re.match(r"^[A-Za-z0-9_]+$", tok):
            self._consume()
            return Var(tok)
        raise ValueError(f"Unexpected token in formula: {tok}")


def parse_symbolic_sequent(text: str) -> Optional[Sequent]:
    try:
        parser = FormulaParser(text, auto_remap_vars=False)
        seq = parser.parse_sequent()
        if parser.pos < len(parser.tokens):
            return None
        return seq
    except Exception:
        return None


# =====================================================================
# NATURAL LANGUAGE CONDITIONAL & SYLLOGISM COMPILER
# =====================================================================

RESERVED_WORDS = {"and", "or", "not", "iff", "implies", "therefore", "falsum"}


def safe_identifier(name: str, report: Optional[dict] = None) -> str:
    """Renames identifiers that the formula parser would read as operators or falsum (e.g. 'or', '0')."""
    if not name:
        return name
    if name.lower() in RESERVED_WORDS or name[0].isdigit():
        new = f"f_{name}"
        if report is not None:
            report.setdefault("renamed_identifiers", {})[name] = new
        return new
    return name


def normalize_word(w: str) -> str:
    low = w.lower()
    if len(w) == 1 and w.isalpha():
        return w.upper()

    stopwords = {
        "is", "are", "in", "the", "an", "it", "did", "do", "does",
        "were", "was", "then", "of", "to", "there", "should", "could",
        "would", "can", "will", "shall", "must", "may", "might", "a",
    }
    if low in stopwords:
        return ""
    if low.endswith("ing") and len(low) > 4:
        low = low[:-3]
    elif low.endswith("ed") and len(low) > 3:
        low = low[:-2]
    elif low.endswith("es") and len(low) > 3:
        low = low[:-2]
    elif low.endswith("s") and len(low) > 2 and not low.endswith("ss"):
        low = low[:-1]
    return low.capitalize()


def clean_term(phrase: str) -> Formula:
    phrase_clean = phrase.strip()
    if not phrase_clean:
        return Var("X")

    # 1. Exact valid identifier (e.g. 'kyc_ok', 'is_adult', 'tx_amount'): preserve verbatim
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", phrase_clean):
        return Var(safe_identifier(phrase_clean))

    # 2. Single letter: uppercase
    if len(phrase_clean) == 1 and phrase_clean.isalpha():
        return Var(phrase_clean.upper())

    # 3. Clean multi-word phrase into snake_case identifier
    words = re.findall(r"[A-Za-z0-9]+", phrase_clean)
    if not words:
        return Var("X")

    # Filter only purely grammatical articles if phrase is long
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
    return Var(safe_identifier(name if name else phrase_clean))


def compile_to_formula(expr: str) -> Formula:
    """
    Robust semantic compiler converting symbolic expressions, business rules,
    or proposition identifiers into formal Formula AST nodes.
    """
    expr_clean = expr.strip()
    if not expr_clean:
        return Var("X")

    # 1. Symbolic formula / sequent
    if any(tok in expr_clean for tok in ("=>", "->", "|-", "&", "|", "~", "<->", "<=>", "⊥")):
        seq = parse_symbolic_sequent(f"0 |- {expr_clean}" if "|-" not in expr_clean else expr_clean)
        if seq is not None:
            if seq.delta:
                return seq.delta[0]
            if seq.gamma:
                return seq.gamma[0]
        try:
            FormulaParser(f"0 |- {expr_clean}").parse_sequent()
        except ValueError as e:
            raise ValueError(f"cannot parse rule {expr_clean!r}: {e}") from None
        raise ValueError(f"cannot parse rule {expr_clean!r}")

    # 2. Natural language conditional ("If A then B", "A implies B")
    if re.search(r"^(?:if|assuming|suppose|given|whenever|when|provided)\b|\b(?:implies|leads to|means)\b", expr_clean, re.IGNORECASE):
        if len([s for s in re.split(r"(?<!\d)[.!?\n]+(?!\d)", expr_clean) if s.strip()]) > 1:
            raise ValueError(f"rule {expr_clean!r} has several sentences; pass one rule per string")
        nl_res = parse_natural_language(expr_clean)
        if nl_res is not None:
            seq = nl_res[0] if isinstance(nl_res, tuple) else nl_res
            if seq.gamma:
                return seq.gamma[0]
            if seq.delta:
                return seq.delta[0]

    # 3. Clean atomic / identifier fallback
    return clean_term(expr_clean)


def _safe_compare(op_fn, a, b):
    try:
        return op_fn(a, b)
    except TypeError:
        try:
            return op_fn(float(a), float(b))
        except (ValueError, TypeError):
            return op_fn(str(a), str(b))


GENERIC_OPS = {
    "==": lambda a, b: a == b or str(a).strip().lower() == str(b).strip().lower(),
    "!=": lambda a, b: not (a == b or str(a).strip().lower() == str(b).strip().lower()),
    ">": lambda a, b: _safe_compare(operator.gt, a, b),
    ">=": lambda a, b: _safe_compare(operator.ge, a, b),
    "<": lambda a, b: _safe_compare(operator.lt, a, b),
    "<=": lambda a, b: _safe_compare(operator.le, a, b),
    "in": lambda a, b: (str(a).lower() in [str(x).lower() for x in b]) if isinstance(b, (list, tuple, set)) else (str(a).lower() in str(b).lower() if b is not None else False),
    "not in": lambda a, b: not ((str(a).lower() in [str(x).lower() for x in b]) if isinstance(b, (list, tuple, set)) else (str(a).lower() in str(b).lower() if b is not None else False)),
    "contains": lambda a, b: (str(b).lower() in [str(x).lower() for x in a]) if isinstance(a, (list, tuple, set)) else (str(b).lower() in str(a).lower() if a is not None else False),
    "exists": lambda a, _: a is not None and a != "",
    "regex": lambda a, b: bool(re.search(str(b), str(a), re.IGNORECASE)) if a is not None else False,
}


def resolve_dot_path(data: Any, path: str) -> Tuple[bool, Any]:
    """Resolves nested dot-separated or bracket paths across dicts and lists."""
    if not path or path == "$text":
        return True, str(data)
    curr = data
    parts = path.split(".")
    for part in parts:
        if isinstance(curr, dict) and part in curr:
            curr = curr[part]
        elif isinstance(curr, (list, tuple)) and part.isdigit() and int(part) < len(curr):
            curr = curr[int(part)]
        else:
            return False, None
    return True, curr


def ground_state_generic(
    state: Any,
    predicates: Optional[Dict[str, Any]] = None,
    report: Optional[dict] = None,
) -> Tuple[List[Formula], List[str]]:
    """
    Production-grade domain-agnostic state grounder.
    Converts arbitrary nested JSON objects and explicit/inferred predicates
    into certified factual Formula AST propositions without domain hardcoding.
    """
    facts: List[Formula] = []
    missing: List[str] = []

    # 1. Explicit Predicate Declarations (PredicateSpec / dict)
    if predicates and isinstance(predicates, dict):
        for atom_name, spec in predicates.items():
            if hasattr(spec, "field"):
                field_path, op_str, expected_val = spec.field, spec.op, spec.value
            elif isinstance(spec, dict):
                field_path = spec.get("field", atom_name)
                op_str = spec.get("op", "==")
                expected_val = spec.get("value", True)
            else:
                field_path, op_str, expected_val = atom_name, "==", spec

            found, actual_val = resolve_dot_path(state, field_path)
            atom_name = safe_identifier(atom_name, report)
            if not found or actual_val is None:
                missing.append(atom_name)
                continue

            op_fn = GENERIC_OPS.get(op_str, operator.eq)
            try:
                holds = bool(op_fn(actual_val, expected_val))
                facts.append(Var(atom_name) if holds else Not(Var(atom_name)))
            except Exception:
                missing.append(atom_name)

    # 2. Automatic Schema Traversal for Arbitrary JSON States
    if isinstance(state, dict):
        import hashlib

        def _ungrounded(path, why):
            if report is not None:
                report.setdefault("ungrounded_fields", []).append({"field": path, "reason": why})

        def _sign_atoms(base, v):
            sign = "positive" if v > 0 else ("zero" if v == 0 else "negative")
            for s in ("positive", "zero", "negative"):
                a = Var(safe_identifier(f"{base}_{s}", report))
                facts.append(a if s == sign else Not(a))

        def _walk_dict(d: dict, prefix: str = ""):
            for k, v in d.items():
                pfx = f"{prefix}_{k}" if prefix else k
                pfx_clean = safe_identifier(re.sub(r"[^A-Za-z0-9_]", "_", str(pfx)).strip("_"), report)
                if predicates and k in predicates:
                    continue
                if not pfx_clean:
                    _ungrounded(pfx, "key has no usable characters")
                elif isinstance(v, bool):
                    facts.append(Var(pfx_clean) if v else Not(Var(pfx_clean)))
                elif isinstance(v, (int, float)):
                    _sign_atoms(pfx_clean, v)
                elif isinstance(v, str):
                    val_clean = re.sub(r"[^A-Za-z0-9_]", "_", v.lower()).strip("_")
                    if not val_clean:
                        _ungrounded(pfx, "string has no usable characters")
                    elif len(val_clean) >= 32:
                        h = hashlib.sha1(v.encode("utf-8")).hexdigest()[:10]
                        facts.append(Var(f"{pfx_clean}_h{h}"))
                    else:
                        facts.append(Var(f"{pfx_clean}_{val_clean}"))
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        if isinstance(item, bool):
                            facts.append(Var(f"{pfx_clean}_has_{str(item).lower()}"))
                        elif isinstance(item, (int, float)):
                            facts.append(Var(f"{pfx_clean}_has_{str(item).replace('-', 'm').replace('.', 'p')}"))
                        elif isinstance(item, str):
                            item_clean = re.sub(r"[^A-Za-z0-9_]", "_", item.lower()).strip("_")
                            if item_clean:
                                facts.append(Var(f"{pfx_clean}_has_{item_clean[:40]}"))
                            else:
                                _ungrounded(pfx, "list item has no usable characters")
                        else:
                            _ungrounded(pfx, f"list item of type {type(item).__name__}")
                elif isinstance(v, dict):
                    _walk_dict(v, pfx_clean)
                elif v is None:
                    _ungrounded(pfx, "null")
                else:
                    _ungrounded(pfx, f"unsupported type {type(v).__name__}")

        _walk_dict(state)

    # Eliminate duplicate facts while preserving order
    seen: Set[str] = set()
    unique_facts: List[Formula] = []
    for f in facts:
        s = f.to_str()
        if s not in seen:
            seen.add(s)
            unique_facts.append(f)

    return unique_facts, missing


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


def parse_natural_language(text: str) -> Optional[Tuple[Sequent, str]]:
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
# UNIVERSAL TEXT & RUBRIC CRITERIA GROUNDER (ANY REAL-WORLD DOMAIN)
# =====================================================================

_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "can", "will", "should", "would", "could",
    "may", "might", "must", "into", "over", "under", "between", "within", "when",
    "where", "which", "who", "whom", "whose", "any", "all", "some", "such", "than",
    "then", "too", "very", "just", "only", "also", "more", "most", "other", "another",
    "each", "every", "either", "neither", "both", "about", "above", "after", "again",
    "against", "because", "before", "below", "during", "further", "here", "once",
    "same", "through", "until", "while", "does", "did", "doing", "done", "been",
    "being", "there", "their", "they", "them", "these", "those", "its", "our", "your",
}


def _stem_token(w: str) -> str:
    w = w.lower()
    for suf in ("ation", "ition", "ment", "ness", "able", "ible", "ingly", "edly", "ing", "ed", "ies", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            if suf == "ies":
                return w[:-3] + "y"
            return w[:-len(suf)]
    return w


def _extract_content_stems(text: str) -> List[Tuple[str, str]]:
    words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    out: List[Tuple[str, str]] = []
    for w in words:
        if len(w) >= 3 and w not in _STOPWORDS:
            out.append((w, _stem_token(w)))
    return out


def _state_to_flat_text(state) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return " . ".join(_state_to_flat_text(item) for item in state)
    if isinstance(state, dict):
        parts: List[str] = []

        def _walk(obj, prefix: str = ""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _walk(v, f"{prefix} {k}".strip())
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item, prefix)
            elif isinstance(obj, bool):
                parts.append(f"{prefix} {'true yes active' if obj else 'false no zero none'}")
            elif obj is None:
                parts.append(f"{prefix} none null missing zero")
            else:
                parts.append(f"{prefix} {obj}")

        _walk(state)
        return " . ".join(parts)
    return str(state)


def compile_criteria_and_state(
    state,
    q_id: str,
    q_type: str,
    options: List[str],
    criteria=None,
    instructions: Optional[str] = None,
    target_atom: Optional[str] = None,
) -> Tuple[List[Formula], List[Formula]]:
    """
    Universal zero-shot grounder for any real-world text-based or JSON-based decision task
    that provides `criteria` (dict for noul/choice, list for score) or `instructions`
    (such as benchmark datasets, customer tickets, security logs, medical notes, or audits).

    Extracts matching evidence atoms from `state` and synthesizes a 2-hop Horn
    chain (`(ev_1 & ev_2) => warrant_opt`, `warrant_opt => opt`) for Gentzen proof search.
    """
    if not criteria and not instructions:
        return [], []

    state_text = _state_to_flat_text(state)
    state_pairs = _extract_content_stems(state_text)
    state_stems = {st for _, st in state_pairs}
    state_words = {w for w, _ in state_pairs}

    crit_map: Dict[str, str] = {}
    if isinstance(criteria, dict):
        for k, v in criteria.items():
            crit_map[str(k)] = str(v)
    elif isinstance(criteria, list):
        for idx, v in enumerate(criteria):
            key = str(options[idx]) if idx < len(options) else str(idx)
            crit_map[key] = str(v)
    elif instructions and q_type == "noul":
        crit_map["true"] = str(instructions) if not isinstance(instructions, dict) else " . ".join(f"{k}: {v}" for k, v in instructions.items())

    if not crit_map:
        return [], []

    best_key: Optional[str] = None
    best_score = -1.0
    best_matched_words: List[str] = []

    for cand_key, desc in crit_map.items():
        sub_clauses = [
            c.strip()
            for c in re.split(r"\bor\b|;|,|\.", desc, flags=re.IGNORECASE)
            if len(c.strip()) >= 4
        ] or [desc]

        cand_best_clause_score = 0.0
        cand_matched: List[str] = []
        for cl in sub_clauses:
            cl_pairs = _extract_content_stems(cl)
            if not cl_pairs:
                continue
            matched = [
                w for w, st in cl_pairs if (st in state_stems or w in state_words)
            ]
            cl_has_neg = bool(
                re.search(
                    r"\b(no|not|none|never|without|zero|routine|benign|normal|clean)\b",
                    cl.lower(),
                )
            )
            state_has_neg = bool(
                re.search(
                    r"\b(no|not|none|never|without|zero|routine|benign|normal|clean)\b",
                    state_text.lower(),
                )
            )
            neg_bonus = 0.25 if (cl_has_neg and state_has_neg) else 0.0

            coverage = len(matched) / max(len(cl_pairs), 1)
            score = len(matched) * 1.0 + coverage * 1.5 + neg_bonus
            if score > cand_best_clause_score:
                cand_best_clause_score = score
                cand_matched = matched

        all_pairs = _extract_content_stems(desc)
        all_matched: List[str] = []
        for w, st in all_pairs:
            if (st in state_stems or w in state_words) and w not in all_matched:
                all_matched.append(w)
        total_score = cand_best_clause_score + 0.35 * len(all_matched)

        if total_score > best_score:
            best_score = total_score
            best_key = cand_key
            best_matched_words = all_matched[:3] if all_matched else cand_matched[:3]

    if best_key is None or best_score <= 0.2:
        return [], []

    safe_q = re.sub(r"[^A-Za-z0-9_]", "_", q_id).strip("_") or "q"
    safe_k = re.sub(r"[^A-Za-z0-9_]", "_", str(best_key)).strip("_") or "opt"

    ev_atoms: List[str] = []
    for idx, w in enumerate(best_matched_words[:2]):
        clean_w = re.sub(r"[^A-Za-z0-9_]", "", w)[:18] or f"ev{idx}"
        atom_name = f"ev_{safe_q}_{clean_w}"
        if atom_name not in ev_atoms:
            ev_atoms.append(atom_name)
    if not ev_atoms:
        ev_atoms = [f"ev_{safe_q}_matched"]

    facts: List[Formula] = [Var(a) for a in ev_atoms]
    prem_ast: Formula = Var(ev_atoms[0])
    for a in ev_atoms[1:]:
        prem_ast = And(prem_ast, Var(a))

    warrant_var = Var(f"warrant_{safe_q}_{safe_k}")

    if q_type == "noul":
        goal_name = target_atom or q_id
        goal_var = compile_to_formula(goal_name)
        if str(best_key).lower() in ("true", "yes", "1"):
            rules = [Imp(prem_ast, warrant_var), Imp(warrant_var, goal_var)]
        else:
            rules = [Imp(prem_ast, warrant_var), Imp(warrant_var, Not(goal_var))]
    elif q_type == "choice":
        goal_var = compile_to_formula(str(best_key))
        rules = [Imp(prem_ast, warrant_var), Imp(warrant_var, goal_var)]
    elif q_type == "score":
        goal_var = compile_to_formula(f"{safe_q}_{safe_k}")
        rules = [Imp(prem_ast, warrant_var), Imp(warrant_var, goal_var)]
    else:
        rules = []

    return facts, rules


# =====================================================================
# MODULAR DOMAIN REGISTRY INTEGRATION & UNIVERSAL CALIBRATION
# =====================================================================

from .domain_registry import (
    REGISTRY,
    register_domain,
    benchmark_detect_workflow,
    compile_benchmark_structured_decision,
    BENCHMARK_CALIBRATED_DISTRIBUTIONS,
)

# Backward-compatible aliases for legacy benchmark callers
detect_workflow = benchmark_detect_workflow
CALIBRATED_DISTRIBUTIONS = BENCHMARK_CALIBRATED_DISTRIBUTIONS


def get_calibrated_distribution(
    workflow: Optional[str],
    q_id: str,
    proven_label: str,
    options: List[str],
    state: Optional[dict] = None,
    epistemic_probs: Optional[dict] = None,
    temp: float = 0.68,
) -> Optional[dict]:
    """
    Universal Epistemic Temperature Calibration.
    Uses domain-specific registered empirical priors if available;
    otherwise falls back to dynamic Dirichlet/Epistemic calibration for unseen domains.
    """
    if not options:
        return None
    key = f"{workflow}/{q_id}/{proven_label}" if workflow else f"{q_id}/{proven_label}"

    base_dist = None
    if hasattr(REGISTRY, "get_calibrated_distribution"):
        base_dist = REGISTRY.get_calibrated_distribution(key)
    # Phase 0: no direct lookup in BENCHMARK_CALIBRATED_DISTRIBUTIONS; only enabled packs supply priors.

    if base_dist is not None and all(str(o) in base_dist for o in options):
        dist = {str(o): float(base_dist[str(o)]) for o in options}
        if isinstance(epistemic_probs, dict) and str(proven_label) in dist:
            p_true = float(epistemic_probs.get("true", 0.75))
            p_undec = float(epistemic_probs.get("undecided", 0.15))
            boost = 0.045 * (p_true - 0.70) - 0.02 * p_undec
            dist[str(proven_label)] = max(0.01, min(0.99, dist[str(proven_label)] + boost))

        scaled = {k: max(v, 1e-6) ** (1.0 / temp) for k, v in dist.items()}
        total = sum(scaled.values())
        return {str(o): round(scaled[str(o)] / total, 4) for o in options}

    # Universal Epistemic Temperature Calibration fallback for open/unseen domains
    p_true = float(epistemic_probs.get("true", 0.75)) if isinstance(epistemic_probs, dict) else 0.75
    p_undec = float(epistemic_probs.get("undecided", 0.15)) if isinstance(epistemic_probs, dict) else 0.15
    k = len(options)
    if k == 1:
        return {str(options[0]): 1.0}

    # Dynamic proven confidence calibrated to empirical accuracy (~0.75-0.85)
    p_win = max(0.40, min(0.92, 0.70 + 0.15 * (p_true - 0.5) - 0.10 * p_undec))
    rem = max(0.01, 1.0 - p_win)
    dist = {}
    for o in options:
        so = str(o)
        if so == str(proven_label):
            dist[so] = p_win
        else:
            dist[so] = rem / (k - 1)

    scaled = {k_opt: max(v, 1e-6) ** (1.0 / temp) for k_opt, v in dist.items()}
    total = sum(scaled.values())
    return {str(o): round(scaled[str(o)] / total, 4) for o in options}


def compile_structured_decision(
    state: dict, questions: Optional[dict] = None
) -> Tuple[List[Formula], List[Formula], Optional[str]]:
    """
    Compiles structured JSON state and question schemas into grounded
    propositional facts (`Var`) and Gentzen implication rules (`Imp`)
    for constructive verification in `kernel.py` + `NeuralProofSearch`.

    Delegates to registered domain compiler packs if matched, or uses
    domain-agnostic state grounder as universal fallback.
    """
    name, compiler = REGISTRY.match(state)
    if compiler is not None:
        return compiler(state, questions)

    # Universal domain-agnostic fallback:
    facts, _ = ground_state_generic(state)
    return facts, [], None
