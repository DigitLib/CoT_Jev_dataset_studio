"""
logic_syntax.py - the symbolic formula / sequent language of the nanoGentzen verification engine.

Shared by both dataset pipelines (ClearCoT reasoning and Jev decisions) because it is the engine's own
syntax, not a task-specific front end. Task-specific front ends live in:
    nanogentzen/cot/parser_cot.py   natural language and KG paths  -> sequents
    nanogentzen/jev/parser_jev.py   Jev state JSON + policy rules  -> facts, rules, questions
"""

import operator
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .kernel import (
    And,
    Atom,
    Formula,
    Imp,
    Not,
    Or,
    Sequent,
    Var,
)


# =====================================================================
# 1. FORMULA & SEQUENT RECURSIVE DESCENT PARSER
# =====================================================================

class FormulaParser:
    """Recursive descent parser for propositional formulas and Gentzen sequents."""

    VALID_VARS = ("P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "A", "B", "D", "E", "C")

    def __init__(self, text: str, auto_remap_vars: bool = False):
        text = (
            text.replace("⟶", "|-")
            .replace("⊢", "|-")
            .replace("<=>", "<->")
            .replace("⇔", "<->")
            .replace("->", "=>")
            .replace("→", "=>")
            .replace("⇒", "=>")
            .replace("⊃", "=>")
            .replace("&&", "&")
            .replace("∧", "&")
            .replace("||", "|")
            .replace("∨", "|")
            .replace("¬", "~")
            .replace("!", "~")
        )

        open_count = text.count("(")
        close_count = text.count(")")
        if open_count > close_count:
            text = text + ")" * (open_count - close_count)
        elif close_count > open_count:
            text = "(" * (close_count - open_count) + text

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
            ("IFF", r"<->|\biff\b"),
            ("IMP", r"=>|\bimplies\b|\btherefore\b"),
            ("LPAREN", r"[\(\[\{]"),
            ("RPAREN", r"[\)\]\}]"),
            ("AND", r"&|\band\b"),
            ("OR", r"\||\bv\b|\bor\b"),
            ("NOT", r"~|\bnot\b"),
            ("COMMA", r",|;"),
            ("ZERO", r"\b0\b|\bfalse\b|\bBOT\b|\bFALSUM\b|⊥"),
            ("NUM", r"\b[0-9]+\b"),
            ("VAR", r"[A-Za-z_][A-Za-z0-9_]*"),
            ("SKIP", r"\s+"),
        ]
        tok_regex = "|".join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_spec)
        tokens: List[str] = []
        for mo in re.finditer(tok_regex, text, re.IGNORECASE):
            kind = mo.lastgroup
            val = mo.group()
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
                if self._peek() == "0":
                    self._consume()
                    if self._peek() == "|-":
                        break
                    gamma.append(Atom("0"))
                else:
                    gamma.append(self.parse_formula())
                if self._peek() == ",":
                    self._consume(",")
                else:
                    break

        if self._peek() == "|-":
            self._consume("|-")

        if self._peek():
            if self._peek() == "0":
                self._consume()
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
        if tok == "(":
            self._consume("(")
            expr = self.parse_formula()
            self._consume(")")
            return expr
        elif tok and re.match(r"^[A-Za-z0-9_]+$", tok):
            self._consume()
            return Atom(tok)
        raise ValueError(f"Unexpected token in formula: {tok}")


# Backward-compatible Parser alias
class Parser(FormulaParser):
    def __init__(self, text: str):
        super().__init__(text, auto_remap_vars=False)

    def peek(self) -> Optional[str]:
        p = self._peek()
        if p is None:
            return None
        type_map = {
            "=>": "IMP",
            "&": "AND",
            "|": "OR",
            "~": "NOT",
            "(": "LPAREN",
            ")": "RPAREN",
            "|-": "TURNSTILE",
            ",": "COMMA",
        }
        return type_map.get(p, "ATOM")

    def consume(self, expected_type: Optional[str] = None) -> Tuple[str, str]:
        tok = self._consume()
        return (self.peek() or "ATOM", tok)


def parse_formula(text: str) -> Formula:
    """Parse a single formula string."""
    return FormulaParser(text).parse_formula()


def parse_sequent(text: str) -> Sequent:
    """Parse a sequent string (e.g. 'A, B |- C' or 'P -> Q, P ⊢ Q')."""
    return FormulaParser(text).parse_sequent()


def parse_symbolic_sequent(text: str) -> Optional[Sequent]:
    """Safe symbolic sequent parser returning None on syntax error."""
    try:
        parser = FormulaParser(text, auto_remap_vars=False)
        seq = parser.parse_sequent()
        if parser.pos < len(parser.tokens):
            return None
        return seq
    except Exception:
        return None


__all__ = ['FormulaParser', 'Parser', 'parse_formula', 'parse_sequent', 'parse_symbolic_sequent']
