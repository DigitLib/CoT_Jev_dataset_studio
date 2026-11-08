# Vendored from the nanoGentzen v4 model folder by tools/vendor_v4_engine.py. Do not edit by hand.
"""
kernel.py (nanoGentzen-v3, Phase 1)
Deterministic sequent-calculus kernel for intuitionistic propositional logic.

  1. Formula AST with a dedicated falsum node Bot() (no magic atom names)
  2. Normalised sequents: gamma de-duplicated and sorted, (A => Bot) written as ~A, goal Bot = empty delta
  3. Dyckhoff (1992) G4ip rules, with the negation cases of ~A = (A => Bot) spelled out
  4. g4ip_prove: complete, terminating decision procedure that returns a verifiable proof tree
  5. Strict parent -> child proof-tree verifier
  6. Kripke models: evaluation, and a search for checked countermodels (rooted frames up to 3 worlds)

The 11 LJ rule names in RULES are kept because the released policy network was trained on them;
search.py maps them onto G4ip moves. L_CONTR is kept only so that old trees remain checkable.
"""
import itertools
import re
import sys as _sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

if _sys.getrecursionlimit() < 10000:
    _sys.setrecursionlimit(10000)

# Phase 1: no identifier is ever read as falsum. Kept (empty) for import compatibility.
RESERVED_FALSUM: Set[str] = set()
FALSUM_SYMBOL = "⊥"  # printed form of Bot; the model input uses "0" as during training
CANONICAL_VARS: Tuple[str, ...] = ("P", "Q", "R", "S") + tuple(f"v{i}" for i in range(64))
TRAINING_VOCAB_VARS: Set[str] = {"P", "Q", "R", "S"} | {f"v{i}" for i in range(28)}


# =====================================================================
# 1. FORMULA AST & SEQUENTS
# =====================================================================

class Formula:
    def to_str(self, model: bool = False) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.to_str()

    def __eq__(self, other: Any) -> bool:
        return hasattr(other, "to_str") and self.to_str() == other.to_str()

    def __hash__(self) -> int:
        return hash(self.to_str())

    def __lt__(self, other: Any) -> bool:
        return self.to_str() < other.to_str()


@dataclass(frozen=True, eq=False)
class Var(Formula):
    name: str

    def to_str(self, model: bool = False) -> str:
        return self.name


@dataclass(frozen=True, eq=False)
class Bot(Formula):
    def to_str(self, model: bool = False) -> str:
        return "0" if model else FALSUM_SYMBOL


@dataclass(frozen=True, eq=False)
class Not(Formula):
    inner: Formula

    def to_str(self, model: bool = False) -> str:
        return f"~{self.inner.to_str(model)}"


@dataclass(frozen=True, eq=False)
class And(Formula):
    left: Formula
    right: Formula

    def to_str(self, model: bool = False) -> str:
        return f"({self.left.to_str(model)} & {self.right.to_str(model)})"


@dataclass(frozen=True, eq=False)
class Or(Formula):
    left: Formula
    right: Formula

    def to_str(self, model: bool = False) -> str:
        return f"({self.left.to_str(model)} | {self.right.to_str(model)})"


@dataclass(frozen=True, eq=False)
class Imp(Formula):
    left: Formula
    right: Formula

    def to_str(self, model: bool = False) -> str:
        return f"({self.left.to_str(model)} => {self.right.to_str(model)})"


BOT = Bot()


@dataclass(frozen=True, eq=False)
class Sequent:
    gamma: Tuple[Formula, ...]  # antecedents
    delta: Tuple[Formula, ...]  # succedent, |delta| <= 1; empty = Bot

    def __eq__(self, other: Any) -> bool:
        return hasattr(other, "gamma") and hasattr(other, "delta") and self.to_str() == other.to_str()

    def __hash__(self) -> int:
        return hash(self.to_str())

    def is_axiom(self) -> bool:
        """Identity (A in gamma, A the goal), ex falso (Bot in gamma), and A, ~A in gamma."""
        gset = set(self.gamma)
        if self.delta and self.delta[0] in gset:
            return True
        if BOT in gset:
            return True
        return any(isinstance(f, Not) and f.inner in gset for f in self.gamma)

    def to_str(self, model: bool = False) -> str:
        g = ", ".join(f.to_str(model) for f in self.gamma) if self.gamma else "0"
        d = ", ".join(f.to_str(model) for f in self.delta) if self.delta else "0"
        return f"{g} |- {d}"

    def model_str(self) -> str:
        """The string the policy network reads (falsum as '0', as in its training data)."""
        return self.to_str(model=True)

    def __repr__(self) -> str:
        return self.to_str()


def normalize_formula(f: Formula) -> Formula:
    """(A => Bot) becomes ~A, recursively. Everything else is unchanged."""
    if isinstance(f, (Var, Bot)):
        return f
    if isinstance(f, Not):
        return Not(normalize_formula(f.inner))
    l, r = normalize_formula(f.left), normalize_formula(f.right)
    if isinstance(f, Imp) and isinstance(r, Bot):
        return Not(l)
    return type(f)(l, r)


def normalize_sequent(seq: Sequent) -> Sequent:
    """Canonical form: normalised formulas, gamma without duplicates and sorted, goal Bot as empty delta."""
    gamma = sorted({normalize_formula(g) for g in seq.gamma}, key=lambda x: x.to_str())
    delta = tuple(normalize_formula(d) for d in seq.delta[:1])
    if delta and isinstance(delta[0], Bot):
        delta = ()
    return Sequent(tuple(gamma), delta)


def _seq(gamma, delta) -> Sequent:
    return normalize_sequent(Sequent(tuple(gamma), tuple(delta)))


# 11 LJ rule names (the policy head's output classes; do not reorder)
RULES: List[str] = ["AXIOM", "R_IMP", "L_IMP", "R_AND", "L_AND", "R_OR_1", "R_OR_2", "L_OR", "R_NOT", "L_NOT", "L_CONTR"]

# G4ip rule set used by search and by g4ip_prove
G4IP_RULES: List[str] = [
    "AXIOM", "R_IMP", "R_AND", "R_NOT", "R_OR_1", "R_OR_2", "L_AND", "L_OR",
    "L_IMP_BOT", "L_NOT_BOT", "L_IMP_ATOM", "L_IMP_AND", "L_IMP_OR", "L_NOT_AND", "L_NOT_OR",
    "L_IMP_IMP", "L_IMP_NOT", "L_NOT_IMP", "L_NOT_NOT",
]
INVERTIBLE_RIGHT_RULES: Tuple[str, ...] = ("R_IMP", "R_AND", "R_NOT")
INVERTIBLE_LEFT_RULES: Tuple[str, ...] = ("L_AND", "L_OR", "L_IMP_BOT", "L_NOT_BOT", "L_IMP_ATOM",
                                          "L_IMP_AND", "L_IMP_OR", "L_NOT_AND", "L_NOT_OR")
CHOICE_LEFT_RULES: Tuple[str, ...] = ("L_IMP_IMP", "L_IMP_NOT", "L_NOT_IMP", "L_NOT_NOT")
LEFT_RULE_NAMES = set(INVERTIBLE_LEFT_RULES) | set(CHOICE_LEFT_RULES) | {"L_IMP", "L_NOT", "L_CONTR"}


# =====================================================================
# 2. VARIABLES & BIJECTIVE ALPHA-RENAMING
# =====================================================================

def extract_ordered_vars(f: Formula) -> List[str]:
    if isinstance(f, Var):
        return [f.name]
    if isinstance(f, Bot):
        return []
    if isinstance(f, Not):
        return extract_ordered_vars(f.inner)
    return extract_ordered_vars(f.left) + extract_ordered_vars(f.right)


def extract_vars_set(f: Formula) -> Set[str]:
    return set(extract_ordered_vars(f))


def rename_formula(f: Formula, mapping: Dict[str, str]) -> Formula:
    if not mapping or isinstance(f, Bot):
        return f
    if isinstance(f, Var):
        return Var(mapping.get(f.name, f.name))
    if isinstance(f, Not):
        return Not(rename_formula(f.inner, mapping))
    return type(f)(rename_formula(f.left, mapping), rename_formula(f.right, mapping))


def canonicalize_sequent(seq: Sequent, force: bool = False) -> Tuple[Sequent, Dict[str, str], Dict[str, str]]:
    """Maps descriptive names onto the training symbols (goal variables first, then gamma in order)."""
    ordered: List[str] = []
    for f in list(seq.delta) + list(seq.gamma):
        for v in extract_ordered_vars(f):
            if v not in ordered:
                ordered.append(v)
    if not force and all(v in TRAINING_VOCAB_VARS for v in ordered):
        return seq, {}, {}
    fwd = {o: (CANONICAL_VARS[i] if i < len(CANONICAL_VARS) else f"v{i}") for i, o in enumerate(ordered)}
    rev = {c: o for o, c in fwd.items()}
    return (Sequent(tuple(rename_formula(g, fwd) for g in seq.gamma), tuple(rename_formula(d, fwd) for d in seq.delta)),
            fwd, rev)


def _shape(f: Formula) -> str:
    """Formula string with every variable replaced by x (name-independent ordering key)."""
    if isinstance(f, Var):
        return "x"
    if isinstance(f, Bot):
        return "0"
    if isinstance(f, Not):
        return "~" + _shape(f.inner)
    op = {"And": "&", "Or": "|", "Imp": "=>"}[type(f).__name__]
    return f"({_shape(f.left)} {op} {_shape(f.right)})"


def _colored(f: Formula, col: Dict[str, int], mark: Optional[str] = None) -> str:
    """Formula string with each variable replaced by its colour (and `mark` by *)."""
    if isinstance(f, Var):
        return "*" if f.name == mark else f"c{col[f.name]}"
    if isinstance(f, Bot):
        return "0"
    if isinstance(f, Not):
        return "~" + _colored(f.inner, col, mark)
    op = {"And": "&", "Or": "|", "Imp": "=>"}[type(f).__name__]
    return f"({_colored(f.left, col, mark)} {op} {_colored(f.right, col, mark)})"


def _refine(roles, occ, col: Dict[str, int]) -> Dict[str, int]:
    """Colour refinement: a variable is described by its colour and the shapes (with colours) of the formulas
    it occurs in, until the partition is stable. Colours are ranks, so the result is name-independent."""
    while True:
        sig = {v: (col[v], tuple(sorted(roles[i][0] + _colored(roles[i][1], col, v) for i in occ[v]))) for v in col}
        keys = sorted(set(sig.values()))
        rank = {k: i for i, k in enumerate(keys)}
        new = {v: rank[sig[v]] for v in col}
        if len(keys) == len(set(col.values())):
            return new
        col = new


def _var_order(seq: Sequent) -> List[str]:
    """Name-independent order of the variables: colour refinement, then individualisation of the first
    ambiguous class, choosing the member that gives the smallest coloured sequent (greedy, so tied members
    are interchangeable whenever they are symmetric, which is the common case: unrelated facts)."""
    roles = [("g", f) for f in seq.gamma] + [("d", f) for f in seq.delta]
    vs = sorted({v for _, f in roles for v in extract_ordered_vars(f)})
    occ = {v: [i for i, (_, f) in enumerate(roles) if v in extract_ordered_vars(f)] for v in vs}
    col = _refine(roles, occ, {v: 0 for v in vs})
    picture = lambda c: (tuple(sorted(_colored(f, c) for f in seq.gamma)), tuple(_colored(f, c) for f in seq.delta))  # noqa: E731
    while len(set(col.values())) < len(vs):
        counts: Dict[int, int] = {}
        for v in vs:
            counts[col[v]] = counts.get(col[v], 0) + 1
        target = min(c for c, n in counts.items() if n > 1)
        best = None
        for v in [x for x in vs if col[x] == target]:
            trial = {x: 2 * c + (1 if c == target and x != v else 0) for x, c in col.items()}
            trial = _refine(roles, occ, trial)
            key = picture(trial)
            if best is None or key < best[0]:
                best = (key, trial)
        col = best[1]
    return sorted(vs, key=lambda v: col[v])


def _var_name(i: int) -> str:
    return CANONICAL_VARS[i] if i < len(CANONICAL_VARS) else f"v{i}"


def model_view_sequent(seq: Sequent) -> Tuple[Sequent, List[int]]:
    """
    The sequent exactly as v4 models read it, and the premise order used: view.gamma[k] is a renaming of
    normalize_sequent(seq).gamma[order[k]]. Variables are named P, Q, R, S, v0, v1, ... in a name-independent
    order (_var_order), premises are sorted by shape and then by renamed text, so sequents that differ only
    in variable names and premise order get the same view.
    """
    seq = normalize_sequent(seq)
    fwd = {v: _var_name(i) for i, v in enumerate(_var_order(seq))}
    g = [rename_formula(f, fwd) for f in seq.gamma]
    order = sorted(range(len(g)), key=lambda i: (_shape(g[i]), g[i].to_str()))
    return Sequent(tuple(g[i] for i in order), tuple(rename_formula(d, fwd) for d in seq.delta)), order


def restore_sequent_str(seq_str: str, rev_map: Dict[str, str]) -> str:
    if not rev_map:
        return seq_str
    keys = sorted(rev_map.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")
    return pattern.sub(lambda m: rev_map.get(m.group(0), m.group(0)), seq_str)


def restore_proof_tree(node: Optional[dict], rev_map: Dict[str, str]) -> Optional[dict]:
    if node is None or not rev_map:
        return node
    return {"sequent": restore_sequent_str(node.get("sequent", ""), rev_map), "rule": node.get("rule", ""),
            "branches": [restore_proof_tree(b, rev_map) for b in node.get("branches", [])]}


# =====================================================================
# 3. INFERENCE RULES (backwards: conclusion -> premises)
# =====================================================================

def f_type(f: Optional[Formula]) -> str:
    return type(f).__name__ if f is not None else ""


def apply_rule(seq: Sequent, rule: str, idx: int = 0) -> Optional[List[Sequent]]:
    """Premises of `seq` under `rule` (pivot gamma[idx] for left rules), normalised; None if not applicable."""
    gamma, delta = list(seq.gamma), list(seq.delta)
    if rule == "AXIOM":
        return [] if seq.is_axiom() else None
    d0 = delta[0] if delta else None

    # ---- right rules
    if rule == "R_IMP" and isinstance(d0, Imp):
        return [_seq([d0.left] + gamma, [d0.right])]
    if rule == "R_AND" and isinstance(d0, And):
        return [_seq(gamma, [d0.left]), _seq(gamma, [d0.right])]
    if rule == "R_NOT" and isinstance(d0, Not):
        return [_seq([d0.inner] + gamma, [])]
    if rule == "R_OR_1" and isinstance(d0, Or):
        return [_seq(gamma, [d0.left])]
    if rule == "R_OR_2" and isinstance(d0, Or):
        return [_seq(gamma, [d0.right])]

    if idx < 0 or idx >= len(gamma):
        return None
    t = gamma[idx]
    rest = [f for i, f in enumerate(gamma) if i != idx]

    # ---- shared left rules
    if rule == "L_AND" and isinstance(t, And):
        return [_seq([t.left, t.right] + rest, delta)]
    if rule == "L_OR" and isinstance(t, Or):
        return [_seq([t.left] + rest, delta), _seq([t.right] + rest, delta)]

    # ---- legacy LJ rules (not used by search; kept so older trees can still be checked)
    if rule == "L_IMP" and isinstance(t, Imp):
        return [_seq(rest, [t.left]), _seq([t.right] + rest, delta)]
    if rule == "L_NOT" and isinstance(t, Not):
        return [_seq(rest, [t.inner])]
    if rule == "L_CONTR":
        return [Sequent(tuple([t, t] + rest), tuple(delta))]

    # ---- G4ip: implications by the shape of the antecedent
    if isinstance(t, Imp):
        a, b = t.left, t.right
        if rule == "L_IMP_BOT" and isinstance(a, Bot):
            return [_seq(rest, delta)]
        if rule == "L_IMP_ATOM" and isinstance(a, Var) and a in rest:
            return [_seq([b] + rest, delta)]
        if rule == "L_IMP_AND" and isinstance(a, And):
            return [_seq([Imp(a.left, Imp(a.right, b))] + rest, delta)]
        if rule == "L_IMP_OR" and isinstance(a, Or):
            return [_seq([Imp(a.left, b), Imp(a.right, b)] + rest, delta)]
        if rule == "L_IMP_IMP" and isinstance(a, Imp):
            return [_seq([a.left, Imp(a.right, b)] + rest, [a.right]), _seq([b] + rest, delta)]
        if rule == "L_IMP_NOT" and isinstance(a, Not):        # (~C => B): C, (Bot => B) |- Bot
            return [_seq([a.inner] + rest, []), _seq([b] + rest, delta)]

    # ---- G4ip: negations ~A = (A => Bot)
    if isinstance(t, Not):
        a = t.inner
        if rule == "L_NOT_BOT" and isinstance(a, Bot):
            return [_seq(rest, delta)]
        if rule == "L_NOT_AND" and isinstance(a, And):
            return [_seq([Imp(a.left, Not(a.right))] + rest, delta)]
        if rule == "L_NOT_OR" and isinstance(a, Or):
            return [_seq([Not(a.left), Not(a.right)] + rest, delta)]
        if rule == "L_NOT_IMP" and isinstance(a, Imp):        # second G4ip premise is Bot |- G (closed)
            return [_seq([a.left, Not(a.right)] + rest, [a.right])]
        if rule == "L_NOT_NOT" and isinstance(a, Not):        # ~~C: C |- Bot (second premise closed)
            return [_seq([a.inner] + rest, [])]
    return None


def applicable_moves(seq: Sequent, rules) -> List[Tuple[str, int]]:
    out = []
    for r in rules:
        if r.startswith("R_"):
            if apply_rule(seq, r, 0) is not None:
                out.append((r, 0))
        else:
            for i in range(len(seq.gamma)):
                if apply_rule(seq, r, i) is not None:
                    out.append((r, i))
    return out


def rule_label(rule: str, idx: int) -> str:
    return f"{rule}_{idx}" if rule.startswith("L_") else rule


# =====================================================================
# 4. G4ip DECISION PROCEDURE (complete, terminating, proof-producing)
# =====================================================================

class SearchTimeout(Exception):
    pass


def g4ip_prove(seq: Sequent, deadline_s: Optional[float] = None,
               memo: Optional[Dict[str, Optional[dict]]] = None) -> Tuple[Optional[dict], bool]:
    """
    Decides `seq` in G4ip. Returns (proof_tree or None, timed_out).
    timed_out=False with None means the sequent is NOT intuitionistically provable.
    """
    end = None if deadline_s is None else time.perf_counter() + deadline_s
    memo = {} if memo is None else memo

    def go(s: Sequent) -> Optional[dict]:
        key = s.to_str()
        if key in memo:
            return memo[key]
        if end is not None and time.perf_counter() > end:
            raise SearchTimeout()
        if s.is_axiom():
            memo[key] = {"sequent": key, "rule": "AXIOM", "branches": []}
            return memo[key]
        # invertible rules: the first applicable one decides the sequent
        for r, i in _first_invertible(s):
            prem = apply_rule(s, r, i)
            subs = []
            for p in prem:
                t = go(p)
                if t is None:
                    memo[key] = None
                    return None
                subs.append(t)
            memo[key] = {"sequent": key, "rule": rule_label(r, i), "branches": subs}
            return memo[key]
        # choice points
        for r, i in applicable_moves(s, ("R_OR_1", "R_OR_2") + CHOICE_LEFT_RULES):
            prem = apply_rule(s, r, i)
            subs = []
            for p in prem:
                t = go(p)
                if t is None:
                    break
                subs.append(t)
            if len(subs) == len(prem):
                memo[key] = {"sequent": key, "rule": rule_label(r, i), "branches": subs}
                return memo[key]
        memo[key] = None
        return None

    try:
        return go(normalize_sequent(seq)), False
    except SearchTimeout:
        return None, True


def _first_invertible(s: Sequent):
    for r in ("L_AND", "L_OR", "L_IMP_BOT", "L_NOT_BOT", "L_IMP_ATOM", "L_IMP_AND", "L_IMP_OR", "L_NOT_AND", "L_NOT_OR"):
        for i in range(len(s.gamma)):
            if apply_rule(s, r, i) is not None:
                return [(r, i)]
    for r in INVERTIBLE_RIGHT_RULES:
        if apply_rule(s, r, 0) is not None:
            return [(r, 0)]
    return []


def g4ip_provable(gamma, goal=None, deadline_s: Optional[float] = None) -> Optional[bool]:
    """True / False, or None when the deadline was hit."""
    tree, to = g4ip_prove(Sequent(tuple(gamma), (goal,) if goal is not None else ()), deadline_s)
    return None if to else tree is not None


# =====================================================================
# 5. STRICT PROOF-TREE VERIFIER
# =====================================================================

def verify_proof_tree_strict(node: dict, parser_fn: Optional[Callable] = None) -> bool:
    """Every node: sequent parses; AXIOM leaves are axioms; children equal apply_rule(parent, rule, pivot)."""
    if not isinstance(node, dict):
        return False
    if parser_fn is None:
        from .parser import parse_symbolic_sequent as parser_fn
    seq = parser_fn(node.get("sequent", ""))
    rule_raw, branches = node.get("rule", ""), node.get("branches", [])
    if seq is None:
        return False
    if rule_raw == "AXIOM":
        return len(branches) == 0 and seq.is_axiom()
    if rule_raw in ("R_OR_1", "R_OR_2"):
        name, indices = rule_raw, [0]
    elif "_" in rule_raw and rule_raw.rsplit("_", 1)[-1].isdigit():
        name, i = rule_raw.rsplit("_", 1)
        indices = [int(i)]
    else:
        name, indices = rule_raw, list(range(max(1, len(seq.gamma))))
    for i in indices:
        expected = apply_rule(seq, name, i)
        if expected is not None and len(expected) == len(branches) and expected:
            if all(parser_fn(b.get("sequent", "")) == e for b, e in zip(branches, expected)):
                return all(verify_proof_tree_strict(b, parser_fn) for b in branches)
    return False


verify_proof_tree = verify_proof_tree_strict


# =====================================================================
# 6. KRIPKE MODELS
# =====================================================================

@dataclass
class KripkeWorld:
    name: str
    verified_atoms: Set[str] = field(default_factory=set)

    def is_verified(self, atom: str) -> bool:
        return atom in self.verified_atoms


@dataclass
class KripkeModel:
    """Rooted intuitionistic Kripke model (reflexive, transitive accessibility; monotone valuation)."""
    origin: str
    worlds: Dict[str, KripkeWorld] = field(default_factory=dict)
    accessibility: Set[Tuple[str, str]] = field(default_factory=set)

    def add_world(self, name: str, atoms: Optional[Set[str]] = None) -> KripkeWorld:
        w = KripkeWorld(name=name, verified_atoms=set(atoms or set()))
        self.worlds[name] = w
        self.accessibility.add((name, name))
        return w

    def add_relation(self, h1: str, h2: str):
        self.accessibility.add((h1, h2))
        changed = True
        while changed:
            changed = False
            for (a, b) in list(self.accessibility):
                for (c, d) in list(self.accessibility):
                    if b == c and (a, d) not in self.accessibility:
                        self.accessibility.add((a, d)); changed = True
        for (a, b) in self.accessibility:
            if a in self.worlds and b in self.worlds:
                self.worlds[b].verified_atoms.update(self.worlds[a].verified_atoms)

    def evaluate(self, f: Formula, w: str) -> bool:
        if isinstance(f, Bot):
            return False
        if isinstance(f, Var):
            return self.worlds[w].is_verified(f.name)
        if isinstance(f, And):
            return self.evaluate(f.left, w) and self.evaluate(f.right, w)
        if isinstance(f, Or):
            return self.evaluate(f.left, w) or self.evaluate(f.right, w)
        succ = [b for (a, b) in self.accessibility if a == w]
        if isinstance(f, Imp):
            return all((not self.evaluate(f.left, v)) or self.evaluate(f.right, v) for v in succ)
        if isinstance(f, Not):
            return all(not self.evaluate(f.inner, v) for v in succ)
        return False

    def is_countermodel_for(self, seq: Sequent) -> bool:
        g = self.origin
        if not all(self.evaluate(p, g) for p in seq.gamma):
            return False
        return not any(self.evaluate(d, g) for d in seq.delta)  # empty delta (Bot) is never forced


def _frames(n: int):
    if n == 1:
        yield [{0}]
    elif n == 2:
        yield [{0, 1}, {1}]
    elif n == 3:
        yield [{0, 1, 2}, {1, 2}, {2}]          # chain
        yield [{0, 1, 2}, {1}, {2}]             # fork


def _upsets(up):
    n = len(up)
    for r in range(n + 1):
        for s in itertools.combinations(range(n), r):
            s = set(s)
            if all(up[w] <= s for w in s):
                yield frozenset(s)


def find_kripke_countermodel(seq: Sequent, max_worlds: int = 3, max_atoms: int = 8,
                             deadline_s: float = 0.05) -> Optional[Dict[str, Any]]:
    """Smallest rooted Kripke countermodel with up to `max_worlds` worlds, checked; None if none found in time."""
    seq = normalize_sequent(seq)
    atoms = sorted(set().union(*[extract_vars_set(f) for f in list(seq.gamma) + list(seq.delta)], set()))
    if len(atoms) > max_atoms:
        return None
    end = time.perf_counter() + deadline_s
    for n in range(1, max_worlds + 1):
        for up in _frames(n):
            ups = list(_upsets(up))
            for combo in itertools.product(ups, repeat=len(atoms)):
                if time.perf_counter() > end:
                    return None
                val = dict(zip(atoms, combo))
                m = KripkeModel(origin="w0")
                for w in range(n):
                    m.add_world(f"w{w}", {a for a in atoms if w in val[a]})
                for w in range(n):
                    for v in up[w]:
                        m.accessibility.add((f"w{w}", f"w{v}"))
                if m.is_countermodel_for(seq):
                    return {
                        "is_countermodel": True,
                        "n_worlds": n,
                        "origin": "w0",
                        "worlds": {k: sorted(v.verified_atoms) for k, v in m.worlds.items()},
                        "accessibility": sorted([a, b] for a, b in m.accessibility if a != b),
                        "refutes": seq.to_str(),
                        "kripke_structure": {"origin": "w0",
                                             "worlds": {k: sorted(v.verified_atoms) for k, v in m.worlds.items()},
                                             "accessibility": sorted(m.accessibility)},
                    }
    return None


def extract_kripke_countermodel(seq: Sequent, max_depth: int = 6) -> Dict[str, Any]:
    """Compatibility wrapper: returns a checked countermodel or {'is_countermodel': False}."""
    cm = find_kripke_countermodel(seq)
    if cm is not None:
        cm.setdefault("sequent", seq.to_str())
        cm.setdefault("diagnosis", "INTUITIONISTIC_UNDECIDED")
        return cm
    return {"sequent": seq.to_str(), "is_countermodel": False, "diagnosis": "UNKNOWN",
            "note": "no countermodel with <= 3 worlds found within the time limit"}


# =====================================================================
# 7. RELEVANCE & CURRY-HOWARD (unchanged in Phase 1; rebuilt in Phase 3)
# =====================================================================

def extract_relevance_witnesses(root_seq: Sequent, proof_tree: dict) -> Dict[str, Any]:
    """
    Premises of root_seq used by this proof: every rule pivot, the atom consumed by L_IMP_ATOM, the formula
    matched by an identity axiom, falsum and both halves of an A, ~A axiom. Proof-relative (another proof may
    use other premises). Sequents are parsed, not split on commas.
    """
    prem = [f.to_str() for f in root_seq.gamma]
    if not proof_tree or not isinstance(proof_tree, dict):
        return {"relevant_witnesses": [], "unused_premises": prem, "is_strictly_relevant": False, "relevance_ratio": 0.0}
    from .parser import parse_symbolic_sequent as parse
    used: Set[str] = set()

    def walk(node: dict):
        s = parse(node.get("sequent", ""))
        rule = node.get("rule", "")
        if s is not None:
            gset = set(s.gamma)
            if rule == "AXIOM":
                if s.delta and s.delta[0] in gset:
                    used.add(s.delta[0].to_str())
                if BOT in gset:
                    used.add(BOT.to_str())
                for f in s.gamma:
                    if isinstance(f, Not) and f.inner in gset:
                        used.update({f.to_str(), f.inner.to_str()})
            elif rule.startswith("L_") and rule.rsplit("_", 1)[-1].isdigit():
                name, i = rule.rsplit("_", 1)
                i = int(i)
                if 0 <= i < len(s.gamma):
                    piv = s.gamma[i]
                    used.add(piv.to_str())
                    if name == "L_IMP_ATOM" and isinstance(piv, Imp):
                        used.add(piv.left.to_str())
        for b in node.get("branches", []):
            walk(b)

    walk(proof_tree)
    rel = [p for p in prem if p in used]
    unused = [p for p in prem if p not in used]
    return {"relevant_witnesses": rel, "unused_premises": unused, "is_strictly_relevant": not unused,
            "relevance_ratio": round(len(rel) / len(prem), 4) if prem else 1.0}


def proof_tree_to_curry_howard(node: dict, simplify: bool = True) -> str:
    """Typed proof term of a proof tree (Phase 3, certificates.certify); legacy sketch only if that fails."""
    if not node or not isinstance(node, dict):
        return "nil"
    try:
        from . import certificates as _c
        cert = _c.certify(node)
        if cert.get("checked"):
            return cert["term_str"]
    except Exception:
        pass
    return _legacy_term_sketch(node)


def _legacy_term_sketch(node: dict) -> str:
    if not node or not isinstance(node, dict):
        return "nil"
    rule, s, br = node.get("rule", ""), node.get("sequent", ""), node.get("branches", [])
    sub = [_legacy_term_sketch(b) for b in br]
    clean = lambda x: (re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]", "_", x.strip())).strip("_") or "x")
    if rule == "AXIOM":
        d = s.split("|-")[1].strip() if "|-" in s else ""
        return clean(d) if d and d != "0" else "abort(contradiction)"
    if rule == "R_IMP":
        d = s.split("|-")[1].strip() if "|-" in s else ""
        ant = d.split("=>")[0].strip("( ") if "=>" in d else "premise"
        return f"λ({clean(ant)}: {ant}) -> {sub[0] if sub else 'x'}"
    if rule.startswith(("L_IMP", "L_NOT")):
        return f"apply({', '.join(sub)})" if sub else "apply()"
    if rule == "R_AND":
        return f"pair({', '.join(sub)})"
    if rule == "R_OR_1":
        return f"inl({sub[0]})"
    if rule == "R_OR_2":
        return f"inr({sub[0]})"
    if rule.startswith("L_OR"):
        return f"case(left => {sub[0]}, right => {sub[1]})"
    if rule == "R_NOT":
        return f"λ(x) -> {sub[0] if sub else 'abort'}"
    return sub[0] if sub else "qed"
