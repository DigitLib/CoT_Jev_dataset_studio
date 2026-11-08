"""
kernel.py (nanoGentzen-v3 Kripke + Jev Unified Sequent Calculus Kernel)
100% Deterministic Gentzen Sequent Calculus Engine for Intuitionistic Logic (LI)
featuring:
  1. Bijective Alpha-Renaming (Canonical Symbol Compression)
  2. Dyckhoff G4ip Contraction-Free and Standard Gentzen LI Rule Sets
  3. Backward-compatible Atom/Var and Sequent (gamma/delta & antecedents/succedent) APIs
  4. Strict Parent->Child Recursive Proof-Tree Verifier (supporting both ProofNode and Dict trees)
  5. Saul Kripke (1965) Intuitionistic Model Theory & Semantic Tableaux Countermodel Extractor
  6. Intuitionistic Relevance Logic (IR) Witness Tracking
  7. Curry-Howard Typed Lambda-Calculus Proof Certificate Serialization
"""
import copy
import re
import sys as _sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

_sys.modules.setdefault("kernel", _sys.modules[__name__])
_sys.modules.setdefault("nanogentzen.kernel", _sys.modules[__name__])

RESERVED_FALSUM: Set[str] = {"0", "FALSUM", "false", "BOT", "_|_", "⊥"}
CANONICAL_VARS: Tuple[str, ...] = ("P", "Q", "R", "S") + tuple(f"v{i}" for i in range(64))
TRAINING_VOCAB_VARS: Set[str] = {"P", "Q", "R", "S"} | {f"v{i}" for i in range(28)}


# =====================================================================
# 1. FORMULA AST & SEQUENT DATASTRUCTURES
# =====================================================================

class Formula:
    def to_str(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.to_str()

    def __str__(self) -> str:
        return self.to_str()

    def __eq__(self, other: Any) -> bool:
        return hasattr(other, "to_str") and self.to_str() == other.to_str()

    def __hash__(self) -> int:
        return hash(self.to_str())

    def get_atoms(self) -> Set[str]:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class Var(Formula):
    name: str

    def to_str(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Var('{self.name}')"

    def __eq__(self, other: Any) -> bool:
        return hasattr(other, "to_str") and self.to_str() == other.to_str()

    def __hash__(self) -> int:
        return hash(self.to_str())

    def get_atoms(self) -> Set[str]:
        return {self.name} if self.name not in RESERVED_FALSUM else set()


@dataclass(frozen=True, eq=False)
class Atom(Var):
    def __repr__(self) -> str:
        return f"Atom('{self.name}')"

    def __eq__(self, other: Any) -> bool:
        return hasattr(other, "to_str") and self.to_str() == other.to_str()

    def __hash__(self) -> int:
        return hash(self.to_str())


@dataclass(frozen=True, eq=False)
class Not(Formula):
    inner: Formula

    def to_str(self) -> str:
        return f"~{self.inner.to_str()}"

    def __str__(self) -> str:
        return f"¬{self.inner}"

    def __repr__(self) -> str:
        return f"Not({repr(self.inner)})"

    def __eq__(self, other: Any) -> bool:
        return hasattr(other, "to_str") and self.to_str() == other.to_str()

    def __hash__(self) -> int:
        return hash(self.to_str())

    def get_atoms(self) -> Set[str]:
        return self.inner.get_atoms()


@dataclass(frozen=True, eq=False)
class And(Formula):
    left: Formula
    right: Formula

    def to_str(self) -> str:
        return f"({self.left.to_str()} & {self.right.to_str()})"

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"

    def __repr__(self) -> str:
        return f"And({repr(self.left)}, {repr(self.right)})"

    def __eq__(self, other: Any) -> bool:
        return hasattr(other, "to_str") and self.to_str() == other.to_str()

    def __hash__(self) -> int:
        return hash(self.to_str())

    def get_atoms(self) -> Set[str]:
        return self.left.get_atoms() | self.right.get_atoms()


@dataclass(frozen=True, eq=False)
class Or(Formula):
    left: Formula
    right: Formula

    def to_str(self) -> str:
        return f"({self.left.to_str()} | {self.right.to_str()})"

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"

    def __repr__(self) -> str:
        return f"Or({repr(self.left)}, {repr(self.right)})"

    def __eq__(self, other: Any) -> bool:
        return hasattr(other, "to_str") and self.to_str() == other.to_str()

    def __hash__(self) -> int:
        return hash(self.to_str())

    def get_atoms(self) -> Set[str]:
        return self.left.get_atoms() | self.right.get_atoms()


@dataclass(frozen=True, eq=False)
class Imp(Formula):
    left: Formula
    right: Formula

    def to_str(self) -> str:
        return f"({self.left.to_str()} => {self.right.to_str()})"

    def __str__(self) -> str:
        return f"({self.left} → {self.right})"

    def __repr__(self) -> str:
        return f"Imp({repr(self.left)}, {repr(self.right)})"

    def __eq__(self, other: Any) -> bool:
        return hasattr(other, "to_str") and self.to_str() == other.to_str()

    def __hash__(self) -> int:
        return hash(self.to_str())

    def get_atoms(self) -> Set[str]:
        return self.left.get_atoms() | self.right.get_atoms()


@dataclass(frozen=True)
class Sequent:
    gamma: Tuple[Formula, ...] = field(default_factory=tuple)  # Antecedents (Premises)
    delta: Tuple[Formula, ...] = field(default_factory=tuple)  # Succedents (|Delta| <= 1 for LI)

    def __init__(
        self,
        gamma: Any = None,
        delta: Any = None,
        antecedents: Any = None,
        succedent: Any = None,
    ):
        # Resolve gamma / antecedents
        g = gamma if gamma is not None else antecedents
        if g is None:
            g_tuple = ()
        elif isinstance(g, (list, tuple)):
            g_tuple = tuple(g)
        elif isinstance(g, Formula):
            g_tuple = (g,)
        else:
            g_tuple = tuple(g)

        # Resolve delta / succedent
        d = delta if delta is not None else succedent
        if d is None:
            d_tuple = ()
        elif isinstance(d, (list, tuple)):
            d_tuple = tuple(d)
        elif isinstance(d, Formula):
            d_tuple = (d,)
        else:
            d_tuple = tuple(d)

        object.__setattr__(self, "gamma", g_tuple)
        object.__setattr__(self, "delta", d_tuple)

    @property
    def antecedents(self) -> Tuple[Formula, ...]:
        return self.gamma

    @property
    def succedent(self) -> Optional[Formula]:
        return self.delta[0] if self.delta else None

    def is_axiom(self) -> bool:
        """Identity Axiom: Gamma, A |- A and Ex Falso Quodlibet: 0, Gamma |- Delta."""
        delta_set = set(self.delta)
        if any(f in delta_set for f in self.gamma):
            return True
        if any(isinstance(f, Var) and f.name in RESERVED_FALSUM for f in self.gamma):
            return True
        return False

    def to_str(self) -> str:
        g = ", ".join(f.to_str() for f in self.gamma) if self.gamma else "0"
        d = ", ".join(f.to_str() for f in self.delta) if self.delta else "0"
        return f"{g} |- {d}"

    def __str__(self) -> str:
        g_str = ", ".join(str(f) for f in self.gamma) if self.gamma else "∅"
        d_str = str(self.succedent) if self.succedent is not None else "⊥"
        return f"{g_str} ⊢ {d_str}"

    def __repr__(self) -> str:
        return self.to_str()

    def __eq__(self, other: Any) -> bool:
        if not hasattr(other, "gamma") or not hasattr(other, "delta"):
            return False
        return self.to_str() == other.to_str()

    def __hash__(self) -> int:
        return hash(self.to_str())

    def get_atoms(self) -> Set[str]:
        atoms = set()
        for f in self.gamma:
            atoms.update(f.get_atoms())
        for f in self.delta:
            atoms.update(f.get_atoms())
        return atoms


# 11 Standard Gentzen LI rules (matching configuration_nanogentzen.py num_rules=11)
RULES: List[str] = [
    "AXIOM",
    "R_IMP",
    "L_IMP",
    "R_AND",
    "L_AND",
    "R_OR_1",
    "R_OR_2",
    "L_OR",
    "R_NOT",
    "L_NOT",
    "L_CONTR",
]

# Legacy rule names mapping for studio v2 backward compatibility
RULE_NAMES: List[str] = [
    "Axiom",   # 0
    "Imp_R",   # 1
    "Imp_L",   # 2
    "And_R",   # 3
    "And_L1",  # 4
    "And_L2",  # 5
    "Or_R1",   # 6
    "Or_R2",   # 7
    "Or_L",    # 8
    "Not_R",   # 9
    "Not_L",   # 10
]

# Contraction-free Dyckhoff LJT (G4ip) rules
G4IP_RULES: List[str] = [
    "AXIOM",
    "R_IMP",
    "R_AND",
    "L_AND",
    "R_OR_1",
    "R_OR_2",
    "L_OR",
    "R_NOT",
    "L_NOT",
    "L_IMP_ATOM",
    "L_IMP_AND",
    "L_IMP_OR",
    "L_IMP_IMP",
]

INVERTIBLE_RIGHT_RULES: Tuple[str, ...] = ("R_IMP", "R_AND", "R_NOT")
INVERTIBLE_LEFT_RULES: Tuple[str, ...] = ("L_AND", "L_OR", "L_IMP_AND", "L_IMP_OR", "L_IMP_ATOM")


# ----------------------------------------------------------------------
# Proof Tree Node
# ----------------------------------------------------------------------

@dataclass
class ProofNode:
    rule: str
    conclusion: Sequent
    premises: List[Sequent] = field(default_factory=list)
    children: List['ProofNode'] = field(default_factory=list)
    pivot: Optional[int] = None
    provability_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "conclusion": str(self.conclusion),
            "pivot": self.pivot,
            "score": round(self.provability_score, 3),
            "children": [c.to_dict() for c in self.children],
        }


# =====================================================================
# 2. VARIABLE ANALYSIS & BIJECTIVE ALPHA-RENAMING
# =====================================================================

def extract_ordered_vars(f: Formula) -> List[str]:
    """Extracts variable names in deterministic left-to-right pre-order."""
    if isinstance(f, Var):
        return [] if f.name in RESERVED_FALSUM else [f.name]
    if isinstance(f, Not):
        return extract_ordered_vars(f.inner)
    if isinstance(f, (And, Or, Imp)):
        return extract_ordered_vars(f.left) + extract_ordered_vars(f.right)
    return []


def extract_vars_set(f: Formula) -> Set[str]:
    return set(extract_ordered_vars(f))


def rename_formula(f: Formula, mapping: Dict[str, str]) -> Formula:
    """Applies a variable renaming map to a Formula AST."""
    if not mapping:
        return f
    if isinstance(f, Var):
        if f.name in RESERVED_FALSUM:
            return f
        cls = Atom if isinstance(f, Atom) else Var
        return cls(mapping.get(f.name, f.name))
    if isinstance(f, Not):
        return Not(rename_formula(f.inner, mapping))
    if isinstance(f, And):
        return And(rename_formula(f.left, mapping), rename_formula(f.right, mapping))
    if isinstance(f, Or):
        return Or(rename_formula(f.left, mapping), rename_formula(f.right, mapping))
    if isinstance(f, Imp):
        return Imp(rename_formula(f.left, mapping), rename_formula(f.right, mapping))
    return f


def canonicalize_sequent(
    seq: Sequent, force: bool = False
) -> Tuple[Sequent, Dict[str, str], Dict[str, str]]:
    """
    Bijective Alpha-Renaming:
    Maps arbitrary descriptive variable names (e.g. 'kyc_verified', 'risk_limit_ok')
    into compact training-distribution symbols ('P', 'Q', 'R', 'S', 'v0', 'v1', ...).
    Goal variables in Delta are assigned first ('P'), followed by Gamma in order.
    """
    ordered_vars: List[str] = []
    for d in seq.delta:
        for v in extract_ordered_vars(d):
            if v not in ordered_vars:
                ordered_vars.append(v)
    for g in seq.gamma:
        for v in extract_ordered_vars(g):
            if v not in ordered_vars:
                ordered_vars.append(v)

    if not force and all(v in TRAINING_VOCAB_VARS for v in ordered_vars):
        return seq, {}, {}

    fwd_map: Dict[str, str] = {}
    rev_map: Dict[str, str] = {}
    for idx, orig_name in enumerate(ordered_vars):
        canon_name = CANONICAL_VARS[idx] if idx < len(CANONICAL_VARS) else f"v{idx}"
        fwd_map[orig_name] = canon_name
        rev_map[canon_name] = orig_name

    canon_seq = Sequent(
        gamma=tuple(rename_formula(g, fwd_map) for g in seq.gamma),
        delta=tuple(rename_formula(d, fwd_map) for d in seq.delta),
    )
    return canon_seq, fwd_map, rev_map


def restore_sequent_str(seq_str: str, rev_map: Dict[str, str]) -> str:
    """Restores human-readable variable names inside a serialized sequent string."""
    if not rev_map:
        return seq_str
    keys = sorted(rev_map.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")
    return pattern.sub(lambda m: rev_map.get(m.group(0), m.group(0)), seq_str)


def restore_proof_tree(node: Optional[dict], rev_map: Dict[str, str]) -> Optional[dict]:
    """Recursively restores original variable names across a verified proof tree."""
    if node is None or not rev_map:
        return node
    return {
        "sequent": restore_sequent_str(node.get("sequent", ""), rev_map),
        "rule": node.get("rule", ""),
        "branches": [restore_proof_tree(b, rev_map) for b in node.get("branches", [])],
    }


# =====================================================================
# 3. GENTZEN LI & DYCKHOFF G4ip INFERENCE TRANSITIONS
# =====================================================================

def f_type(f: Optional[Formula]) -> str:
    return getattr(f, "__class__", type(f)).__name__ if f is not None else ""


def apply_rule(seq: Sequent, rule: str, idx: int = 0) -> Optional[List[Sequent]]:
    """
    Applies inverse Gentzen LI rules backwards to reduce sequents into premises.
    Supports standard 11 LI rules as well as Dyckhoff LJT (G4ip) rules.
    """
    gamma, delta = list(seq.gamma), list(seq.delta)

    if rule == "AXIOM":
        return [] if seq.is_axiom() else None

    d0 = delta[0] if delta else None
    d0_t = f_type(d0)

    # (|- =>) : Gamma |- (A => B) decomposes to A, Gamma |- B
    if rule == "R_IMP" and d0 and (isinstance(d0, Imp) or d0_t == "Imp"):
        return [Sequent(tuple([d0.left] + gamma), (d0.right,))]

    # (|- &) : Gamma |- (A & B) decomposes to Gamma |- A and Gamma |- B
    if rule == "R_AND" and d0 and (isinstance(d0, And) or d0_t == "And"):
        return [
            Sequent(tuple(gamma), (d0.left,)),
            Sequent(tuple(gamma), (d0.right,)),
        ]

    # (|- ~) : Gamma |- ~A decomposes to A, Gamma |- 0
    if rule == "R_NOT" and d0 and (isinstance(d0, Not) or d0_t == "Not"):
        return [Sequent(tuple([d0.inner] + gamma), ())]

    # (|- v)1 : Gamma |- (A v B) decomposes to Gamma |- A
    if rule == "R_OR_1" and d0 and (isinstance(d0, Or) or d0_t == "Or"):
        return [Sequent(tuple(gamma), (d0.left,))]

    # (|- v)2 : Gamma |- (A v B) decomposes to Gamma |- B
    if rule == "R_OR_2" and d0 and (isinstance(d0, Or) or d0_t == "Or"):
        return [Sequent(tuple(gamma), (d0.right,))]

    # --- LEFT RULES (Requires valid index in Gamma) ---
    if idx < 0 or idx >= len(gamma):
        return None

    target = gamma[idx]
    rest = [f for i, f in enumerate(gamma) if i != idx]
    t_t = f_type(target)

    # (=> |-) Standard LI: (A => B), Gamma |- Delta decomposes to Gamma |- A and B, Gamma |- Delta
    if rule == "L_IMP" and (isinstance(target, Imp) or t_t == "Imp"):
        return [
            Sequent(tuple(rest), (target.left,)),
            Sequent(tuple([target.right] + rest), tuple(delta)),
        ]

    # (& |-) : (A & B), Gamma |- Delta decomposes to A, B, Gamma |- Delta
    if rule == "L_AND" and (isinstance(target, And) or t_t == "And"):
        return [Sequent(tuple([target.left, target.right] + rest), tuple(delta))]

    # (v |-) : (A v B), Gamma |- Delta decomposes to A, Gamma |- Delta and B, Gamma |- Delta
    if rule == "L_OR" and (isinstance(target, Or) or t_t == "Or"):
        return [
            Sequent(tuple([target.left] + rest), tuple(delta)),
            Sequent(tuple([target.right] + rest), tuple(delta)),
        ]

    # (~ |-) : ~A, Gamma |- Delta decomposes to Gamma |- A
    if rule == "L_NOT" and (isinstance(target, Not) or t_t == "Not"):
        return [Sequent(tuple(rest), (target.inner,))]

    # Structural Contraction (contr |-)
    if rule == "L_CONTR":
        return [
            Sequent(
                tuple([target, target] + rest),
                tuple(delta),
            )
        ]

    # --- DYCKHOFF LJT (G4ip) LEFT-IMPLICATION RULES ---
    if isinstance(target, Imp) or t_t == "Imp":
        ant, cons = target.left, target.right

        # 1. Atomic Antecedent: (P => B), P, Gamma |- Delta --> B, P, Gamma |- Delta
        if rule == "L_IMP_ATOM" and isinstance(ant, Var):
            if any(isinstance(f, Var) and f.name == ant.name for f in rest):
                return [Sequent(tuple([cons] + rest), tuple(delta))]

        # 2. Conjunctive Antecedent: ((C & D) => B), Gamma |- Delta --> (C => (D => B)), Gamma |- Delta
        elif rule == "L_IMP_AND" and isinstance(ant, And):
            curried = Imp(ant.left, Imp(ant.right, cons))
            return [Sequent(tuple([curried] + rest), tuple(delta))]

        # 3. Disjunctive Antecedent: ((C | D) => B), Gamma |- Delta --> (C => B), (D => B), Gamma |- Delta
        elif rule == "L_IMP_OR" and isinstance(ant, Or):
            return [
                Sequent(tuple([Imp(ant.left, cons), Imp(ant.right, cons)] + rest), tuple(delta))
            ]

        # 4. Implicational Antecedent: ((C => D) => B), Gamma |- Delta --> C, (D => B), Gamma |- D and B, Gamma |- Delta
        elif rule == "L_IMP_IMP" and isinstance(ant, Imp):
            d_imp_b = Imp(ant.right, cons)
            branch_1 = Sequent(tuple([ant.left, d_imp_b] + rest), (ant.right,))
            branch_2 = Sequent(tuple([cons] + rest), tuple(delta))
            return [branch_1, branch_2]

    return None


# =====================================================================
# 4. STRICT PROOF-TREE VERIFIER
# =====================================================================

def verify_proof_tree_strict(node: dict, parser_fn: Optional[Callable] = None, max_depth: int = 20) -> bool:
    """
    Recursively verifies every parent->child transition via apply_rule on dictionary trees.
    Validates that:
      1. Every node's sequent parses cleanly.
      2. If rule == "AXIOM", branches must be empty and sequent.is_axiom() must be True.
      3. For any inference rule, the child branches must strictly match the expected
         premises produced by apply_rule(seq, rule, idx).
      4. All leaf nodes terminate in valid axioms.
    """
    if not isinstance(node, dict) or max_depth <= 0:
        return False
    if parser_fn is None:
        try:
            from nanogentzen.logic_syntax import parse_symbolic_sequent
            parser_fn = parse_symbolic_sequent
        except ImportError:
            try:
                from parser import parse_symbolic_sequent
                parser_fn = parse_symbolic_sequent
            except ImportError:
                return False

    seq = parser_fn(node.get("sequent", ""))
    rule_raw = node.get("rule", "")
    branches = node.get("branches", [])
    if seq is None:
        return False

    if rule_raw == "AXIOM":
        return len(branches) == 0 and seq.is_axiom()
    if not branches:
        return False

    # Parse rule name and pivot index (e.g., "L_IMP_2" -> ("L_IMP", 2))
    if rule_raw in ("R_OR_1", "R_OR_2"):
        rule_name = rule_raw
        indices = [0]
    elif rule_raw.startswith("R_"):
        rule_name = rule_raw
        indices = [0]
    elif "_" in rule_raw and rule_raw.rsplit("_", 1)[-1].isdigit():
        rule_name, idx_str = rule_raw.rsplit("_", 1)
        indices = [int(idx_str)]
    else:
        rule_name = rule_raw
        indices = list(range(max(1, len(seq.gamma))))

    for idx in indices:
        expected = apply_rule(seq, rule_name, idx)
        if expected is not None and len(expected) == len(branches):
            if all(
                parser_fn(b.get("sequent", "")) == exp
                for b, exp in zip(branches, expected)
            ):
                return all(verify_proof_tree_strict(b, parser_fn, max_depth=max_depth - 1) for b in branches)
    return False


def verify_proof_tree_node(node: ProofNode) -> bool:
    """
    Recursively verify that a ProofNode tree is mathematically valid in Gentzen LI.
    Returns True only if every node strictly obeys the inference rules.
    """
    if not node:
        return False

    rule = node.rule
    seq = node.conclusion
    gamma = list(seq.antecedents)
    delta = seq.succedent
    gamma_set = set(gamma)

    # Axiom: A ⊢ A (A in gamma and delta == A or ex falso)
    if rule in ("Axiom", "AXIOM"):
        if delta is None:
            return seq.is_axiom()
        return delta in gamma_set or seq.is_axiom()

    # Right rules operate on delta
    if rule in ("Imp_R", "R_IMP"):
        if not isinstance(delta, Imp) or len(node.children) != 1:
            return False
        child = node.children[0]
        if set(child.conclusion.antecedents) != set(gamma + [delta.left]) or child.conclusion.succedent != delta.right:
            return False
        return verify_proof_tree_node(child)

    if rule in ("And_R", "R_AND"):
        if not isinstance(delta, And) or len(node.children) != 2:
            return False
        c1, c2 = node.children[0], node.children[1]
        if (
            set(c1.conclusion.antecedents) != gamma_set
            or c1.conclusion.succedent != delta.left
            or set(c2.conclusion.antecedents) != gamma_set
            or c2.conclusion.succedent != delta.right
        ):
            return False
        return verify_proof_tree_node(c1) and verify_proof_tree_node(c2)

    if rule in ("Or_R1", "R_OR_1"):
        if not isinstance(delta, Or) or len(node.children) != 1:
            return False
        child = node.children[0]
        return (
            set(child.conclusion.antecedents) == gamma_set
            and child.conclusion.succedent == delta.left
            and verify_proof_tree_node(child)
        )

    if rule in ("Or_R2", "R_OR_2"):
        if not isinstance(delta, Or) or len(node.children) != 1:
            return False
        child = node.children[0]
        return (
            set(child.conclusion.antecedents) == gamma_set
            and child.conclusion.succedent == delta.right
            and verify_proof_tree_node(child)
        )

    if rule in ("Not_R", "R_NOT"):
        if not isinstance(delta, Not) or len(node.children) != 1:
            return False
        child = node.children[0]
        return (
            set(child.conclusion.antecedents) == set(gamma + [delta.inner])
            and (child.conclusion.succedent is None or len(child.conclusion.delta) == 0)
            and verify_proof_tree_node(child)
        )

    # Left rules operate on a pivot antecedent
    if node.pivot is None or node.pivot < 0 or node.pivot >= len(gamma):
        return False

    pivot_form = gamma[node.pivot]
    gamma_without_pivot = [g for i, g in enumerate(gamma) if i != node.pivot]
    rest_set = set(gamma_without_pivot)

    if rule in ("And_L1", "L_AND"):
        if not isinstance(pivot_form, And) or len(node.children) != 1:
            return False
        child = node.children[0]
        return (
            set(child.conclusion.antecedents) == set(gamma_without_pivot + [pivot_form.left])
            and child.conclusion.succedent == delta
            and verify_proof_tree_node(child)
        )

    if rule == "And_L2":
        if not isinstance(pivot_form, And) or len(node.children) != 1:
            return False
        child = node.children[0]
        return (
            set(child.conclusion.antecedents) == set(gamma_without_pivot + [pivot_form.right])
            and child.conclusion.succedent == delta
            and verify_proof_tree_node(child)
        )

    if rule in ("Or_L", "L_OR"):
        if not isinstance(pivot_form, Or) or len(node.children) != 2:
            return False
        c1, c2 = node.children[0], node.children[1]
        return (
            set(c1.conclusion.antecedents) == set(gamma_without_pivot + [pivot_form.left])
            and c1.conclusion.succedent == delta
            and set(c2.conclusion.antecedents) == set(gamma_without_pivot + [pivot_form.right])
            and c2.conclusion.succedent == delta
            and verify_proof_tree_node(c1)
            and verify_proof_tree_node(c2)
        )

    if rule in ("Imp_L", "L_IMP", "L_IMP_ATOM"):
        if not isinstance(pivot_form, Imp) or len(node.children) not in (1, 2):
            return False
        # Dyckhoff L_IMP_ATOM single-premise case:  A, B, rest |- delta  /  A, A->B, rest |- delta
        # The antecedent A must itself be in the context; otherwise (A->C) |- C would "verify".
        if len(node.children) == 1:
            child = node.children[0]
            return (
                pivot_form.left in gamma_without_pivot
                and set(child.conclusion.antecedents) == set(gamma_without_pivot + [pivot_form.right])
                and child.conclusion.succedent == delta
                and verify_proof_tree_node(child)
            )
        c1, c2 = node.children[0], node.children[1]
        c1_gamma_set = set(c1.conclusion.antecedents)
        c2_gamma_set = set(c2.conclusion.antecedents)
        return (
            c1_gamma_set in (gamma_set, rest_set)
            and c1.conclusion.succedent == pivot_form.left
            and c2_gamma_set == set(gamma_without_pivot + [pivot_form.right])
            and c2.conclusion.succedent == delta
            and verify_proof_tree_node(c1)
            and verify_proof_tree_node(c2)
        )

    if rule in ("Not_L", "L_NOT"):
        if not isinstance(pivot_form, Not) or len(node.children) != 1:
            return False
        child = node.children[0]
        return (
            set(child.conclusion.antecedents) in (gamma_set, rest_set)
            and child.conclusion.succedent == pivot_form.inner
            and verify_proof_tree_node(child)
        )

    return False


def verify_proof_tree(target: Any, parser_fn: Optional[Callable] = None, max_depth: int = 20) -> bool:
    """
    Unified entrypoint for proof tree verification.
    Dispatches to verify_proof_tree_node if target is a ProofNode,
    or verify_proof_tree_strict if target is a dictionary representation.
    """
    if isinstance(target, ProofNode):
        return verify_proof_tree_node(target)
    if isinstance(target, dict):
        return verify_proof_tree_strict(target, parser_fn=parser_fn, max_depth=max_depth)
    return False


# =====================================================================
# 5. SAUL KRIPKE (1965) INTUITIONISTIC MODEL THEORY & TABLEAUX
# =====================================================================

@dataclass
class KripkeWorld:
    """An evidential state / world H in K with verified proposition letters."""
    name: str
    verified_atoms: Set[str] = field(default_factory=set)

    def is_verified(self, atom: str) -> bool:
        return atom in self.verified_atoms


@dataclass
class KripkeModel:
    """
    Intuitionistic Model Structure (G, K, R) with valuation phi(P, H).
    Satisfies Kripke 1965 Monotonicity/Hereditary condition:
      If phi(P, H) = T and H R H', then phi(P, H') = T.
    """
    origin: str
    worlds: Dict[str, KripkeWorld] = field(default_factory=dict)
    accessibility: Set[Tuple[str, str]] = field(default_factory=set)

    def add_world(self, name: str, atoms: Optional[Set[str]] = None) -> KripkeWorld:
        world = KripkeWorld(name=name, verified_atoms=atoms or set())
        self.worlds[name] = world
        self.accessibility.add((name, name))  # Reflexive
        return world

    def add_relation(self, h1: str, h2: str):
        self.accessibility.add((h1, h2))
        # Enforce transitivity and hereditary atom transmission
        changed = True
        while changed:
            changed = False
            for (w1, w2) in list(self.accessibility):
                for (w2_b, w3) in list(self.accessibility):
                    if w2 == w2_b and (w1, w3) not in self.accessibility:
                        self.accessibility.add((w1, w3))
                        changed = True
        # Hereditary condition: what is verified at w1 must hold at w2
        for (w1, w2) in self.accessibility:
            if w1 in self.worlds and w2 in self.worlds:
                self.worlds[w2].verified_atoms.update(self.worlds[w1].verified_atoms)

    def evaluate(self, f: Formula, world_name: str) -> bool:
        """
        Saul Kripke (1965) Inductive Truth Definition:
          - Var: phi(P, H) = T iff verified in H
          - And: phi(A & B, H) = T iff phi(A, H) and phi(B, H)
          - Or:  phi(A | B, H) = T iff phi(A, H) or phi(B, H)
          - Imp: phi(A => B, H) = T iff forall H' (H R H' => (not phi(A, H') or phi(B, H')))
          - Not: phi(~A, H) = T iff forall H' (H R H' => not phi(A, H'))
        """
        world = self.worlds.get(world_name)
        if world is None:
            return False

        if isinstance(f, Var):
            if f.name in RESERVED_FALSUM:
                return False
            return world.is_verified(f.name)

        if isinstance(f, And):
            return self.evaluate(f.left, world_name) and self.evaluate(f.right, world_name)

        if isinstance(f, Or):
            return self.evaluate(f.left, world_name) or self.evaluate(f.right, world_name)

        if isinstance(f, Imp):
            accessible_worlds = [h2 for (h1, h2) in self.accessibility if h1 == world_name]
            for w in accessible_worlds:
                if self.evaluate(f.left, w) and not self.evaluate(f.right, w):
                    return False
            return True

        if isinstance(f, Not):
            accessible_worlds = [h2 for (h1, h2) in self.accessibility if h1 == world_name]
            for w in accessible_worlds:
                if self.evaluate(f.inner, w):
                    return False
            return True

        return False

    def is_countermodel_for(self, seq: Sequent) -> bool:
        """Returns True if origin G verifies all Gamma but refutes Delta."""
        g = self.origin
        # All premises in Gamma must evaluate to True at origin G
        for premise in seq.gamma:
            if not self.evaluate(premise, g):
                return False
        # The goal in Delta must evaluate to False at origin G
        for goal in seq.delta:
            if self.evaluate(goal, g):
                return False
        return True


def extract_kripke_countermodel(seq: Sequent, max_depth: int = 6) -> Dict[str, Any]:
    """
    Constructs a Kripke semantic tableau following Saul Kripke (1965).
    If the sequent is unprovable in intuitionistic logic:
      1. Extracts missing ungrounded atomic facts.
      2. Constructs an explicit Kripke countermodel structure (G, K, R) refuting the sequent.
      3. Classifies whether the sequent is INTUITIONISTIC_UNDECIDED (Class 1) or PROVEN_FALSE (Class 0).
    """
    known_atoms: Set[str] = {f.name for f in seq.gamma if isinstance(f, Var) and f.name not in RESERVED_FALSUM}
    neg_atoms: Set[str] = {
        f.inner.name
        for f in seq.gamma
        if isinstance(f, Not) and isinstance(f.inner, Var) and f.inner.name not in RESERVED_FALSUM
    }

    # Propagate forward Horn implications and contrapositive (Modus Tollens) refutations
    pos_closure = set(known_atoms)
    neg_closure = set(neg_atoms)
    changed = True
    while changed:
        changed = False
        for g in seq.gamma:
            if isinstance(g, Imp):
                lhs, rhs = g.left, g.right
                # Check if lhs is satisfied under current closure
                lhs_holds = False
                if isinstance(lhs, Var) and lhs.name in pos_closure:
                    lhs_holds = True
                elif isinstance(lhs, And):
                    def _sat(node: Formula) -> bool:
                        if isinstance(node, Var):
                            return node.name in pos_closure
                        if isinstance(node, And):
                            return _sat(node.left) and _sat(node.right)
                        return False
                    lhs_holds = _sat(lhs)

                if lhs_holds:
                    if isinstance(rhs, Var) and rhs.name not in pos_closure:
                        pos_closure.add(rhs.name)
                        changed = True
                    elif isinstance(rhs, Not) and isinstance(rhs.inner, Var) and rhs.inner.name not in neg_closure:
                        neg_closure.add(rhs.inner.name)
                        changed = True

                # Contrapositive propagation (A => B, ~B |- ~A)
                if isinstance(rhs, Var) and rhs.name in neg_closure:
                    if isinstance(lhs, Var) and lhs.name not in neg_closure:
                        neg_closure.add(lhs.name)
                        changed = True

    # Identify missing base facts necessary to prove goal
    missing_facts: List[str] = []
    goal_atoms = [v for d in seq.delta for v in extract_ordered_vars(d)]
    for g_atom in goal_atoms:
        if g_atom not in pos_closure and g_atom not in RESERVED_FALSUM:
            missing_facts.append(g_atom)

    # Detect constructive refutation (Contradiction / Modus Tollens)
    is_refuted = False
    refutation_reason = ""

    if pos_closure & neg_closure:
        is_refuted = True
        culprit = list(pos_closure & neg_closure)[0]
        refutation_reason = f"Explicit contradiction derived: {culprit} and ~{culprit} both hold"

    if seq.delta:
        delta_goal = seq.delta[0]
        neg_delta = Not(delta_goal)
        if any(f == neg_delta for f in seq.gamma):
            is_refuted = True
            refutation_reason = f"Premise contradicts goal {delta_goal.to_str()}"
        elif isinstance(delta_goal, Var) and delta_goal.name in neg_closure:
            is_refuted = True
            refutation_reason = f"Premises constructively refute goal {delta_goal.name}"
        elif isinstance(delta_goal, Not) and isinstance(delta_goal.inner, Var) and delta_goal.inner.name in pos_closure:
            is_refuted = True
            refutation_reason = f"Goal is negation of derived fact {delta_goal.inner.name}"
        elif isinstance(delta_goal, And):
            if (isinstance(delta_goal.left, Var) and delta_goal.left.name in neg_closure) or (
                isinstance(delta_goal.right, Var) and delta_goal.right.name in neg_closure
            ):
                is_refuted = True
                refutation_reason = f"Conjunct of goal {delta_goal.to_str()} is refuted"
        elif isinstance(delta_goal, Or):
            if (
                isinstance(delta_goal.left, Var)
                and delta_goal.left.name in neg_closure
                and isinstance(delta_goal.right, Var)
                and delta_goal.right.name in neg_closure
            ):
                is_refuted = True
                refutation_reason = f"All disjuncts of goal {delta_goal.to_str()} are refuted"

    # Build canonical 2-point Kripke countermodel (G R H)
    model = KripkeModel(origin="G")
    world_g_facts = sorted(list(pos_closure))
    world_h_facts = sorted(list(pos_closure | set(missing_facts)))
    model.add_world("G", set(world_g_facts))
    model.add_world("H", set(world_h_facts))
    model.add_relation("G", "H")

    is_valid_cm = (not is_refuted) and model.is_countermodel_for(seq)
    diagnosis = "PROVEN_FALSE" if is_refuted else ("INTUITIONISTIC_UNDECIDED" if is_valid_cm or missing_facts else "UNKNOWN")

    return {
        "sequent": seq.to_str(),
        "is_countermodel": is_valid_cm,
        "diagnosis": diagnosis,
        "epistemic_class": 0 if is_refuted else 1,
        "missing_facts": missing_facts,
        "refutation_reason": refutation_reason,
        "world_g_known_facts": world_g_facts,
        "world_h_accessible_facts": world_h_facts,
        "kripke_structure": {
            "origin": "G",
            "worlds": {
                w_name: sorted(list(w.verified_atoms))
                for w_name, w in model.worlds.items()
            },
            "accessibility": sorted(list(model.accessibility)),
        },
    }


# =====================================================================
# 6. INTUITIONISTIC RELEVANCE (IR) & CURRY-HOWARD SERIALIZATION
# =====================================================================

def extract_relevance_witnesses(root_seq: Sequent, proof_tree: dict) -> Dict[str, Any]:
    """
    Intuitionistic Relevance Logic (IR) Witness Tracker:
    Traverses the proof tree to identify which premises in Gamma were actively used
    at Axiom leaves or left-decomposition steps (eliminating Weakening redundancy).
    
    Returns:
      - relevant_witnesses: list of premise strings actively used in the proof.
      - unused_premises: list of premise strings present in Gamma but never used.
      - is_strictly_relevant: True if every premise in Gamma contributed to the proof.
      - relevance_ratio: float in [0.0, 1.0] representing ratio of used premises.
    """
    if not proof_tree or not isinstance(proof_tree, dict):
        return {
            "relevant_witnesses": [],
            "unused_premises": [f.to_str() for f in root_seq.gamma],
            "is_strictly_relevant": False,
            "relevance_ratio": 0.0,
        }

    used_formulas: Set[str] = set()

    def _collect_used_nodes(node: dict):
        rule = node.get("rule", "")
        rule_upper = rule.upper()
        seq_str = node.get("sequent") or node.get("conclusion", "")
        branches = node.get("branches") or node.get("children", [])

        turnstile = "|-" if "|-" in seq_str else ("\\vdash" if "\\vdash" in seq_str else "⊢")

        # 1. Leaf Axiom
        if rule_upper in ("AXIOM", "AXIOM_LEAF") or rule in ("Axiom", "AXIOM"):
            if turnstile in seq_str:
                parts = seq_str.split(turnstile)
                g_str, d_str = parts[0].strip(), parts[1].strip()
                g_items = [x.strip() for x in g_str.split(",") if x.strip()]
                if d_str and d_str != "0":
                    used_formulas.add(d_str)
                if "0" in g_items:
                    used_formulas.add("0")

        # 2. Left Decomposition rule (Modus Ponens / Left Elimination)
        if rule_upper.startswith("L_") or rule in ("Imp_L", "And_L1", "And_L2", "Or_L", "Not_L"):
            pivot_idx = None
            if "_" in rule:
                parts = rule.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    pivot_idx = int(parts[1])
            if pivot_idx is None and "pivot" in node:
                pivot_idx = node["pivot"]

            if turnstile in seq_str:
                g_part = seq_str.split(turnstile)[0].strip()
                g_items = [x.strip() for x in g_part.split(",") if x.strip() and x.strip() != "0"]
                if pivot_idx is not None and 0 <= pivot_idx < len(g_items):
                    pivot_str = g_items[pivot_idx]
                    used_formulas.add(pivot_str)
                    # When an implication is detached (L_IMP / L_IMP_ATOM / Imp_L), the antecedent was also used
                    for arrow in ["=>", "->", "→", "\\to"]:
                        if arrow in pivot_str:
                            ant = pivot_str.split(arrow)[0].strip(" ()")
                            if ant and ant in g_items:
                                used_formulas.add(ant)
                            break

        for b in branches:
            if isinstance(b, dict):
                _collect_used_nodes(b)

    _collect_used_nodes(proof_tree)

    def _normalize_symbolic(s: str) -> str:
        s = s.strip()
        s = re.sub(r'->|\\to|→', '=>', s)
        s = re.sub(r'\s+', '', s)
        return s

    root_premises = [f.to_str() for f in root_seq.gamma]
    root_norm_map = {_normalize_symbolic(p): p for p in root_premises}
    used_norm = {_normalize_symbolic(u) for u in used_formulas}

    relevant_witnesses: List[str] = []
    unused_premises: List[str] = []

    for prem_str in root_premises:
        norm_p = _normalize_symbolic(prem_str)
        if norm_p in used_norm or prem_str in used_formulas:
            relevant_witnesses.append(prem_str)
        else:
            unused_premises.append(prem_str)

    total = len(root_premises)
    ratio = (len(relevant_witnesses) / total) if total > 0 else 1.0
    strictly_relevant = (len(unused_premises) == 0)

    return {
        "relevant_witnesses": relevant_witnesses,
        "unused_premises": unused_premises,
        "is_strictly_relevant": strictly_relevant,
        "relevance_ratio": round(ratio, 4),
    }


def proof_tree_to_curry_howard(node: dict, simplify: bool = True) -> str:
    """
    Converts a verified Gentzen proof tree into a typed Curry-Howard lambda term.
    
    Isomorphism mappings:
      - AXIOM (A |- A)            --> variable witness (e.g. `x_A` or atomic name)
      - R_IMP (Gamma, A |- B)      --> lambda abstraction `λ(x: A) -> M`
      - L_IMP (A => B, N: A)       --> application `let y = f(N) in M` or `f(N)`
      - R_AND (A, B)               --> pair `pair(M, N)`
      - L_AND ((A & B))            --> unpairing `let (x, y) = p in M`
      - R_OR_1 (A)                 --> injection `inl(M)`
      - R_OR_2 (B)                 --> injection `inr(N)`
      - L_OR (A | B)               --> case analysis `match s with inl(x) => M | inr(y) => N`
      - R_NOT / L_NOT (~A)         --> negation abstraction / falsum application
      - Ex Falso (0 |- C)          --> abort / ex falso eliminator `abort(w)`
    """
    if not node or not isinstance(node, dict):
        return "nil"

    rule = node.get("rule", "")
    rule_upper = rule.upper()
    seq_str = node.get("sequent") or node.get("conclusion", "")
    branches = node.get("branches") or node.get("children", [])
    turnstile = "|-" if "|-" in seq_str else ("\\vdash" if "\\vdash" in seq_str else "⊢")

    def _clean_var(s: str) -> str:
        s = s.strip()
        s = re.sub(r"[^A-Za-z0-9_]", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s if s else "x"

    # 1. Leaf Axiom
    if rule_upper in ("AXIOM", "AXIOM_LEAF") or rule in ("Axiom", "AXIOM"):
        if turnstile in seq_str:
            d_part = seq_str.split(turnstile)[1].strip()
            if d_part and d_part != "0":
                return _clean_var(d_part)
            g_part = seq_str.split(turnstile)[0].strip()
            if "0" in [x.strip() for x in g_part.split(",")]:
                return "abort(contradiction)"
        return "x"

    # 2. Right Implication (Deduction theorem)
    if rule_upper in ("R_IMP", "IMP_R"):
        ant = "premise"
        if turnstile in seq_str:
            d_part = seq_str.split(turnstile)[1].strip()
            for arrow in ["=>", "->", "→", "\\to"]:
                if arrow in d_part:
                    ant = d_part.split(arrow)[0].strip()
                    break
        body = proof_tree_to_curry_howard(branches[0]) if branches else "x"
        var_name = _clean_var(ant)
        return f"λ({var_name}: {ant}) -> {body}"

    # 3. Left Implication (Modus Ponens)
    if rule_upper.startswith("L_IMP") or rule == "Imp_L":
        arg_term = proof_tree_to_curry_howard(branches[0]) if len(branches) > 0 else "x"
        if len(branches) > 1:
            rest_term = proof_tree_to_curry_howard(branches[1])
            if simplify and (rest_term == "x" or len(rest_term) < 15):
                return f"apply({arg_term})"
            return f"let y = apply({arg_term}) in {rest_term}"
        return f"apply({arg_term})"

    # 4. Right Conjunction (Pairing)
    if rule_upper in ("R_AND", "AND_R"):
        t1 = proof_tree_to_curry_howard(branches[0]) if len(branches) > 0 else "x"
        t2 = proof_tree_to_curry_howard(branches[1]) if len(branches) > 1 else "y"
        return f"pair({t1}, {t2})"

    # 5. Left Conjunction (Unpairing)
    if rule_upper.startswith("L_AND") or rule in ("And_L1", "And_L2"):
        sub = proof_tree_to_curry_howard(branches[0]) if branches else "x"
        return sub

    # 6. Right Disjunction (Injections)
    if rule_upper in ("R_OR_1", "OR_R1"):
        sub = proof_tree_to_curry_howard(branches[0]) if branches else "x"
        return f"inl({sub})"
    if rule_upper in ("R_OR_2", "OR_R2"):
        sub = proof_tree_to_curry_howard(branches[0]) if branches else "x"
        return f"inr({sub})"

    # 7. Left Disjunction (Case Analysis)
    if rule_upper.startswith("L_OR") or rule == "Or_L":
        c1 = proof_tree_to_curry_howard(branches[0]) if len(branches) > 0 else "x"
        c2 = proof_tree_to_curry_howard(branches[1]) if len(branches) > 1 else "y"
        return f"case(left => {c1}, right => {c2})"

    # 8. Negation Rules
    if rule_upper in ("R_NOT", "NOT_R"):
        sub = proof_tree_to_curry_howard(branches[0]) if branches else "abort"
        return f"λ(x) -> {sub}"
    if rule_upper.startswith("L_NOT") or rule == "Not_L":
        sub = proof_tree_to_curry_howard(branches[0]) if branches else "x"
        return f"refute({sub})"

    if branches:
        return proof_tree_to_curry_howard(branches[0])
    return "qed"
