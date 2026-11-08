"""
parser_jev.py - Jev front end: decision-scenario state JSON and policy rules -> facts and rules.

Only the Jev decision pipeline uses this module. It shares nothing with the ClearCoT front end
(nanogentzen/cot/parser_cot.py) except the engine syntax in nanogentzen/logic_syntax.py.

Atom names here are state keys (lowercase snake_case), not the Capitalized KG concept atoms of ClearCoT.
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

from ..logic_syntax import FormulaParser, parse_formula  # noqa: F401


# =====================================================================
# 5. CONE-OF-INFLUENCE ANTECEDENT RELEVANCE FILTER (UP TO 32 PIVOTS)
# =====================================================================

def extract_vars(node: Any) -> Set[str]:
    """Extracts atomic variable identifiers from a Formula AST."""
    if isinstance(node, Var):
        return {node.name}
    out: Set[str] = set()
    for attr in ("left", "right", "inner", "a", "b"):
        child = getattr(node, attr, None)
        if child is not None:
            out |= extract_vars(child)
    return out


def select_relevant_gamma(
    facts: List[Formula],
    rules: List[Formula],
    goal: Formula,
    max_antecedents: int = 32,
) -> Tuple[Formula, ...]:
    """
    Selects relevant premises for goal via polarity-aware backward cone-of-influence traversal.
    Supports up to 32 antecedents matching the Span Pointer cross-attention capacity.
    """
    all_items = facts + rules
    atomic_facts = [x for x in all_items if isinstance(x, (Var, Not))]
    compound_rules = [x for x in all_items if not isinstance(x, (Var, Not))]

    goal_vars = set(extract_vars(goal))
    cone_vars = set(goal_vars)
    neg_fact_vars = {v for f in atomic_facts if isinstance(f, Not) for v in extract_vars(f)}
    relevant_rules: List[Formula] = []
    changed = True

    while changed:
        changed = False
        for r in compound_rules:
            if r in relevant_rules:
                continue
            rhs = r.right if isinstance(r, Imp) else r
            rhs_vars = extract_vars(rhs)
            lhs_vars = extract_vars(r.left) if isinstance(r, Imp) else set()

            if isinstance(goal, Var) and isinstance(rhs, Not) and rhs_vars == goal_vars:
                continue
            if isinstance(goal, Not) and isinstance(rhs, Var) and rhs_vars == goal_vars:
                continue

            if (rhs_vars & cone_vars) or ((lhs_vars & cone_vars) and (rhs_vars & (neg_fact_vars | cone_vars))):
                relevant_rules.append(r)
                new_vars = extract_vars(r) - cone_vars
                if new_vars:
                    cone_vars |= new_vars
                    changed = True

    relevant_facts = [f for f in atomic_facts if extract_vars(f) & cone_vars]
    combined = relevant_facts + list(reversed(relevant_rules))
    if not combined:
        combined = relevant_facts

    return tuple(combined[:max_antecedents])


# =====================================================================
# 6. UNIVERSAL JSON STATE & RUBRIC GROUNDER
# =====================================================================

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
) -> Tuple[List[Formula], List[str]]:
    """
    Production-grade domain-agnostic state grounder.
    Converts arbitrary nested JSON objects and explicit/inferred predicates
    into certified factual Formula AST propositions without domain hardcoding.
    """
    facts: List[Formula] = []
    missing: List[str] = []

    # 1. Explicit Predicate Declarations
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
            if not found or actual_val is None:
                missing.append(atom_name)
                continue

            op_fn = GENERIC_OPS.get(op_str, operator.eq)
            try:
                holds = bool(op_fn(actual_val, expected_val))
                facts.append(Atom(atom_name) if holds else Not(Atom(atom_name)))
            except Exception:
                missing.append(atom_name)

    # 2. Automatic Schema Traversal for Arbitrary JSON States
    if isinstance(state, dict):
        def _walk_dict(d: dict, prefix: str = ""):
            for k, v in d.items():
                pfx = f"{prefix}_{k}" if prefix else k
                pfx_clean = re.sub(r"[^A-Za-z0-9_]", "_", pfx).strip("_")

                if isinstance(v, bool):
                    if not predicates or k not in predicates:
                        facts.append(Atom(pfx_clean) if v else Not(Atom(pfx_clean)))
                elif isinstance(v, (int, float)):
                    if not predicates or k not in predicates:
                        if v > 0:
                            facts.append(Atom(f"{pfx_clean}_positive"))
                        elif v == 0:
                            facts.append(Atom(f"{pfx_clean}_zero"))
                elif isinstance(v, str):
                    if not predicates or k not in predicates:
                        val_clean = re.sub(r"[^A-Za-z0-9_]", "_", v.lower()).strip("_")
                        if val_clean and len(val_clean) < 32:
                            facts.append(Atom(f"{pfx_clean}_{val_clean}"))
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        if isinstance(item, str):
                            item_clean = re.sub(r"[^A-Za-z0-9_]", "_", item.lower()).strip("_")
                            if item_clean:
                                facts.append(Atom(f"{pfx_clean}_has_{item_clean}"))
                elif isinstance(v, dict):
                    _walk_dict(v, pfx_clean)

        _walk_dict(state)

    seen: Set[str] = set()
    unique_facts: List[Formula] = []
    for f in facts:
        s = f.to_str()
        if s not in seen:
            seen.add(s)
            unique_facts.append(f)

    return unique_facts, missing


# =====================================================================
# 7. DOMAIN REGISTRY INTEGRATION (LAZY TO AVOID CIRCULAR IMPORT)
# =====================================================================

def _get_domain_registry():
    try:
        from ..domain_registry import (
            REGISTRY,
            benchmark_detect_workflow,
            compile_benchmark_structured_decision,
            BENCHMARK_CALIBRATED_DISTRIBUTIONS,
        )
        return REGISTRY, benchmark_detect_workflow, compile_benchmark_structured_decision, BENCHMARK_CALIBRATED_DISTRIBUTIONS
    except ImportError:
        try:
            from nanogentzen.domain_registry import (
                REGISTRY,
                benchmark_detect_workflow,
                compile_benchmark_structured_decision,
                BENCHMARK_CALIBRATED_DISTRIBUTIONS,
            )
            return REGISTRY, benchmark_detect_workflow, compile_benchmark_structured_decision, BENCHMARK_CALIBRATED_DISTRIBUTIONS
        except ImportError:
            return None, None, None, {}


def detect_workflow(state: dict, questions: Optional[dict] = None) -> Optional[str]:
    _, detect_fn, _, _ = _get_domain_registry()
    return detect_fn(state, questions) if detect_fn else None


def get_calibrated_distribution(
    workflow: Optional[str],
    q_id: str,
    proven_label: str,
    options: List[str],
    state: Optional[dict] = None,
    epistemic_probs: Optional[dict] = None,
    temp: float = 0.68,
) -> Optional[dict]:
    """Universal Epistemic Temperature Calibration."""
    if not options:
        return None
    key = f"{workflow}/{q_id}/{proven_label}" if workflow else f"{q_id}/{proven_label}"

    registry, _, _, calibrated_dists = _get_domain_registry()
    base_dist = None
    if registry and hasattr(registry, "get_calibrated_distribution"):
        base_dist = registry.get_calibrated_distribution(key)
    if base_dist is None and calibrated_dists and key in calibrated_dists:
        base_dist = dict(calibrated_dists[key])

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

    p_true = float(epistemic_probs.get("true", 0.75)) if isinstance(epistemic_probs, dict) else 0.75
    p_undec = float(epistemic_probs.get("undecided", 0.15)) if isinstance(epistemic_probs, dict) else 0.15
    k = len(options)
    if k == 1:
        return {str(options[0]): 1.0}

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
    """Compiles structured JSON state and question schemas into facts and rules."""
    registry, _, _, _ = _get_domain_registry()
    if registry:
        name, compiler = registry.match(state)
        if compiler is not None:
            return compiler(state, questions)

    facts, _ = ground_state_generic(state)
    return facts, [], None


# =====================================================================
# 8. JEV SCENARIO FRONT END (v4): rules, predicates, questions, validation
# =====================================================================
# These used to be private methods of JevLogicalArchitect. They are pure functions of the scenario JSON and
# compile to nanoGentzen v4 formulas; the architect only hands their output to nanogentzen.verify.

# atoms of the example in the old scenario prompt; a rule using them was copied, not grounded in the text
PROMPT_TEMPLATE_ATOMS = frozenset({"is_scientific_theory", "uses_processes"})

UNDECIDED_OPTION_RE = re.compile(
    r"^(?:undecided|undetermined|indeterminate|insufficient(?:_\w+)?|unknown|unclear|inconclusive|"
    r"cannot_(?:be_)?determine\w*|not_determinable|no_decision|open)$", re.IGNORECASE)

# issues that make a scenario unusable as training data (the others are reported only)
BLOCKING_ISSUES = frozenset({"no_rules", "template_rule", "contradictory_rules", "no_questions"})


def undecided_option(options) -> Optional[str]:
    """The 'undecided'-type option of a question, if any."""
    return next((o for o in options if UNDECIDED_OPTION_RE.match(str(o).strip())), None)


def norm_name(name: str) -> str:
    """Case- and separator-insensitive key ('Compatible' == 'compatible', 'high-risk' == 'high_risk')."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def leaf_paths(state: Any, prefix: Tuple[str, ...] = ()) -> List[Tuple[Tuple[str, ...], Any]]:
    out: List[Tuple[Tuple[str, ...], Any]] = []
    if isinstance(state, dict):
        for k, v in state.items():
            if isinstance(v, dict):
                out.extend(leaf_paths(v, prefix + (str(k),)))
            else:
                out.append((prefix + (str(k),), v))
    return out


def _has_imp(f) -> bool:
    if type(f).__name__ == "Imp":
        return True
    return any(_has_imp(c) for c in (getattr(f, "left", None), getattr(f, "right", None),
                                      getattr(f, "inner", None)) if c is not None)


_RULE_OPERATOR_RE = re.compile(r"(<->|<=>|=>|->|→|↔|&|∧|\||∨|~|¬|\(|\)|⊥)")
_UNICODE_OPS = {"→": "=>", "↔": "<->", "∧": "&", "∨": "|", "¬": "~"}
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def snake_atom(text: str) -> str:
    """'Predictive Processing framework' -> 'predictive_processing_framework' (identifiers are kept as they are)."""
    t = str(text).strip()
    if _IDENT_RE.match(t):
        return t
    t = re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")
    if t and t[0].isdigit():
        t = "f_" + t
    return t


def normalize_rule_atoms(rule: str) -> str:
    """
    Rewrite the operands of a symbolic rule as snake_case atoms, so an LLM rule written with phrases
    ('(Predictive Processing framework & Hierarchical generative model) => compatible') compiles to the
    atoms the state uses (predictive_processing_framework, ...). Rules without a symbolic operator are
    returned unchanged (they are natural-language conditionals or not rules at all).
    """
    r = str(rule).strip()
    parts = _RULE_OPERATOR_RE.split(r)
    if len(parts) == 1:
        return r
    out = []
    for part in parts:
        if not part:
            continue
        if _RULE_OPERATOR_RE.fullmatch(part):
            out.append(_UNICODE_OPS.get(part, part))
        elif part.strip():
            atom = snake_atom(part)
            if not atom:
                return r          # an operand with no letters or digits: leave the rule for the parser to reject
            out.append(atom)
    text = " ".join(out)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return re.sub(r"~\s+", "~", text)


def compile_rules_v4(policy_rules: List[str]) -> Tuple[List[str], List[Dict[str, str]], Set[str]]:
    """
    Rules that compile in the v4 parser (atoms normalized by normalize_rule_atoms; these normalized strings
    are returned), the rejected ones (with the reason), and all their atoms.
    """
    from nanogentzen_v4.parser import compile_to_formula
    from nanogentzen_v4.kernel import extract_vars_set

    ok, rejected, atoms = [], [], set()
    for raw in policy_rules or []:
        r = normalize_rule_atoms(str(raw)) if str(raw).strip() else str(raw)
        try:
            f = compile_to_formula(r)
        except Exception as e:
            rejected.append({"rule": str(raw), "error": str(e)})
            continue
        if not _has_imp(f):
            # the v4 parser reads plain text as one atom; as a "rule" it would silently become a fact
            rejected.append({"rule": str(raw), "error": f"not a conditional rule (compiles to {f.to_str()})"})
            continue
        ok.append(r)
        atoms |= extract_vars_set(f)
    return ok, rejected, atoms


def bind_predicates_v4(state: Any, atoms: Set[str]) -> Dict[str, Dict[str, Any]]:
    """
    Bind rule atoms to nested boolean state fields. v4 grounding names nested keys by their path
    (attributes.kyc -> attributes_kyc), while scenario rules use the leaf key (kyc). A predicate
    {"kyc": {"field": "attributes.kyc", "op": "==", "value": true}} makes the rule atom refer to that field.
    """
    if not isinstance(state, dict):
        return {}
    leaves = leaf_paths(state)
    top_level = {path[0] for path, _ in leaves if len(path) == 1}
    preds: Dict[str, Dict[str, Any]] = {}
    for a in sorted(atoms):
        if a in top_level:
            continue  # already a fact under its own name
        cands = [(p, v) for p, v in leaves if isinstance(v, bool) and len(p) > 1
                 and (p[-1] == a or norm_name(p[-1]) == norm_name(a))]
        if not cands:
            continue
        cands.sort(key=lambda pv: (pv[0][-1] != a, "attributes" not in pv[0], len(pv[0])))
        preds[a] = {"field": ".".join(cands[0][0]), "op": "==", "value": True}
    return preds


def question_specs_v4(questions: Dict[str, Any], rule_atoms: Set[str]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
    """Studio questions -> v4 QuestionSpecs; option keys aligned case-insensitively to rule atoms."""
    by_norm = {norm_name(a): a for a in rule_atoms}
    specs: Dict[str, Any] = {}
    back: Dict[str, Dict[str, str]] = {}
    for q_id, q in (questions or {}).items():
        if not isinstance(q, dict):
            continue
        qtype = q.get("type", "choice")
        if qtype not in ("choice", "noul", "score"):
            qtype = "choice"
        spec: Dict[str, Any] = {"type": qtype}
        for k in ("instructions", "criteria", "require_relevance", "exclusive", "rules"):
            if q.get(k) is not None:
                spec[k] = q[k]
        if qtype == "choice":
            crit = q.get("criteria", {})
            options = q.get("options") or (list(crit.keys()) if isinstance(crit, dict) else [])
            mapping = {}
            for o in options:
                o = str(o)
                mapping[by_norm.get(norm_name(o), o) if o not in rule_atoms else o] = o
            spec["options"] = list(mapping.keys())
            back[q_id] = mapping
        elif qtype == "noul":
            spec["target"] = q.get("target") or q_id
        else:
            for k in ("levels", "level_atoms"):
                if q.get(k) is not None:
                    spec[k] = q[k]
        specs[str(q_id)] = spec
    return specs, back


def _split_rule(rule: str) -> Optional[Tuple[str, str]]:
    """'(a & b) => c' -> ('a&b' normalized antecedent, 'c'); None if not a top-level implication."""
    m = re.match(r"^\s*(.+?)\s*(?:=>|->|→)\s*(~?\s*[A-Za-z_][A-Za-z0-9_]*)\s*$", str(rule))
    if not m:
        return None
    ante = re.sub(r"[\s()]", "", m.group(1)).lower()
    parts = sorted(p for p in re.split(r"[&∧]", ante) if p)   # conjunct order does not matter
    return "&".join(parts), re.sub(r"\s", "", m.group(2))


def rule_conclusions(policy_rules: List[str]) -> Set[str]:
    """Normalized option names that some rule concludes positively."""
    out = set()
    for r in policy_rules or []:
        sp = _split_rule(r)
        if sp and not sp[1].startswith("~"):
            out.add(norm_name(sp[1]))
    return out


def uncovered_options(questions: Dict[str, Any], policy_rules: List[str]) -> Dict[str, List[str]]:
    """Per question: the non-'undecided' options that no rule concludes (the kernel can never prove them)."""
    concl = rule_conclusions(policy_rules)
    out: Dict[str, List[str]] = {}
    for q_id, q in (questions or {}).items():
        if not isinstance(q, dict) or q.get("type", "choice") != "choice":
            continue
        crit = q.get("criteria") or {}
        opts = [str(o) for o in (q.get("options") or (list(crit) if isinstance(crit, dict) else []))]
        und = undecided_option(opts)
        missing = [o for o in opts if o != und and norm_name(o) not in concl]
        if missing:
            out[str(q_id)] = missing
    return out


def validate_scenario(state: Any, questions: Dict[str, Any], policy_rules: List[str]) -> List[Dict[str, Any]]:
    """
    Static checks on an LLM-written Jev scenario, before any proof search. Each issue is
    {"code", "detail", "blocking"}; blocking codes are in BLOCKING_ISSUES.

      no_questions         no usable choice question
      no_rules             no policy rule compiles: the label could only come from a heuristic
      rejected_rules       some rules do not compile and are dropped (reported)
      template_rule        a rule uses the prompt's example atoms (copied, not read from the evidence)
      contradictory_rules  the same antecedent concludes two different options
      uncovered_options    an option no rule concludes (the kernel can never prove it; reported)
      ungrounded_atoms     rule antecedent atoms with no boolean field in the state (reported)
      nonstandard_options  option labels of 4+ words, i.e. phrases rather than labels (reported)
    """
    issues: List[Dict[str, Any]] = []

    def add(code, detail):
        issues.append({"code": code, "detail": detail, "blocking": code in BLOCKING_ISSUES})

    if not any(isinstance(q, dict) for q in (questions or {}).values()):
        add("no_questions", "no usable question")
    raw_rules = [str(r) for r in (policy_rules or []) if str(r).strip()]
    # judge the rules the kernel will actually get: a rule list that compiles to nothing would make
    # jev_server fall back to criteria_heuristic, which invents its own atoms and "proves" any label
    rules, rejected, _ = compile_rules_v4(raw_rules) if raw_rules else ([], [], set())
    if not raw_rules:
        add("no_rules", "policy_rules is empty")
    elif not rules:
        add("no_rules", f"none of the {len(raw_rules)} policy rules compiles: " +
            "; ".join(f"{x['rule']!r} ({x['error']})" for x in rejected[:3]))
    elif rejected:
        add("rejected_rules", [x["rule"] for x in rejected])

    atoms_by_rule = {r: set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", r)) for r in rules}
    tmpl = [r for r, a in atoms_by_rule.items() if a & PROMPT_TEMPLATE_ATOMS]
    if tmpl:
        add("template_rule", tmpl)

    by_ante: Dict[str, Set[str]] = {}
    parsed_rules = []
    for r in rules:
        sp = _split_rule(r)
        if sp:
            ante, head = sp
            clean_head = norm_name(head) if not head.startswith("~") else head
            by_ante.setdefault(ante, set()).add(clean_head)
            if "~" not in ante:
                pos_atoms = set(norm_name(a) for a in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ante))
                parsed_rules.append((r, pos_atoms, clean_head))

    clash = {a: sorted(c) for a, c in by_ante.items() if len({x for x in c if not x.startswith("~")}) > 1}
    for i in range(len(parsed_rules)):
        for j in range(i + 1, len(parsed_rules)):
            r1, a1, h1 = parsed_rules[i]
            r2, a2, h2 = parsed_rules[j]
            if h1 != h2 and not h1.startswith("~") and not h2.startswith("~"):
                if a1 and a2 and (a1 == a2 or a1.issubset(a2) or a2.issubset(a1)):
                    clash.setdefault(f"{r1} <-> {r2}", sorted([h1, h2]))
    if clash:
        add("contradictory_rules", clash)

    unc = uncovered_options(questions, rules)
    if unc:
        add("uncovered_options", unc)

    bool_keys = {norm_name(p[-1]) for p, v in leaf_paths(state) if isinstance(v, bool)} if isinstance(state, dict) else set()
    option_names = {norm_name(o) for q in (questions or {}).values() if isinstance(q, dict)
                    for o in (q.get("criteria") or {})}
    heads = {norm_name(sp[1].lstrip("~")) for sp in (_split_rule(r) for r in rules) if sp}  # derived atoms
    ungrounded = set()
    for r in rules:
        sp = _split_rule(r)
        ante = sp[0] if sp else r
        for a in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ante):
            if norm_name(a) not in bool_keys | option_names | heads:
                ungrounded.add(a)
    if ungrounded:
        add("ungrounded_atoms", sorted(ungrounded))

    # option labels that are phrases rather than labels ("compatible_occupations_with_..._score"): they make
    # the label vocabulary of a dataset open-ended (reported, not blocking)
    long_opts = sorted({o for q in (questions or {}).values() if isinstance(q, dict)
                        for o in (q.get("criteria") or {}) if len(re.split(r"[_\s-]+", str(o).strip())) >= 4})
    if long_opts:
        add("nonstandard_options", long_opts)
    return issues


def blocking_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [i for i in issues or [] if i.get("blocking")]
