# Vendored from the nanoGentzen v4 model folder by tools/vendor_v4_engine.py. Do not edit by hand.
"""
jev_server.py (nanoGentzen-v3 Production Server)
Production-grade neuro-symbolic reasoning and formal verification server featuring:
  1. Domain-Agnostic JSON State Grounding (Dynamic AST extraction for arbitrary business domains)
  2. Fallback to Structured Workflow Compilers when detecting known schemas (e.g. typed-decisions)
  3. Saul Kripke (1965) 3-Class Epistemic Valuation (Refuted / Undecided / Proven True in < 3ms)
  4. Gerhard Gentzen LJ / Roy Dyckhoff G4ip Contraction-Free Constructive Proof Search
  5. Intuitionistic Relevance Logic (IR) Witness Tracking (Isolating used vs unused distractor premises)
  6. Curry-Howard Typed Lambda-Calculus Proof Certificate Serialization (Compact 1-line terms)
  7. Semantic Tableau Countermodel Extraction for Incomplete / Undecided Goals
  8. Calibrated Epistemic Distribution Blending (Empirical soft priors modulated by epistemic sharpness)
"""

from __future__ import annotations

import inspect
import itertools
import json
import math
import os
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

try:
    import torch
except ImportError:          # NumPy backend (numpy_model.py) needs no torch
    torch = None
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:          # in-process use inside the Studio needs no web server
    class HTTPException(Exception):
        def __init__(self, status_code: int = 400, detail: Any = None):
            super().__init__(detail)
            self.status_code, self.detail = status_code, detail

    class FastAPI:
        def __init__(self, *a, **k):
            pass

        def add_middleware(self, *a, **k):
            pass

        def get(self, *a, **k):
            return lambda fn: fn

        post = get

    CORSMiddleware = None
from pydantic import BaseModel, Field

from . import certificates as _certs
from . import parser as parser_module
from .backend import as_input, to_numpy, no_grad as _no_grad
from .kernel import (
    BOT,
    CHOICE_LEFT_RULES,
    applicable_moves,
    find_kripke_countermodel,
    g4ip_prove,
    model_view_sequent,
    normalize_sequent,
    And,
    Formula,
    Imp,
    Not,
    Or,
    RULES,
    Sequent,
    Var,
    apply_rule,
    canonicalize_sequent,
    extract_kripke_countermodel,
    extract_relevance_witnesses,
    proof_tree_to_curry_howard,
    verify_proof_tree_strict,
)
from .parser import (
    compile_to_formula,
    ground_state_generic,
    parse_symbolic_sequent,
    detect_workflow,
    compile_structured_decision,
    get_calibrated_distribution,
)
from .search import NeuralProofSearch
from .tokenization_nanogentzen import LogicTokenizer

try:
    from .modeling_nanogentzen import GentzenConfig, GentzenPolicyValueModel
except ImportError:          # torch missing: callers must pass `model=` (e.g. NumpyGentzenModel)
    GentzenConfig = GentzenPolicyValueModel = None

_HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("NANOGENTZEN_DIR", str(_HERE.parent / "model"))).resolve()
if not (MODEL_DIR / "config.json").exists() and (_HERE.parent / "model_v4").exists():
    MODEL_DIR = (_HERE.parent / "model_v4").resolve()


# ============================================================================
# 1. CONE-OF-INFLUENCE ANTECEDENT SELECTOR (SCALES UP TO 32 PIVOTS)
# ============================================================================

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
    max_antecedents: Optional[int] = None,
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

    return tuple(combined if max_antecedents is None else combined[:max_antecedents])


# ============================================================================
# PHASE 0: explicit run configuration (no benchmark logic unless asked for)
# ============================================================================
# Environment defaults; each can be overridden per request with the same field name.
#   NANOGENTZEN_DOMAIN_PACK   ""  | "typed-decisions"   benchmark rule pack (+ workflow detection)
#   NANOGENTZEN_PRIORS        off | on                  replace proven answers' probabilities by priors
#   NANOGENTZEN_NETWORK       on  | off                 off = fixed-order proof search, neutral System 1
#   NANOGENTZEN_SYSTEM2       on  | off                 off = never run proof search
#   NANOGENTZEN_PROVEN_EPS    0.0                       mass moved off a proven label when priors are off
KNOWN_PACKS = ("typed-decisions",)
BACKSTOP_S = float(os.getenv("NANOGENTZEN_BACKSTOP_MS", "500")) / 1000.0   # time cap of the complete G4ip prover


def _literal_fastpath_unprovable(seq) -> bool:
    """Gamma of atoms / negated atoms only, goal an atom or negated atom not present: not derivable."""
    if seq.is_axiom() or not seq.delta:
        return False
    lit = lambda f: isinstance(f, Var) or (isinstance(f, Not) and isinstance(f.inner, Var))  # noqa: E731
    if not all(lit(f) for f in seq.gamma) or not lit(seq.delta[0]):
        return False
    d0 = seq.delta[0]
    pos = {f.name for f in seq.gamma if isinstance(f, Var)}
    neg = {f.inner.name for f in seq.gamma if isinstance(f, Not)}
    return (isinstance(d0, Var) and d0.name not in pos) or (isinstance(d0, Not) and d0.inner.name not in neg)


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "on", "true", "yes")


def resolve_run_config(req) -> Dict[str, Any]:
    pack = getattr(req, "domain_pack", None)
    if pack is None:
        pack = os.getenv("NANOGENTZEN_DOMAIN_PACK", "").strip() or None
    if pack is not None and pack not in KNOWN_PACKS:
        raise HTTPException(400, f"Unknown domain_pack '{pack}'. Known: {list(KNOWN_PACKS)}")
    cfg = {
        "domain_pack": pack,
        "use_priors": getattr(req, "use_priors", None),
        "network": getattr(req, "network", None),
        "system2": getattr(req, "system2", None),
        "proven_eps": float(os.getenv("NANOGENTZEN_PROVEN_EPS", "0.0")),
    }
    pri = cfg["use_priors"]
    if pri is None:
        pri = os.getenv("NANOGENTZEN_PRIORS", "off").strip().lower()
    if pri is True or pri in ("on", "replace", "1", "true", "yes"):
        cfg["priors"] = "replace"
    elif pri in ("field",):
        cfg["priors"] = "field"
    else:
        cfg["priors"] = "off"
    cfg["use_priors"] = cfg["priors"] != "off"
    if cfg["network"] is None:
        cfg["network"] = _env_flag("NANOGENTZEN_NETWORK", "on")
    if cfg["system2"] is None:
        cfg["system2"] = _env_flag("NANOGENTZEN_SYSTEM2", "on")
    _activate_pack(pack)
    return cfg


_PACK_LOCK = threading.Lock()
_ACTIVE_PACK = ["<unset>"]


def _activate_pack(pack: Optional[str]) -> None:
    """Reconfigure the shared domain REGISTRY only when the requested pack changes, under a lock, so concurrent
    requests for the same pack (the Studio's batch threads) never see a half-cleared registry."""
    with _PACK_LOCK:
        if _ACTIVE_PACK[0] == pack:
            return
        _domain_registry_module().disable_all_packs()
        if pack == "typed-decisions":
            _domain_registry_module().enable_typed_decisions_pack()
        _ACTIVE_PACK[0] = pack


def _domain_registry_module():
    # the exact module whose REGISTRY the parser reads (the train repo has two copies of the file)
    return sys.modules[parser_module.REGISTRY.__class__.__module__]


def _proven_dist(options: List[str], label: str, eps: float) -> Dict[str, float]:
    k = len(options)
    if k <= 1 or eps <= 0:
        return {str(o): (1.0 if str(o) == str(label) else 0.0) for o in options}
    return {str(o): round(1.0 - eps if str(o) == str(label) else eps / (k - 1), 6) for o in options}


class FixedOrderSearch(NeuralProofSearch):
    """Proof search without the network: rules in RULES order, pivots left to right, no pruning."""

    def rank_actions(self, seq):
        return applicable_moves(seq, ("R_OR_1", "R_OR_2") + tuple(CHOICE_LEFT_RULES)), None, 1.0

NEUTRAL_S1 = {"epistemic": {"false": 0.3333, "undecided": 0.3334, "true": 0.3333}, "value": 0.5, "confidence": 0.5}


# ============================================================================
# 2. PYDANTIC SCHEMAS (GENERIC, DATASET-AGNOSTIC)
# ============================================================================

class PredicateSpec(BaseModel):
    field: str
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "contains", "regex", "exists"] = "=="
    value: Any = True


class QuestionSpec(BaseModel):
    model_config = {"extra": "allow"}
    type: Literal["noul", "choice", "score"]
    target: Optional[str] = None
    options: Optional[List[str]] = None
    levels: Optional[List[Union[int, str]]] = None
    level_atoms: Optional[List[str]] = None
    rules: Optional[List[str]] = None
    require_proof: bool = True
    require_relevance: bool = False
    instructions: Optional[Any] = None
    criteria: Optional[Any] = None


class DecideRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str = "nanoGentzen-v3"
    state: Any = None
    predicates: Dict[str, Union[PredicateSpec, Dict[str, Any]]] = Field(default_factory=dict)
    policy_rules: List[str] = Field(default_factory=list)
    questions: Dict[str, QuestionSpec]
    max_depth: int = 8
    search_budget: int = 64
    include_proof_tree: bool = True
    include_lambda_term: bool = True
    include_relevance: bool = True
    include_countermodel: bool = True
    # Phase 0 run configuration (None = environment default, see resolve_run_config)
    domain_pack: Optional[str] = None
    use_priors: Optional[Union[bool, str]] = None   # off | field | replace (True = replace)
    network: Optional[bool] = None
    system2: Optional[bool] = None


# ============================================================================
# 3. PRODUCTION NEURO-SYMBOLIC GENTZEN REASONING ENGINE
# ============================================================================

class ProductionGentzenEngine:
    """
    Production-ready Neuro-Symbolic Gentzen Reasoning Engine.
    Combines System 1 (single-pass RoPE Span Pointer Transformer) with
    System 2 (Dyckhoff G4ip strictly verified constructive proof search),
    Curry-Howard lambda-term serialization, and Intuitionistic Relevance (IR) tracking.
    """

    def __init__(
        self,
        base_dir: Path = MODEL_DIR,
        temperature: float = 1.15,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
    ):
        self.device = device or os.getenv("DEVICE") or (
            "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
        self.temperature = temperature
        self.base_dir = base_dir

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = LogicTokenizer()

        if model is not None:
            self.model = model.to(self.device).eval()
        else:
            print(f"[*] Initializing nanoGentzen-v3 engine on {self.device} from: {base_dir}")
            self.model = GentzenPolicyValueModel.from_pretrained(
                str(base_dir),
                local_files_only=True,
                trust_remote_code=True,
            ).to(self.device).eval()

        if tokenizer is None and getattr(self.model.config, "tokenizer_type", "char") == "word":
            from_dir = Path(base_dir) / "vocab.json"
            try:
                from .tokenization_nanogentzen import tokenizer_for_config
            except ImportError:
                raise
            self.tokenizer = tokenizer_for_config(self.model.config, str(from_dir) if from_dir.exists() else None)
        self.temperature = float(getattr(self.model.config, "s1_temperature", self.temperature))
        self.searcher = NeuralProofSearch(
            self.model,
            self.tokenizer,
            device=self.device,
            value_prune_threshold=0.03,
            use_invertible_fastpath=True,
        )
        self.fixed_searcher = FixedOrderSearch(self.model, self.tokenizer, device=self.device,
                                               use_invertible_fastpath=True)
        print("[+] nanoGentzen-v4 Engine successfully loaded and ready for production.")

    @_no_grad
    def system1_evaluate_batch(self, seqs: List[Sequent], network: bool = True) -> List[Dict[str, Any]]:
        """
        Batched forward pass over a list of sequents.
        Applies fast subformula pruning / axiom checks to shortcuts,
        and batches all neural model inferences into a single forward pass.
        """
        if not seqs:
            return []
        if not network:
            return [dict(NEUTRAL_S1) for _ in seqs]

        results: List[Optional[Dict[str, Any]]] = [None] * len(seqs)
        pending_indices: List[int] = []
        pending_encoded: List[List[int]] = []

        for idx, seq in enumerate(seqs):
            if hasattr(seq, "is_axiom") and seq.is_axiom():
                results[idx] = {
                    "epistemic": {"false": 0.0, "undecided": 0.0, "true": 1.0},
                    "value": 0.999,
                    "confidence": 0.999,
                }
                continue

            # literal-only gamma, literal goal absent: not derivable (sound shortcut)
            if _literal_fastpath_unprovable(normalize_sequent(seq)):
                results[idx] = {
                    "epistemic": {"false": 0.1, "undecided": 0.8, "true": 0.1},
                    "value": 0.25,
                    "confidence": 0.25,
                }
                continue

            # System 1 reads the sequent in its original premise order, exactly as before Phase 1
            # (only proof search needs the normalised order, for pivot indices).
            if getattr(self.model.config, "input_format", "") == "normalized_canonical":
                eval_seq = model_view_sequent(seq)[0]   # v4 checkpoints
            else:
                eval_seq = canonicalize_sequent(seq)[0]
            encoded = self.tokenizer.encode(eval_seq.model_str())
            if len(encoded) > self.model.config.block_size:
                results[idx] = dict(NEUTRAL_S1, window_overflow=True)   # the model cannot see the whole sequent
                continue
            pending_indices.append(idx)
            pending_encoded.append(encoded)

        if pending_indices:
            block_size = self.model.config.block_size
            pad_id = self.tokenizer.pad_id
            batch_padded = [
                enc + [pad_id] * (block_size - len(enc))
                for enc in pending_encoded
            ]
            inp = as_input(batch_padded, self.device, self.model)

            rule_probs, pivot_probs, ep_probs, val_score = self.model.predict_epistemic(inp)
            ep_cpu = to_numpy(ep_probs)
            val_cpu = to_numpy(val_score)

            for b_i, orig_idx in enumerate(pending_indices):
                p_f = round(float(ep_cpu[b_i, 0].item()), 4)
                p_u = round(float(ep_cpu[b_i, 1].item()), 4)
                p_t = round(float(ep_cpu[b_i, 2].item()), 4)
                raw_val = float(val_cpu[b_i].item())

                # Temperature-calibrated confidence
                p = min(max(raw_val, 1e-6), 1.0 - 1e-6)
                logit = math.log(p / (1.0 - p)) / self.temperature
                conf = round(1.0 / (1.0 + math.exp(-logit)), 4)

                epi = {"false": p_f, "undecided": p_u, "true": p_t}
                if ep_cpu.shape[1] > 3:   # v4: 4 classes, temperature fitted on the whole distribution
                    q = [max(float(ep_cpu[b_i, k].item()), 1e-12) ** (1.0 / self.temperature) for k in range(4)]
                    z = sum(q)
                    q = [x / z for x in q]
                    epi = {"false": round(q[0], 4), "undecided": round(q[1], 4), "true": round(q[2], 4),
                           "inconsistent": round(q[3], 4)}
                    conf = round(q[2] + 0.5 * q[1], 4)
                results[orig_idx] = {
                    "epistemic": epi,
                    "value": round(raw_val, 4),
                    "confidence": conf,
                }

        return [r for r in results if r is not None]

    @_no_grad
    def system1_evaluate(self, seq: Sequent, network: bool = True) -> Dict[str, Any]:
        """
        Ultra-fast single forward pass (< 3ms) returning Kripke triality and value score.
        """
        return self.system1_evaluate_batch([seq], network=network)[0]

    def system2_prove(
        self, seq: Sequent, max_depth: int = 8, budget: int = 64, network: bool = True
    ) -> Tuple[bool, Optional[dict], Optional[dict], Dict[str, Any], str, Dict[str, Any]]:
        """
        Phase 1 proof pipeline. Returns (proved, tree, countermodel, relevance, term, info).
          1. literal-only fast path (sound refutation of the obvious cases)
          2. neural (or fixed-order) G4ip search within max_depth choice points / `budget` nodes
          3. if that search stopped at a limit: the complete G4ip prover with a time cap
        info["exhausted"] is True only when the time cap was hit, so "not proved" otherwise means
        "not intuitionistically derivable". Every returned tree passed verify_proof_tree_strict.
        """
        seq = normalize_sequent(seq)
        empty_rel = {"relevant_witnesses": [], "unused_premises": [f.to_str() for f in seq.gamma],
                     "is_strictly_relevant": False, "relevance_ratio": 0.0}
        info: Dict[str, Any] = {"prover": None, "nodes": 0, "exhausted": False}
        if _literal_fastpath_unprovable(seq):
            info["prover"] = "literal_fastpath"
            return False, None, find_kripke_countermodel(seq), empty_rel, "nil", info
        searcher = self.searcher if network else self.fixed_searcher
        proof, _ = searcher.prove(seq, max_depth=max_depth, max_nodes=budget, extract_countermodel_on_failure=False)
        info.update(prover="neural_search" if network else "fixed_order_search", nodes=searcher.last_nodes,
                    window_overflow=getattr(searcher, "last_window_overflow", False))
        if proof is None and searcher.last_truncated:
            proof, timed_out = g4ip_prove(seq, deadline_s=BACKSTOP_S)
            info["prover"] = "g4ip_backstop"
            info["exhausted"] = timed_out
        if proof is not None:
            if verify_proof_tree_strict(proof):
                # Phase 3: typed proof term, checked by the independent checker; relevance from its free hypotheses
                try:
                    cert = _certs.certify(proof)
                except Exception as e:  # never lose a verified proof because of the certificate layer
                    cert = {"checked": False, "check_error": f"{type(e).__name__}: {e}"}
                if cert.get("checked"):
                    info["certificate"] = {k: cert[k] for k in ("context", "type", "term", "checked")}
                    return True, proof, None, _certs.relevance(seq, cert), cert["term_str"], info
                info["certificate_error"] = cert.get("check_error")
                return True, proof, None, extract_relevance_witnesses(seq, proof), "nil", info
            info["verification_failed"] = True
            return False, None, None, empty_rel, "nil", info
        cm = None if info["exhausted"] else find_kripke_countermodel(seq)
        return False, None, cm, empty_rel, "nil", info

    def decide(self, req: Union[DecideRequest, Dict[str, Any]]) -> Dict[str, Any]:
        """
        System 2 decision (Phase 1+2):
          * premises are checked for consistency first (union of the goal and ~goal cones)
          * statuses: PROVEN_TRUE, PROVEN_FALSE, INTUITIONISTIC_UNDECIDED, SEARCH_BUDGET_EXHAUSTED,
            INCONSISTENT_PREMISES, SYSTEM1_ONLY, INTUITIONISTIC_WEAKENED_REJECTED (noul);
            PROVEN, MULTIPLE_PROVEN, NOT_PROVEN, ... (choice / score)
          * proven answers carry probability 1.0 on the proven label (priors, if requested, go to
            `prior_distribution`; NANOGENTZEN_PRIORS=replace keeps the pre-Phase-2 behaviour)
        """
        if isinstance(req, dict):
            req = DecideRequest(**req)
        t0 = time.perf_counter()
        cfg = resolve_run_config(req)
        report: Dict[str, Any] = {}
        global_rules: List[Formula] = [compile_to_formula(r) for r in req.policy_rules]
        facts: List[Formula] = []
        wf = (detect_workflow(req.state)
              if cfg["domain_pack"] == "typed-decisions" and isinstance(req.state, dict) else None)
        if wf and not req.predicates and hasattr(parser_module, "compile_structured_decision"):
            s_facts, s_rules, _ = parser_module.compile_structured_decision(req.state)
            facts.extend(s_facts)
            global_rules.extend(s_rules)
        g_facts, ungrounded_predicates = ground_state_generic(req.state, req.predicates, report)
        for f in g_facts:
            if f not in facts:
                facts.append(f)

        results: Dict[str, Any] = {}
        for q_id, q in req.questions.items():
            q_facts = list(facts)
            rules = list(global_rules) + [compile_to_formula(r) for r in (q.rules or [])]
            grounding = "rule_pack" if wf else "rules"
            if (wf is None and (q.criteria or q.instructions) and not req.policy_rules and not q.rules
                    and hasattr(parser_module, "compile_criteria_and_state")):
                c_opts = (q.options or (list(q.criteria.keys()) if isinstance(q.criteria, dict) else None)
                          or ([str(l) for l in q.levels] if q.levels is not None else
                              ([str(i) for i in range(len(q.criteria))] if isinstance(q.criteria, list) else ["false", "true"])))
                c_facts, c_rules = parser_module.compile_criteria_and_state(
                    state=req.state, q_id=q_id, q_type=q.type, options=[str(o) for o in c_opts],
                    criteria=q.criteria, instructions=q.instructions, target_atom=q.target or q_id)
                q_facts.extend(c_facts)
                rules.extend(c_rules)
                grounding = "criteria_heuristic"
            prove_on = q.require_proof and cfg["system2"]
            cone = lambda g: select_relevant_gamma(q_facts, rules, g)  # noqa: E731

            if q.type == "noul":
                res = self._decide_noul(q_id, q, cone, cfg, wf, req, prove_on, q_facts, rules)
            elif q.type == "choice":
                res = self._decide_choice(q_id, q, cone, cfg, wf, req, prove_on, q_facts, rules)
            else:
                res = self._decide_score(q_id, q, cone, cfg, wf, req, prove_on, q_facts, rules)
            res["grounding"] = grounding
            if grounding == "criteria_heuristic" and res.get("verified"):
                # premises were generated from the criteria text, not taken from the state: the proof is about
                # those heuristic premises, so it does not certify the answer (verification keeps the kernel result)
                res["verified"] = False
                res["verification_note"] = "proof over heuristic premises compiled from the criteria text"
            results[q_id] = res

        latency = round((time.perf_counter() - t0) * 1000, 2)
        approx_in = len(str(req.state).split()) + sum(len(str(q).split()) for q in req.questions.values())
        return {
            "model": req.model,
            "mode": "system_two_verified",
            "run_config": {k: v for k, v in cfg.items()},
            "workflow_used": wf,
            "latency_ms": latency,
            "usage": {"input_tokens": max(approx_in, 1), "output_tokens": max(len(results) * 16, 1),
                      "requests": 1, "retries": 0},
            "ungrounded_predicates": ungrounded_predicates,
            "ungrounded_fields": report.get("ungrounded_fields", []),
            "renamed_identifiers": report.get("renamed_identifiers", {}),
            "answers": results,
            "decisions": results,
        }

    # ---------------------------------------------------------------- shared helpers
    def _conflict(self, premises: List[Formula]) -> Optional[List[str]]:
        """Minimal (irredundant) inconsistent subset of `premises`, or None if they are consistent."""
        prem = list(dict.fromkeys(premises))
        tree, to = g4ip_prove(Sequent(tuple(prem), ()), deadline_s=BACKSTOP_S)
        if tree is None and to:
            # consistency unknown: never treat as consistent (a proof could otherwise succeed ex falso)
            return [f"consistency check timed out after {BACKSTOP_S}s; premises not certified consistent"]
        if tree is None:
            return None
        cur = list(prem)
        for x in list(cur):
            trial = [y for y in cur if y != x]
            t2, to2 = g4ip_prove(Sequent(tuple(trial), ()), deadline_s=BACKSTOP_S)
            if t2 is not None:
                cur = trial
        return [f.to_str() for f in cur]

    def _union_cones(self, cone, targets: List[Formula]) -> List[Formula]:
        out: List[Formula] = []
        for t in targets:
            for g in list(cone(t)) + list(cone(Not(t))):
                if g not in out:
                    out.append(g)
        return out

    def _missing_facts(self, goal: Formula, q_facts, rules) -> List[str]:
        """Irredundant set of base literals (not the goal, not derivable heads) that would make goal provable."""
        gamma = list(select_relevant_gamma(q_facts, rules, goal))
        known = set()
        for f in gamma:
            if isinstance(f, Var):
                known.add(f.name)
            elif isinstance(f, Not) and isinstance(f.inner, Var):
                known.add(f.inner.name)
        heads, cands = set(), []
        for r in gamma:
            if isinstance(r, Imp):
                heads |= extract_vars(r.right)

        def lits(f, pos=True):
            if isinstance(f, Var):
                yield f if pos else Not(f)
            elif isinstance(f, Not):
                yield from lits(f.inner, not pos)
            elif isinstance(f, (And, Or)):
                yield from lits(f.left, pos)
                yield from lits(f.right, pos)
        goal_vars = extract_vars(goal)
        for r in gamma:
            if isinstance(r, Imp):
                for l in lits(r.left):
                    v = list(extract_vars(l))[0]
                    if v not in known and v not in heads and v not in goal_vars and l not in cands:
                        cands.append(l)
        cands = cands[:16]
        if not cands:
            return []

        def works(extra):
            t, _ = g4ip_prove(Sequent(tuple(gamma + extra), (goal,)), deadline_s=BACKSTOP_S)
            if t is None:
                return False
            t2, to2 = g4ip_prove(Sequent(tuple(gamma + extra), ()), deadline_s=BACKSTOP_S)
            return t2 is None and not to2   # a timed-out consistency check is not "consistent"
        if works(cands):
            cur = list(cands)
            for x in list(cur):
                trial = [y for y in cur if y != x]
                if works(trial):
                    cur = trial
            return [f.to_str() for f in cur]
        for k in (1, 2):
            for combo in itertools.combinations(cands[:10], k):
                if works(list(combo)):
                    return [f.to_str() for f in combo]
        return []

    def _probs_for_proven(self, cfg, wf, q_id, options, label, req, epistemic):
        """Degenerate distribution on the proven label; priors per cfg['priors']."""
        probs = _proven_dist(options, label, cfg["proven_eps"])
        prior = None
        if cfg["priors"] != "off" and hasattr(parser_module, "get_calibrated_distribution"):
            prior = parser_module.get_calibrated_distribution(
                wf, q_id, str(label), options, state=req.state if isinstance(req.state, dict) else None,
                epistemic_probs=epistemic)
        if prior and cfg["priors"] == "replace":
            return prior, None
        return probs, prior

    # ---------------------------------------------------------------- noul
    def _decide_noul(self, q_id, q, cone, cfg, wf, req, prove_on, q_facts, rules):
        goal = compile_to_formula(q.target or q_id)
        seq_pos = Sequent(cone(goal), (goal,))
        s1 = self.system1_evaluate(seq_pos, network=cfg["network"])
        res: Dict[str, Any] = {"type": "noul", "epistemic": s1["epistemic"]}
        if not prove_on:
            v = s1["confidence"]          # temperature-calibrated System 1 value, as before Phase 1
            res.update(answer=None, verification="SYSTEM1_ONLY", confidence=v, noul=round(v, 4),
                       probabilities={"false": round(1 - v, 4), "true": round(v, 4)}, system1_estimate=v)
            return res
        conflict = self._conflict(self._union_cones(cone, [goal]))
        if conflict:
            res.update(answer=None, verification="INCONSISTENT_PREMISES", conflicting_premises=conflict,
                       confidence=0.0, noul=0.5, probabilities={"false": 0.5, "true": 0.5})
            return res
        ok_pos, tree, cm_pos, rel, term, info_pos = self.system2_prove(seq_pos, req.max_depth, req.search_budget, network=cfg["network"])
        status, info = None, {"positive": info_pos}
        if ok_pos:
            status = "PROVEN_TRUE"
        else:
            neg = Not(goal)
            ok_neg, tree_n, cm_n, rel_n, term_n, info_neg = self.system2_prove(
                Sequent(cone(neg), (neg,)), req.max_depth, req.search_budget, network=cfg["network"])
            info["negative"] = info_neg
            if ok_neg:
                status, tree, rel, term = "PROVEN_FALSE", tree_n, rel_n, term_n
            elif info_pos.get("exhausted") or info_neg.get("exhausted"):
                status = "SEARCH_BUDGET_EXHAUSTED"
            else:
                status = "INTUITIONISTIC_UNDECIDED"
        if q.require_relevance and status == "PROVEN_TRUE" and not rel.get("is_strictly_relevant", True):
            status = "INTUITIONISTIC_WEAKENED_REJECTED"
        res["verification"] = status
        res["search"] = info
        if status in ("PROVEN_TRUE", "PROVEN_FALSE"):
            lbl = "true" if status == "PROVEN_TRUE" else "false"
            probs, prior = self._probs_for_proven(cfg, wf, q_id, ["false", "true"], lbl, req, s1["epistemic"])
            res.update(answer=status == "PROVEN_TRUE", confidence=1.0, probabilities=probs, noul=probs["true"])
            if prior:
                res["prior_distribution"] = prior
        else:
            v = s1["value"]
            res.update(answer=None, confidence=v, system1_estimate=v, noul=round(v, 4),
                       probabilities={"false": round(1 - v, 4), "true": round(v, 4)})
            if status in ("INTUITIONISTIC_UNDECIDED", "SEARCH_BUDGET_EXHAUSTED", "INTUITIONISTIC_WEAKENED_REJECTED"):
                res["missing_facts"] = self._missing_facts(goal, q_facts, rules)
            if status == "INTUITIONISTIC_UNDECIDED" and req.include_countermodel and cm_pos:
                res["countermodel"] = cm_pos
        if req.include_relevance and rel:
            res["relevance"] = rel
        if req.include_lambda_term and term and term != "nil":
            res["curry_howard_term"] = term
            proven_info = info.get("negative") if status == "PROVEN_FALSE" else info.get("positive")
            if proven_info and proven_info.get("certificate"):
                res["certificate"] = proven_info["certificate"]
        if req.include_proof_tree and tree:
            res["proof_tree"] = tree
        return res

    # ---------------------------------------------------------------- choice
    def _decide_choice(self, q_id, q, cone, cfg, wf, req, prove_on, q_facts, rules):
        options = q.options or (list(q.criteria.keys()) if isinstance(q.criteria, dict) else [])
        if not options:
            raise HTTPException(400, f"Question '{q_id}' requires 'options' or dict 'criteria'.")
        forms = {opt: compile_to_formula(opt) for opt in options}
        seqs = {opt: Sequent(cone(forms[opt]), (forms[opt],)) for opt in options}
        s1_batch = self.system1_evaluate_batch([seqs[o] for o in options], network=cfg["network"])
        s1_scores = {o: r["confidence"] for o, r in zip(options, s1_batch)}
        ep_by_opt = {o: r["epistemic"] for o, r in zip(options, s1_batch)}
        logits = [math.log(max(p, 1e-6) / max(1.0 - p, 1e-6)) for p in s1_scores.values()]
        mx = max(logits)
        ex = [math.exp(l - mx) for l in logits]
        probs = {o: round(e / sum(ex), 4) for o, e in zip(options, ex)}
        ranked = sorted(options, key=lambda o: probs[o], reverse=True)
        res: Dict[str, Any] = {"type": "choice"}
        if not prove_on:
            res.update(choice=ranked[0], verified=False, verification="SYSTEM1_ONLY", confidence=probs[ranked[0]],
                       probabilities=probs, epistemic=ep_by_opt.get(ranked[0], {}))
            return res
        conflict = self._conflict(self._union_cones(cone, list(forms.values())))
        if conflict:
            res.update(choice=None, verified=False, verification="INCONSISTENT_PREMISES", conflicting_premises=conflict,
                       confidence=0.0, probabilities={o: round(1 / len(options), 4) for o in options})
            return res
        proven, trees, exhausted = [], {}, False
        for opt in ranked:
            ok, tree, cm, rel, term, info = self.system2_prove(seqs[opt], req.max_depth, req.search_budget, network=cfg["network"])
            exhausted |= info.get("exhausted", False)
            if ok:
                proven.append(opt)
                trees[opt] = (tree, rel, term, info.get("certificate"))
        exclusive = bool(getattr(q, "exclusive", False))
        if len(proven) > 1 and exclusive:
            res.update(choice=None, verified=False, verification="INCONSISTENT_PREMISES", proven_options=proven,
                       conflicting_premises=[f"options declared exclusive, but {proven} are all provable"],
                       confidence=0.0, probabilities={o: round(1 / len(options), 4) for o in options})
            return res
        if len(proven) == 1:
            chosen = proven[0]
            pr, prior = self._probs_for_proven(cfg, wf, q_id, options, chosen, req, ep_by_opt[chosen])
            res.update(choice=chosen, verified=True, verification="PROVEN", confidence=1.0, probabilities=pr)
            if prior:
                res["prior_distribution"] = prior
        elif len(proven) > 1:
            chosen = proven[0]
            res.update(choice=chosen, verified=False, verification="MULTIPLE_PROVEN", proven_options=proven,
                       confidence=round(1 / len(proven), 4),
                       probabilities={o: (round(1 / len(proven), 4) if o in proven else 0.0) for o in options})
        else:
            chosen = ranked[0]
            res.update(choice=chosen, verified=False,
                       verification="SEARCH_BUDGET_EXHAUSTED" if exhausted else "NOT_PROVEN",
                       confidence=probs[chosen], probabilities=probs,
                       missing_facts={o: self._missing_facts(forms[o], q_facts, rules) for o in ranked[:3]})
        res["epistemic"] = ep_by_opt.get(chosen, {})
        if chosen in trees:
            tree, rel, term, cert = trees[chosen]
            if req.include_relevance and rel:
                res["relevance"] = rel
            if req.include_lambda_term and term and term != "nil":
                res["curry_howard_term"] = term
                if cert:
                    res["certificate"] = cert
            if req.include_proof_tree and tree:
                res["proof_tree"] = tree
        diff = round(1.0 - sum(res["probabilities"].values()), 4)
        if diff:
            top = max(res["probabilities"], key=res["probabilities"].get)
            res["probabilities"][top] = round(res["probabilities"][top] + diff, 4)
        return res

    # ---------------------------------------------------------------- score
    def _decide_score(self, q_id, q, cone, cfg, wf, req, prove_on, q_facts, rules):
        levels = (q.levels if q.levels is not None else
                  (list(range(len(q.criteria))) if isinstance(q.criteria, list) and q.criteria else [0, 1, 2, 3]))
        atoms = q.level_atoms or [f"{q_id}_{lvl}" for lvl in levels]
        pairs = list(zip(levels, atoms))
        try:
            pairs.sort(key=lambda x: float(x[0]), reverse=True)      # numeric levels: highest first
        except (TypeError, ValueError):
            pairs = list(reversed(pairs))                              # otherwise: declared order, last first
        forms = {lvl: compile_to_formula(a) for lvl, a in pairs}
        seqs = {lvl: Sequent(cone(forms[lvl]), (forms[lvl],)) for lvl, _ in pairs}
        s1_batch = self.system1_evaluate_batch([seqs[l] for l, _ in pairs], network=cfg["network"])
        s1_by = {l: r for (l, _), r in zip(pairs, s1_batch)}
        res: Dict[str, Any] = {"type": "score"}
        if prove_on:
            conflict = self._conflict(self._union_cones(cone, list(forms.values())))
            if conflict:
                k = len(levels)
                res.update(score=None, proven_tier=None, verified=False, verification="INCONSISTENT_PREMISES",
                           conflicting_premises=conflict, probabilities={str(l): round(1 / k, 4) for l in levels})
                return res
        proven_lvl, proof_tree, rel_info, lambda_term, exhausted, cert_lvl = None, None, {}, "", False, None
        weights = []
        for lvl, _ in pairs:
            w = s1_by[lvl]["confidence"]
            if prove_on and proven_lvl is None:
                ok, tr, cm, rel, term, info = self.system2_prove(seqs[lvl], req.max_depth, req.search_budget, network=cfg["network"])
                exhausted |= info.get("exhausted", False)
                if ok:
                    proven_lvl, proof_tree, rel_info, lambda_term, w = lvl, tr, rel, term, 1.0
                    cert_lvl = info.get("certificate")
            weights.append((lvl, w))
        order = [l for l, _ in reversed(pairs)]
        wmap = dict(weights)
        ex = [math.exp(4.0 * wmap[l]) for l in order]
        probs = {str(l): round(e / sum(ex), 4) for l, e in zip(order, ex)}
        str_levels = [str(l) for l in levels]
        if proven_lvl is not None:
            probs, prior = self._probs_for_proven(cfg, wf, q_id, str_levels, str(proven_lvl), req, s1_by[proven_lvl]["epistemic"])
            if prior:
                res["prior_distribution"] = prior
        diff = round(1.0 - sum(probs.values()), 4)
        if probs and diff:
            top = max(probs, key=probs.get)
            probs[top] = round(probs[top] + diff, 4)
        if wf is not None:
            score_val = round(sum(float(k) * v for k, v in probs.items()), 4)
        elif proven_lvl is not None:
            score_val = float(proven_lvl)
        else:
            score_val = round(sum(float(k) * v for k, v in probs.items()), 2)
        res.update(score=score_val, proven_tier=proven_lvl, verified=proven_lvl is not None, probabilities=probs,
                   verification=("PROVEN" if proven_lvl is not None else
                                 ("SYSTEM1_ONLY" if not prove_on else ("SEARCH_BUDGET_EXHAUSTED" if exhausted else "NOT_PROVEN"))))
        if req.include_relevance and rel_info:
            res["relevance"] = rel_info
        if req.include_lambda_term and lambda_term and lambda_term != "nil":
            res["curry_howard_term"] = lambda_term
            if cert_lvl:
                res["certificate"] = cert_lvl
        if req.include_proof_tree and proof_tree:
            res["proof_tree"] = proof_tree
        return res

    def system_one(self, req: Union[DecideRequest, Dict[str, Any]]) -> Dict[str, Any]:
        """
        API: Single-pass System 1 Epistemic Valuation (< 3ms per item).
        """
        if isinstance(req, dict):
            req = DecideRequest(**req)

        t0 = time.perf_counter()
        cfg = resolve_run_config(req)
        facts: List[Formula] = []
        global_rules = [compile_to_formula(r) for r in req.policy_rules]

        wf = (
            detect_workflow(req.state)
            if cfg["domain_pack"] == "typed-decisions" and isinstance(req.state, dict)
            else None
        )
        if wf and not req.predicates and hasattr(parser_module, "compile_structured_decision"):
            s_facts, s_rules, _ = parser_module.compile_structured_decision(req.state)
            facts.extend(s_facts)
            global_rules.extend(s_rules)

        report: Dict[str, Any] = {}
        g_facts, ungrounded_predicates = ground_state_generic(req.state, req.predicates, report)
        for f in g_facts:
            if f not in facts:
                facts.append(f)

        answers: Dict[str, Any] = {}
        query_plan = []  # List of tuples: (q_id, item_type, key, Sequent)

        for q_id, q in req.questions.items():
            q_facts = list(facts)
            rules = list(global_rules) + [compile_to_formula(r) for r in (q.rules or [])]

            if (
                wf is None
                and (q.criteria or q.instructions)
                and not req.policy_rules
                and not q.rules
                and hasattr(parser_module, "compile_criteria_and_state")
            ):
                c_opts = (
                    q.options
                    or (list(q.criteria.keys()) if isinstance(q.criteria, dict) else None)
                    or (
                        [str(l) for l in q.levels]
                        if q.levels is not None
                        else (
                            [str(i) for i in range(len(q.criteria))]
                            if isinstance(q.criteria, list)
                            else ["false", "true"]
                        )
                    )
                )
                c_facts, c_rules = parser_module.compile_criteria_and_state(
                    state=req.state,
                    q_id=q_id,
                    q_type=q.type,
                    options=[str(o) for o in c_opts],
                    criteria=q.criteria,
                    instructions=q.instructions,
                    target_atom=q.target or q_id,
                )
                q_facts.extend(c_facts)
                rules.extend(c_rules)

            if q.type == "noul":
                goal = compile_to_formula(q.target or q_id)
                seq = Sequent(select_relevant_gamma(q_facts, rules, goal), (goal,))
                query_plan.append((q_id, "noul", "target", seq))

            elif q.type == "choice":
                options = q.options or (list(q.criteria.keys()) if isinstance(q.criteria, dict) else [])
                for opt in options:
                    f_opt = compile_to_formula(opt)
                    seq_opt = Sequent(select_relevant_gamma(q_facts, rules, f_opt), (f_opt,))
                    query_plan.append((q_id, "choice", opt, seq_opt))

            elif q.type == "score":
                levels = (
                    [str(l) for l in q.levels]
                    if q.levels is not None
                    else (
                        [str(i) for i in range(len(q.criteria))]
                        if isinstance(q.criteria, list) and q.criteria
                        else ["0", "1", "2", "3"]
                    )
                )
                for lvl in levels:
                    f_lvl = compile_to_formula(f"{q_id}_{lvl}")
                    seq_lvl = Sequent(select_relevant_gamma(q_facts, rules, f_lvl), (f_lvl,))
                    query_plan.append((q_id, "score", lvl, seq_lvl))

        # Single intra-request batched forward pass across all candidate sequents
        all_seqs = [item[3] for item in query_plan]
        all_results = self.system1_evaluate_batch(all_seqs)

        plan_results: Dict[str, Dict[str, Any]] = {}
        for (q_id, q_type, key, _), res in zip(query_plan, all_results):
            if q_id not in plan_results:
                plan_results[q_id] = {}
            plan_results[q_id][key] = res

        for q_id, q in req.questions.items():
            if q.type == "noul":
                s1 = plan_results.get(q_id, {}).get("target", {
                    "epistemic": {"false": 0.5, "undecided": 0.5, "true": 0.0},
                    "confidence": 0.5,
                })
                p_true = s1["epistemic"]["true"]
                noul_probs = {"false": round(1.0 - p_true, 4), "true": round(p_true, 4)}
                diff_n = round(1.0 - sum(noul_probs.values()), 4)
                if diff_n != 0:
                    noul_probs["true"] = round(noul_probs["true"] + diff_n, 4)

                answers[q_id] = {
                    "type": "noul",
                    "answer": p_true >= 0.5,
                    "confidence": s1["confidence"],
                    "noul": noul_probs["true"],
                    "epistemic": s1["epistemic"],
                    "probabilities": noul_probs,
                }

            elif q.type == "choice":
                options = q.options or (list(q.criteria.keys()) if isinstance(q.criteria, dict) else [])
                q_res = plan_results.get(q_id, {})
                probs = {opt: q_res.get(opt, {}).get("confidence", 0.0) for opt in options}
                total_p = sum(probs.values()) or 1.0
                norm_probs = {o: round(v / total_p, 4) for o, v in probs.items()}
                diff_c = round(1.0 - sum(norm_probs.values()), 4)
                if norm_probs and diff_c != 0:
                    w = max(norm_probs, key=norm_probs.get)
                    norm_probs[w] = round(norm_probs[w] + diff_c, 4)
                winner = max(norm_probs, key=norm_probs.get) if norm_probs else None
                answers[q_id] = {
                    "type": "choice",
                    "choice": winner,
                    "probabilities": norm_probs,
                }

            elif q.type == "score":
                levels = (
                    [str(l) for l in q.levels]
                    if q.levels is not None
                    else (
                        [str(i) for i in range(len(q.criteria))]
                        if isinstance(q.criteria, list) and q.criteria
                        else ["0", "1", "2", "3"]
                    )
                )
                q_res = plan_results.get(q_id, {})
                probs = {lvl: q_res.get(lvl, {}).get("confidence", 0.0) for lvl in levels}
                total_p = sum(probs.values()) or 1.0
                norm_probs = {lvl: round(v / total_p, 4) for lvl, v in probs.items()}
                diff_s = round(1.0 - sum(norm_probs.values()), 4)
                if norm_probs and diff_s != 0:
                    w = max(norm_probs, key=norm_probs.get)
                    norm_probs[w] = round(norm_probs[w] + diff_s, 4)
                exp_score = sum(float(l) * p for l, p in norm_probs.items() if l.replace(".", "", 1).isdigit())
                answers[q_id] = {
                    "type": "score",
                    "score": round(exp_score, 2),
                    "probabilities": norm_probs,
                }

        latency = round((time.perf_counter() - t0) * 1000, 2)
        approx_in = len(str(req.state).split()) + sum(len(str(q).split()) for q in req.questions.values())
        usage = {
            "input_tokens": max(approx_in, 1),
            "output_tokens": max(len(answers) * 16, 1),
            "requests": 1,
            "retries": 0,
        }
        return {
            "model": req.model,
            "mode": "system_one_fast",
            "run_config": {k: v for k, v in cfg.items()},
            "workflow_used": wf,
            "ungrounded_predicates": ungrounded_predicates,
            "ungrounded_fields": report.get("ungrounded_fields", []),
            "renamed_identifiers": report.get("renamed_identifiers", {}),
            "latency_ms": latency,
            "usage": usage,
            "answers": answers,
            "decisions": answers,
        }


def _client_errors(fn):
    def wrapped(self, req, *a, **k):
        try:
            return fn(self, req, *a, **k)
        except ValueError as e:
            raise HTTPException(400, str(e))
    wrapped.__doc__ = fn.__doc__
    return wrapped


ProductionGentzenEngine.decide = _client_errors(ProductionGentzenEngine.decide)
ProductionGentzenEngine.system_one = _client_errors(ProductionGentzenEngine.system_one)


# Backward compatibility alias
LocalGentzenJevEngine = ProductionGentzenEngine


# ============================================================================
# 4. FASTAPI APPLICATION SETUP
# ============================================================================

ENGINE: Optional[ProductionGentzenEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE
    ENGINE = ProductionGentzenEngine(base_dir=MODEL_DIR)
    yield


app = FastAPI(
    title="nanoGentzen-v3 Neuro-Symbolic Jev Server",
    description="Domain-agnostic formal verification and decision server with Saul Kripke semantics, "
                "Intuitionistic Relevance Logic, and Curry-Howard Proof Terms.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 5. API ENDPOINTS & PYTHON FUNCTIONS
# ============================================================================

@app.get("/health")
def health():
    if ENGINE is None:
        raise HTTPException(503, "Engine not initialized.")
    return {
        "status": "healthy",
        "engine": "nanoGentzen-v3",
        "device": ENGINE.device,
        "max_antecedents": getattr(ENGINE.model.config, "max_antecedents", 32),
        "num_epistemic_classes": getattr(ENGINE.model.config, "num_epistemic_classes", 3),
        "pointer_attention": getattr(ENGINE.model.config, "use_pointer_pivot", True),
    }


@app.post("/v1/systemone")
def systemone_api(req: DecideRequest):
    global ENGINE
    if ENGINE is None:
        ENGINE = ProductionGentzenEngine(base_dir=MODEL_DIR)
    return ENGINE.system_one(req)


@app.post("/v1/decide")
def decide_api(req: DecideRequest):
    global ENGINE
    if ENGINE is None:
        ENGINE = ProductionGentzenEngine(base_dir=MODEL_DIR)
    return ENGINE.decide(req)


def get_engine(device: Optional[str] = None) -> ProductionGentzenEngine:
    """Returns or initializes the singleton ProductionGentzenEngine."""
    global ENGINE
    if ENGINE is None:
        ENGINE = ProductionGentzenEngine(base_dir=MODEL_DIR, device=device)
    return ENGINE


def decide(req: Union[DecideRequest, Dict[str, Any]]) -> Dict[str, Any]:
    """In-process Python interface for benchmark evaluation and unit tests."""
    return get_engine().decide(req)


def systemone(req: Union[DecideRequest, Dict[str, Any]]) -> Dict[str, Any]:
    """In-process Python interface for System-1 benchmark evaluation."""
    return get_engine().system_one(req)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)
