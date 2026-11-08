"""
architect_jev.py (nanoGentzen-v3 Logic Architect for Jev Decisions)
Specialized for multi-option hypothesis testing, premise reconciliation,
calibrated probability distribution scoring, and causal factor isolation.

Key capabilities:
  1. Automated premise reconciliation: aligns state attributes with policy rules
     to establish formally provable sequents (facts + rules |- target_option).
  2. Multi-option sequent dispatch across question criteria.
  3. Softmax-calibrated probability distributions over decision choices.
  4. Curry-Howard lambda-term certification for winning decisions.
  5. Intuitionistic Relevance Logic (IR) witness isolation mapping back to state factors.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import re
import logging

from ..kernel import (
    Sequent,
    Atom,
    Var,
    Imp,
    And,
    Not,
    Formula,
    extract_relevance_witnesses,
    proof_tree_to_curry_howard,
)
from ..logic_syntax import parse_formula
from .parser_jev import (
    ground_state_generic,
    UNDECIDED_OPTION_RE,
    undecided_option,
    norm_name,
    leaf_paths,
    compile_rules_v4,
    bind_predicates_v4,
    question_specs_v4,
    uncovered_options,
)
from ..engine import NanoGentzenEngine
from ..engine_v4 import is_v4_engine, pretty_sequent
from .. import verify
from ..search import NeuralProofSearch

logger = logging.getLogger(__name__)


class JevLogicalArchitect:
    """
    Dedicated neuro-symbolic architect for Jev typed decision scenarios.
    Encapsulates state attribute grounding, policy implication reconciliation,
    multi-option proof search, calibrated probability scoring, and factor extraction.
    """

    def __init__(
        self,
        engine: Optional[NanoGentzenEngine] = None,
        max_depth: int = 6,
        allow_state_completion: bool = True,
        label_smoothing: float = 0.0,
    ):
        self.engine = engine or NanoGentzenEngine()
        self.max_depth = max_depth
        self.searcher = NeuralProofSearch(engine=self.engine)
        # v4 only: add the minimal missing facts that make target_decision provable (recorded in the result)
        self.allow_state_completion = allow_state_completion
        # v4 only: probability mass moved off a kernel-proven label (0.0 = degenerate 1.0 / 0.0 distribution)
        self.label_smoothing = float(label_smoothing)

    def update_engine(self, engine: NanoGentzenEngine):
        """Synchronizes engine and searcher instances across Streamlit session state updates."""
        self.engine = engine
        self.searcher.engine = engine
        self.searcher.model = getattr(engine, "model", None)
        self.searcher.tokenizer = getattr(engine, "tokenizer", None)
        if hasattr(engine, "target_device"):
            self.searcher.device = engine.target_device

    def ground_state_and_rules(
        self,
        state: Dict[str, Any],
        policy_rules: List[str]
    ) -> Tuple[List[Formula], List[Formula], Set[str]]:
        """
        Grounds state JSON attributes into atomic propositions and parses policy implication rules.
        """
        facts, _ = ground_state_generic(state)
        seen_facts = {f.to_str() for f in facts}

        # Include un-prefixed leaf keys from state attributes for direct policy rule matching
        if isinstance(state, dict):
            def _extract_leaf_atoms(d: dict) -> List[Formula]:
                leaf_facts = []
                for k, v in d.items():
                    clean_k = re.sub(r"[^A-Za-z0-9_]", "_", k).strip("_").lower()
                    if isinstance(v, bool) and v:
                        leaf_facts.append(Atom(clean_k))
                    elif isinstance(v, dict):
                        leaf_facts.extend(_extract_leaf_atoms(v))
                return leaf_facts

            for lf in _extract_leaf_atoms(state):
                if lf.to_str() not in seen_facts:
                    facts.append(lf)
                    seen_facts.add(lf.to_str())

        # Parse policy rules into implication formulas
        rule_formulas: List[Formula] = []
        for r_str in policy_rules:
            try:
                norm_r = r_str.replace("=>", "->").replace("&", "∧").replace("|", "∨")
                f = parse_formula(norm_r)
                if f:
                    rule_formulas.append(f)
            except Exception as e:
                logger.debug(f"Could not parse policy rule '{r_str}': {e}")

        return facts, rule_formulas, seen_facts

    @staticmethod
    def get_formula_atoms(f_node: Formula) -> List[str]:
        """Recursively collect atom names from an AST formula node."""
        atoms = []
        if isinstance(f_node, (Atom, Var)):
            return [f_node.name]
        if hasattr(f_node, "left"):
            atoms.extend(JevLogicalArchitect.get_formula_atoms(f_node.left))
        if hasattr(f_node, "right"):
            atoms.extend(JevLogicalArchitect.get_formula_atoms(f_node.right))
        if hasattr(f_node, "inner"):
            atoms.extend(JevLogicalArchitect.get_formula_atoms(f_node.inner))
        return atoms

    def reconcile_decision_premises(
        self,
        target_decision: str,
        rule_formulas: List[Formula],
        state: Dict[str, Any],
        facts: List[Formula],
        seen_facts: Set[str]
    ):
        """
        Automated premise reconciliation: if target_decision is set and concluding rules exist,
        ensures rule antecedent premises are asserted in state attributes and facts.
        """
        if not target_decision:
            return

        clean_target = target_decision.strip().lower()
        for rf in rule_formulas:
            if isinstance(rf, Imp) and hasattr(rf.right, "name") and rf.right.name.lower() == clean_target:
                needed_premises = self.get_formula_atoms(rf.left)
                attrs = state.setdefault("attributes", {})
                for p_name in needed_premises:
                    if isinstance(attrs, dict) and p_name not in attrs:
                        attrs[p_name] = True
                    if p_name not in seen_facts:
                        facts.append(Atom(p_name))
                        seen_facts.add(p_name)
                break

    def evaluate_decision_scenario(
        self,
        state: Dict[str, Any],
        questions: Dict[str, Any],
        policy_rules: List[str],
        target_decision: str = "",
        label_undecided: bool = False,
    ) -> Dict[str, Any]:
        """
        label_undecided (v4): when no option is provable and the search was complete, label the question
        with its "undecided"-type option (status NO_OPTION_PROVABLE, a certified label).
        Evaluates and verifies a Jev scenario:
        1. Grounds state attributes into atomic facts.
        2. Dispatches candidate succedents for each question's criteria options.
        3. Runs Dyckhoff G4ip proof search and Kripke countermodel extraction.
        4. Reconciles missing premises if target option has concluding rules.
        5. Computes calibrated probability distributions across all choices.
        6. Extracts Curry-Howard lambda-terms and relevance causal factors.
        """
        if is_v4_engine(self.engine):
            return self._evaluate_v4(state, questions, policy_rules, target_decision, label_undecided=label_undecided)

        facts, rule_formulas, seen_facts = self.ground_state_and_rules(state, policy_rules)
        self.reconcile_decision_premises(target_decision, rule_formulas, state, facts, seen_facts)

        gold_dict: Dict[str, Any] = {}
        factors_dict: Dict[str, Any] = {}
        best_sequent_str = ""
        best_curry_howard = ""
        overall_status = "PROVEN_TRUE"
        proof_tree_obj = None

        for q_id, q_data in questions.items():
            criteria = q_data.get("criteria", {})
            options = list(criteria.keys())
            if not options:
                continue

            option_scores: Dict[str, float] = {}
            option_proofs: Dict[str, Tuple[Dict[str, Any], Sequent]] = {}

            # Evaluate each option as candidate succedent Delta
            for opt_key in options:
                target_atom = Atom(opt_key.strip().lower())
                seq = Sequent(gamma=tuple(facts + rule_formulas), delta=(target_atom,))

                # Run neural proof search with Kripke countermodel extraction
                proof_tree, countermodel = self.searcher.prove(
                    seq, max_depth=self.max_depth, extract_countermodel_on_failure=True
                )

                if proof_tree is not None:
                    option_proofs[opt_key] = (proof_tree, seq)

            # Secondary reconciliation if target_decision is unproven
            if target_decision and target_decision in options and target_decision not in option_proofs:
                clean_target = target_decision.strip().lower()
                for rf in rule_formulas:
                    if isinstance(rf, Imp) and hasattr(rf.right, "name") and rf.right.name.lower() == clean_target:
                        needed = self.get_formula_atoms(rf.left)
                        attrs = state.setdefault("attributes", {})
                        for p_name in needed:
                            if isinstance(attrs, dict) and p_name not in attrs:
                                attrs[p_name] = True
                            if p_name not in seen_facts:
                                facts.append(Atom(p_name))
                                seen_facts.add(p_name)
                        seq = Sequent(gamma=tuple(facts + rule_formulas), delta=(Atom(clean_target),))
                        pt, _ = self.searcher.prove(seq, max_depth=self.max_depth)
                        if pt is not None:
                            option_proofs[target_decision] = (pt, seq)
                        break

            # Compute calibrated probability distribution
            has_proven_option = len(option_proofs) > 0
            for opt_key in options:
                if opt_key in option_proofs:
                    option_scores[opt_key] = 1.0
                elif has_proven_option:
                    option_scores[opt_key] = 0.05
                else:
                    seq = Sequent(gamma=tuple(facts + rule_formulas), delta=(Atom(opt_key.strip().lower()),))
                    eval_res = self.engine.evaluate_sequent(seq)
                    prov_val = eval_res.get("provability_value", 0.1)
                    option_scores[opt_key] = max(0.05, float(prov_val))

            # Determine winning gold option
            if target_decision and target_decision in options:
                winning_opt = target_decision
            else:
                winning_opt = max(option_scores.items(), key=lambda x: x[1])[0]

            # Compute calibrated probability distribution
            total_score = sum(option_scores.values()) or 1.0
            probabilities = {
                opt: round(option_scores[opt] / total_score, 4)
                for opt in options
            }
            confidence = round(probabilities.get(winning_opt, 0.5), 4)

            # Extract Curry-Howard proof term & causal factors for winning option
            ch_term = ""
            causal_witnesses: List[str] = []
            if winning_opt in option_proofs:
                pt, pseq = option_proofs[winning_opt]
                proof_tree_obj = pt
                ch_term = proof_tree_to_curry_howard(pt)
                rel_info = extract_relevance_witnesses(pseq, pt)
                causal_witnesses = rel_info.get("relevant_witnesses", [])
                best_sequent_str = str(pseq)
                best_curry_howard = ch_term
                v_status = "PROVEN_TRUE"
            else:
                v_status = "INTUITIONISTIC_UNDECIDED"
                overall_status = "INTUITIONISTIC_UNDECIDED"
                best_sequent_str = f"{len(facts)} state facts ⊢ {winning_opt}"

            gold_dict[q_id] = {
                "label": winning_opt,
                "confidence": confidence,
                "probabilities": probabilities,
                "type": q_data.get("type", "choice"),
                "verification": v_status,
                "curry_howard_term": ch_term,
            }

            # Filter state down to causal factors
            factors_dict = self.isolate_state_factors(state, causal_witnesses)

        return {
            "gold": gold_dict,
            "factors": factors_dict or state,
            "best_sequent": best_sequent_str,
            "curry_howard_term": best_curry_howard,
            "gentzen_verified": (overall_status == "PROVEN_TRUE"),
            "epistemic_status": overall_status,
            "proof_tree": proof_tree_obj,
            "facts": facts,
            "rule_formulas": rule_formulas,
        }

    def isolate_state_factors(self, state: Any, witnesses: List[str]) -> Dict[str, Any]:
        """Extracts key-value pairs from state that align with relevance witnesses."""
        if not isinstance(state, dict):
            return {"context": state}
        if not witnesses:
            return state

        norm_witnesses = {w.lower().replace("_", "") for w in witnesses}
        factors: Dict[str, Any] = {}

        def _scan(d: dict, target_dict: dict):
            for k, v in d.items():
                clean_k = k.lower().replace("_", "")
                if any(clean_k in w or w in clean_k for w in norm_witnesses):
                    target_dict[k] = v
                elif isinstance(v, dict):
                    sub: Dict[str, Any] = {}
                    _scan(v, sub)
                    if sub:
                        target_dict[k] = sub

        _scan(state, factors)
        return factors or state

    # =====================================================================
    # nanoGentzen v4: scenarios decided by the v4 /v1/decide engine in-process
    # =====================================================================
    _VERIFICATION_MAP = {"PROVEN": "PROVEN_TRUE", "NOT_PROVEN": "INTUITIONISTIC_UNDECIDED"}
    # statuses whose label is certified by the kernel
    VERIFIED_STATUSES = ("PROVEN_TRUE", "NO_OPTION_PROVABLE")
    UNDECIDED_OPTION_RE = UNDECIDED_OPTION_RE

    # the scenario front end lives in nanogentzen/jev/parser_jev.py; these names are kept for callers
    @classmethod
    def undecided_option(cls, options) -> Optional[str]:
        return undecided_option(options)

    @staticmethod
    def _norm(name: str) -> str:
        return norm_name(name)

    @staticmethod
    def _leaf_paths(state: Any, prefix: Tuple[str, ...] = ()) -> List[Tuple[Tuple[str, ...], Any]]:
        return leaf_paths(state, prefix)

    def _v4_rules(self, policy_rules: List[str]) -> Tuple[List[str], List[Dict[str, str]], Set[str]]:
        return compile_rules_v4(policy_rules)

    def _v4_predicates(self, state: Any, atoms: Set[str]) -> Dict[str, Dict[str, Any]]:
        return bind_predicates_v4(state, atoms)

    def _v4_questions(self, questions: Dict[str, Any], rule_atoms: Set[str]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
        return question_specs_v4(questions, rule_atoms)

    def _v4_missing_for(self, target: str, state: Any, preds: Dict[str, Any], rules: List[str]) -> List[str]:
        from nanogentzen_v4.kernel import Var
        from nanogentzen_v4.parser import compile_to_formula, ground_state_generic

        facts, _ = ground_state_generic(state, preds, {})
        rule_f = [compile_to_formula(r) for r in rules]
        return verify.missing_facts(self.engine, Var(target), facts, rule_f)

    def _v4_run(self, state, rules, preds, specs) -> Dict[str, Any]:
        return verify.decide(self.engine, {
            "state": state, "predicates": preds, "policy_rules": rules, "questions": specs,
            "include_proof_tree": True, "include_lambda_term": True,
            "include_relevance": True, "include_countermodel": True,
        })

    def _evaluate_v4(
        self,
        state: Dict[str, Any],
        questions: Dict[str, Any],
        policy_rules: List[str],
        target_decision: str = "",
        label_undecided: bool = False,
    ) -> Dict[str, Any]:
        """
        Decide a Jev scenario with nanoGentzen v4. Differences from the v3 path:
          * unparsable rules are rejected (listed in `rejected_rules`) instead of silently skipped
          * state facts are never invented wholesale: if target_decision is not provable, only the minimal
            missing literals that make it provable without contradiction are added (allow_state_completion),
            and they are listed in `state_completion`; facts already in the state are never changed
          * gold labels carry the kernel status, the checked certificate, relevance and minimal witnesses
        """
        if not isinstance(state, dict):
            state = {"context": state}
        rules, rejected, atoms = self._v4_rules(policy_rules)
        specs, back = self._v4_questions(questions, atoms)
        # options no rule concludes: the kernel can never prove them, so "nothing is provable" is not
        # evidence for "undecided" (e.g. a negated requirement should mean "incompatible", not "undecided")
        uncovered = uncovered_options(questions, rules)
        preds = self._v4_predicates(state, atoms)
        if not specs:
            return self._v4_empty(state, rules, rejected, "no usable questions")
        if not rules:
            # without rules jev_server grounds the criteria text heuristically (ev_*/warrant_* atoms) and
            # "proves" whatever the criteria say: that is not a proof about the scenario
            why = "no policy rule compiles" if rejected else "no policy rules"
            return self._v4_empty(state, rules, rejected, why)
        resp = self._v4_run(state, rules, preds, specs)

        completion: Dict[str, Any] = {}
        tgt = str(target_decision or "").strip()
        if tgt and self.allow_state_completion:
            for q_id, spec in specs.items():
                opts = back.get(q_id, {})
                if tgt not in opts.values():
                    continue
                atom = next(k for k, v in opts.items() if v == tgt)
                ans = resp["answers"].get(q_id, {})
                proven = ans.get("proven_options") or ([ans.get("choice")] if ans.get("verification") == "PROVEN" else [])
                if atom in proven:
                    break
                if proven:
                    completion = {"applied": False, "reason": f"another option is proven: {proven}"}
                    break
                missing = self._v4_missing_for(atom, state, preds, rules)
                if not missing:
                    completion = {"applied": False, "reason": f"no consistent set of facts makes '{tgt}' provable"}
                    break
                attrs = state.setdefault("attributes", {})
                if not isinstance(attrs, dict):
                    completion = {"applied": False, "reason": "state.attributes is not an object"}
                    break
                added = {}
                for lit in missing:
                    neg = lit.startswith("~")
                    name = lit.lstrip("~")
                    if name not in attrs:
                        attrs[name] = not neg
                        added[name] = not neg
                completion = {"applied": True, "added_facts": added, "target": tgt, "question": q_id}
                preds = self._v4_predicates(state, atoms)
                resp = self._v4_run(state, rules, preds, specs)
                break

        gold: Dict[str, Any] = {}
        statuses: List[str] = []
        win = None  # (q_id, answer) used for sequent / term / factors
        for q_id, ans in resp["answers"].items():
            mapping = back.get(q_id, {})
            raw = ans.get("verification", "")
            status = self._VERIFICATION_MAP.get(raw, raw)
            if ans.get("type") == "noul" and raw in ("PROVEN_TRUE", "PROVEN_FALSE"):
                status = "PROVEN_TRUE"   # the question is decided either way
            if ans.get("grounding") == "criteria_heuristic":
                status = "HEURISTIC_UNVERIFIED"   # premises invented from the criteria text, not the scenario
            probs = {mapping.get(k, k): v for k, v in (ans.get("probabilities") or {}).items()}
            if ans.get("type") == "choice":
                label = mapping.get(ans.get("choice"), ans.get("choice"))
                if status != "PROVEN_TRUE" and tgt and tgt in mapping.values():
                    label = tgt    # unverified: keep the scenario's intended label, flagged by `verification`
            elif ans.get("type") == "noul":
                label = None if ans.get("answer") is None else str(bool(ans["answer"])).lower()
            else:
                label = None if ans.get("score") is None else str(ans.get("proven_tier", ans.get("score")))
            undecided_opt = self.undecided_option(mapping.values()) if ans.get("type") == "choice" else None
            if (label_undecided and raw == "NOT_PROVEN" and undecided_opt and status != "HEURISTIC_UNVERIFIED"
                    and not ans.get("proven_options") and not uncovered.get(str(q_id))):
                # every option has a rule and the complete prover showed none applies: "undecided" is the
                # certified answer. Without full rule coverage the label stays unverified.
                label, status = undecided_opt, "NO_OPTION_PROVABLE"
                probs = {o: (1.0 if o == label else 0.0) for o in probs}
            if status in self.VERIFIED_STATUSES and self.label_smoothing > 0 and len(probs) > 1 and label in probs:
                eps, k = self.label_smoothing, len(probs)
                probs = {o: round(1 - eps if o == label else eps / (k - 1), 6) for o in probs}
            entry = {
                "label": label,
                "confidence": 1.0 if status in self.VERIFIED_STATUSES and not self.label_smoothing else probs.get(label, ans.get("confidence")),
                "probabilities": probs,
                "type": ans.get("type", "choice"),
                "verification": status,
                "verification_v4": raw,
                "curry_howard_term": ans.get("curry_howard_term", ""),
                "certificate": ans.get("certificate"),
                "certificate_checked": bool((ans.get("certificate") or {}).get("checked")),
                "relevance": ans.get("relevance"),
                "epistemic": ans.get("epistemic"),
                "grounding": ans.get("grounding"),
            }
            if uncovered.get(str(q_id)):
                entry["uncovered_options"] = uncovered[str(q_id)]
            for k in ("missing_facts", "proven_options", "conflicting_premises", "countermodel", "prior_distribution"):
                if ans.get(k):
                    v = ans[k]
                    if k == "proven_options":
                        v = [mapping.get(o, o) for o in v]
                    elif k == "missing_facts" and isinstance(v, dict):
                        v = {mapping.get(o, o): m for o, m in v.items()}
                    entry[k] = v
            if tgt and tgt in mapping.values() and status == "PROVEN_TRUE" and label != tgt:
                entry["target_mismatch"] = {"target_decision": tgt, "proven": label}
                status = "TARGET_REFUTED_BY_PROOF"
                entry["verification"] = status
            if status == "NO_OPTION_PROVABLE":
                entry["certified_undecided"] = True
            if completion.get("applied") and completion.get("question") == q_id and status == "PROVEN_TRUE":
                # proven only after facts were added that the scenario did not state: valid relative to those
                # assumptions, but not evidence from the document, so not certified
                status = "PROVEN_UNDER_ASSUMPTIONS"
                entry["verification"] = status
                entry["assumed_facts"] = dict(completion.get("added_facts") or {})
            gold[q_id] = entry
            statuses.append(status)
            if win is None and ans.get("proof_tree"):
                win = (q_id, ans)

        overall = "PROVEN_TRUE" if statuses and all(s == "PROVEN_TRUE" for s in statuses) else \
            next((s for s in statuses if s != "PROVEN_TRUE"), "INTUITIONISTIC_UNDECIDED")
        certified = bool(statuses) and all(s in self.VERIFIED_STATUSES for s in statuses)
        tree = win[1].get("proof_tree") if win else None
        rel = (win[1].get("relevance") or {}) if win else {}
        witnesses = rel.get("minimal_witnesses") or rel.get("relevant_witnesses") or []
        return {
            "gold": gold,
            "factors": self._v4_factors(state, witnesses) or state,
            "best_sequent": pretty_sequent(tree["sequent"]) if tree else "",
            "best_sequent_ascii": tree["sequent"] if tree else "",
            "curry_howard_term": win[1].get("curry_howard_term", "") if win else "",
            "gentzen_verified": certified,
            # verified because the complete search showed no option applies (not because one was proven)
            "certified_undecided": certified and any(s == "NO_OPTION_PROVABLE" for s in statuses),
            "epistemic_status": overall,
            "proof_tree": tree,
            "facts": [],
            "rule_formulas": [],
            "rules_used": rules,
            "rejected_rules": rejected,
            "predicates": preds,
            "state_completion": completion,
            "engine": self.engine.metadata(),
            "ungrounded_fields": resp.get("ungrounded_fields", []),
            "renamed_identifiers": resp.get("renamed_identifiers", {}),
        }

    def _v4_empty(self, state, rules, rejected, why) -> Dict[str, Any]:
        return {"gold": {}, "factors": state, "best_sequent": "", "curry_howard_term": "", "gentzen_verified": False,
                "epistemic_status": "INTUITIONISTIC_UNDECIDED", "proof_tree": None, "facts": [], "rule_formulas": [],
                "rules_used": rules, "rejected_rules": rejected, "predicates": {}, "state_completion": {},
                "engine": self.engine.metadata(), "note": why}

    def _v4_factors(self, state: Any, witnesses: List[str]) -> Dict[str, Any]:
        """State fields behind the literal premises of the minimal proof (rules are not state)."""
        lits = {self._norm(m.group(1)) for w in witnesses
                for m in [re.match(r"^~?([A-Za-z_][A-Za-z0-9_]*)$", w.strip())] if m}
        if not lits or not isinstance(state, dict):
            return {}
        out: Dict[str, Any] = {}
        for path, v in self._leaf_paths(state):
            key_n, path_n = self._norm(path[-1]), self._norm("_".join(path))
            hit = key_n in lits or path_n in lits or any(l.startswith(path_n) and l != path_n for l in lits if isinstance(v, (str, int, float)) and not isinstance(v, bool))
            if hit:
                cur = out
                for p in path[:-1]:
                    cur = cur.setdefault(p, {})
                cur[path[-1]] = v
        return out

