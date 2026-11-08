"""
architect_cot.py (nanoGentzen-v3 Logic Tree Generator for CoT)
Specialized for natural deduction verbalization and ClearCoT dataset creation.
Builds the best formal Gentzen proof tree (or intuitionistic derivation)
which serves as the structural logical backbone on which ClearCoT is built.

Key capabilities:
  1. Relational chain derivation for multi-hop Knowledge Graph paths:
     E0, (E0 -> E1), (E1 -> E2), ..., (E_{k-1} -> Ek) |- Ek
  2. Synthesizes paragraph deductive bridges:
     P1, ..., Pk, ((P1 & ... & Pk) -> C) |- C
  3. Dual Gentzen Kernel Verification (ProofNode AST + dict tree validation)
  4. Saul Kripke 3-class epistemic valuation (PROVEN_TRUE, INTUITIONISTIC_UNDECIDED, PROVEN_FALSE)
  5. Intuitionistic Relevance Logic (IR) witness isolation (causal vs distractor premises)
  6. Curry-Howard typed lambda-term serialization
  7. Dyckhoff G4ip contraction-free proof search with Andreoli invertible reduction
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

from ..kernel import (
    Sequent,
    ProofNode,
    verify_proof_tree,
    verify_proof_tree_strict,
    Formula,
    Atom,
    Var,
    And,
    Imp,
    extract_relevance_witnesses,
    proof_tree_to_curry_howard,
    extract_kripke_countermodel,
)
from ..logic_syntax import parse_sequent
from .parser_cot import (
    text_to_sequent,
    path_to_sequent,
    path_logic_report,
    path_missing_links,
    clean_atom_name,
)
from ..engine import NanoGentzenEngine
from ..engine_v4 import is_v4_engine, pretty_sequent
from .. import verify
from ..search import NeuralProofSearch, dict_to_proof_node, proof_node_to_dict

logger = logging.getLogger(__name__)

RULE_DESCRIPTIONS = {
    "Axiom": "Axiom (Direct Premise / Ground Truth)",
    "AXIOM": "Axiom (Direct Premise / Ground Truth)",
    "Imp_R": "Conditional Proof (Assume antecedent to derive consequent)",
    "R_IMP": "Conditional Proof (Assume antecedent to derive consequent)",
    "Imp_L": "Modus Ponens (Applying established implication to antecedent)",
    "L_IMP": "Modus Ponens (Applying established implication to antecedent)",
    "L_IMP_ATOM": "Dyckhoff Modus Ponens (Direct atomic detachment)",
    "L_IMP_AND": "Curried Implication Decomposition",
    "L_IMP_OR": "Disjunctive Implication Split",
    "L_IMP_IMP": "Higher-Order Implication Reduction",
    "And_R": "Conjunction Introduction (Both subgoals proven)",
    "R_AND": "Conjunction Introduction (Both subgoals proven)",
    "And_L1": "Conjunction Elimination (Extracting first component)",
    "And_L2": "Conjunction Elimination (Extracting second component)",
    "L_AND": "Conjunction Elimination (Extracting components)",
    "Or_R1": "Disjunction Introduction (Establishing first branch)",
    "R_OR_1": "Disjunction Introduction (Establishing first branch)",
    "Or_R2": "Disjunction Introduction (Establishing second branch)",
    "R_OR_2": "Disjunction Introduction (Establishing second branch)",
    "Or_L": "Proof by Cases (Verifying under all alternatives)",
    "L_OR": "Proof by Cases (Verifying under all alternatives)",
    "Not_R": "Proof by Contradiction / Negation Introduction",
    "R_NOT": "Proof by Contradiction / Negation Introduction",
    "Not_L": "Contradiction Detachment",
    "L_NOT": "Contradiction Detachment",
    # Dyckhoff G4ip moves used by nanoGentzen v4
    "L_IMP_BOT": "Vacuous Implication Removal (from falsum)",
    "L_NOT_BOT": "Trivial Negation Removal",
    "L_IMP_NOT": "Negated-Antecedent Implication Reduction",
    "L_NOT_AND": "Negated Conjunction Decomposition",
    "L_NOT_OR": "Negated Disjunction Split (both disjuncts refuted)",
    "L_NOT_IMP": "Negated Implication Reduction",
    "L_NOT_NOT": "Double Negation Reduction",
}


def _rule_base(label: str) -> str:
    """'L_IMP_ATOM_3' -> 'L_IMP_ATOM'; right rules carry no pivot index ('R_OR_1' stays)."""
    if label.startswith("L_") and "_" in label and label.rsplit("_", 1)[-1].isdigit():
        return label.rsplit("_", 1)[0]
    return label


def _proves_sequent(root: Any, seq: Any) -> bool:
    """True if proof `root` concludes exactly `seq` (same antecedent set, same succedent)."""
    concl = getattr(root, "conclusion", None)
    if concl is None:
        return False
    try:
        return (set(concl.antecedents) == set(seq.antecedents)
                and concl.succedent == seq.succedent)
    except Exception:
        return False


class CotLogicalArchitect:
    """
    Dedicated logical architect for ClearCoT reasoning chains.
    Synthesizes, searches, verifies, and linearizes formal Gentzen LI proof trees
    to serve as the verified logical backbone for LLM Chain-of-Thought verbalization.
    """

    def __init__(
        self,
        engine: Optional[NanoGentzenEngine] = None,
        max_depth: int = 8,
        max_nodes: int = 500,
        beam_width: int = 6,
    ):
        self.engine = engine or NanoGentzenEngine()
        self.searcher = NeuralProofSearch(
            engine=self.engine,
            max_depth=max_depth,
            max_nodes=max_nodes,
            prune_threshold=0.05,
            beam_width=beam_width,
            use_invertible_fastpath=True,
        )

    def update_engine(self, engine: NanoGentzenEngine):
        """Synchronizes engine and searcher instances across Streamlit session state updates."""
        self.engine = engine
        self.searcher.engine = engine
        self.searcher.model = getattr(engine, "model", None)
        self.searcher.tokenizer = getattr(engine, "tokenizer", None)
        if hasattr(engine, "target_device"):
            self.searcher.device = engine.target_device

    def generate_best_logic_tree(
        self,
        premises: List[str],
        conclusion: str,
        symbolic_sequent: Optional[str] = None,
        path_nodes: Optional[List[str]] = None,
        path_repr: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Logic tree for a sample, labelled with what the proof is actually about (`logic_basis`):

          symbolic             a sequent given explicitly
          conditional_premises premises contain their own conditionals; the proof uses them
          kg_entailment_chain  every KG hop is a forward inference (implies, requires, excludes ...)
          assumed_bridge       paragraph seed: the rule (P1 & ... & Pk) => C is ADDED, so any claim follows
          kg_relational        KG path walked against an edge or through a non-inferential relation
          kg_unlabelled        KG path without relation labels

        Only the first three can be `is_verified`. For assumed_bridge the (valid) proof is kept for the
        CoT guide, but `is_verified` is False and `epistemic_status` is BRIDGE_ASSUMED, because the proof
        shows nothing about the text. kg_relational / kg_unlabelled paths are decided honestly and are
        normally INTUITIONISTIC_UNDECIDED.
        """
        basis, path_report = self._logic_basis(premises, conclusion, symbolic_sequent, path_nodes, path_repr)
        res = self._generate_best_logic_tree_core(premises, conclusion, symbolic_sequent, path_nodes, path_repr,
                                                  allow_chain_synthesis=(basis == "kg_entailment_chain"))
        res["logic_basis"] = basis
        if path_report is not None:
            res["path_edges"] = path_report.get("edges", [])
            res["path_broken_at"] = path_report.get("broken_at")
            # repair signal for relational paths: which hop needs which rule (missing_facts covers base
            # literals only, so it is empty for a path whose hops carry no rules at all)
            res["missing_links"] = path_missing_links(path_report)
        res["proof_valid"] = bool(res.get("is_verified"))
        if res.get("is_verified") and basis in ("assumed_bridge", "kg_unlabelled"):
            res["is_verified"] = False
            res["epistemic_status"] = "BRIDGE_ASSUMED" if basis == "assumed_bridge" else "RELATIONS_UNLABELLED"
            note = ("**Logic basis:** assumed bridge: the rule (premises ⇒ conclusion) was added, so the proof "
                    "does not show that the text supports the conclusion."
                    if basis == "assumed_bridge" else
                    "**Logic basis:** KG path without relation labels: hops cannot be read as inferences.")
            res["logic_tree_text"] = res.get("logic_tree_text", "").replace(
                "Formally Verified", "Valid proof of an assumed sequent (not counted as verified)", 1) + "\n\n" + note
        elif basis == "kg_relational" and path_report is not None and path_report.get("broken_at") is not None:
            e = path_report["edges"][path_report["broken_at"]]
            why = ("about a degenerate concept (not a proposition)" if e.get("degenerate")
                   else "walked against the edge direction" if e.get("direction") != "forward"
                   else f"relation '{e.get('predicate')}' licenses no inference" if e.get("logic") == "none"
                   else "nothing follows from an excluded concept")
            res["logic_tree_text"] = res.get("logic_tree_text", "") + (
                f"\n\n**Logic basis:** relational path; hop {path_report['broken_at'] + 1} "
                f"({e.get('u')} → {e.get('v')}) is {why}.")
        return res

    def _logic_basis(self, premises, conclusion, symbolic_sequent, path_nodes, path_repr):
        if symbolic_sequent:
            return "symbolic", None
        if path_nodes and len(path_nodes) >= 2:
            rep = path_logic_report(path_nodes, path_repr)
            return rep["logic_basis"], rep
        with_bridge = text_to_sequent(premises, conclusion, include_deductive_bridge=True)
        without = text_to_sequent(premises, conclusion, include_deductive_bridge=False)
        return ("assumed_bridge" if with_bridge.to_str() != without.to_str() else "conditional_premises"), None

    def _generate_best_logic_tree_core(
        self,
        premises: List[str],
        conclusion: str,
        symbolic_sequent: Optional[str] = None,
        path_nodes: Optional[List[str]] = None,
        path_repr: Optional[str] = None,
        allow_chain_synthesis: bool = False,
    ) -> Dict[str, Any]:
        """
        Build the best logical tree for the given premises and conclusion.
        Supports multi-hop KG paths and paragraph deductive seeds.
        nanoGentzen searches for the soundest, most compact derivation tree,
        strictly verified by the Gentzen LI kernel.
        """
        if symbolic_sequent:
            try:
                seq = parse_sequent(symbolic_sequent)
            except Exception as e:
                logger.warning(f"Failed to parse symbolic sequent '{symbolic_sequent}': {e}")
                seq = text_to_sequent(premises, conclusion, include_deductive_bridge=True)
        elif path_nodes and len(path_nodes) >= 2:
            seq = path_to_sequent(path_nodes, path_repr=path_repr)
        else:
            seq = text_to_sequent(premises, conclusion, include_deductive_bridge=True)

        # nanoGentzen v4: the kernel decides and certifies; no constructive synthesizers needed
        if is_v4_engine(self.engine):
            return self._generate_v4(seq, premises, conclusion, symbolic_sequent, path_nodes)

        # 1. Run neural-guided proof search (Andreoli invertible reduction + Dyckhoff G4ip)
        proof_tree_dict, countermodel = self.searcher.prove(seq, extract_countermodel_on_failure=True)
        proof_root = dict_to_proof_node(proof_tree_dict) if proof_tree_dict else None

        # 2. Dual verification check on search result
        is_verified = False
        if proof_root is not None and verify_proof_tree(proof_root):
            is_verified = True
        elif proof_tree_dict is not None and verify_proof_tree_strict(proof_tree_dict):
            is_verified = True
            if proof_root is None:
                proof_root = dict_to_proof_node(proof_tree_dict)

        # 3. If proof search did not find or verify a closed proof, apply constructive proof synthesizer
        if not is_verified and not symbolic_sequent:
            if path_nodes and len(path_nodes) >= 2 and allow_chain_synthesis:
                try:
                    atoms = [Atom(clean_atom_name(n)) for n in path_nodes]
                    synth_root = self._construct_chain_proof(atoms)
                    if synth_root is not None and not _proves_sequent(synth_root, seq):
                        synth_root = None   # a proof of some other sequent certifies nothing about this one
                    if synth_root is not None:
                        synth_dict = proof_node_to_dict(synth_root)
                        if verify_proof_tree(synth_root) or (synth_dict and verify_proof_tree_strict(synth_dict)):
                            proof_root = synth_root
                            proof_tree_dict = synth_dict
                            is_verified = True
                except Exception as e:
                    logger.debug(f"Chain constructive proof failed: {e}")

            if not is_verified and not (path_nodes and len(path_nodes) >= 2):
                try:
                    p_atoms = [Atom(clean_atom_name(p, prefix=f"P{i}")) for i, p in enumerate(premises, 1)]
                    c_atom = Atom(clean_atom_name(conclusion, prefix="C"))
                    synth_root = self._construct_deductive_proof(p_atoms, c_atom)
                    if synth_root is not None and not _proves_sequent(synth_root, seq):
                        synth_root = None   # it proves P1..Pk, bridge |- C with renamed atoms, not `seq`
                    if synth_root is not None:
                        synth_dict = proof_node_to_dict(synth_root)
                        if verify_proof_tree(synth_root) or (synth_dict and verify_proof_tree_strict(synth_dict)):
                            proof_root = synth_root
                            proof_tree_dict = synth_dict
                            is_verified = True
                except Exception as e:
                    logger.debug(f"Deductive constructive proof failed: {e}")

        # 4. Final verification gate
        if not is_verified:
            if proof_root is not None and verify_proof_tree(proof_root):
                is_verified = True
            elif proof_tree_dict is not None and verify_proof_tree_strict(proof_tree_dict):
                is_verified = True
                if proof_root is None:
                    proof_root = dict_to_proof_node(proof_tree_dict)

        # 5. Build verified tree or fallback to structured heuristic guide
        if is_verified and proof_root is not None:
            steps: List[Dict[str, Any]] = []
            self._linearize_tree(proof_root, steps, step_counter=[1])

            # Extract v3 Curry-Howard term & Relevance Logic witnesses
            proof_dict_effective = proof_tree_dict or proof_node_to_dict(proof_root) or proof_root.to_dict()
            curry_howard_term = proof_tree_to_curry_howard(proof_dict_effective) if proof_dict_effective else "apply(axiom)"
            relevance = extract_relevance_witnesses(seq, proof_dict_effective) if proof_dict_effective else {
                "relevant_witnesses": [f.to_str() for f in seq.gamma],
                "unused_premises": [],
                "is_strictly_relevant": True,
                "relevance_ratio": 1.0,
            }

            formatted_tree = self._format_logic_tree(
                root_seq=seq,
                steps=steps,
                root=proof_root,
                ch_term=curry_howard_term,
                relevance=relevance
            )

            depth = self._compute_tree_depth(proof_root)
            node_cnt = self._count_nodes(proof_root)

            # Build dual-compatible proof_tree dictionary for GUI and serializer
            gui_compatible_tree = proof_root.to_dict()
            if proof_tree_dict and "branches" in proof_tree_dict:
                gui_compatible_tree["branches"] = proof_tree_dict["branches"]
                gui_compatible_tree["sequent"] = proof_tree_dict.get("sequent", str(seq))

            return {
                "is_verified": True,
                "epistemic_status": "PROVEN_TRUE",
                "sequent": str(seq),
                "proof_tree": gui_compatible_tree,
                "curry_howard_term": curry_howard_term,
                "relevance": relevance,
                "relevance_ratio": relevance.get("relevance_ratio", 1.0),
                "is_strictly_relevant": relevance.get("is_strictly_relevant", True),
                "countermodel": None,
                "kripke_countermodel": None,
                "logic_tree_text": formatted_tree,
                "derivation_steps": steps,
                "provability_score": 1.0,
                "depth": depth,
                "node_count": node_cnt,
            }
        else:
            # Fallback heuristic tree when formal closed proof is not established
            steps = [
                {
                    "step_id": i,
                    "rule": "Premise",
                    "rule_desc": "Given Fact",
                    "conclusion": p,
                    "score": 1.0
                }
                for i, p in enumerate(premises, 1)
            ]
            steps.append({
                "step_id": len(premises) + 1,
                "rule": "Deduction",
                "rule_desc": "Synthesized Conclusion",
                "conclusion": conclusion,
                "score": 0.85
            })

            formatted_tree = self._format_heuristic_tree(premises, conclusion, countermodel)
            eval_res = self.engine.evaluate_sequent(seq)
            diag = countermodel.get("diagnosis", "INTUITIONISTIC_UNDECIDED") if countermodel else "INTUITIONISTIC_UNDECIDED"

            return {
                "is_verified": False,
                "epistemic_status": diag,
                "sequent": str(seq),
                "proof_tree": None,
                "curry_howard_term": "",
                "relevance": {
                    "relevant_witnesses": [],
                    "unused_premises": [f.to_str() for f in seq.gamma],
                    "is_strictly_relevant": False,
                    "relevance_ratio": 0.0,
                },
                "relevance_ratio": 0.0,
                "is_strictly_relevant": False,
                "countermodel": countermodel,
                "kripke_countermodel": countermodel,
                "logic_tree_text": formatted_tree,
                "derivation_steps": steps,
                "provability_score": eval_res.get("provability_value", 0.0),
                "depth": 2,
                "node_count": len(steps),
            }

    # ------------------------------------------------------------------ nanoGentzen v4
    def _generate_v4(
        self,
        seq: Sequent,
        premises: List[str],
        conclusion: str,
        symbolic_sequent: Optional[str],
        path_nodes: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        v4 path. The sequent is decided by the G4ip kernel (neural search ordered by the v4 network, complete
        prover as backstop), every proof is strictly verified and certified by an independent type checker.
        Unlike v3, nothing is "synthesized" for a sequent that does not hold: the status says what holds.
        """
        eng = self.engine
        res = verify.prove_sequent(eng, seq)
        sequent_source = "symbolic" if symbolic_sequent else ("kg_path" if path_nodes and len(path_nodes) >= 2 else "text_premises")

        status = res["status"]
        tree = res.get("proof_tree")
        relevance = dict(res.get("relevance") or {})
        s1 = res.get("system1") or {}
        common = {
            "sequent": pretty_sequent(res["sequent"]),
            "sequent_ascii": res["sequent"],
            "sequent_source": sequent_source,
            "engine": eng.metadata(),
            "system1_epistemic": s1.get("epistemic"),
            "search": res.get("search"),
            "certificate": res.get("certificate"),
            "certificate_checked": bool(res.get("certificate_checked")),
            "minimal_witnesses": relevance.get("minimal_witnesses", []),
            "missing_facts": res.get("missing_facts", []),
            "conflicting_premises": res.get("conflicting_premises", []),
            "refutation": res.get("refutation"),
        }

        if status == "PROVEN_TRUE" and tree:
            steps = self._linearize_dict_tree(tree)
            term = res.get("curry_howard_term", "")
            text = self._format_logic_tree_v4(res, steps, relevance, term)
            return {
                **common,
                "is_verified": True,
                "epistemic_status": "PROVEN_TRUE",
                "proof_tree": tree,
                "curry_howard_term": term,
                "relevance": relevance,
                "relevance_ratio": relevance.get("relevance_ratio", 1.0),
                "is_strictly_relevant": relevance.get("is_strictly_relevant", True),
                "countermodel": None,
                "kripke_countermodel": None,
                "logic_tree_text": text,
                "derivation_steps": steps,
                "provability_score": 1.0,
                "depth": eng.tree_depth(tree),
                "node_count": eng.tree_size(tree),
            }

        # not proven: PROVEN_FALSE / INTUITIONISTIC_UNDECIDED / INCONSISTENT_PREMISES / SEARCH_BUDGET_EXHAUSTED
        diag = {"diagnosis": status, "missing_facts": res.get("missing_facts", [])}
        if status == "PROVEN_FALSE":
            diag["refutation_reason"] = "The negation of the conclusion is provable from the premises."
        elif status == "INCONSISTENT_PREMISES":
            diag["refutation_reason"] = ("The premises contradict each other: "
                                         + ", ".join(res.get("conflicting_premises", [])))
        cm = res.get("countermodel")
        if cm:
            diag.update(cm)
        steps = [{"step_id": i, "rule": "Premise", "rule_desc": "Given Fact", "conclusion": p, "score": 1.0}
                 for i, p in enumerate(premises, 1)]
        steps.append({"step_id": len(premises) + 1, "rule": "Deduction", "rule_desc": "Synthesized Conclusion",
                      "conclusion": conclusion, "score": round(float(s1.get("confidence", 0.0) or 0.0), 3)})
        return {
            **common,
            "is_verified": False,
            "epistemic_status": status,
            "proof_tree": None,
            "curry_howard_term": "",
            "relevance": relevance,
            "relevance_ratio": 0.0,
            "is_strictly_relevant": False,
            "countermodel": diag,
            "kripke_countermodel": diag,
            "logic_tree_text": self._format_heuristic_tree(premises, conclusion, diag).replace(
                "[nanoGentzen-v3 Logic Tree", "[nanoGentzen-v4 Logic Tree"),
            "derivation_steps": steps,
            "provability_score": float(s1.get("confidence", 0.0) or 0.0),
            "depth": 2,
            "node_count": len(steps),
        }

    @staticmethod
    def _linearize_dict_tree(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Post-order steps (subgoals before the goals they establish) from a v4 proof-tree dict."""
        steps: List[Dict[str, Any]] = []

        def walk(node: Dict[str, Any]):
            for b in node.get("branches", []) or []:
                walk(b)
            label = node.get("rule", "")
            base = _rule_base(label)
            steps.append({
                "step_id": len(steps) + 1,
                "rule": label,
                "rule_desc": RULE_DESCRIPTIONS.get(base, base),
                "conclusion": pretty_sequent(node.get("sequent", "")),
                "score": 1.0,
            })

        walk(tree)
        return steps

    def _format_logic_tree_v4(self, res: Dict[str, Any], steps: List[Dict[str, Any]],
                              relevance: Dict[str, Any], term: str) -> str:
        tree = res.get("proof_tree")
        n_prem = len(relevance.get("relevant_witnesses", [])) + len(relevance.get("unused_premises", []))
        cert_line = ("checked by the independent type checker" if res.get("certificate_checked")
                     else f"NOT checked ({res.get('certificate_error') or 'no certificate'}); proof tree verified")
        prover = ((res.get("search") or {}).get("positive") or {}).get("prover", "")
        lines = [
            "### [nanoGentzen-v4 Logic Tree]",
            f"**Root Goal:** `{pretty_sequent(res['sequent'])}`",
            "**Status:** Formally Verified (Intuitionistic G4ip, kernel-decided: PROVEN_TRUE)",
            f"**Certificate:** {cert_line}",
        ]
        if term:
            lines.append(f"**Curry-Howard λ-term:** `{term}`")
        if n_prem:
            lines.append(f"**Relevance Logic:** `{len(relevance.get('relevant_witnesses', []))}/{n_prem} premises used "
                         f"({int(relevance.get('relevance_ratio', 1.0) * 100)}% relevant)`")
        if relevance.get("minimal_witnesses"):
            lines.append(f"**Minimal Witnesses:** `{', '.join(relevance['minimal_witnesses'])}`")
        lines.extend([
            f"**Depth:** {self.engine.tree_depth(tree)} | **Nodes:** {self.engine.tree_size(tree)} | **Prover:** {prover}",
            "",
            "**Deductive Steps to follow in CoT:**",
        ])
        for s in steps:
            lines.append(f"{s['step_id']}. [{s['rule_desc']}] ➔ Established: `{s['conclusion']}`")
        if relevance.get("unused_premises"):
            lines.extend(["", "**Unused Distractor Premises (Omit from CoT):**"])
            lines.extend(f"- `{u}`" for u in relevance["unused_premises"])
        return "\n".join(lines)

    def _construct_chain_proof(self, nodes: List[Atom]) -> Optional[ProofNode]:
        """
        Construct a verified closed derivation tree for a multi-hop path:
        E0, (E0 -> E1), (E1 -> E2), ..., (E_{k-1} -> Ek) |- Ek
        """
        if not nodes:
            return None
        if len(nodes) == 1:
            seq = Sequent((nodes[0],), (nodes[0],))
            return ProofNode(rule="Axiom", conclusion=seq, provability_score=1.0)

        e0 = nodes[0]
        ek = nodes[-1]
        imps = [Imp(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        full_gamma = tuple([e0] + imps)

        def prove_subgoal(current_gamma: Tuple[Formula, ...], target: Atom, remaining_imps: List[Imp]) -> ProofNode:
            if target in current_gamma:
                return ProofNode(rule="Axiom", conclusion=Sequent(current_gamma, (target,)), provability_score=1.0)
            for idx, imp in enumerate(current_gamma):
                if isinstance(imp, Imp) and imp.left in current_gamma and imp in remaining_imps:
                    pivot_idx = idx
                    p1 = ProofNode(rule="Axiom", conclusion=Sequent(current_gamma, (imp.left,)), provability_score=1.0)
                    gamma_rest = tuple([g for i, g in enumerate(current_gamma) if i != pivot_idx] + [imp.right])
                    rem_rest = [item for item in remaining_imps if item != imp]
                    p2 = prove_subgoal(gamma_rest, target, rem_rest)
                    return ProofNode(
                        rule="Imp_L",
                        conclusion=Sequent(current_gamma, (target,)),
                        premises=[p1.conclusion, p2.conclusion],
                        children=[p1, p2],
                        pivot=pivot_idx,
                        provability_score=1.0
                    )
            raise ValueError(f"Cannot resolve subgoal for {target}")

        try:
            return prove_subgoal(full_gamma, ek, imps)
        except Exception as e:
            logger.debug(f"Chain proof construction failed: {e}")
            return None

    def _construct_deductive_proof(self, premises: List[Atom], conclusion: Atom) -> Optional[ProofNode]:
        """
        Construct a verified closed derivation tree for deductive synthesis:
        P1, ..., Pk, ((... (P1 & P2) & ...) -> C) |- C
        """
        if not premises:
            seq = Sequent((conclusion,), (conclusion,))
            return ProofNode(rule="Axiom", conclusion=seq, provability_score=1.0)

        if len(premises) == 1:
            p1 = premises[0]
            imp = Imp(p1, conclusion)
            root_seq = Sequent((p1, imp), (conclusion,))
            c1 = ProofNode(rule="Axiom", conclusion=Sequent((p1, imp), (p1,)), provability_score=1.0)
            c2 = ProofNode(rule="Axiom", conclusion=Sequent((p1, conclusion), (conclusion,)), provability_score=1.0)
            return ProofNode(
                rule="Imp_L",
                conclusion=root_seq,
                premises=[c1.conclusion, c2.conclusion],
                children=[c1, c2],
                pivot=1,
                provability_score=1.0
            )

        conj = premises[0]
        for p in premises[1:]:
            conj = And(conj, p)

        bridge_imp = Imp(conj, conclusion)
        full_gamma = tuple(premises + [bridge_imp])
        pivot_idx = len(premises)
        root_seq = Sequent(full_gamma, (conclusion,))

        def prove_conj(target_form: Formula) -> ProofNode:
            if isinstance(target_form, (Atom, Var)):
                return ProofNode(rule="Axiom", conclusion=Sequent(full_gamma, (target_form,)), provability_score=1.0)
            elif isinstance(target_form, And):
                left_node = prove_conj(target_form.left)
                right_node = prove_conj(target_form.right)
                return ProofNode(
                    rule="And_R",
                    conclusion=Sequent(full_gamma, (target_form,)),
                    premises=[left_node.conclusion, right_node.conclusion],
                    children=[left_node, right_node],
                    provability_score=1.0
                )
            raise ValueError(f"Unexpected formula: {target_form}")

        try:
            c1 = prove_conj(conj)
            c2 = ProofNode(
                rule="Axiom",
                conclusion=Sequent(tuple(premises + [conclusion]), (conclusion,)),
                provability_score=1.0
            )
            return ProofNode(
                rule="Imp_L",
                conclusion=root_seq,
                premises=[c1.conclusion, c2.conclusion],
                children=[c1, c2],
                pivot=pivot_idx,
                provability_score=1.0
            )
        except Exception as e:
            logger.debug(f"Deductive proof construction failed: {e}")
            return None

    def _linearize_tree(
        self,
        node: ProofNode,
        steps: List[Dict[str, Any]],
        step_counter: List[int]
    ):
        """Traverse tree post-order to establish subgoals before their parent conclusions."""
        for child in node.children:
            self._linearize_tree(child, steps, step_counter)

        curr_id = step_counter[0]
        step_counter[0] += 1

        rule_desc = RULE_DESCRIPTIONS.get(node.rule, node.rule)
        steps.append({
            "step_id": curr_id,
            "rule": node.rule,
            "rule_desc": rule_desc,
            "conclusion": str(node.conclusion),
            "score": round(node.provability_score, 3),
        })

    def _format_logic_tree(
        self,
        root_seq: Sequent,
        steps: List[Dict[str, Any]],
        root: ProofNode,
        ch_term: str = "",
        relevance: Optional[Dict[str, Any]] = None
    ) -> str:
        lines = [
            "### [nanoGentzen-v3 Logic Tree]",
            f"**Root Goal:** `{root_seq}`",
            "**Status:** Formally Verified (Intuitionistic LI - Saul Kripke PROVEN_TRUE)",
        ]

        if ch_term:
            lines.append(f"**Certificate:** `Curry-Howard λ-term: {ch_term}`")

        if relevance:
            witnesses = relevance.get("relevant_witnesses", [])
            ratio = relevance.get("relevance_ratio", 1.0)
            lines.append(
                f"**Relevance Logic:** `{len(witnesses)}/{len(root_seq.gamma)} causal witnesses ({int(ratio * 100)}% relevant)`"
            )

        lines.extend([
            f"**Depth:** {self._compute_tree_depth(root)} | **Nodes:** {self._count_nodes(root)}",
            "",
            "**Deductive Steps to follow in CoT:**",
        ])

        for s in steps:
            lines.append(
                f"{s['step_id']}. [{s['rule_desc']}] ➔ Established: `{s['conclusion']}`"
            )

        if relevance and relevance.get("unused_premises"):
            lines.extend([
                "",
                "**Unused Distractor Premises (Omit from CoT):**",
            ])
            for u in relevance["unused_premises"]:
                lines.append(f"- `{u}`")

        return "\n".join(lines)

    def _format_heuristic_tree(
        self,
        premises: List[str],
        conclusion: str,
        countermodel: Optional[Dict[str, Any]] = None
    ) -> str:
        lines = [
            "### [nanoGentzen-v3 Logic Tree - Open Derivation]",
            f"**Target Conclusion:** {conclusion}",
        ]

        if countermodel:
            diag = countermodel.get("diagnosis", "INTUITIONISTIC_UNDECIDED")
            lines.append(f"**Saul Kripke Status:** `{diag}`")
            if countermodel.get("missing_facts"):
                lines.append(f"**Missing Evidential Facts:** `{', '.join(countermodel['missing_facts'])}`")
            if countermodel.get("refutation_reason"):
                lines.append(f"**Refutation Reason:** {countermodel['refutation_reason']}")

        lines.extend([
            "",
            "**Structural Steps to follow in CoT:**",
        ])
        for idx, p in enumerate(premises, 1):
            lines.append(f"{idx}. [Premise Node] ➔ Base fact: {p}")
        lines.append(f"{len(premises) + 1}. [Deduction Node] ➔ Synthesize base facts to reach: {conclusion}")
        return "\n".join(lines)

    def _compute_tree_depth(self, node: ProofNode) -> int:
        if not node.children:
            return 1
        return 1 + max(self._compute_tree_depth(c) for c in node.children)

    def _count_nodes(self, node: ProofNode) -> int:
        return 1 + sum(self._count_nodes(c) for c in node.children)


# Backward-compatible alias
LogicalArchitect = CotLogicalArchitect


def proof_record(tree_res: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Compact, re-checkable record of a v4 logic-tree result for dataset provenance (None for v3 results).
    `certificate` can be re-checked offline with nanogentzen_v4.checker.check_certificate(certificate).
    """
    if not tree_res or ("engine" not in tree_res and "logic_basis" not in tree_res):
        return None
    return {
        "engine": tree_res.get("engine"),
        "epistemic_status": tree_res.get("epistemic_status"),
        "sequent": tree_res.get("sequent_ascii"),
        "sequent_source": tree_res.get("sequent_source"),
        "logic_basis": tree_res.get("logic_basis"),
        "proof_valid": tree_res.get("proof_valid"),
        "certificate_checked": tree_res.get("certificate_checked", False),
        "certificate": tree_res.get("certificate"),
        "minimal_witnesses": tree_res.get("minimal_witnesses", []),
        "missing_facts": tree_res.get("missing_facts", []),
        "missing_links": tree_res.get("missing_links", []),
        "conflicting_premises": tree_res.get("conflicting_premises", []),
    }


def with_proof_record(provenance: Optional[Dict[str, Any]], tree_res: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    prov = dict(provenance or {})
    rec = proof_record(tree_res)
    if rec:
        prov["nanogentzen"] = rec
    return prov

