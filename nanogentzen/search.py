"""
search.py (nanoGentzen-v3 Neural Proof Search Engine)
Features:
  1. Andreoli Asynchronous Invertible Reduction (Deterministic CPU Fast-Path)
  2. Span Pointer Synchronous Choice Selection (up to 32 antecedent pivots)
  3. Saul Kripke (1965) 3-Class Epistemic Pruning [P(False), P(Undecided), P(True)]
  4. Dyckhoff G4ip Contraction-Free Sequent Transitions (guaranteed termination)
  5. Open Semantic Tableau Countermodel Extraction on Proof Failure
  6. Bijective Alpha-Renaming (Canonical Symbol Compression)
  7. Dual ProofNode & Dict Tree Serialization for Dataset Studio compatibility
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .kernel import (
    And,
    Imp,
    Not,
    Or,
    RULES,
    G4IP_RULES,
    RULE_NAMES,
    Sequent,
    Var,
    Atom,
    ProofNode,
    apply_rule,
    canonicalize_sequent,
    extract_kripke_countermodel,
    restore_proof_tree,
    INVERTIBLE_LEFT_RULES,
    INVERTIBLE_RIGHT_RULES,
)
from .engine import NanoGentzenEngine, NanoGentzenONNXEngine

logger = logging.getLogger(__name__)


# =====================================================================
# 1. PROOF TREE CONVERTERS (DICT <-> PROOFNODE)
# =====================================================================

def dict_to_proof_node(node: Optional[dict]) -> Optional[ProofNode]:
    """Converts a dictionary proof tree into an object-oriented ProofNode."""
    if not node or not isinstance(node, dict):
        return None

    from .logic_syntax import parse_sequent
    rule_str = node.get("rule", "AXIOM")
    seq_str = node.get("sequent", "")
    seq = parse_sequent(seq_str)

    pivot = node.get("pivot", None)
    clean_rule = rule_str
    if "_" in rule_str and rule_str.rsplit("_", 1)[-1].isdigit():
        r_parts = rule_str.rsplit("_", 1)
        clean_rule = r_parts[0]
        if pivot is None:
            pivot = int(r_parts[1])

    # Normalize rule names for studio GUI
    rule_map = {
        "AXIOM": "Axiom",
        "R_IMP": "Imp_R",
        "L_IMP": "Imp_L",
        "R_AND": "And_R",
        "L_AND": "And_L1",
        "R_OR_1": "Or_R1",
        "R_OR_2": "Or_R2",
        "L_OR": "Or_L",
        "R_NOT": "Not_R",
        "L_NOT": "Not_L",
    }
    normalized_rule = rule_map.get(clean_rule, clean_rule)

    children = [dict_to_proof_node(b) for b in node.get("branches", []) if b]
    premises = [c.conclusion for c in children if c]

    return ProofNode(
        rule=normalized_rule,
        conclusion=seq,
        premises=premises,
        children=children,
        pivot=pivot,
        provability_score=node.get("score", 1.0)
    )


def proof_node_to_dict(node: Optional[ProofNode]) -> Optional[dict]:
    """Converts an object-oriented ProofNode into a dictionary proof tree."""
    if not node:
        return None

    rule_map_rev = {
        "Axiom": "AXIOM",
        "Imp_R": "R_IMP",
        "Imp_L": f"L_IMP_{node.pivot}" if node.pivot is not None else "L_IMP",
        "And_R": "R_AND",
        "And_L1": f"L_AND_{node.pivot}" if node.pivot is not None else "L_AND",
        "And_L2": f"L_AND_{node.pivot}" if node.pivot is not None else "L_AND",
        "Or_R1": "R_OR_1",
        "Or_R2": "R_OR_2",
        "Or_L": f"L_OR_{node.pivot}" if node.pivot is not None else "L_OR",
        "Not_R": "R_NOT",
        "Not_L": f"L_NOT_{node.pivot}" if node.pivot is not None else "L_NOT",
    }
    rule = rule_map_rev.get(node.rule, node.rule)
    return {
        "sequent": node.conclusion.to_str(),
        "rule": rule,
        "branches": [proof_node_to_dict(c) for c in node.children if c]
    }


# =====================================================================
# 2. NEURAL PROOF SEARCH ENGINE
# =====================================================================

class NeuralProofSearch:
    def __init__(
        self,
        engine: Optional[NanoGentzenEngine] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: str = "auto",
        max_depth: int = 8,
        max_nodes: int = 500,
        prune_threshold: float = 0.05,
        beam_width: int = 6,
        use_alpha_renaming: bool = True,
        use_invertible_fastpath: bool = True,
    ):
        self.engine = engine or NanoGentzenEngine(device=device)
        self.model = model or getattr(self.engine, "model", None)
        self.tokenizer = tokenizer or getattr(self.engine, "tokenizer", None)
        self.device = device
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.prune_threshold = prune_threshold
        self.beam_width = beam_width
        self.use_alpha_renaming = use_alpha_renaming
        self.use_invertible_fastpath = use_invertible_fastpath
        self.nodes_visited = 0

    def _find_invertible_action(self, seq: Sequent) -> Optional[Tuple[str, int]]:
        """
        Andreoli Asynchronous Phase:
        Deterministically returns the first applicable 100% invertible rule on CPU.
        Invertible rules preserve provability, eliminating choice points and neural overhead.
        """
        # 1. Single-premise left invertible rules (L_AND)
        for idx, target in enumerate(seq.gamma):
            t_name = getattr(target, "__class__", type(target)).__name__
            if (isinstance(target, And) or t_name == "And") and apply_rule(seq, "L_AND", idx) is not None:
                return ("L_AND", idx)

        # 2. Right invertible rules
        if seq.delta:
            d = seq.delta[0]
            d_name = getattr(d, "__class__", type(d)).__name__
            if (isinstance(d, Imp) or d_name == "Imp") and apply_rule(seq, "R_IMP", 0) is not None:
                return ("R_IMP", 0)
            if (isinstance(d, And) or d_name == "And") and apply_rule(seq, "R_AND", 0) is not None:
                return ("R_AND", 0)
            if (isinstance(d, Not) or d_name == "Not") and apply_rule(seq, "R_NOT", 0) is not None:
                return ("R_NOT", 0)

        # 3. Two-premise left invertible rules (L_OR)
        for idx, target in enumerate(seq.gamma):
            t_name = getattr(target, "__class__", type(target)).__name__
            if (isinstance(target, Or) or t_name == "Or") and apply_rule(seq, "L_OR", idx) is not None:
                return ("L_OR", idx)

        return None

    def evaluate_sequent(self, seq: Sequent) -> Dict[str, Any]:
        """Forward pass via engine returning rule_probs, pivot_probs, epistemic_probs, value."""
        return self.engine.evaluate_sequent(seq)

    def rank_actions(self, seq: Sequent) -> Tuple[List[Tuple[str, int]], Optional[np.ndarray], float]:
        """
        Ranks applicable Gentzen rules and Span Pointer pivots by joint score:
        Score = P(Rule) * P(Pivot).
        """
        eval_res = self.evaluate_sequent(seq)
        rule_probs = eval_res["rule_probs"]
        pivot_probs = eval_res["pivot_probs"]
        ep_probs = eval_res.get("epistemic_probs", None)
        val_score = eval_res.get("provability_value", 0.5)

        delta = seq.delta[0] if seq.delta else None
        d_name = getattr(delta, "__class__", type(delta)).__name__ if delta else ""
        gamma = seq.gamma
        gamma_len = len(gamma)
        max_ant = min(32, len(pivot_probs))

        scored_actions: List[Tuple[float, str, int]] = []

        # Right rules
        if isinstance(delta, Imp) or d_name == "Imp":
            scored_actions.append((float(rule_probs[1]), "R_IMP", 0))
        elif isinstance(delta, And) or d_name == "And":
            scored_actions.append((float(rule_probs[3]), "R_AND", 0))
        elif isinstance(delta, Or) or d_name == "Or":
            scored_actions.append((float(rule_probs[6]), "R_OR_1", 0))
            scored_actions.append((float(rule_probs[7]), "R_OR_2", 0))
        elif isinstance(delta, Not) or d_name == "Not":
            scored_actions.append((float(rule_probs[9]), "R_NOT", 0))

        # Left rules
        for p in range(min(gamma_len, max_ant)):
            f = gamma[p]
            f_name = getattr(f, "__class__", type(f)).__name__
            p_prob = float(pivot_probs[p]) if p < len(pivot_probs) else 0.001

            if isinstance(f, Imp) or f_name == "Imp":
                # Check for Dyckhoff L_IMP_ATOM fast-path first
                if isinstance(f.left, Var) and any(isinstance(x, Var) and x.name == f.left.name for i, x in enumerate(gamma) if i != p):
                    scored_actions.append((float(rule_probs[2]) * p_prob * 1.5, "L_IMP_ATOM", p))
                scored_actions.append((float(rule_probs[2]) * p_prob, "L_IMP", p))
            elif isinstance(f, And) or f_name == "And":
                scored_actions.append((float(rule_probs[4]) * p_prob, "L_AND", p))
            elif isinstance(f, Or) or f_name == "Or":
                scored_actions.append((float(rule_probs[8]) * p_prob, "L_OR", p))
            elif isinstance(f, Not) or f_name == "Not":
                scored_actions.append((float(rule_probs[10]) * p_prob, "L_NOT", p))

        scored_actions.sort(key=lambda x: x[0], reverse=True)
        return [(r, idx) for _, r, idx in scored_actions[:self.beam_width]], ep_probs, val_score

    def _prove_recursive(
        self,
        seq: Sequent,
        depth: int,
        max_depth: int,
        budget: List[int],
        max_nodes: int,
        path_visited: Set[str],
        memo: Dict[str, Optional[Dict]],
        truncated: Optional[List[bool]] = None,
    ) -> Optional[Dict]:
        if seq.is_axiom():
            return {"sequent": seq.to_str(), "rule": "AXIOM", "branches": []}

        if depth >= max_depth or budget[0] >= max_nodes:
            if truncated is not None:
                truncated[0] = True
            return None

        seq_key = seq.to_str()
        if seq_key in path_visited:
            if truncated is not None:
                truncated[0] = True
            return None
        if seq_key in memo:
            return memo[seq_key]

        path_visited.add(seq_key)
        budget[0] += 1
        sub_trunc = [False]

        # Phase 1: Andreoli Asynchronous Invertible Reduction
        if self.use_invertible_fastpath:
            inv_action = self._find_invertible_action(seq)
            if inv_action is not None:
                rule, idx = inv_action
                premises = apply_rule(seq, rule, idx=idx)
                if premises is not None:
                    sub_proofs = []
                    failed = False
                    for p in premises:
                        branch_trunc = [False]
                        res = self._prove_recursive(
                            p, depth + 1, max_depth, budget, max_nodes, path_visited, memo, branch_trunc
                        )
                        if branch_trunc[0]:
                            sub_trunc[0] = True
                        if res is None:
                            failed = True
                            break
                        sub_proofs.append(res)
                    path_visited.remove(seq_key)
                    if not failed and len(sub_proofs) == len(premises):
                        tree = {
                            "sequent": seq.to_str(),
                            "rule": f"{rule}_{idx}" if "L_" in rule else rule,
                            "branches": sub_proofs,
                        }
                        memo[seq_key] = tree
                        return tree
                    if not sub_trunc[0]:
                        memo[seq_key] = None
                    elif truncated is not None:
                        truncated[0] = True
                    return None

        # Phase 2: Synchronous Choice Points with Kripke Epistemic Guidance
        candidate_actions, ep_probs, val_score = self.rank_actions(seq)

        # Epistemic pruning: if P(False) is high or constructive value is near zero
        p_false = float(ep_probs[0]) if ep_probs is not None else 0.0
        if depth > 1 and (p_false > 0.90 or val_score < self.prune_threshold):
            path_visited.remove(seq_key)
            if truncated is not None:
                truncated[0] = True
            return None

        for rule, idx in candidate_actions:
            if budget[0] >= max_nodes:
                sub_trunc[0] = True
                break
            premises = apply_rule(seq, rule, idx=idx)
            if premises is not None:
                sub_proofs = []
                failed = False
                for p in premises:
                    branch_trunc = [False]
                    res = self._prove_recursive(
                        p, depth + 1, max_depth, budget, max_nodes, path_visited, memo, branch_trunc
                    )
                    if branch_trunc[0]:
                        sub_trunc[0] = True
                    if res is None:
                        failed = True
                        break
                    sub_proofs.append(res)

                if not failed and len(sub_proofs) == len(premises):
                    tree = {
                        "sequent": seq.to_str(),
                        "rule": f"{rule}_{idx}" if "L_" in rule else rule,
                        "branches": sub_proofs,
                    }
                    path_visited.remove(seq_key)
                    memo[seq_key] = tree
                    return tree

        path_visited.remove(seq_key)
        if not sub_trunc[0]:
            memo[seq_key] = None
        elif truncated is not None:
            truncated[0] = True
        return None

    def prove(
        self,
        seq: Sequent,
        max_depth: Optional[int] = None,
        max_nodes: Optional[int] = None,
        extract_countermodel_on_failure: bool = True,
        return_node: bool = False,
    ) -> Any:
        """
        Top-level proof search entrypoint.
        Returns:
            - If return_node=False: Tuple[Optional[Dict], Optional[Dict]] -> (proof_tree, countermodel)
            - If return_node=True: Optional[ProofNode]
        """
        m_depth = max_depth or self.max_depth
        m_nodes = max_nodes or self.max_nodes
        budget = [0]
        path_visited: Set[str] = set()
        memo: Dict[str, Optional[Dict]] = {}

        if self.use_alpha_renaming:
            canon_seq, _, rev_map = canonicalize_sequent(seq)
            raw_tree = self._prove_recursive(
                canon_seq, 0, m_depth, budget, m_nodes, path_visited, memo
            )
            proof_tree = restore_proof_tree(raw_tree, rev_map)
        else:
            proof_tree = self._prove_recursive(
                seq, 0, m_depth, budget, m_nodes, path_visited, memo
            )

        countermodel = None
        if proof_tree is None and extract_countermodel_on_failure:
            countermodel = extract_kripke_countermodel(seq, max_depth=m_depth)

        if return_node:
            return dict_to_proof_node(proof_tree) if proof_tree else None

        return proof_tree, countermodel


def prove_best_first(
    seq: Sequent,
    engine: Optional[NanoGentzenEngine] = None,
    max_depth: int = 8,
    max_nodes: int = 300,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Convenience wrapper for one-shot neural proof search with Kripke diagnosis."""
    searcher = NeuralProofSearch(engine=engine)
    return searcher.prove(seq, max_depth=max_depth, max_nodes=max_nodes)
