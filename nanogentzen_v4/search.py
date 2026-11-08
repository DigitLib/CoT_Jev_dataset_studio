# Vendored from the nanoGentzen v4 model folder by tools/vendor_v4_engine.py. Do not edit by hand.
"""
search.py (nanoGentzen-v3, Phase 1)
Neural-guided G4ip proof search.

  1. Invertible G4ip rules are applied deterministically (no network call, no choice).
  2. At real choice points (R_OR_1 / R_OR_2, L_IMP_IMP, L_IMP_NOT, L_NOT_IMP, L_NOT_NOT) the released
     policy head ranks the moves: its 11 LJ classes are mapped onto G4ip moves
     (L_IMP -> L_IMP_IMP / L_IMP_NOT, L_NOT -> L_NOT_IMP / L_NOT_NOT), its pointer head ranks pivots.
  3. L_CONTR is gone; G4ip needs no contraction and every branch terminates.
  4. Sequents are normalised, so memo keys are canonical. The network reads the alpha-renamed sequent;
     the search itself works on the original names, so proof trees stay verifiable.
  5. max_depth limits nested choice points; max_nodes limits all expanded nodes.
  6. prove() reports whether a limit was hit (self.last_truncated) and whether the model window
     overflowed (self.last_window_overflow); callers decide what an unfinished search means.
"""
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import torch
except ImportError:  # the search itself does not need torch; the model does
    torch = None

from .kernel import (CHOICE_LEFT_RULES, RULES, Sequent, _first_invertible, apply_rule, applicable_moves,
                     canonicalize_sequent, find_kripke_countermodel, normalize_sequent, rule_label)
from . import kernel as _kernel
from .backend import as_input, to_numpy

# G4ip choice move -> the policy head class trained for it
HEAD_CLASS = {"R_OR_1": "R_OR_1", "R_OR_2": "R_OR_2", "L_IMP_IMP": "L_IMP", "L_IMP_NOT": "L_IMP",
              "L_NOT_IMP": "L_NOT", "L_NOT_NOT": "L_NOT"}
CHOICE_RULES = ("R_OR_1", "R_OR_2") + tuple(CHOICE_LEFT_RULES)


def _no_grad(fn):
    return torch.no_grad()(fn) if torch is not None else fn


class NeuralProofSearch:
    def __init__(self, model: Any, tokenizer: Any, device: str = "cuda", value_prune_threshold: float = 0.05,
                 use_alpha_renaming: bool = True, use_invertible_fastpath: bool = True,
                 epistemic_pruning: bool = False, confidence_threshold: float = 0.30):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.value_prune_threshold = value_prune_threshold
        self.use_alpha_renaming = use_alpha_renaming
        self.use_invertible_fastpath = use_invertible_fastpath  # kept for API compatibility; always on in G4ip
        self.epistemic_pruning = epistemic_pruning
        cfg_thresh = getattr(getattr(model, "config", None), "search_confidence_threshold", None)
        self.confidence_threshold = float(cfg_thresh if cfg_thresh is not None else confidence_threshold)
        self.last_truncated = False
        self.last_window_overflow = False
        self.last_nodes = 0
        if hasattr(self.model, "eval"):
            self.model.eval()

    # ------------------------------------------------------------ network
    def model_input(self, seq: Sequent) -> Tuple[Optional[List[int]], Sequent]:
        self._view_order = None
        if getattr(self.model.config, "input_format", "") == "normalized_canonical":   # v4 checkpoints
            eval_seq, order = _kernel.model_view_sequent(seq)
            self._view_order = {j: k for k, j in enumerate(order)}   # search index -> model slot
        else:
            eval_seq = canonicalize_sequent(seq)[0] if self.use_alpha_renaming else seq
        enc = self.tokenizer.encode(eval_seq.model_str())
        bs = self.model.config.block_size
        if len(enc) > bs:
            return None, eval_seq
        return enc, eval_seq

    @_no_grad
    def evaluate_sequent(self, seq: Sequent):
        """(rule_probs, pivot_probs, epistemic_probs, value) or None when the sequent exceeds the model window."""
        padded, _ = self.model_input(seq)
        if padded is None:
            self.last_window_overflow = True
            return None
        x = as_input([padded], self.device, self.model)
        rp, pp, ep, val = self.model.predict_epistemic(x)   # probabilities, NumPy or torch backend
        return to_numpy(rp)[0], to_numpy(pp)[0], to_numpy(ep)[0], float(to_numpy(val)[0])

    def evaluate_value(self, seq: Sequent) -> float:
        r = self.evaluate_sequent(seq)
        return 0.5 if r is None else r[3]

    def rank_actions(self, seq: Sequent) -> Tuple[List[Tuple[str, int]], Any, float]:
        """G4ip choice moves ordered by P(head class) * P(pivot). Falls back to a fixed order without the network."""
        moves = applicable_moves(seq, CHOICE_RULES)
        if not moves:
            return [], None, 1.0
        ev = self.evaluate_sequent(seq)
        if ev is None:
            return moves, None, 1.0
        view_order = self._view_order
        rule_probs, pivot_probs, ep_probs, val = ev
        n_piv = int(pivot_probs.shape[-1])
        classes = getattr(self.model.config, "rule_classes", None)   # v4 heads predict G4ip moves directly
        scored = []
        for idx_m, (r, i) in enumerate(moves):
            if classes:
                rp = float(rule_probs[classes.index(r)].item()) if r in classes else 1e-3
            else:
                rp = float(rule_probs[RULES.index(HEAD_CLASS[r])].item())
            slot = i if view_order is None else view_order.get(i, n_piv)
            pp = 1.0 if r.startswith("R_") else (float(pivot_probs[slot].item()) if slot < n_piv else 1e-3)
            scored.append((rp * pp, idx_m, r, i))
        if len(scored) > 1 and max(x[0] for x in scored) < self.confidence_threshold:
            return moves, ep_probs, val
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(r, i) for _, _, r, i in scored], ep_probs, val

    def _find_invertible_action(self, seq: Sequent) -> Optional[Tuple[str, int]]:
        inv = _first_invertible(seq)
        return inv[0] if inv else None

    # ------------------------------------------------------------ search
    def _prove_recursive(self, seq: Sequent, depth: int, max_depth: int, budget: List[int], max_nodes: int,
                         path_visited: Set[str], memo: Dict[str, Optional[Dict]],
                         truncated: Optional[List[bool]] = None) -> Optional[Dict]:
        key = seq.to_str()
        if seq.is_axiom():
            return {"sequent": key, "rule": "AXIOM", "branches": []}
        if key in memo:
            return memo[key]
        if depth >= max_depth or budget[0] >= max_nodes or key in path_visited:
            if truncated is not None:
                truncated[0] = True
            return None
        budget[0] += 1
        path_visited.add(key)
        sub_trunc = [False]
        try:
            inv = self._find_invertible_action(seq)
            if inv is not None:
                candidates, ep, val = [inv], None, 1.0
            else:
                candidates, ep, val = self.rank_actions(seq)
                if self.epistemic_pruning and ep is not None and depth > 1:
                    if float(ep[0].item()) > 0.90 or val < self.value_prune_threshold:
                        sub_trunc[0] = True
                        candidates = []
            for r, i in candidates:
                prem = apply_rule(seq, r, i)
                if prem is None:
                    continue
                subs = []
                step = 0 if inv is not None else 1     # depth counts choice points only; budget counts every node
                for p in prem:
                    bt = [False]
                    t = self._prove_recursive(p, depth + step, max_depth, budget, max_nodes, path_visited, memo, bt)
                    sub_trunc[0] |= bt[0]
                    if t is None:
                        break
                    subs.append(t)
                if len(subs) == len(prem):
                    tree = {"sequent": key, "rule": rule_label(r, i), "branches": subs}
                    memo[key] = tree
                    return tree
                if inv is not None:      # invertible rule failed: the sequent is unprovable (if not truncated)
                    break
            if not sub_trunc[0]:
                memo[key] = None
            elif truncated is not None:
                truncated[0] = True
            return None
        finally:
            path_visited.discard(key)

    def prove(self, seq: Sequent, max_depth: int = 8, max_nodes: int = 300,
              extract_countermodel_on_failure: bool = True) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Returns (proof_tree, countermodel). A None proof with self.last_truncated=False means unprovable."""
        self.last_truncated = False
        self.last_window_overflow = False
        root = normalize_sequent(seq)
        budget, trunc = [0], [False]
        tree = self._prove_recursive(root, 0, max_depth, budget, max_nodes, set(), {}, trunc)
        self.last_truncated = trunc[0]
        self.last_nodes = budget[0]
        cm = None
        if tree is None and extract_countermodel_on_failure and not trunc[0]:
            cm = find_kripke_countermodel(root)
        return tree, cm


def prove_best_first(model, tokenizer, seq: Sequent, device: str = "cuda", max_depth: int = 8, max_nodes: int = 300):
    return NeuralProofSearch(model, tokenizer, device=device).prove(seq, max_depth=max_depth, max_nodes=max_nodes)
