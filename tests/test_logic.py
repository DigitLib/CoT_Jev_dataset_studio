"""
Logic core: sequent kernel, formula parser, proof checker, the nanoGentzen v4 engine (backends,
certificates, verdicts, atom renaming), the engine factory and the decision server's safety rules.
"""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tests  # noqa: F401  (redirects all runtime data to a temp dir; see tests/__init__.py)

from nanogentzen.architect_cot import CotLogicalArchitect, proof_record
from nanogentzen.architect_jev import JevLogicalArchitect
from nanogentzen.engine import NanoGentzenEngine, detect_model_version, load_engine
from nanogentzen.engine_v4 import NanoGentzenV4Engine
from nanogentzen.kernel import And, Atom, Imp, Not, Or, ProofNode, RULE_NAMES, Sequent, Var, verify_proof_tree, verify_proof_tree_node
from nanogentzen.logical_architect import LogicalArchitect
from nanogentzen.parser import Parser, clean_atom_name, ground_state_generic, parse_natural_language, parse_sequent, select_relevant_gamma, text_to_sequent
from nanogentzen.search import NeuralProofSearch
from nanogentzen_v4.checker import check_certificate


MODEL_V4 = ROOT / "model"   # nanoGentzen v4 weights (config.MODEL_DIR)

HAS_V4 = (MODEL_V4 / "model.safetensors").exists()


def _rand_formula(rng, atoms, depth):
    if depth == 0 or rng.random() < 0.3:
        a = Atom(rng.choice(atoms))
        return Not(a) if rng.random() < 0.2 else a
    op = rng.choice([And, Or, Imp, Imp, Not])
    if op is Not:
        return Not(_rand_formula(rng, atoms, depth - 1))
    return op(_rand_formula(rng, atoms, depth - 1), _rand_formula(rng, atoms, depth - 1))


HAS_WEIGHTS = (Path(__file__).parent.parent / "model" / "model.safetensors").exists()


class TestKernel(unittest.TestCase):
    def setUp(self):
        self.p = Atom("P")
        self.q = Atom("Q")
        self.r = Atom("R")

    def test_atom_equality_and_hashing(self):
        a1 = Atom("P")
        a2 = Atom("P")
        b = Atom("Q")
        self.assertEqual(a1, a2)
        self.assertNotEqual(a1, b)
        self.assertEqual(hash(a1), hash(a2))

    def test_formula_string_representation(self):
        conj = And(self.p, self.q)
        self.assertEqual(str(conj), "(P ∧ Q)")

        disj = Or(self.p, self.q)
        self.assertEqual(str(disj), "(P ∨ Q)")

        impl = Imp(self.p, self.q)
        self.assertEqual(str(impl), "(P → Q)")

        neg = Not(self.p)
        self.assertEqual(str(neg), "¬P")

    def test_sequent_representation(self):
        seq = Sequent(antecedents=(self.p, Imp(self.p, self.q)), succedent=self.q)
        self.assertIn("P", str(seq))
        self.assertIn("⊢", str(seq))
        self.assertEqual(seq.get_atoms(), {"P", "Q"})

    def test_axiom_verification(self):
        # Γ, A ⊢ A
        seq = Sequent(antecedents=(self.p, self.q), succedent=self.p)
        node = ProofNode(rule="Axiom", conclusion=seq, pivot=0, provability_score=1.0)
        self.assertTrue(verify_proof_tree(node))

        # Invalid axiom (succedent not in antecedents)
        invalid_seq = Sequent(antecedents=(self.p,), succedent=self.q)
        bad_node = ProofNode(rule="Axiom", conclusion=invalid_seq, pivot=0)
        self.assertFalse(verify_proof_tree(bad_node))

    def test_and_right_verification(self):
        # Conclusion: Γ ⊢ (P ∧ Q)
        # Premises: Γ ⊢ P and Γ ⊢ Q
        conc = Sequent(antecedents=(self.p, self.q), succedent=And(self.p, self.q))
        p1 = Sequent(antecedents=(self.p, self.q), succedent=self.p)
        p2 = Sequent(antecedents=(self.p, self.q), succedent=self.q)

        leaf1 = ProofNode(rule="Axiom", conclusion=p1, pivot=0)
        leaf2 = ProofNode(rule="Axiom", conclusion=p2, pivot=1)

        root = ProofNode(
            rule="And_R",
            conclusion=conc,
            premises=[p1, p2],
            children=[leaf1, leaf2]
        )
        self.assertTrue(verify_proof_tree(root))

    def test_rule_names_present(self):
        self.assertIn("Axiom", RULE_NAMES)
        self.assertIn("Imp_L", RULE_NAMES)
        self.assertIn("Imp_R", RULE_NAMES)
        self.assertIn("And_R", RULE_NAMES)


class TestParser(unittest.TestCase):
    def test_parse_atom(self):
        p = Parser("Alpha")
        f = p.parse_formula()
        self.assertIsInstance(f, Atom)
        self.assertEqual(f.name, "Alpha")

    def test_parse_conjunction(self):
        p = Parser("P & Q")
        f = p.parse_formula()
        self.assertIsInstance(f, And)
        self.assertEqual(f.left, Atom("P"))
        self.assertEqual(f.right, Atom("Q"))

    def test_parse_disjunction(self):
        p = Parser("P | Q")
        f = p.parse_formula()
        self.assertIsInstance(f, Or)
        self.assertEqual(f.left, Atom("P"))
        self.assertEqual(f.right, Atom("Q"))

    def test_parse_implication(self):
        p = Parser("P -> Q")
        f = p.parse_formula()
        self.assertIsInstance(f, Imp)
        self.assertEqual(f.left, Atom("P"))
        self.assertEqual(f.right, Atom("Q"))

    def test_parse_negation(self):
        p = Parser("~P")
        f = p.parse_formula()
        self.assertIsInstance(f, Not)
        self.assertEqual(f.inner, Atom("P"))

    def test_parse_sequent_string(self):
        seq = parse_sequent("P, P -> Q |- Q")
        self.assertIsInstance(seq, Sequent)
        self.assertEqual(len(seq.antecedents), 2)
        self.assertEqual(seq.succedent, Atom("Q"))

    def test_clean_atom_name(self):
        name = clean_atom_name("Computational Theory of Mind", prefix="Premise")
        self.assertTrue(name.startswith("Premise_"))
        self.assertTrue(name.replace("_", "").isalnum())

    def test_text_to_sequent(self):
        premises = ["The brain performs computation", "Computation requires state transitions"]
        conclusion = "The brain requires state transitions"
        seq = text_to_sequent(premises, conclusion)
        self.assertIsInstance(seq, Sequent)
        self.assertGreater(len(seq.antecedents), 0)
        self.assertIsNotNone(seq.succedent)


class TestProofChecker(unittest.TestCase):
    def _node(self, gamma):
        a, c = Atom("A"), Atom("C")
        child = ProofNode(rule="Axiom", conclusion=Sequent(gamma=tuple(g for g in gamma if g != Imp(a, c)) + (c,),
                                                           delta=(c,)))
        for i, g in enumerate(gamma):
            if g == Imp(a, c):
                pivot = i
        return ProofNode(rule="L_IMP_ATOM", conclusion=Sequent(gamma=tuple(gamma), delta=(c,)),
                         children=[child], pivot=pivot)

    def test_rejects_missing_antecedent(self):
        a, c = Atom("A"), Atom("C")
        self.assertFalse(verify_proof_tree_node(self._node([Imp(a, c)])), "(A->C) |- C must not verify")

    def test_accepts_with_antecedent(self):
        a, c = Atom("A"), Atom("C")
        self.assertTrue(verify_proof_tree_node(self._node([a, Imp(a, c)])), "A, (A->C) |- C is valid")


class TestAtomRenaming(unittest.TestCase):
    def test_renamed_atom_never_collides(self):
        from nanogentzen.engine_v4 import NanoGentzenV4Engine as E
        names = {}
        renamed = E._safe_name("x y", names)
        self.assertNotEqual(renamed, E._safe_name("a_x_y", names))
        self.assertEqual(E._safe_name("a_x_y", names), "a_x_y", "valid identifiers pass through")
        spoof = E._safe_name(renamed, names)   # an original atom that looks like a renamed one
        self.assertNotEqual(spoof, renamed)
        self.assertEqual(E._safe_name("x y", names), renamed, "stable")


class TestEngineFactory(unittest.TestCase):
    def test_v3_checkpoint_is_refused(self):
        from nanogentzen.engine import detect_model_version, load_engine
        d = Path(tempfile.mkdtemp(prefix="v3ckpt-"))
        (d / "config.json").write_text(json.dumps({"model_type": "nanogentzen", "format_version": 3}))
        self.assertEqual(detect_model_version(str(d)), 3)
        with self.assertRaises(ValueError):
            load_engine(str(d))

    def test_missing_checkpoint_counts_as_v4(self):
        from nanogentzen.engine import detect_model_version
        self.assertEqual(detect_model_version(str(Path(tempfile.mkdtemp()) / "nope")), 4)


class TestDecisionServerSafety(unittest.TestCase):
    def setUp(self):
        from nanogentzen_v4 import jev_server
        self.js = jev_server

    def test_timed_out_consistency_is_not_consistent(self):
        from nanogentzen_v4.kernel import Var
        cls = next((c for c in vars(self.js).values() if isinstance(c, type) and hasattr(c, "_conflict")), None)
        self.assertIsNotNone(cls)
        inst = cls.__new__(cls)
        with mock.patch.object(self.js, "g4ip_prove", return_value=(None, True)):
            out = inst._conflict([Var("a"), Var("b")])
        self.assertIsNotNone(out, "a timed-out consistency check must not report 'consistent' (None)")
        self.assertIn("timed out", out[0])
        with mock.patch.object(self.js, "g4ip_prove", return_value=(None, False)):
            self.assertIsNone(inst._conflict([Var("a")]))


@unittest.skipUnless(HAS_V4, "model/ (nanoGentzen v4 weights) not present")
class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = load_engine(str(MODEL_V4), backend="auto")
        cls.sym = load_engine(str(MODEL_V4), backend="symbolic")

    def test_isolated_from_v3_package(self):
        import nanogentzen.kernel as k3
        import nanogentzen_v4.kernel as k4
        self.assertIsNot(k3, k4)
        self.assertTrue(hasattr(k3, "Atom"))
        self.assertEqual(sys.modules["nanogentzen.kernel"].__file__, k3.__file__)
        for top in ("kernel", "parser", "search", "certificates", "checker", "jev_server"):
            mod = sys.modules.get(top)
            # nanogentzen/ registers `kernel` as an alias of its own module; no alias may point at a v4 module
            leaked = mod is not None and str(getattr(mod, "__name__", "")).startswith("nanogentzen_v4")
            self.assertFalse(leaked, f"v4 module leaked as top-level '{top}'")

    def test_detects_versions(self):
        self.assertIsInstance(self.eng, NanoGentzenV4Engine)
        self.assertEqual(detect_model_version(str(MODEL_V4 / "model.safetensors")), 4)
        if (ROOT / "model" / "config.json").exists():
            self.assertEqual(detect_model_version(str(ROOT / "model")), 4)
        self.assertIn(self.eng.backend, ("v4_torch", "v4_numpy"))
        self.assertEqual(self.sym.backend, "v4_symbolic")
        self.assertTrue(self.eng.status()["all_must_pass"])

    def test_chain_proof_is_certified_independently(self):
        a, b, c = Atom("a"), Atom("b"), Atom("c")
        r = self.eng.prove_sequent(Sequent((a, Imp(a, b), Imp(b, c), Atom("noise")), (c,)))
        self.assertEqual(r["status"], "PROVEN_TRUE")
        self.assertTrue(r["certificate_checked"])
        ok, err = check_certificate(r["certificate"])
        self.assertTrue(ok, err)
        self.assertIn("noise", r["relevance"]["unused_premises"])
        self.assertNotIn("noise", r["relevance"]["minimal_witnesses"])

    def test_statuses(self):
        a, b = Atom("a"), Atom("b")
        self.assertEqual(self.eng.prove_sequent(Sequent((a,), (b,)))["status"], "INTUITIONISTIC_UNDECIDED")
        r = self.eng.prove_sequent(Sequent((a, Imp(a, Not(b))), (b,)))
        self.assertEqual(r["status"], "PROVEN_FALSE")
        self.assertTrue(r["refutation"]["certificate"]["checked"])
        r = self.eng.prove_sequent(Sequent((a, Not(a)), (b,)))
        self.assertEqual(r["status"], "INCONSISTENT_PREMISES")
        self.assertEqual(sorted(r["conflicting_premises"]), ["a", "~a"])
        r = self.eng.prove_sequent(Sequent((), (Or(a, Not(a)),)))      # no excluded middle
        self.assertEqual(r["status"], "INTUITIONISTIC_UNDECIDED")
        r = self.eng.prove_sequent(Sequent((), (Not(Not(Or(a, Not(a)))),)))
        self.assertEqual(r["status"], "PROVEN_TRUE")

    def test_undecided_has_checked_countermodel_and_missing_facts(self):
        k, s, ap = Atom("kyc"), Atom("sanctioned"), Atom("approve")
        r = self.eng.prove_sequent(Sequent((k, Imp(And(k, Not(s)), ap)), (ap,)))
        self.assertEqual(r["status"], "INTUITIONISTIC_UNDECIDED")
        self.assertTrue(r["countermodel"]["is_countermodel"])
        self.assertEqual(r["missing_facts"], ["~sanctioned"])

    def test_reserved_identifiers_survive(self):
        r = self.eng.prove_sequent(Sequent((Atom("or"), Imp(Atom("or"), Atom("not"))), (Atom("not"),)))
        self.assertEqual(r["status"], "PROVEN_TRUE")
        self.assertIn("or", r["relevance"]["minimal_witnesses"])

    def test_network_never_changes_a_verdict(self):
        rng = random.Random(7)
        atoms = ["a", "b", "c", "d"]
        for _ in range(40):
            gamma = tuple(_rand_formula(rng, atoms, 2) for _ in range(rng.randint(1, 4)))
            seq = Sequent(gamma, (_rand_formula(rng, atoms, 2),))
            r1 = self.eng.prove_sequent(seq, want_missing_facts=False)
            r2 = self.sym.prove_sequent(seq, want_missing_facts=False)
            self.assertEqual(r1["status"], r2["status"], seq.to_str())
            if r1["status"] == "PROVEN_TRUE":
                self.assertTrue(check_certificate(r1["certificate"])[0])

    def test_system1_invariant_under_renaming_and_order(self):
        s1 = Sequent((Atom("x"), Imp(Atom("x"), Atom("y")), Atom("z")), (Atom("y"),))
        s2 = Sequent((Atom("q9"), Atom("p1"), Imp(Atom("p1"), Atom("goal"))), (Atom("goal"),))
        e1, e2 = self.eng.evaluate_sequent(s1)["epistemic"], self.eng.evaluate_sequent(s2)["epistemic"]
        for k in e1:
            self.assertAlmostEqual(e1[k], e2[k], places=4)

    def test_decide_api(self):
        res = self.eng.decide({"state": {"kyc": True, "sanctioned": False},
                               "policy_rules": ["(kyc & ~sanctioned) => approve"],
                               "questions": {"approve": {"type": "noul", "target": "approve"}}})
        ans = res["answers"]["approve"]
        self.assertEqual(ans["verification"], "PROVEN_TRUE")
        self.assertTrue(ans["certificate"]["checked"])
        with self.assertRaises(ValueError):
            self.eng.decide({"state": {}, "policy_rules": ["(a & => b"], "questions": {"q": {"type": "noul"}}})


@unittest.skipUnless(HAS_V4, "model/ (nanoGentzen v4 weights) not present")
class TestNumpyTorchParity(unittest.TestCase):
    def test_parity(self):
        try:
            import torch  # noqa: F401
            from nanogentzen_v4.modeling_nanogentzen import GentzenPolicyValueModel
        except Exception:
            self.skipTest("torch / transformers not installed")
        import numpy as np
        from nanogentzen_v4.numpy_model import NumpyGentzenModel
        from nanogentzen_v4.backend import to_numpy
        tm = GentzenPolicyValueModel.from_pretrained(str(MODEL_V4), local_files_only=True).eval()
        nm = NumpyGentzenModel(str(MODEL_V4))
        eng = load_engine(str(MODEL_V4), backend="numpy")
        seq = Sequent((Atom("a"), Imp(Atom("a"), Or(Atom("b"), Atom("c"))), Imp(Atom("b"), Atom("d"))),
                      (Or(Atom("d"), Atom("c")),))
        v4seq, _ = eng.to_v4_sequent(seq)
        from nanogentzen_v4.kernel import model_view_sequent
        ids = eng.tokenizer.encode(model_view_sequent(v4seq)[0].model_str())
        row = ids + [0] * (tm.config.block_size - len(ids))
        t_out = [to_numpy(x) for x in tm.predict_epistemic(torch.tensor([row]))]
        n_out = nm.predict_epistemic(np.array([row]))
        for a, b in zip(t_out, n_out):
            self.assertLess(float(np.abs(np.asarray(a) - np.asarray(b)).max()), 1e-4)


@unittest.skipUnless(HAS_V4, "model/ (nanoGentzen v4 weights) not present")
class TestArchitects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = load_engine(str(MODEL_V4))
        cls.cot = CotLogicalArchitect(engine=cls.eng)
        cls.jev = JevLogicalArchitect(engine=cls.eng)

    V3_KEYS = {"is_verified", "epistemic_status", "sequent", "proof_tree", "curry_howard_term", "relevance",
               "relevance_ratio", "is_strictly_relevant", "countermodel", "kripke_countermodel",
               "logic_tree_text", "derivation_steps", "provability_score", "depth", "node_count"}

    def test_cot_contract_and_chain(self):
        r = self.cot.generate_best_logic_tree(
            premises=["x"], conclusion="Ecumenical Approach",
            path_nodes=["Autonomy", "Algorithmic System", "Ecumenical Approach"],
            path_repr="(Autonomy) --[requires]--> (Algorithmic System) -> (Algorithmic System) --[implies]--> (Ecumenical Approach)")
        self.assertTrue(self.V3_KEYS <= set(r))
        self.assertTrue(r["is_verified"])
        self.assertTrue(r["certificate_checked"])
        self.assertIn("⊢", r["derivation_steps"][-1]["conclusion"])
        self.assertIn("Established:", r["logic_tree_text"])
        rec = proof_record(r)
        self.assertTrue(rec["certificate_checked"] and rec["certificate"]["checked"])

    def test_cot_never_marks_unprovable_as_verified(self):
        r = self.cot.generate_best_logic_tree(premises=["a"], conclusion="c", symbolic_sequent="a, a => b |- c")
        self.assertTrue(self.V3_KEYS <= set(r))
        self.assertFalse(r["is_verified"])
        self.assertEqual(r["epistemic_status"], "INTUITIONISTIC_UNDECIDED")
        self.assertIsNone(r["proof_tree"])

    RULES = ["(is_scientific_theory & uses_processes) => compatible", "is_falsified => incompatible"]

    def _q(self):
        return {"compat": {"instructions": "?", "type": "choice",
                           "criteria": {"Compatible": "...", "incompatible": "...", "undecided": "..."}}}

    def test_jev_nested_attributes_and_case(self):
        state = {"attributes": {"is_scientific_theory": True, "uses_processes": True, "is_falsified": False}}
        out = self.jev.evaluate_decision_scenario(state, self._q(), self.RULES + ["just a sentence"], "Compatible")
        g = out["gold"]["compat"]
        self.assertEqual(out["epistemic_status"], "PROVEN_TRUE")
        self.assertEqual(g["label"], "Compatible")
        self.assertEqual(g["probabilities"]["Compatible"], 1.0)
        self.assertTrue(g["certificate_checked"])
        self.assertEqual([r["rule"] for r in out["rejected_rules"]], ["just a sentence"])
        self.assertEqual(out["factors"], {"attributes": {"is_scientific_theory": True, "uses_processes": True}})

    def test_jev_minimal_state_completion(self):
        state = {"attributes": {"is_scientific_theory": True}}
        out = self.jev.evaluate_decision_scenario(state, self._q(), self.RULES, "Compatible")
        # provable only with the added fact: valid under that assumption, but not certified
        self.assertEqual(out["epistemic_status"], "PROVEN_UNDER_ASSUMPTIONS")
        self.assertFalse(out["gentzen_verified"])
        self.assertEqual(out["gold"]["compat"]["assumed_facts"], {"uses_processes": True})
        self.assertEqual(out["state_completion"]["added_facts"], {"uses_processes": True})
        self.assertEqual(state["attributes"], {"is_scientific_theory": True, "uses_processes": True})

    def test_jev_completion_never_overrides_evidence(self):
        state = {"attributes": {"is_scientific_theory": True, "uses_processes": False}}
        out = self.jev.evaluate_decision_scenario(state, self._q(), self.RULES, "Compatible")
        self.assertFalse(out["gentzen_verified"])
        self.assertFalse(out["state_completion"]["applied"])
        self.assertFalse(state["attributes"]["uses_processes"])

    def test_jev_completion_can_be_disabled(self):
        jev = JevLogicalArchitect(engine=self.eng, allow_state_completion=False)
        out = jev.evaluate_decision_scenario({"attributes": {"is_scientific_theory": True}}, self._q(), self.RULES, "Compatible")
        self.assertEqual(out["epistemic_status"], "INTUITIONISTIC_UNDECIDED")
        self.assertEqual(out["gold"]["compat"]["label"], "Compatible")      # intended label kept, flagged

    def test_jev_target_contradicted_by_proof(self):
        state = {"attributes": {"is_falsified": True}}
        out = self.jev.evaluate_decision_scenario(state, self._q(), self.RULES, "Compatible")
        self.assertEqual(out["epistemic_status"], "TARGET_REFUTED_BY_PROOF")
        self.assertEqual(out["gold"]["compat"]["target_mismatch"]["proven"], "incompatible")

    def test_jev_label_smoothing(self):
        jev = JevLogicalArchitect(engine=self.eng, label_smoothing=0.1)
        state = {"attributes": {"is_scientific_theory": True, "uses_processes": True}}
        g = jev.evaluate_decision_scenario(state, self._q(), self.RULES, "Compatible")["gold"]["compat"]
        self.assertAlmostEqual(g["probabilities"]["Compatible"], 0.9)
        self.assertAlmostEqual(sum(g["probabilities"].values()), 1.0, places=5)


class TestEngineAndSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_dir = Path(__file__).parent.parent / "model"
        cls.engine = NanoGentzenEngine(model_path=str(model_dir / "model.safetensors"))
        cls.architect = LogicalArchitect(engine=cls.engine)

    @unittest.skipUnless(HAS_WEIGHTS, "model/ (nanoGentzen v4 weights) not present")
    def test_engine_backend_is_v4(self):
        self.assertIn(self.engine.backend, ("v4_torch", "v4_numpy", "v4_symbolic"))
        self.assertTrue(self.engine.is_safetensors_loaded)

    @unittest.skipUnless(HAS_WEIGHTS, "model/ (nanoGentzen v4 weights) not present")
    def test_engine_neural_evaluation(self):
        seq = Sequent((Var("P"), Imp(Var("P"), Var("Q"))), (Var("Q"),))
        res = self.engine.evaluate_sequent(seq)
        self.assertIn(res["mode"], ("v4_torch", "v4_numpy", "v4_symbolic"))
        self.assertIn(len(res["rule_probs"]), (6, 11))
        self.assertEqual(len(res["pivot_probs"]), 32)
        self.assertIsNotNone(res["epistemic_probs"])
        self.assertGreater(res["provability_value"], 0.8)

    def test_natural_language_parser(self):
        text = "If user is adult then access is granted. User is adult. Is access granted?"
        res = parse_natural_language(text)
        self.assertIsNotNone(res)
        seq, desc = res
        self.assertEqual(len(seq.gamma), 2)
        self.assertIsInstance(seq.gamma[0], Imp)

    def test_ground_state_generic_nested(self):
        state = {
            "applicant": {
                "income": 85000,
                "verified": True,
            },
            "risk_score": 12,
        }
        predicates = {
            "income_ok": {"field": "applicant.income", "op": ">=", "value": 60000},
            "kyc_ok": {"field": "applicant.verified", "op": "==", "value": True},
        }
        facts, missing = ground_state_generic(state, predicates)
        fact_strs = [f.to_str() for f in facts]
        self.assertIn("income_ok", fact_strs)
        self.assertIn("kyc_ok", fact_strs)
        self.assertEqual(len(missing), 0)

    def test_select_relevant_gamma_cone_of_influence(self):
        facts = [Var("user_active"), Var("income_ok"), Var("distractor_fact")]
        rules = [
            Imp(And(Var("user_active"), Var("income_ok")), Var("loan_approved")),
            Imp(Var("unrelated"), Var("something_else")),
        ]
        goal = Var("loan_approved")
        relevant = select_relevant_gamma(facts, rules, goal, max_antecedents=32)
        rel_strs = [f.to_str() for f in relevant]
        self.assertIn("user_active", rel_strs)
        self.assertIn("income_ok", rel_strs)
        self.assertNotIn("distractor_fact", rel_strs)

    def test_neural_proof_search_andreoli_and_g4ip(self):
        # A, (A => B) |- B
        searcher = NeuralProofSearch(engine=self.engine)
        seq = Sequent((Var("A"), Imp(Var("A"), Var("B"))), (Var("B"),))
        proof_tree, cm = searcher.prove(seq)
        self.assertIsNotNone(proof_tree)
        self.assertIn("rule", proof_tree)

    def test_logical_architect_end_to_end_verified(self):
        premises = [
            "Computation requires state transitions",
            "The brain performs computation",
        ]
        conclusion = "The brain requires state transitions"
        result = self.architect.generate_best_logic_tree(premises, conclusion)

        # Plain factual premises: the bridge (P1 & P2) => C is ADDED, so the valid proof shows nothing
        # about the text. It is kept for the CoT guide but not counted as verified.
        self.assertEqual(result["logic_basis"], "assumed_bridge")
        self.assertTrue(result["proof_valid"])
        self.assertFalse(result["is_verified"])
        self.assertEqual(result["epistemic_status"], "BRIDGE_ASSUMED")
        self.assertIn("curry_howard_term", result)
        self.assertTrue(len(result["curry_howard_term"]) > 0)
        self.assertIn("relevance", result)
        self.assertIn("logic_tree_text", result)
        self.assertIn("Curry-Howard", result["logic_tree_text"])
        self.assertIn("Relevance Logic", result["logic_tree_text"])

    def test_logical_architect_multihop_path(self):
        path = ["Acoustics", "Cultural_Heritage", "Preservation"]
        unlabelled = self.architect.generate_best_logic_tree(
            premises=["Acoustics relates to preservation"], conclusion="Preservation", path_nodes=path)
        self.assertFalse(unlabelled["is_verified"])          # hops without relations are not inferences
        result = self.architect.generate_best_logic_tree(
            premises=["Acoustics relates to preservation"],
            conclusion="Preservation",
            path_nodes=path,
            path_repr="(Acoustics) --[is a kind of]--> (Cultural_Heritage) -> (Cultural_Heritage) --[requires]--> (Preservation)",
        )
        self.assertTrue(result["is_verified"])
        self.assertEqual(result["epistemic_status"], "PROVEN_TRUE")
        self.assertIn("curry_howard_term", result)


if __name__ == "__main__":
    unittest.main()
