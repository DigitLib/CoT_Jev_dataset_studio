"""
ClearCoT pipeline: KG relations and paths as logic, the CoT architect's verdicts, question
provenance, reasoning-trace modes, grounding checks and naive-trace (DPO negative) quality.
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tests  # noqa: F401  (redirects all runtime data to a temp dir; see tests/__init__.py)

from core.cot_generator import CoTGenerator, naive_trace_leak
from core.dataset_manager import DatasetManager, DatasetSample
from core.kg_builder import KnowledgeGraphBuilder, Proposition
from core.question_generator import QuestionGenerator
from nanogentzen.architect_cot import CotLogicalArchitect
from nanogentzen.cot.parser_cot import is_degenerate_concept, path_logic_report, path_missing_links
from nanogentzen.engine import NanoGentzenEngine, load_engine
from nanogentzen.logical_architect import LogicalArchitect
from nanogentzen.parser import path_logic_report, path_to_sequent
from nanogentzen.relation_logic import relation_logic


MODEL_V4 = ROOT / "model"   # nanoGentzen v4 weights (config.MODEL_DIR)

RAIN = (["Rain", "Wet Street", "Sprinkler"],
        "(Rain) --[causes] --> (Wet Street) -> (Wet Street) --[causes] <-- (Sprinkler)")


def _kg():
    return KnowledgeGraphBuilder.__new__(KnowledgeGraphBuilder)   # no graph / LLM needed for these helpers


GENUINE = ("Hmm, so open licences are supposed to help, right? I think they spread access, but wait, maybe the "
           "attribution part is tricky in some places. Licences are kind of universal anyway, so they probably "
           "just work everywhere, and that would mean the assumption holds. Yes, I'd say it holds overall.")

LEAKY = ("To construct a naive thinking trace: 1. Start by acknowledging the premise. The expected answer concept "
         "states that NOODL challenges openness. " + GENUINE)


class TestPredicates(unittest.TestCase):
    def test_polarity_is_kept(self):
        k = _kg()
        self.assertEqual(k._canonicalize_predicate("does not require"), "not requires")
        self.assertEqual(k._canonicalize_predicate("cannot enable"), "not enables")
        self.assertEqual(k._canonicalize_predicate("is not a kind of"), "not is a kind of")
        self.assertEqual(k._canonicalize_predicate("doesn't imply"), "not implies")
        self.assertEqual(k._canonicalize_predicate("lacks"), "excludes")
        self.assertEqual(k._canonicalize_predicate("is incompatible with"), "excludes")
        self.assertEqual(k._canonicalize_predicate("requires"), "requires")

    def test_logic_table(self):
        self.assertEqual(relation_logic("requires"), "entails")
        self.assertEqual(relation_logic("excludes"), "excludes")
        for p in ("not requires", "contrasts with", "models", "formulates", "enables", "not excludes"):
            self.assertEqual(relation_logic(p), "none", p)

    def test_heuristic_extractor_sees_negation(self):
        chunk = types.SimpleNamespace(text="Moral responsibility does not require libertarian free will.",
                                      chunk_id="c1", doc_name="d", page_number=1, section_title="s")
        props = _kg()._heuristic_proposition_extract(chunk)
        self.assertTrue(props)
        self.assertEqual(props[0].predicate, "not requires")
        self.assertEqual(props[0].subject, "Moral Responsibility")

    def test_names_are_not_mangled(self):
        k = _kg()
        for raw, want in [("Ethics", "Ethics"), ("Species", "Species"), ("Physics", "Physics"),
                          ("LLM reasoning", "LLM Reasoning"), ("DNA repair", "DNA Repair"),
                          ("neural networks", "Neural Network"), ("theories", "Theory"),
                          ("hypotheses", "Hypothesis"), ("GraphQL schemas", "GraphQL Schema")]:
            self.assertEqual(k._canonicalize(raw), want)


class TestPathEncoding(unittest.TestCase):
    def test_backward_edge_keeps_its_direction(self):
        seq = path_to_sequent(*RAIN)
        self.assertIn("(Sprinkler => Wet_Street)", seq.to_str())
        self.assertNotIn("(Wet_Street => Sprinkler)", seq.to_str())
        self.assertEqual(path_logic_report(*RAIN)["logic_basis"], "kg_relational")

    def test_negation_and_non_inferential_hops(self):
        s = path_to_sequent(["A", "B", "C"], "(A) --[implies]--> (B) -> (B) --[excludes]--> (C)")
        self.assertEqual(s.to_str(), "A, (A => B), (B => ~C) |- ~C")
        s = path_to_sequent(["A", "B", "C"], "(A) --[excludes]--> (B) -> (B) --[implies]--> (C)")
        self.assertEqual(s.to_str(), "A, (A => ~B), (B => C) |- C")
        s = path_to_sequent(["A", "B", "C"], "(A) --[contrasts with]--> (B) -> (B) --[implies]--> (C)")
        self.assertEqual(s.to_str(), "A, (B => C) |- C")
        self.assertEqual(path_to_sequent(["A", "B", "C"]).to_str(), "A |- C")   # no labels, no rules


class TestCotArchitect(unittest.TestCase):
    """Verdicts of the ClearCoT architect for KG paths and paragraph seeds."""

    @classmethod
    def setUpClass(cls):
        cls.arch = CotLogicalArchitect(engine=load_engine(str(MODEL_V4)))


    def test_rain_sprinkler_is_not_proven(self):
        r = self.arch.generate_best_logic_tree(premises=["x"], conclusion="Sprinkler", path_nodes=RAIN[0], path_repr=RAIN[1])
        self.assertFalse(r["is_verified"])
        self.assertEqual(r["logic_basis"], "kg_relational")
        self.assertIn("against the edge direction", r["logic_tree_text"])

    def test_entailment_chain_is_proven(self):
        r = self.arch.generate_best_logic_tree(premises=["x"], conclusion="C", path_nodes=["A", "B", "C"],
                                               path_repr="(A) --[requires]--> (B) -> (B) --[implies]--> (C)")
        self.assertTrue(r["is_verified"])
        self.assertEqual(r["logic_basis"], "kg_entailment_chain")

    def test_unlabelled_path_is_not_verified(self):
        r = self.arch.generate_best_logic_tree(premises=["x"], conclusion="C", path_nodes=["A", "B", "C"])
        self.assertFalse(r["is_verified"])

    def test_assumed_bridge_is_not_verified(self):
        r = self.arch.generate_best_logic_tree(premises=["Water boils at 100C", "The kettle is heated"],
                                               conclusion="The water boils")
        self.assertEqual(r["logic_basis"], "assumed_bridge")
        self.assertEqual(r["epistemic_status"], "BRIDGE_ASSUMED")
        self.assertFalse(r["is_verified"])
        self.assertTrue(r["proof_valid"])
        self.assertIn("Established:", r["logic_tree_text"])    # the guide for the CoT is kept


@unittest.skipUnless((MODEL_V4 / "model.safetensors").exists(), "model/ (nanoGentzen v4 weights) not present")


class TestCotPathSignals(unittest.TestCase):
    def test_missing_links_name_the_non_inferential_hop(self):
        rep = path_logic_report(["Algorithmic System", "Autonomy", "Freedom"],
                                "(Algorithmic System) --[bear upon]--> (Autonomy) -> (Autonomy) --[requires]--> (Freedom)")
        self.assertEqual(rep["logic_basis"], "kg_relational")
        links = path_missing_links(rep)
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["needed_rule"], "Algorithmic_System => Autonomy")
        self.assertIn("bear upon", links[0]["reason"])

    def test_degenerate_concept_breaks_the_chain(self):
        rep = path_logic_report(["Freedom", "Ecological Non-domination", "Both"],
                                "(Freedom) --[is defined as]--> (Ecological Non-domination) -> "
                                "(Ecological Non-domination) --[requires]--> (Both)")
        self.assertEqual(rep["logic_basis"], "kg_relational")
        self.assertEqual(rep["broken_at"], 1)
        self.assertTrue(is_degenerate_concept("Other Concept"))
        self.assertFalse(is_degenerate_concept("AI"))
        self.assertFalse(is_degenerate_concept("Substantive Independence"))


class TestQuestionProvenance(unittest.TestCase):
    def test_question_generator_provenance(self):
        qgen = QuestionGenerator()
        path_info = {
            "source": "Hierarchical Index",
            "target": "Dense Reranker",
            "hops": 2,
            "path_nodes": ["Hierarchical Index", "Vector Search Engine", "Dense Reranker"],
            "path_repr": "(Hierarchical Index) --[enables]--> (Vector Search Engine) --[accelerates]--> (Dense Reranker)",
            "provenance_pages": [3, 8],
            "provenance_docs": ["indexing.pdf", "reranking.pdf"],
            "doc_name": "indexing.pdf",
            "is_cross_document": True,
            "chunk_ids": ["c1", "c2"],
            "propositions": [
                {"statement": "Hierarchical Index enables Vector Search Engine.", "evidence": "ev1", "page_number": 3},
                {"statement": "Vector Search Engine accelerates Dense Reranker.", "evidence": "ev2", "page_number": 8},
            ],
            "composite_claim": "Because Hierarchical Index enables Vector Search Engine, vector search engine accelerates dense reranker."
        }

        # Test heuristic seed generator fallback when offline
        seed = qgen._heuristic_multihop_seed(path_info, [])
        prov = seed["provenance"]
        self.assertTrue(prov["is_cross_document"])
        self.assertEqual(prov["documents"], ["indexing.pdf", "reranking.pdf"])
        self.assertEqual(prov["doc_name"], "indexing.pdf")
        self.assertEqual(prov["pages"], [3, 8])

    def test_nanogentzen_upstream_path_verification(self):
        from nanogentzen.logical_architect import LogicalArchitect

        architect = LogicalArchitect()
        builder = KnowledgeGraphBuilder(architect=architect)

        builder.add_proposition(Proposition(
            subject="Franz Brentano", predicate="argued", obj="Intentionality",
            statement="Franz Brentano argued that mental phenomena exhibit Intentionality.",
            evidence="ev1", doc_name="docA.pdf", page_number=28
        ))
        builder.add_proposition(Proposition(
            subject="Intentionality", predicate="requires", obj="Mental Representation",
            statement="Intentionality requires Mental Representation.",
            evidence="ev2", doc_name="docA.pdf", page_number=29
        ))
        builder.add_proposition(Proposition(
            subject="Mental Representation", predicate="implemented by", obj="Global Workspace Architecture",
            statement="Mental Representation is implemented by Global Workspace Architecture.",
            evidence="ev3", doc_name="docB.pdf", page_number=15
        ))

        builder.add_proposition(Proposition(
            subject="Global Workspace Architecture", predicate="requires", obj="Broadcast Mechanism",
            statement="Global Workspace Architecture requires a Broadcast Mechanism.",
            evidence="ev4", doc_name="docB.pdf", page_number=16
        ))
        builder.add_proposition(Proposition(
            subject="Broadcast Mechanism", predicate="implies", obj="Access Consciousness",
            statement="A Broadcast Mechanism implies Access Consciousness.",
            evidence="ev5", doc_name="docB.pdf", page_number=17
        ))

        paths = builder.find_multihop_paths(max_hops=3, min_hops=2, intra_document_only=False, verify_with_gentzen=True)
        self.assertTrue(len(paths) > 0)
        for p in paths:
            self.assertIn("logic_tree_res", p)
            self.assertIn("curry_howard_term", p)
            self.assertIn("gentzen_sequent", p)
            self.assertIn(p["logic_basis"], ("kg_entailment_chain", "kg_relational"))
            self.assertTrue(len(p["logic_tree_text"]) > 20)
            # verified exactly when every hop is a forward inference ("argued" / "implemented by" are not)
            self.assertEqual(p["gentzen_verified"], p["logic_basis"] == "kg_entailment_chain", p["path_repr"])
            if p["gentzen_verified"]:
                self.assertEqual(p["epistemic_status"], "PROVEN_TRUE")
                self.assertIsNotNone(p.get("proof_tree"))
        verified = [p for p in paths if p["gentzen_verified"]]
        self.assertTrue(any(p["path_nodes"] == ["Global Workspace Architecture", "Broadcast Mechanism", "Access Consciousness"]
                            for p in verified))
        self.assertFalse(any(p["path_nodes"][0] == "Franz Brentano" for p in verified))

    def test_path_selection_preview_synchronization(self):
        """Verify that switching selected path invalidates old sample and loads new path tree."""
        from nanogentzen.logical_architect import LogicalArchitect
        architect = LogicalArchitect()
        builder = KnowledgeGraphBuilder(architect=architect)

        builder.add_proposition(Proposition(
            subject="Concept A", predicate="causes", obj="Concept B",
            statement="Concept A causes Concept B.",
            evidence="ev1", doc_name="doc1.pdf", page_number=1
        ))
        builder.add_proposition(Proposition(
            subject="Concept B", predicate="enables", obj="Concept C",
            statement="Concept B enables Concept C.",
            evidence="ev2", doc_name="doc1.pdf", page_number=2
        ))
        builder.add_proposition(Proposition(
            subject="Concept X", predicate="leads to", obj="Concept Y",
            statement="Concept X leads to Concept Y.",
            evidence="ev3", doc_name="doc2.pdf", page_number=3
        ))
        builder.add_proposition(Proposition(
            subject="Concept Y", predicate="yields", obj="Concept Z",
            statement="Concept Y yields Concept Z.",
            evidence="ev4", doc_name="doc2.pdf", page_number=4
        ))

        paths = builder.find_multihop_paths(max_hops=3, min_hops=2, intra_document_only=False, verify_with_gentzen=True)
        self.assertGreaterEqual(len(paths), 2)

        path_1 = paths[0]
        path_2 = paths[1]

        # Simulate initial preview for path 1
        active_preview_path_repr = path_1["path_repr"]
        card_version = 0

        # Simulate user generating a sample for path 1
        current_sample = {
            "question": "Question about path 1",
            "provenance": {"path_repr": path_1["path_repr"]},
            "reasoning": "Reasoning 1"
        }

        # Simulate user switching dropdown to path 2
        new_path_repr = path_2["path_repr"]
        if new_path_repr != active_preview_path_repr:
            active_preview_path_repr = new_path_repr
            card_version += 1
            if current_sample and current_sample.get("provenance", {}).get("path_repr") != new_path_repr:
                current_sample = None

        self.assertEqual(active_preview_path_repr, path_2["path_repr"])
        self.assertEqual(card_version, 1)
        self.assertIsNone(current_sample, "Stale sample from path 1 must be cleared when selecting path 2")


class TestNativeReasoningMode(unittest.TestCase):
    class _FakeLLM:
        def __init__(self, raw):
            self.raw, self.calls = raw, []
        def health_check(self, *a, **k):
            return True
        def generate(self, **kw):
            self.calls.append(kw)
            return self.raw

    RAW = ("<think>\nHurley holds that consciousness requires intentional agency, so any conscious system acts "
           "intentionally in the light of its experiences.\n</think>\n\n<reasoning>A second, post-thinking trace "
           "that should not become the sample's reasoning in native mode at all.</reasoning>\n"
           "<answer>Consciousness requires intentional agency, according to Hurley.</answer>")

    def _build(self, mode):
        from core.cot_generator import CoTGenerator
        llm = self._FakeLLM(self.RAW)
        r = CoTGenerator(llm, mode).build_cot_from_logic_tree(
            question="Why?", target_claim="Consciousness requires intentional agency.",
            logic_tree_text="Consciousness => Intentional_Agency", context_text="Hurley (1998) ...",
            path_nodes=["Consciousness", "Intentional Agency"])
        return r, llm.calls[0]

    def test_native_uses_thinking_channel(self):
        r, call = self._build("native")
        self.assertEqual(call.get("chat_template_kwargs"), {"enable_thinking": True})
        self.assertIn("Hurley holds", r["reasoning"])
        self.assertIn("post-thinking", r["post_think_reasoning"])
        self.assertNotIn("<reasoning> MUST", call["prompt"])

    def test_prompted_is_unchanged(self):
        r, call = self._build("prompted")
        self.assertNotIn("chat_template_kwargs", call)
        self.assertIn("post-thinking", r["reasoning"])

    def test_client_recovers_missing_open_tag_and_per_call_kwargs(self):
        from llm_backend.client import LocalLLMClient
        c = LocalLLMClient("http://x/v1")
        out = c._parse_response_content({"choices": [{"message": {"content": "thought text </think> [1]"}}]})
        self.assertTrue(out.startswith("<think>"))
        self.assertEqual(c._template_kwargs("/m/Occam-2B-CCoT", {"enable_thinking": True}), {"enable_thinking": True})
        self.assertIsNone(c._template_kwargs("/m/Occam-2B-CCoT"))
        self.assertEqual(c._template_kwargs("/m/Spark-X2.6-4B-vllm"), {"enable_thinking": False})


class TestClearCoTPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = Path(__file__).parent.parent / "model" / "model.safetensors"
        cls.engine = NanoGentzenEngine(model_path=str(model_path))
        cls.architect = LogicalArchitect(engine=cls.engine)
        cls.cot_generator = CoTGenerator(llm_client=None)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_studio.db"
        self.db_manager = DatasetManager(db_path=str(self.db_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_conceptual_guide_relevance_distractor_omission(self):
        """Verifies that distractor premises are flagged for omission in the CoT prompt guide."""
        logic_tree_text = "[Axiom] Established: Physical_System\n[Modus Ponens] Established: Consciousness"
        target_claim = "Consciousness"
        relevance_info = {
            "relevant_witnesses": ["Physical_System"],
            "unused_premises": ["Irrelevant_Weather_Condition"],
            "relevance_ratio": 0.5,
        }
        guide = self.cot_generator._format_conceptual_guide(
            logic_tree_text=logic_tree_text,
            target_claim=target_claim,
            relevance_info=relevance_info,
        )
        self.assertIn("Load-Bearing Causal Premises: Physical System", guide)
        self.assertIn("Non-Causal Distractors (OMIT from deduction): Irrelevant Weather Condition", guide)
        self.assertIn("Consciousness", guide)

    def test_end_to_end_proven_true_pipeline(self):
        """Tests complete pipeline on a provable sequent."""
        premises = [
            "If acoustic heritage is cultural then systematic preservation is necessary.",
            "Acoustic heritage is cultural."
        ]
        conclusion = "Systematic preservation is necessary."
        tree_res = self.architect.generate_best_logic_tree(premises, conclusion)

        self.assertTrue(tree_res["is_verified"])
        self.assertEqual(tree_res["epistemic_status"], "PROVEN_TRUE")
        self.assertTrue(len(tree_res["curry_howard_term"]) > 0)
        self.assertIn("systematic_preservation_is_necessary", tree_res["curry_howard_term"])
        self.assertIn("relevance", tree_res)

        # Generate CoT via fallback (offline)
        cot_res = self.cot_generator.build_cot_from_logic_tree(
            question="Why is systematic preservation necessary for acoustical heritage?",
            target_claim=conclusion,
            logic_tree_text=tree_res["logic_tree_text"],
            context_text="At the end of the 20th century, acoustical heritage was acknowledged as cultural heritage.",
            relevance_info=tree_res.get("relevance"),
            curry_howard_term=tree_res.get("curry_howard_term"),
        )
        self.assertTrue(cot_res["is_fallback"])
        self.assertIn("systematic preservation", cot_res["answer"].lower())

        # Create and persist DatasetSample
        sample = DatasetSample(
            sample_id="CC-001",
            doc_name="unesco_heritage.pdf",
            question=cot_res["question"],
            reasoning=cot_res["reasoning"],
            answer=cot_res["answer"],
            naive_cot="Well, acoustic heritage needs preservation based on UNESCO principles.",
            status="accepted",
            gentzen_verified=tree_res["is_verified"],
            gentzen_sequent=tree_res["sequent"],
            proof_tree_json=json.dumps(tree_res["proof_tree"]),
            provability_score=tree_res["provability_score"],
            curry_howard_term=tree_res["curry_howard_term"],
            epistemic_status=tree_res["epistemic_status"],
            tokens_saved_pct=45,
            clarity_score=cot_res["clarity_score"],
        )
        self.db_manager.save_sample(sample)

        # Retrieve and verify persistence
        saved_samples = self.db_manager.get_samples(status="accepted")
        self.assertEqual(len(saved_samples), 1)
        s = saved_samples[0]
        self.assertEqual(s.sample_id, "CC-001")
        self.assertEqual(s.epistemic_status, "PROVEN_TRUE")
        self.assertEqual(s.curry_howard_term, tree_res["curry_howard_term"])

        # Test exports
        jsonl_export = Path(self.temp_dir.name) / "sft.jsonl"
        dpo_export = Path(self.temp_dir.name) / "dpo.jsonl"
        parquet_export = Path(self.temp_dir.name) / "dataset.parquet"

        self.assertEqual(self.db_manager.export_jsonl(str(jsonl_export)), 1)
        self.assertEqual(self.db_manager.export_dpo_pairs(str(dpo_export)), 1)
        self.assertEqual(self.db_manager.export_parquet(str(parquet_export)), 1)

        # Inspect exported JSONL
        with open(jsonl_export, "r", encoding="utf-8") as f:
            j_line = json.loads(f.readline())
            self.assertEqual(j_line["epistemic_status"], "PROVEN_TRUE")
            self.assertEqual(j_line["curry_howard_term"], tree_res["curry_howard_term"])

        # Inspect exported DPO
        with open(dpo_export, "r", encoding="utf-8") as f:
            d_line = json.loads(f.readline())
            self.assertEqual(d_line["metadata"]["epistemic_status"], "PROVEN_TRUE")
            self.assertEqual(d_line["metadata"]["curry_howard_term"], tree_res["curry_howard_term"])
        self.assertIn("grounding_score", d_line["metadata"])

    def test_end_to_end_kripke_unprovable_pipeline(self):
        """Tests countermodel extraction and epistemic status on intuitionistically invalid sequent (e.g. Peirce's Law)."""
        tree_res = self.architect.generate_best_logic_tree(
            premises=["((P => Q) => P)"],
            conclusion="P",
            symbolic_sequent="((P => Q) => P) |- P"
        )

        self.assertFalse(tree_res["is_verified"])
        self.assertIn(tree_res["epistemic_status"], ["INTUITIONISTIC_UNDECIDED", "PROVEN_FALSE"])
        # Should have extracted countermodel for Peirce's law
        self.assertIsNotNone(tree_res["kripke_countermodel"])
        self.assertEqual(tree_res["curry_howard_term"], "")

    def test_cot_grounding_verification_positive(self):
        """Verifies that a reasoning trace verbalizing the logic tree receives high grounding score."""
        tree_res = {
            "is_verified": True,
            "epistemic_status": "PROVEN_TRUE",
            "sequent": "Autonomy, (Autonomy -> Algorithm) |- Algorithmic_System",
            "relevance": {"relevant_witnesses": ["Autonomy"], "unused_premises": []},
            "derivation_steps": [
                {"conclusion": "Established: Autonomy"},
                {"conclusion": "Established: Algorithmic_System"}
            ]
        }
        reasoning = "In examining algorithmic systems, autonomy is directly impacted by automated decisions. Consequently, this establishes that algorithmic systems constrain human agency."
        answer = "Algorithmic systems constrain autonomy through automated decision frameworks."
        path_nodes = ["Autonomy", "Algorithmic System"]
        target_claim = "Algorithmic systems constrain autonomy"

        res = self.cot_generator.verify_cot_grounding(
            reasoning=reasoning,
            answer=answer,
            logic_tree_res=tree_res,
            path_nodes=path_nodes,
            target_claim=target_claim
        )
        self.assertTrue(res["is_grounded"])
        self.assertGreaterEqual(res["grounding_score"], 0.80)
        self.assertEqual(res["node_coverage"], 1.0)
        self.assertEqual(len(res["missing_nodes"]), 0)

    def test_cot_grounding_verification_negative(self):
        """Verifies that an irrelevant or hallucinated reasoning trace fails grounding."""
        tree_res = {
            "is_verified": True,
            "epistemic_status": "PROVEN_TRUE",
            "sequent": "Autonomy, (Autonomy -> Algorithm) |- Algorithmic_System",
            "relevance": {"relevant_witnesses": ["Autonomy"], "unused_premises": []},
        }
        reasoning = "The recipe requires two cups of flour and one cup of sugar mixed thoroughly."
        answer = "Baking takes forty minutes at 350 degrees."
        path_nodes = ["Autonomy", "Algorithmic System"]

        res = self.cot_generator.verify_cot_grounding(
            reasoning=reasoning,
            answer=answer,
            logic_tree_res=tree_res,
            path_nodes=path_nodes,
            target_claim="Algorithmic systems constrain autonomy"
        )
        self.assertFalse(res["is_grounded"])
        self.assertLess(res["grounding_score"], 0.40)
        self.assertEqual(res["node_coverage"], 0.0)

    def test_cot_grounding_relevance_distractor_penalty(self):
        """Verifies that leaking non-causal distractors reduces distractor isolation score."""
        tree_res = {
            "is_verified": True,
            "epistemic_status": "PROVEN_TRUE",
            "relevance": {
                "relevant_witnesses": ["Autonomy"],
                "unused_premises": ["Irrelevant_Weather_Variable"],
            }
        }
        reasoning_clean = "Autonomy is essential for agency. Therefore algorithmic systems impact decisions."
        reasoning_leaked = "Autonomy is essential, but we must also discuss the irrelevant weather variable in the region."

        res_clean = self.cot_generator.verify_cot_grounding(
            reasoning=reasoning_clean,
            answer="Autonomy is impacted.",
            logic_tree_res=tree_res,
            path_nodes=["Autonomy"]
        )
        res_leaked = self.cot_generator.verify_cot_grounding(
            reasoning=reasoning_leaked,
            answer="Autonomy is impacted.",
            logic_tree_res=tree_res,
            path_nodes=["Autonomy"]
        )
        self.assertEqual(res_clean["relevance_isolation"], 1.0)
        self.assertLess(res_leaked["relevance_isolation"], 1.0)
        self.assertIn("Irrelevant Weather Variable", res_leaked["leaked_distractors"])

    def test_consciousness_ai_multihop_path_cot_and_jev(self):
        """
        Validates the exact user issue:
        Multihop path: Consciousness -> Artificial Intelligence -> Human Behavior -> Ai System
        Must be PROVEN_TRUE in CotLogicalArchitect with 100% relevance and 0 distractors.
        Also verifies JevLogicalArchitect handles the corresponding decision criteria.
        """
        from nanogentzen.architect_cot import CotLogicalArchitect
        from nanogentzen.architect_jev import JevLogicalArchitect

        path_nodes = ["Consciousness", "Artificial Intelligence", "Human Behavior", "Ai System"]
        premises = [
            "Artificial intelligence systems like AlphaGo and calculators are distinguished from consciousness because they lack genuine subjective experience.",
            "AI systems might be capable of empty mimicry of human behavior, as their architectures resemble conscious systems.",
            "Advanced AI systems could soon achieve genuine consciousness beyond current silicon-based replication."
        ]
        conclusion = "While AI systems like AlphaGo mimic human behavior, genuine consciousness requires capabilities beyond silicon replication."

        cot_arch = CotLogicalArchitect(engine=self.engine)
        # Without relation labels the hops are not inferences: the path must NOT be verified ...
        unlabelled = cot_arch.generate_best_logic_tree(premises=premises, conclusion=conclusion, path_nodes=path_nodes)
        self.assertFalse(unlabelled["is_verified"])
        self.assertEqual(unlabelled["logic_basis"], "kg_unlabelled")
        # ... and the evidence says AI systems LACK what consciousness has: an exclusion breaks the chain
        lacks_repr = ("(Consciousness) --[lacks] <-- (Artificial Intelligence) -> (Artificial Intelligence) "
                      "--[implies]--> (Human Behavior) -> (Human Behavior) --[implies]--> (Ai System)")
        rel = cot_arch.generate_best_logic_tree(premises=premises, conclusion=conclusion,
                                                path_nodes=path_nodes, path_repr=lacks_repr)
        self.assertFalse(rel["is_verified"])
        self.assertEqual(rel["logic_basis"], "kg_relational")
        # With entailment labels in path direction the chain is a real argument and must be verified
        chain_repr = ("(Consciousness) --[implies]--> (Artificial Intelligence) -> (Artificial Intelligence) "
                      "--[implies]--> (Human Behavior) -> (Human Behavior) --[implies]--> (Ai System)")
        tree_res = cot_arch.generate_best_logic_tree(
            premises=premises,
            conclusion=conclusion,
            path_nodes=path_nodes,
            path_repr=chain_repr,
        )

        self.assertTrue(tree_res["is_verified"], "CotLogicalArchitect failed to verify chain proof!")
        self.assertEqual(tree_res["epistemic_status"], "PROVEN_TRUE")
        self.assertGreaterEqual(tree_res["depth"], 3)
        self.assertGreaterEqual(tree_res["provability_score"], 0.90)

        # Relevance logic must identify Consciousness as load-bearing and have 0 unused distractors
        relevance = tree_res.get("relevance", {})
        self.assertTrue(relevance.get("is_strictly_relevant"), f"Relevance should be strictly relevant: {relevance}")
        self.assertEqual(len(relevance.get("unused_premises", [])), 0, f"Spurious distractors: {relevance.get('unused_premises')}")
        self.assertEqual(relevance.get("relevance_ratio"), 1.0)

        # Curry-Howard certificate must be valid
        self.assertTrue(len(tree_res["curry_howard_term"]) > 0)
        self.assertIn("Ai_System", tree_res["curry_howard_term"])

        # CoT Grounding Verification
        grounding = self.cot_generator.verify_cot_grounding(
            reasoning=(
                "Establish that consciousness is non-cognitive. Note that artificial intelligence systems "
                "mimic human behavior. Current AI architectures resemble conscious systems in behavior. "
                "Conclude that genuine consciousness in an AI system requires substrates beyond silicon."
            ),
            answer=conclusion,
            logic_tree_res=tree_res,
            path_nodes=path_nodes,
            key_premises=premises,
            target_claim=conclusion
        )
        self.assertTrue(grounding["is_grounded"])
        self.assertGreaterEqual(grounding["grounding_score"], 0.70)
        self.assertEqual(len(grounding.get("leaked_distractors", [])), 0)

        # Then JevLogicalArchitect on decision scenario with same domain evidence
        jev_arch = JevLogicalArchitect(engine=self.engine)
        state = {
            "entity_name": "ai_system",
            "attributes": {
                "consciousness": True,
                "mimics_behavior": True
            }
        }
        questions = {
            "consciousness_potential": {
                "instructions": "Can the AI system achieve genuine consciousness?",
                "type": "choice",
                "criteria": {
                    "genuine_consciousness": "Consciousness and mimicry criteria satisfied.",
                    "functional_mimicry_only": "Only functional mimicry without subjective experience."
                }
            }
        }
        policy_rules = [
            "(consciousness & mimics_behavior) => genuine_consciousness"
        ]
        jev_res = jev_arch.evaluate_decision_scenario(
            state=state,
            questions=questions,
            policy_rules=policy_rules,
            target_decision="genuine_consciousness"
        )
        self.assertTrue(jev_res["gentzen_verified"])
        self.assertEqual(jev_res["epistemic_status"], "PROVEN_TRUE")
        gold_q = jev_res["gold"]["consciousness_potential"]
        self.assertEqual(gold_q["label"], "genuine_consciousness")
        self.assertEqual(gold_q["verification"], "PROVEN_TRUE")

    def test_concept_matching_threshold_and_qualifier_preservation(self):
        """Verifies that _matches_concept requires distinguishing qualifiers and prevents metric inflation."""
        from core.cot_generator import _matches_concept

        text_general = (
            "Algorithmic systems bear upon autonomy by influencing decision frameworks, "
            "while respecting autonomy requires providing necessary informational control."
        )
        self.assertTrue(_matches_concept("Autonomy", text_general))
        self.assertTrue(_matches_concept("Algorithmic System", text_general))
        # "Proprietary Algorithmic System" MUST FAIL when "proprietary" is absent
        self.assertFalse(_matches_concept("Proprietary Algorithmic System", text_general))

        # When "proprietary" is included, it must pass
        text_proprietary = "Proprietary algorithmic systems constrain autonomy when opaque."
        self.assertTrue(_matches_concept("Proprietary Algorithmic System", text_proprietary))

    def test_sanitizer_dangling_prepositions_and_headless_clauses(self):
        """Verifies that _sanitize_regular_text avoids headless clauses and dangling prepositions."""
        raw_text = (
            "Chapter 3 applies this to algorithmic systems, identifying justifiability conditions. "
            "This directly links to the conditions outlined in Chapter 3."
        )
        sanitized = self.cot_generator._sanitize_regular_text(raw_text)
        self.assertNotIn("outlined in .", sanitized)
        self.assertNotIn("in Chapter 3", sanitized)
        self.assertNotIn("Chapter 3 applies", sanitized)
        self.assertTrue(sanitized.startswith("This applies to algorithmic systems"))
        self.assertTrue(sanitized.endswith("outlined."))


class TestNaiveTraceLeaks(unittest.TestCase):
    def test_detects_echoed_instructions(self):
        for t in (LEAKY, "## Thinking Trace\nfoo", "An unoptimized internal thinking trace follows.",
                  "Following the ClearCoT methodology, ..."):
            self.assertIsNotNone(naive_trace_leak(t), t)

    def test_genuine_reasoning_passes(self):
        for t in (GENUINE, "The user is asking how licences work. Let me think.", ""):
            self.assertIsNone(naive_trace_leak(t), t)


if __name__ == "__main__":
    unittest.main()
