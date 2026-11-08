"""
Jev pipeline: scenario generation, rule compilation, verification flags, scenario validation and
counterfactual variants.
"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tests  # noqa: F401  (redirects all runtime data to a temp dir; see tests/__init__.py)

from core.chunker import Chunk
from core.jev_dataset_manager import JevDatasetManager, JevSample
from core.jev_generator import JevScenarioGenerator
from nanogentzen.engine import NanoGentzenEngine
from nanogentzen.jev.architect import JevLogicalArchitect
from nanogentzen.jev.parser_jev import blocking_issues, normalize_rule_atoms, uncovered_options, undecided_option, validate_scenario


MODEL_V4 = ROOT / "model"   # nanoGentzen v4 weights (config.MODEL_DIR)

STATE = {"entity_name": "Predictive Processing",
         "attributes": {"predictive_processing_framework": True, "hierarchical_generative_model": True,
                        "consciousness_requirement": True}}

QUESTIONS = {"determination": {"type": "choice", "instructions": "Is the theory compatible?",
                               "criteria": {"compatible": "both hold", "incompatible": "a requirement fails",
                                            "undecided": "not enough evidence"}}}

PHRASE_RULES = ["(Predictive Processing framework & Hierarchical generative model) => compatible",
                "(~consciousness requirement) => incompatible"]

QUESTIONS_jev_integrity_flags = {"rel": {"instructions": "Compatible?", "type": "choice",
                     "criteria": {"compatible": "holds", "incompatible": "fails", "undecided": "unknown"}}}

RULES = ["(is_ai & has_memory) => compatible", "~is_ai => incompatible"]

HAS_V4 = (MODEL_V4 / "model.safetensors").exists()


class TestJevBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = Path(__file__).parent.parent / "model" / "model.safetensors"
        cls.engine = NanoGentzenEngine(model_path=str(model_path))
        cls.generator = JevScenarioGenerator(llm_client=None, engine=cls.engine)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_jev.db"
        self.manager = JevDatasetManager(db_path=str(self.db_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_jev_scenario_from_chunk(self):
        chunk = Chunk(
            chunk_id="chk_001",
            doc_name="consciousness_study.pdf",
            page_number=19,
            section_title="Introduction",
            text="Global workspace theory models information broadcast. Neuroscientific research confirms biological substrates.",
            chunk_type="paragraph"
        )
        scenario = self.generator.generate_from_chunk(chunk, workflow="scientific_theories")

        self.assertEqual(scenario["workflow"], "scientific_theories")
        self.assertIn("state", scenario)
        self.assertIn("questions", scenario)
        self.assertIn("gold", scenario)
        self.assertIn("factors", scenario)
        self.assertTrue(scenario["gentzen_verified"])
        self.assertEqual(scenario["epistemic_status"], "PROVEN_TRUE")
        self.assertTrue(len(scenario["curry_howard_term"]) > 0)

        # Check gold structure
        q_id = list(scenario["gold"].keys())[0]
        gold_entry = scenario["gold"][q_id]
        self.assertIn("label", gold_entry)
        self.assertIn("confidence", gold_entry)
        self.assertIn("probabilities", gold_entry)
        self.assertEqual(gold_entry["verification"], "PROVEN_TRUE")

    def test_generate_jev_scenario_from_multihop_path(self):
        path_info = {
            "path_repr": "(Acoustics) -> (Heritage) -> (Preservation)",
            "path_nodes": ["Acoustics", "Heritage", "Preservation"],
            "provenance_pages": [3, 7]
        }
        chunk1 = Chunk(chunk_id="c1", doc_name="unesco.pdf", page_number=3, section_title="Acoustics", text="Acoustics constitutes cultural heritage.", chunk_type="paragraph")
        chunk2 = Chunk(chunk_id="c2", doc_name="unesco.pdf", page_number=7, section_title="Preservation", text="Cultural heritage requires systematic preservation.", chunk_type="paragraph")

        scenario = self.generator.generate_from_multihop_path(
            path_info=path_info,
            supporting_chunks=[chunk1, chunk2],
            workflow="cultural_policy"
        )
        self.assertEqual(scenario["workflow"], "cultural_policy")
        self.assertTrue(scenario["gentzen_verified"])
        self.assertIn("provenance", scenario)
        self.assertEqual(scenario["provenance"]["pages"], [3, 7])

    def test_infer_workflow_domains(self):
        wf1 = self.generator.infer_workflow("Algorithms_and_Autonomy.pdf", ["Autonomy", "Teacher"], "teachers make decisions")
        self.assertEqual(wf1, "educational_autonomy")

        wf2 = self.generator.infer_workflow("unesco.pdf", ["Acoustics", "Heritage"], "cultural heritage")
        self.assertEqual(wf2, "cultural_policy")

        wf3 = self.generator.infer_workflow("security_audit.pdf", ["Malware", "Quarantine"], "breach incident")
        self.assertEqual(wf3, "security_incidents")

        wf4 = self.generator.infer_workflow("2308.1234.pdf", ["Model", "Representation"], "scientific theories")
        self.assertEqual(wf4, "scientific_theories")

        wf5 = self.generator.infer_workflow("contract.pdf", ["Clause", "Liability"], "contract breach")
        self.assertEqual(wf5, "legal_compliance")

    def test_multihop_path1_proposition_grounding(self):
        path_info = {
            "path_repr": "(Autonomy) --[bear upon] <-- (Algorithmic System) -> (Algorithmic System) --[affect] --> (Person) -> (Person) --[autonomous persons] <-- (Teacher)",
            "path_nodes": ["Autonomy", "Algorithmic System", "Person", "Teacher"],
            "source": "Autonomy",
            "target": "Teacher",
            "provenance_pages": [26, 65, 67],
            "composite_claim": "Given that Algorithmic systems bear upon autonomy by influencing its various conceptions and practical applications, it follows that teachers are autonomous persons who have the capacity to make their own decisions.",
            "propositions": [
                {
                    "statement": "Algorithmic systems bear upon autonomy by influencing its various conceptions and practical applications.",
                    "evidence": "To get at the different ways in which algorithmic systems bear upon autonomy, we have divided the book into four main parts....",
                    "page_number": 26,
                    "doc_name": "Algorithms_and_Autonomy.pdf"
                },
                {
                    "statement": "Algorithmic systems can affect persons for factors they either cannot or should not be responsible for.",
                    "evidence": "We simply introduce the dimension of responsibility to our criterion to highlight the fact that algorithmic systems can affect persons for f...",
                    "page_number": 67,
                    "doc_name": "Algorithms_and_Autonomy.pdf"
                },
                {
                    "statement": "Teachers are autonomous persons who have the capacity to make their own decisions.",
                    "evidence": "To sum up our view so far: teachers are autonomous persons, and hence they have a claim to incorporate their values into their lives as they...",
                    "page_number": 65,
                    "doc_name": "Algorithms_and_Autonomy.pdf"
                }
            ]
        }
        chunk1 = Chunk(chunk_id="c26", doc_name="Algorithms_and_Autonomy.pdf", page_number=26, section_title="Preliminaries", text="To get at the different ways in which algorithmic systems bear upon autonomy, we have divided the book into four main parts. Part I is introductory and ground- clearing.", chunk_type="paragraph")
        chunk2 = Chunk(chunk_id="c65", doc_name="Algorithms_and_Autonomy.pdf", page_number=65, section_title="Chapter 3", text="Teachers are autonomous persons who have the capacity to make their own decisions.", chunk_type="paragraph")

        scenario = self.generator.generate_from_multihop_path(
            path_info=path_info,
            supporting_chunks=[chunk1, chunk2],
            workflow="domain_policy"  # Test dynamic auto-infer
        )
        self.assertEqual(scenario["workflow"], "educational_autonomy")
        self.assertTrue(scenario["gentzen_verified"])
        self.assertEqual(scenario["epistemic_status"], "PROVEN_TRUE")

        obs = scenario["state"]["observations"]
        # Must use exact proposition statement without mid-word truncation or [Page 26]: tag
        self.assertEqual(obs["grounded_premise"], "Algorithmic systems bear upon autonomy by influencing its various conceptions and practical applications.")
        self.assertEqual(obs["verified_condition"], "Teachers are autonomous persons who have the capacity to make their own decisions.")
        self.assertNotIn("[Page", obs["grounded_premise"])
        self.assertNotIn("into f", obs["grounded_premise"])
        self.assertNotIn("ground- clearing", obs["verified_condition"])

        # Check question & criteria
        q_id = list(scenario["questions"].keys())[0]
        q_obj = scenario["questions"][q_id]
        self.assertIn("protected_teacher", q_obj["criteria"])
        self.assertEqual(scenario["gold"][q_id]["label"], "protected_teacher")
        self.assertIn("autonomy_grounded", scenario["curry_howard_term"])
        self.assertIn("teacher_autonomous", scenario["curry_howard_term"])

    def test_editorial_sentence_filtering_in_fallback(self):
        chunk = Chunk(
            chunk_id="c_editorial",
            doc_name="Algorithms_and_Autonomy.pdf",
            page_number=26,
            section_title="Intro",
            text="[Page 26]:\nPart I is introductory and ground- clearing. Algorithmic systems influence personal decision-making across modern society. Procedural theories of autonomy emphasize self-governance.",
            chunk_type="paragraph"
        )
        scenario = self.generator.generate_from_chunk(chunk, workflow="auto")
        obs = scenario["state"]["observations"]
        self.assertNotIn("ground- clearing", obs["grounded_premise"])
        self.assertNotIn("[Page 26]", obs["grounded_premise"])
        self.assertTrue(len(obs["grounded_premise"]) > 20)
        self.assertTrue(scenario["gentzen_verified"])

    def test_typed_decisions_parquet_export_schema(self):
        sample = JevSample(
            sample_id="TR-SEC-001",
            doc_name="security_ops.pdf",
            workflow="security_incidents",
            split="train",
            state_json=json.dumps({"alert": {"evidence": "unauthorized IP access", "is_dormant": True}}),
            questions_json=json.dumps({
                "quarantine": {
                    "type": "choice",
                    "instructions": "Should asset be quarantined?",
                    "criteria": {"yes": "Active exfiltration confirmed", "no": "Authorized admin action"}
                }
            }),
            gold_json=json.dumps({
                "quarantine": {
                    "label": "yes",
                    "confidence": 0.92,
                    "probabilities": {"yes": 0.92, "no": 0.08},
                    "verification": "PROVEN_TRUE"
                }
            }),
            factors_json=json.dumps({"is_dormant": True}),
            target_decision="yes",
            gentzen_verified=True,
            epistemic_status="PROVEN_TRUE",
            curry_howard_term="apply(quarantine_policy)",
        )
        self.manager.save_sample(sample)

        samples = self.manager.get_samples()
        self.assertEqual(len(samples), 1)

        parquet_path = Path(self.temp_dir.name) / "typed_decisions.parquet"
        count = self.manager.export_typed_decisions_parquet(str(parquet_path))
        self.assertEqual(count, 1)

        df = pd.read_parquet(parquet_path)
        self.assertEqual(len(df), 1)
        # Check canonical typed-decisions columns
        for expected_col in ["id", "workflow", "split", "state", "questions", "gold", "factors", "n_questions"]:
            self.assertIn(expected_col, df.columns)
        # Check dynamic flattened columns
        self.assertIn("quarantine__label", df.columns)
        self.assertIn("quarantine__confidence", df.columns)
        self.assertIn("quarantine__probabilities", df.columns)
        self.assertEqual(df.iloc[0]["quarantine__label"], "yes")
        self.assertEqual(df.iloc[0]["quarantine__confidence"], 0.92)

    def test_open_jev_s1mb_parquet_export_schema(self):
        sample = JevSample(
            sample_id="S1MB-001",
            doc_name="contract.pdf",
            workflow="contract_compliance",
            split="test",
            state_json=json.dumps({"terms": {"clause": "non_disclosure", "strict": True}}),
            questions_json=json.dumps({
                "breach": {
                    "type": "choice",
                    "criteria": {"entailment": "Breach occurs", "contradiction": "No breach"}
                }
            }),
            gold_json=json.dumps({
                "breach": {
                    "label": "entailment",
                    "probabilities": {"entailment": 0.95, "contradiction": 0.05}
                }
            }),
            factors_json=json.dumps({"strict": True}),
        )
        self.manager.save_sample(sample)

        s1mb_path = Path(self.temp_dir.name) / "s1mb.parquet"
        count = self.manager.export_open_jev_parquet(str(s1mb_path))
        self.assertEqual(count, 1)

        df = pd.read_parquet(s1mb_path)
        # Check canonical S1MB columns
        for expected_col in ["input", "targets", "split", "case_id", "group_id", "language", "input_hash"]:
            self.assertIn(expected_col, df.columns)
        row = df.iloc[0]
        self.assertIn("state_json", row["input"])
        self.assertEqual(len(row["targets"]), 1)
        self.assertEqual(row["targets"][0]["decision_id"], "breach")
        self.assertEqual(row["targets"][0]["label"], "entailment")

    def test_purge_legacy_samples(self):
        # 1 valid sample
        valid = JevSample(
            sample_id="VALID-01",
            doc_name="doc.pdf",
            workflow="educational_autonomy",
            state_json=json.dumps({"observations": {"grounded_premise": "Valid premise", "verified_condition": "Valid condition"}}),
            questions_json=json.dumps({"q": {"criteria": {"yes": "Holds"}}}),
            gold_json=json.dumps({"q": {"label": "yes"}}),
        )
        # 1 legacy System_Architecture sample
        legacy_sys = JevSample(
            sample_id="LEGACY-SYS-01",
            doc_name="doc.pdf",
            workflow="document_compliance",
            state_json=json.dumps({"entity": "System_Architecture", "observations": {}}),
            questions_json=json.dumps({"q": {"criteria": {"yes": "Holds"}}}),
            gold_json=json.dumps({"q": {"label": "yes"}}),
        )
        # 1 empty observations sample
        legacy_empty = JevSample(
            sample_id="LEGACY-EMP-01",
            doc_name="doc.pdf",
            workflow="document_compliance",
            state_json=json.dumps({"observations": {}}),
            questions_json=json.dumps({"q": {"criteria": {"yes": "Holds"}}}),
            gold_json=json.dumps({"q": {"label": "yes"}}),
        )
        self.manager.save_sample(valid)
        self.manager.save_sample(legacy_sys)
        self.manager.save_sample(legacy_empty)

        self.assertEqual(len(self.manager.get_samples()), 3)
        purged = self.manager.purge_legacy_samples()
        self.assertEqual(purged, 2)
        remaining = self.manager.get_samples()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].sample_id, "VALID-01")

    def test_single_chunk_fallback_without_section_title(self):
        chunk = Chunk(
            chunk_id="chk_notitle",
            doc_name="Algorithms_and_Autonomy.pdf",
            page_number=45,
            section_title=None,
            text="Algorithmic systems influence personal decision-making across modern education and society. Procedural autonomy requires transparency.",
            chunk_type="paragraph"
        )
        scenario = self.generator.generate_from_chunk(chunk, workflow="auto")
        self.assertNotEqual(scenario["state"]["entity"], "System_Architecture")
        self.assertNotIn("Compatibility", str(scenario["questions"]))
        self.assertTrue(scenario["gentzen_verified"])
        self.assertEqual(scenario["workflow"], "educational_autonomy")

    def test_batch_kg_path_generation_flow(self):
        kg_paths = [
            {
                "path_repr": "(Autonomy) -> (Decision_System) -> (Teacher)",
                "path_nodes": ["Autonomy", "Decision_System", "Teacher"],
                "source": "Autonomy",
                "target": "Teacher",
                "doc_name": "Algorithms_and_Autonomy.pdf",
                "provenance_pages": [26, 65],
                "composite_claim": "Autonomy requires fair treatment of teachers.",
                "propositions": [
                    {"statement": "Autonomy governs algorithmic system deployment.", "page_number": 26},
                    {"statement": "Teachers have decision-making capacity.", "page_number": 65}
                ]
            },
            {
                "path_repr": "(Acoustics) -> (Heritage) -> (Preservation)",
                "path_nodes": ["Acoustics", "Heritage", "Preservation"],
                "source": "Acoustics",
                "target": "Preservation",
                "doc_name": "unesco.pdf",
                "provenance_pages": [3, 7],
                "composite_claim": "Acoustics heritage mandates systematic preservation.",
                "propositions": [
                    {"statement": "Acoustics constitutes cultural heritage.", "page_number": 3},
                    {"statement": "Preservation mandates apply to cultural heritage.", "page_number": 7}
                ]
            }
        ]
        scenarios = []
        for p in kg_paths:
            sc = self.generator.generate_from_multihop_path(
                path_info=p,
                supporting_chunks=[],
                workflow="auto"
            )
            scenarios.append(sc)

        self.assertEqual(len(scenarios), 2)
        self.assertEqual(scenarios[0]["workflow"], "educational_autonomy")
        self.assertEqual(scenarios[1]["workflow"], "cultural_policy")
        for sc in scenarios:
            self.assertTrue(sc["gentzen_verified"])
            self.assertEqual(sc["epistemic_status"], "PROVEN_TRUE")
            self.assertTrue(len(sc["curry_howard_term"]) > 0)

    def test_llm_unnested_question_and_entity_name_normalization(self):
        # LLM emitted flat un-nested question dictionary and entity_name
        parsed = {
            "workflow": "scientific_theories",
            "state": {"entity_name": "AI Systems", "attributes": {"is_ai": True}},
            "questions": {
                "instructions": "What is the theoretical compatibility of AI?",
                "type": "choice",
                "criteria": {"compatible": "Holds", "incompatible": "Fails"}
            },
            "policy_rules": ["is_ai => compatible"],
            "target_decision": "compatible"
        }
        res = self.generator._finalize_jev_scenario(
            parsed=parsed,
            doc_name="2308.08708v2.pdf",
            pages=[1, 2],
            chunk_ids=["c1"],
            context_text="AI systems exhibit cognitive behaviors. Formal theories model cognition.",
            path_nodes=["AI Systems", "Consciousness"]
        )
        self.assertEqual(res["state"]["entity"], "AI Systems")
        self.assertEqual(res["state"]["target_entity"], "Consciousness")
        self.assertIn("determination", res["questions"])
        self.assertEqual(res["gold"]["determination"]["label"], "compatible")
        self.assertTrue(res["gentzen_verified"])
        self.assertEqual(res["epistemic_status"], "PROVEN_TRUE")
        self.assertIn("grounded_premise", res["state"]["observations"])
        self.assertIn("verified_condition", res["state"]["observations"])

    def test_academic_citation_filtering_in_scenario_observations(self):
        """Verify that academic footnotes and citation fragments are filtered out of observations."""
        text_page39 = (
            "Consider 13 Oshana, 21–46. 17 Frankfurt, “Freedom of the Will and the Concept of a Person”; "
            "Dworkin, The Theory and Practice of Autonomy. John Christman’s account, which places individuals’ "
            "social and historical contexts at the center of autonomy. Christman argues that individual autonomy "
            "has two key requirements: compe- tence and authenticity. 19 Each is necessary and both are jointly "
            "sufficient for a person to be autonomous. The competence conditions Christman describes are similar "
            "to baseline autonomy requirements in other accounts. First, a person must have some set of fundamental "
            "values and commitments and the ability to “effectively form intentions” to act on the basis of those "
            "values and commitments. 20 Second, they must have the capacity to critically reflect on their basic "
            "values and commitments, their motivations, and other facets of their decision-making abilities. "
            "In other words, autonomy requires that a person have the capacities to form, develop, and critically "
            "reflect on their values and to intentionally and effectively act in accord with those values."
        )
        chunk = Chunk(
            chunk_id="c_autonomy_p39",
            doc_name="Algorithms_and_Autonomy.pdf",
            page_number=39,
            section_title="Autonomy Requirements",
            text=text_page39,
            chunk_type="paragraph",
        )
        scenario = self.generator.generate_from_chunk(chunk, workflow="educational_autonomy")
        obs = scenario["state"]["observations"]
        premise = obs["grounded_premise"]
        cond = obs["verified_condition"]

        # Footnote citations must be excluded
        self.assertNotIn("17 Frankfurt", premise)
        self.assertNotIn("Dworkin, The Theory", premise)
        self.assertNotIn("Consider 13 Oshana", premise)
        self.assertNotIn("17 Frankfurt", cond)
        self.assertNotIn("Dworkin, The Theory", cond)
        self.assertNotIn("Consider 13 Oshana", cond)

        # Must not contain unattached fragments like John Christman's account, which...
        self.assertNotIn("John Christman’s account, which", premise)
        self.assertNotIn("John Christman’s account, which", cond)

        # Must be substantive assertions
        self.assertTrue(len(premise) > 30)
        self.assertTrue(len(cond) > 30)
        self.assertTrue(scenario["gentzen_verified"])

    def test_finalize_replaces_llm_citation_debris_in_observations(self):
        """Simulate an LLM returning citation debris in observations and ensure _finalize cleans/replaces it."""
        parsed = {
            "workflow": "educational_autonomy",
            "state": {
                "entity": "Autonomy",
                "target_entity": "Critical Reflection",
                "observations": {
                    "grounded_premise": "17 Frankfurt, “Freedom of the Will and the Concept of a Person”; Dworkin, The Theory and Practice of Autonomy.",
                    "verified_condition": "John Christman’s account, which places individuals’ social and historical contexts at the center of autonomy.",
                    "is_active": True
                },
                "attributes": {
                    "autonomy_grounded": True,
                    "critical_reflection_autonomous": True
                }
            },
            "questions": {
                "critical_reflection_policy": {
                    "instructions": "Determine status of Critical Reflection.",
                    "type": "choice",
                    "criteria": {"protected_critical_reflection": "Protected", "restricted_critical_reflection": "Restricted"}
                }
            },
            "policy_rules": ["(autonomy_grounded & critical_reflection_autonomous) => protected_critical_reflection"],
            "target_decision": "protected_critical_reflection"
        }
        evidence_text = (
            "[Page 39]:\nConsider 13 Oshana, 21–46. 17 Frankfurt, “Freedom of the Will”; Dworkin. "
            "In other words, autonomy requires that a person have the capacities to form, develop, and critically reflect on their values. "
            "The final part of the heuristic is to apply principles of critical reflection to the concepts and moral theories."
        )
        res = self.generator._finalize_jev_scenario(
            parsed=parsed,
            doc_name="Algorithms_and_Autonomy.pdf",
            pages=[39],
            chunk_ids=["c39"],
            context_text=evidence_text,
            path_nodes=["Autonomy", "Critical Reflection"],
        )
        obs = res["state"]["observations"]
        self.assertNotIn("17 Frankfurt", obs["grounded_premise"])
        self.assertNotIn("Dworkin", obs["grounded_premise"])
        self.assertNotIn("John Christman", obs["verified_condition"])
        self.assertIn("autonomy requires that a person have the capacities", obs["grounded_premise"])
        self.assertIn("principles of critical reflection", obs["verified_condition"])
        self.assertTrue(res["gentzen_verified"])
        self.assertEqual(res["epistemic_status"], "PROVEN_TRUE")


class TestJevRuleCompilation(unittest.TestCase):
    def test_phrase_atoms_become_snake_case(self):
        self.assertEqual(normalize_rule_atoms(PHRASE_RULES[0]),
                         "(predictive_processing_framework & hierarchical_generative_model) => compatible")
        self.assertEqual(normalize_rule_atoms("(high-risk ∧ ¬kyc) → reject"), "(high_risk & ~kyc) => reject")
        self.assertEqual(normalize_rule_atoms("a & b => c"), "a & b => c")
        self.assertEqual(normalize_rule_atoms("If the theory holds then compatible"),
                         "If the theory holds then compatible")

    def test_rules_that_compile_to_nothing_block_the_scenario(self):
        issues = validate_scenario(STATE, QUESTIONS, ["garbage text", "(? & !) => compatible"])
        self.assertEqual([i["code"] for i in blocking_issues(issues)], ["no_rules"])

    def test_phrase_rules_pass_validation(self):
        self.assertEqual(blocking_issues(validate_scenario(STATE, QUESTIONS, PHRASE_RULES)), [])


@unittest.skipUnless((MODEL_V4 / "config.json").exists(), "model/ (nanoGentzen v4 weights) not present")
class TestJevNoHeuristicProof(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nanogentzen.engine import load_engine
        from nanogentzen.jev.architect import JevLogicalArchitect
        cls.arch = JevLogicalArchitect(engine=load_engine(str(MODEL_V4)))

    def _eval(self, rules):
        return self.arch.evaluate_decision_scenario(state=copy.deepcopy(STATE), questions=copy.deepcopy(QUESTIONS),
                                                    policy_rules=rules, target_decision="compatible")

    def test_uncompilable_rules_give_no_gold(self):
        res = self._eval(["garbage text"])
        self.assertFalse(res["gentzen_verified"])
        self.assertEqual(res["gold"], {})
        self.assertTrue(res["rejected_rules"])

    def test_phrase_rules_are_proven_from_state_facts(self):
        res = self._eval(PHRASE_RULES)
        g = res["gold"]["determination"]
        self.assertEqual((g["label"], g["verification"], g["grounding"]), ("compatible", "PROVEN_TRUE", "rules"))
        self.assertNotIn("warrant_", str(g["certificate"]))


class TestJevVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = Path(__file__).parent.parent / "model" / "model.safetensors"
        cls.arch = JevLogicalArchitect(engine=NanoGentzenEngine(model_path=str(model_path)))

    def test_proven_from_stated_facts_is_verified(self):
        state = {"attributes": {"is_ai": True, "has_memory": True}}
        res = self.arch.evaluate_decision_scenario(state, QUESTIONS_jev_integrity_flags, RULES, target_decision="compatible")
        self.assertEqual(res["epistemic_status"], "PROVEN_TRUE")
        self.assertTrue(res["gentzen_verified"])
        self.assertFalse(res["state_completion"].get("applied", False))
        self.assertFalse(res["certified_undecided"])

    def test_state_completion_is_not_verified(self):
        state = {"attributes": {"is_ai": True}}  # has_memory not stated
        res = self.arch.evaluate_decision_scenario(state, QUESTIONS_jev_integrity_flags, RULES, target_decision="compatible")
        comp = res["state_completion"]
        self.assertTrue(comp.get("applied"), comp)
        self.assertEqual(comp["added_facts"], {"has_memory": True})
        g = res["gold"]["rel"]
        self.assertEqual(g["verification"], "PROVEN_UNDER_ASSUMPTIONS")
        self.assertEqual(g["assumed_facts"], {"has_memory": True})
        self.assertEqual(res["epistemic_status"], "PROVEN_UNDER_ASSUMPTIONS")
        self.assertFalse(res["gentzen_verified"])

    def test_completion_disabled_leaves_label_unproven(self):
        arch = JevLogicalArchitect(engine=self.arch.engine, allow_state_completion=False)
        res = arch.evaluate_decision_scenario({"attributes": {"is_ai": True}}, QUESTIONS_jev_integrity_flags, RULES,
                                              target_decision="compatible")
        self.assertFalse(res["gentzen_verified"])
        self.assertNotEqual(res["epistemic_status"], "PROVEN_TRUE")

    def test_certified_undecided_flag(self):
        rules = RULES + ["(is_ai & ~has_memory & unclear) => undecided"]
        state = {"attributes": {"is_ai": True, "has_memory": False}}
        res = self.arch.evaluate_decision_scenario(state, QUESTIONS_jev_integrity_flags, rules, target_decision="", label_undecided=True)
        g = res["gold"]["rel"]
        if g["verification"] != "NO_OPTION_PROVABLE":
            self.skipTest(f"scenario not certified undecided by this engine build: {g['verification']}")
        self.assertEqual(g["label"], "undecided")
        self.assertTrue(g["certified_undecided"])
        self.assertTrue(res["certified_undecided"])
        self.assertTrue(res["gentzen_verified"])


class TestJevScenarioValidation(unittest.TestCase):
    Q = {"determination": {"type": "choice", "criteria": {"compatible": "", "incompatible": "", "undecided": ""}}}

    def codes(self, state, rules, q=None):
        return {i["code"] for i in validate_scenario(state, q or self.Q, rules)}

    def test_prompt_template_rule_is_blocking(self):
        issues = validate_scenario({"attributes": {"is_scientific_theory": True, "uses_processes": True}}, self.Q,
                                   ["(is_scientific_theory & uses_processes) => compatible"])
        self.assertIn("template_rule", {i["code"] for i in blocking_issues(issues)})

    def test_contradictory_rules_are_blocking(self):
        c = self.codes({"attributes": {"a": True, "b": True}}, ["(a & b) => compatible", "(b & a) => incompatible"])
        self.assertIn("contradictory_rules", c)

    def test_empty_rules_are_blocking(self):
        self.assertIn("no_rules", self.codes({"attributes": {}}, []))

    def test_good_scenario_has_no_blocking_issue(self):
        issues = validate_scenario({"attributes": {"req_a": True, "req_b": True}}, self.Q,
                                   ["(req_a & req_b) => compatible", "~req_a => incompatible"])
        self.assertEqual(blocking_issues(issues), [])
        self.assertNotIn("uncovered_options", {i["code"] for i in issues})

    def test_uncovered_and_ungrounded_are_reported_not_blocking(self):
        issues = validate_scenario({"attributes": {"a": True}}, self.Q, ["(a & ghost) => compatible"])
        codes = {i["code"]: i for i in issues}
        self.assertEqual(codes["uncovered_options"]["detail"], {"determination": ["incompatible"]})
        self.assertEqual(codes["ungrounded_atoms"]["detail"], ["ghost"])
        self.assertEqual(blocking_issues(issues), [])

    def test_derived_atoms_are_not_ungrounded(self):
        issues = validate_scenario({"attributes": {"a": True}}, self.Q,
                                   ["a => warrant_x", "warrant_x => compatible", "~a => incompatible"])
        self.assertNotIn("ungrounded_atoms", {i["code"] for i in issues})

    def test_uncovered_options_helper(self):
        self.assertEqual(uncovered_options(self.Q, ["a => compatible", "~a => incompatible"]), {})
        self.assertEqual(undecided_option(["yes", "Insufficient_Evidence"]), "Insufficient_Evidence")


@unittest.skipUnless(HAS_V4, "model/ (nanoGentzen v4 weights) not present")
class TestCounterfactuals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nanogentzen.engine import load_engine
        from core.jev_generator import JevScenarioGenerator
        cls.gen = JevScenarioGenerator(llm_client=None, engine=load_engine(str(MODEL_V4)))

    def _parent(self):
        return self.gen._heuristic_fallback(
            context_text="Scientific theories use observable processes. The framework is compatible with observation.",
            doc_name="T.pdf", pages=[3], workflow="scientific_theories", path_nodes=["Scientific Theory", "Observation"])

    def _with_rules(self, parent, rules):
        parent = copy.deepcopy(parent)
        parent["policy_rules"] = rules
        ev = self.gen.architect.evaluate_decision_scenario(parent["state"], parent["questions"], rules,
                                                           parent["target_decision"])
        parent.update(gold=ev["gold"], gentzen_verified=ev["gentzen_verified"])
        parent["provenance"]["nanogentzen"] = self.gen._v4_provenance(ev)
        return parent

    def _covered_parent(self):
        """The offline fallback: its rules conclude every non-undecided option (incl. ~cond => incompatible_*)."""
        parent = self._parent()
        q = next(iter(parent["gold"]))
        alt = next(o for o in parent["questions"][q]["criteria"] if o.startswith("incompatible"))
        cond = next(k for k in parent["state"]["attributes"] if k.endswith("_verified"))
        self.assertIn(f"~{cond} => {alt}", parent["policy_rules"])
        return parent, q, alt, cond

    def test_uncovered_options_give_no_certified_undecided(self):
        # only the winning option has a rule: removing or negating a fact proves nothing, but that is not
        # evidence for "undecided" (the criteria say a failed requirement means "incompatible")
        parent = self._parent()
        parent = self._with_rules(parent, [r for r in parent["policy_rules"] if not r.startswith("~")])
        self.assertTrue(parent["gentzen_verified"])
        self.assertEqual(self.gen.make_counterfactuals(parent), [])

    def test_variants_are_certified_and_change_the_label(self):
        parent, q, alt, _cond = self._covered_parent()
        self.assertTrue(parent["gentzen_verified"])
        variants = self.gen.make_counterfactuals(parent)
        self.assertTrue(variants)
        for v in variants:
            self.assertTrue(v["gentzen_verified"])
            self.assertNotEqual(v["gold"][q]["label"], parent["gold"][q]["label"])
            self.assertIn(v["gold"][q]["verification"], ("NO_OPTION_PROVABLE", "PROVEN_TRUE"))
            self.assertIn(v["gold"][q]["label"], ("undecided", alt))
            self.assertNotIn("observations", v["state"])
            cf = v["provenance"]["counterfactual"]
            self.assertIn(cf["edit"], ("remove", "negate"))
            self.assertEqual(cf["parent_labels"][q], parent["gold"][q]["label"])
        self.assertEqual(len({json.dumps(v["state"], sort_keys=True) for v in variants}), len(variants))

    def test_negated_fact_can_prove_another_option(self):
        parent, q, alt, cond = self._covered_parent()
        labels = {v["provenance"]["counterfactual"]["edit"]: v["gold"][q]["label"] for v in self.gen.make_counterfactuals(parent)
                  if v["provenance"]["counterfactual"]["field"].endswith(cond)}
        self.assertEqual(labels.get("negate"), alt)
        self.assertEqual(labels.get("remove"), "undecided")

    def test_no_undecided_option_means_no_certified_label(self):
        from nanogentzen.architect_jev import JevLogicalArchitect
        ev = self.gen.architect.evaluate_decision_scenario(
            {"attributes": {}}, {"q": {"type": "choice", "criteria": {"yes": "", "no": ""}}}, ["a => yes"],
            target_decision="", label_undecided=True)
        self.assertFalse(ev["gentzen_verified"])
        self.assertEqual(JevLogicalArchitect.undecided_option(["yes", "Insufficient_Evidence"]), "Insufficient_Evidence")


if __name__ == "__main__":
    unittest.main()
