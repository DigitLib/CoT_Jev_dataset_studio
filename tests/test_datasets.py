"""
Datasets: SQLite storage, train / validation / test splits, ClearCoT and Jev exports, export filters
and the quality gates applied before export.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tests  # noqa: F401  (redirects all runtime data to a temp dir; see tests/__init__.py)

from core.batch_generator import decide_sample_status
from core.cot_generator import CoTGenerator, naive_trace_leak
from core.dataset_manager import DatasetManager, DatasetSample
from core.dataset_splits import assign_splits, describe_split, document_groups
from core.jev_dataset_manager import JevDatasetManager, JevSample


def _jev(sid, doc, cf_of=None, path="(A) --[implies]--> (B)", gold=None, qid="decision"):
    prov = {"doc_name": doc, "path_repr": path}
    if cf_of:
        prov["counterfactual"] = {"of": cf_of}
    return JevSample(
        sample_id=sid, doc_name=doc, workflow="wf",
        state_json=json.dumps({"attributes": {"a": True}}),
        questions_json=json.dumps({qid: {"type": "choice", "instructions": "Which?",
                                         "criteria": {"yes": "y", "no": "n", "undecided": "u"}}}),
        gold_json=json.dumps(gold or {qid: {"label": "yes", "confidence": 1.0, "verification": "PROVEN_TRUE",
                                            "probabilities": {"yes": 1.0, "no": 0.0, "undecided": 0.0}}}),
        policy_rules_json=json.dumps(["a => yes"]), gentzen_verified=True, provenance_json=json.dumps(prov))


GENUINE = ("Hmm, so open licences are supposed to help, right? I think they spread access, but wait, maybe the "
           "attribution part is tricky in some places. Licences are kind of universal anyway, so they probably "
           "just work everywhere, and that would mean the assumption holds. Yes, I'd say it holds overall.")

LEAKY = ("To construct a naive thinking trace: 1. Start by acknowledging the premise. The expected answer concept "
         "states that NOODL challenges openness. " + GENUINE)


class TestDatasetManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_clearcot.db"
        self.manager = DatasetManager(db_path=str(self.db_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_db_creates_table(self):
        self.assertTrue(self.db_path.exists())
        samples = self.manager.get_samples()
        self.assertEqual(len(samples), 0)

    def test_save_and_retrieve_sample(self):
        sample = DatasetSample(
            sample_id="TEST-001",
            doc_name="cognition.pdf",
            question="What is the Computational Theory of Mind?",
            reasoning="Step 1: Mind is defined as a computational system.\nStep 2: Brain executes transitions.",
            answer="The mind is a physical computing system.",
            naive_cot="Well, let me think about this question...",
            status="accepted",
            gentzen_verified=True,
            gentzen_sequent="Mind_System ⊢ Mind_System",
            provability_score=1.0,
            tokens_saved_pct=42,
            clarity_score=0.95
        )
        self.manager.save_sample(sample)

        samples = self.manager.get_samples()
        self.assertEqual(len(samples), 1)
        retrieved = samples[0]
        self.assertEqual(retrieved.sample_id, "TEST-001")
        self.assertEqual(retrieved.doc_name, "cognition.pdf")
        self.assertTrue(retrieved.gentzen_verified)
        self.assertEqual(retrieved.tokens_saved_pct, 42)

    def test_filter_by_status(self):
        s1 = DatasetSample(
            sample_id="S-1",
            doc_name="doc.pdf",
            question="Q1",
            reasoning="R1",
            answer="A1",
            status="accepted"
        )
        s2 = DatasetSample(
            sample_id="S-2",
            doc_name="doc.pdf",
            question="Q2",
            reasoning="R2",
            answer="A2",
            status="rejected"
        )
        self.manager.save_sample(s1)
        self.manager.save_sample(s2)

        accepted = self.manager.get_samples(status="accepted")
        rejected = self.manager.get_samples(status="rejected")
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(accepted[0].sample_id, "S-1")
        self.assertEqual(rejected[0].sample_id, "S-2")

    def test_delete_sample(self):
        sample = DatasetSample(
            sample_id="TO-DEL",
            doc_name="doc.pdf",
            question="Q",
            reasoning="R",
            answer="A"
        )
        self.manager.save_sample(sample)
        self.assertEqual(len(self.manager.get_samples()), 1)

        self.manager.delete_sample("TO-DEL")
        self.assertEqual(len(self.manager.get_samples()), 0)

    def test_epistemic_status_and_curry_howard_export(self):
        sample = DatasetSample(
            sample_id="CH-001",
            doc_name="intuitionistic.pdf",
            question="Does A imply A?",
            reasoning="A is assumed. Hence A holds by identity.",
            answer="A holds constructively.",
            naive_cot="Well, A might imply A maybe.",
            status="accepted",
            gentzen_verified=True,
            gentzen_sequent="A ⊢ A",
            curry_howard_term="(λx:A. x)",
            epistemic_status="PROVEN_TRUE",
            clarity_score=0.96,
            tokens_saved_pct=50
        )
        self.manager.save_sample(sample)

        samples = self.manager.get_samples()
        self.assertEqual(len(samples), 1)
        retrieved = samples[0]
        self.assertEqual(retrieved.curry_howard_term, "(λx:A. x)")
        self.assertEqual(retrieved.epistemic_status, "PROVEN_TRUE")

        # Test JSONL export
        jsonl_path = Path(self.temp_dir.name) / "export.jsonl"
        count = self.manager.export_jsonl(str(jsonl_path))
        self.assertEqual(count, 1)
        import json
        with open(jsonl_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
            self.assertEqual(data["curry_howard_term"], "(λx:A. x)")
            self.assertEqual(data["epistemic_status"], "PROVEN_TRUE")

        # Test DPO export
        dpo_path = Path(self.temp_dir.name) / "export_dpo.jsonl"
        count_dpo = self.manager.export_dpo_pairs(str(dpo_path))
        self.assertEqual(count_dpo, 1)
        with open(dpo_path, "r", encoding="utf-8") as f:
            dpo_data = json.loads(f.readline())
            self.assertEqual(dpo_data["metadata"]["curry_howard_term"], "(λx:A. x)")
            self.assertEqual(dpo_data["metadata"]["epistemic_status"], "PROVEN_TRUE")

        # Test Parquet export
        parquet_path = Path(self.temp_dir.name) / "export.parquet"
        count_parquet = self.manager.export_parquet(str(parquet_path))
        self.assertEqual(count_parquet, 1)
        self.assertTrue(parquet_path.exists())


class TestSplits(unittest.TestCase):
    def test_groups_never_cross_splits_and_ratios_hold(self):
        items = [(f"s{i}", {"document": f"doc{i % 10}", "path": f"p{i}", "family": f"s{i}"}) for i in range(200)]
        m, info = assign_splits(items)
        self.assertEqual(info["level"], "document")
        by_doc = {}
        for sid, keys in items:
            by_doc.setdefault(keys["document"], set()).add(m[sid])
        self.assertTrue(all(len(v) == 1 for v in by_doc.values()))
        self.assertEqual(set(m.values()), {"train", "validation", "test"})
        self.assertGreaterEqual(info["counts"]["train"], 140)

    def test_deterministic(self):
        items = [(f"s{i}", {"document": f"doc{i % 7}"}) for i in range(50)]
        self.assertEqual(assign_splits(items)[0], assign_splits(list(reversed(items)))[0])

    def test_single_document_falls_back_to_paths(self):
        items = [(f"s{i}", {"document": "one.pdf", "path": f"p{i % 12}", "family": f"s{i}"}) for i in range(60)]
        m, info = assign_splits(items)
        self.assertEqual(info["level"], "path")
        self.assertEqual(set(m.values()), {"train", "validation", "test"})

    def test_stored_keeps_existing(self):
        self.assertEqual(assign_splits([("a", {})], "stored")[0], {})


class TestClearCoTSplits(unittest.TestCase):
    def test_sft_and_dpo_carry_group_splits(self):
        with tempfile.TemporaryDirectory() as d:
            db = DatasetManager(db_path=str(Path(d) / "t.db"))
            naive = ("Hmm, so I guess I should start by recalling what I know about this, and honestly there are "
                     "many things here, maybe the first idea is right, maybe not, let me go around it a few times.")
            for i in range(30):
                db.save_sample(DatasetSample(sample_id=f"S{i}", doc_name=f"doc{i % 5}.pdf", question=f"Why {i}?",
                                             reasoning="Because A requires B, B holds.", answer="B holds because A.",
                                             naive_cot=naive, status="accepted",
                                             provenance_json=json.dumps({"path_repr": f"p{i}"})))
            out = Path(d) / "sft.jsonl"
            db.export_jsonl(str(out))
            rows = [json.loads(l) for l in out.read_text().splitlines()]
            per_doc = {}
            for r in rows:
                per_doc.setdefault(r["document"], set()).add(r["split"])
            self.assertTrue(all(len(v) == 1 for v in per_doc.values()))
            dpo = Path(d) / "dpo.jsonl"
            db.export_dpo_pairs(str(dpo))
            self.assertTrue(all("split" in json.loads(l) for l in dpo.read_text().splitlines()))


class TestJevExports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = JevDatasetManager(db_path=str(Path(self.tmp.name) / "j.db"))
        for d in range(6):
            self.m.save_sample(_jev(f"P{d}", f"doc{d}.pdf", path=f"(A{d}) --[implies]--> (B{d})"))
            for k in range(2):        # counterfactual variants of P{d}
                self.m.save_sample(_jev(f"P{d}-cf{k}", f"doc{d}.pdf", cf_of=f"P{d}", path=f"(A{d}) --[implies]--> (B{d})",
                                        gold={"decision": {"label": "undecided", "verification": "NO_OPTION_PROVABLE",
                                                           "confidence": 1.0, "probabilities": {"undecided": 1.0}}}))

    def tearDown(self):
        self.tmp.cleanup()

    def _capture(self, fn, **kw):
        import pandas as pd
        captured = {}
        with mock.patch.object(pd.DataFrame, "to_parquet", lambda self, *a, **k: captured.setdefault("df", self)):
            fn(str(Path(self.tmp.name) / "out.parquet"), **kw)
        return captured["df"]

    def test_counterfactuals_share_parent_split(self):
        out = Path(self.tmp.name) / "j.jsonl"
        self.m.export_jev_jsonl(str(out))
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        split = {r["id"]: r["split"] for r in rows}
        for d in range(6):
            self.assertEqual({split[f"P{d}"], split[f"P{d}-cf0"], split[f"P{d}-cf1"]}, {split[f"P{d}"]})
        self.assertGreater(len(set(split.values())), 1)

    def test_s1mb_input_has_decisions_and_rules(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("pandas not installed")
        df = self._capture(self.m.export_open_jev_parquet)
        inp = json.loads(df.iloc[0]["input"]["state_json"])
        self.assertEqual(inp["state"], {"attributes": {"a": True}})
        self.assertEqual(inp["decisions"][0]["decision_id"], "decision")
        self.assertEqual(inp["decisions"][0]["options"], ["yes", "no", "undecided"])
        self.assertEqual(inp["policy_rules"], ["a => yes"])
        # same group id for a scenario and its variants, different across documents
        g = dict(zip(df["case_id"], df["group_id"]))
        self.assertEqual(g["wf:P0"], g["wf:P0-cf0"])
        self.assertNotEqual(g["wf:P0"], g["wf:P1"])
        legacy = self._capture(self.m.export_open_jev_parquet, include_decisions=False, include_rules=False)
        self.assertEqual(json.loads(legacy.iloc[0]["input"]["state_json"]), {"attributes": {"a": True}})

    def test_positional_columns(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("pandas not installed")
        self.m.save_sample(_jev("X1", "docx.pdf", qid="compat"))
        self.m.save_sample(_jev("X2", "docy.pdf", qid="determination"))
        wide = self._capture(self.m.export_typed_decisions_parquet)
        self.assertIn("compat__label", wide.columns)
        self.assertIn("determination__label", wide.columns)
        pos = self._capture(self.m.export_typed_decisions_parquet, column_mode="positional")
        self.assertNotIn("compat__label", pos.columns)
        self.assertEqual(set(pos["q1__id"]), {"decision", "compat", "determination"})
        self.assertIn("q1__verification", pos.columns)

    def test_only_verified(self):
        s = _jev("U1", "docu.pdf")
        s.gentzen_verified = False
        self.m.save_sample(s)
        out = Path(self.tmp.name) / "v.jsonl"
        self.assertEqual(self.m.export_jev_jsonl(str(out), only_verified=True), 18)
        self.assertEqual(self.m.export_jev_jsonl(str(out)), 19)


class TestSplitBalance(unittest.TestCase):
    def _items(self):
        sizes = {"a.pdf": 108, "b.pdf": 22, "c.pdf": 1}
        return [(f"{d}-{i}", {"document": d, "path": f"p{d}{i}", "family": f"f{d}{i}"})
                for d, n in sizes.items() for i in range(n)]

    def test_auto_skips_an_unbalanced_document_split(self):
        m, info = assign_splits(self._items())
        self.assertEqual(info["level"], "path")
        self.assertGreaterEqual(info["counts"]["test"], 10)
        self.assertEqual(info["levels_skipped"][0]["level"], "document")
        self.assertNotIn("warning", info)

    def test_explicit_document_split_is_kept_but_warned(self):
        m, info = assign_splits(self._items(), "document")
        self.assertEqual((info["level"], info["counts"]["test"]), ("document", 1))
        self.assertIn("unbalanced", describe_split(info)[1])

    def test_balanced_document_split_is_still_preferred(self):
        items = [(f"{d}-{i}", {"document": d, "path": f"p{d}{i}"}) for d in "abcdefghij" for i in range(10)]
        self.assertEqual(assign_splits(items)[1]["level"], "document")


class TestLinkedDocumentSplits(unittest.TestCase):
    def test_cross_document_samples_join_their_documents(self):
        g = document_groups([["a.pdf"], ["b.pdf"], ["a.pdf", "b.pdf"], ["c.pdf"], ["d.pdf"], ["e.pdf"]])
        self.assertEqual(g["a.pdf"], g["b.pdf"])
        self.assertNotEqual(g["a.pdf"], g["c.pdf"])

    def test_clearcot_split_never_separates_linked_documents(self):
        from core.dataset_manager import DatasetManager, DatasetSample
        with tempfile.TemporaryDirectory() as d:
            db = DatasetManager(db_path=str(Path(d) / "t.db"))
            docs = [f"doc{i}.pdf" for i in range(8)]
            for i in range(40):
                doc = docs[i % 8]
                prov = {"documents": [doc, docs[(i + 1) % 8]]} if i % 8 in (0, 1) else {}   # doc0-doc1-doc2 linked
                db.save_sample(DatasetSample(sample_id=f"S{i}", doc_name=doc, question="q?", reasoning="r", answer="a",
                                             status="accepted", provenance_json=json.dumps(prov)))
            out = Path(d) / "sft.jsonl"
            db.export_jsonl(str(out))
            split = {}
            for line in out.read_text().splitlines():
                r = json.loads(line)
                split.setdefault(r["document"], set()).add(r["split"])
            linked = split["doc0.pdf"] | split["doc1.pdf"] | split["doc2.pdf"]
            self.assertEqual(len(linked), 1, split)


class TestClearCoTExports(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="cc-export-"))
        self.m = DatasetManager(db_path=str(self.dir / "t.db"))
        for i in range(40):
            accepted = i % 3 != 0          # DPO exports accepted samples only: a different subset than SFT
            self.m.save_sample(DatasetSample(
                sample_id=f"S-{i:04d}", doc_name=f"doc{i % 5}.pdf", question=f"Question {i}?",
                reasoning=f"Reasoning {i} is direct and short.", answer=f"Answer {i}.",
                naive_cot=(LEAKY if i == 4 else GENUINE + f" ({i})"), status="accepted" if accepted else "draft",
                gentzen_verified=True, tokens_saved_pct=40,
                provenance_json=json.dumps({"path_nodes": [f"A{i}", f"B{i}", f"C{i}"]})))

    def _read(self, name):
        lines = (self.dir / name).read_text(encoding="utf-8").splitlines()
        return {json.loads(l)["id"]: json.loads(l)["split"] for l in lines}

    def test_same_split_in_every_file(self):
        self.m.export_jsonl(str(self.dir / "sft.jsonl"))
        self.m.export_dpo_pairs(str(self.dir / "dpo.jsonl"))
        sft, dpo = self._read("sft.jsonl"), self._read("dpo.jsonl")
        self.assertTrue(set(dpo) < set(sft), "DPO is a strict subset of SFT here")
        self.assertEqual({i: sft[i] for i in dpo}, dpo)
        try:
            import pandas as pd
        except ImportError:
            return
        self.m.export_parquet(str(self.dir / "cc.parquet"))
        df = pd.read_parquet(self.dir / "cc.parquet")
        self.assertEqual(dict(zip(df.id, df.split)), sft)

    def test_leaky_negative_is_not_exported(self):
        self.m.export_dpo_pairs(str(self.dir / "dpo.jsonl"))
        rows = [json.loads(l) for l in (self.dir / "dpo.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertNotIn("S-0004", {r["id"] for r in rows})
        self.assertEqual(self.m.last_dpo_skipped["meta_leak"], 1)
        self.assertFalse(any(naive_trace_leak(r["rejected"]) for r in rows))

    def test_bad_provenance_json_does_not_abort_export(self):
        self.m.save_sample(DatasetSample(sample_id="S-BAD", doc_name="x.pdf", question="Q?", reasoning="R.",
                                         answer="A.", provenance_json="{not json"))
        n = self.m.export_jsonl(str(self.dir / "sft.jsonl"))
        self.assertEqual(n, 41)


class TestExportFilters(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_cot_only_verified_and_negative_savings(self):
        db = DatasetManager(db_path=str(Path(self.tmp.name) / "c.db"))
        for i, (ver, saved) in enumerate([(True, 50), (False, 60), (True, -20)]):
            db.save_sample(DatasetSample(sample_id=f"S{i}", doc_name="d.pdf", question=f"Q{i}?", reasoning="r",
                                         answer="a", status="accepted", gentzen_verified=ver, tokens_saved_pct=saved,
                                         provenance_json=json.dumps({"path_repr": f"p{i}"})))
        out = Path(self.tmp.name) / "sft.jsonl"
        self.assertEqual(db.export_jsonl(str(out)), 2)                      # negative savings dropped
        self.assertEqual(db.export_jsonl(str(out), only_verified=True), 1)
        self.assertEqual(json.loads(out.read_text().splitlines()[0])["id"], "S0")

    def test_jev_unverified_label_is_not_exported_as_gold(self):
        m = JevDatasetManager(db_path=str(Path(self.tmp.name) / "j.db"))
        m.save_sample(JevSample(
            sample_id="U1", doc_name="d.pdf", workflow="wf", state_json=json.dumps({"attributes": {"a": True}}),
            questions_json=json.dumps({"q": {"type": "choice", "criteria": {"yes": "", "no": ""}}}),
            gold_json=json.dumps({"q": {"label": "yes", "confidence": 0.5, "verification": "MULTIPLE_PROVEN",
                                        "probabilities": {"yes": 0.5, "no": 0.5}}}),
            policy_rules_json=json.dumps(["a => yes", "a => no"]), gentzen_verified=False,
            epistemic_status="MULTIPLE_PROVEN", provenance_json="{}"))
        out = Path(self.tmp.name) / "j.jsonl"
        self.assertEqual(m.export_jev_jsonl(str(out), only_verified=True), 0)
        self.assertEqual(m.export_jev_jsonl(str(out)), 1)
        g = json.loads(out.read_text())["gold"]["q"]
        self.assertIsNone(g["label"])
        self.assertEqual(g["intended_label"], "yes")


class TestDatasetQuality(unittest.TestCase):
    def test_status_gate(self):
        self.assertEqual(decide_sample_status("accepted", True, True, False), ("accepted", []))
        st, why = decide_sample_status("accepted", False, True, False)
        self.assertEqual(st, "draft")
        self.assertIn("proof not verified", why)
        self.assertEqual(decide_sample_status("accepted", True, True, True)[0], "draft")
        self.assertEqual(decide_sample_status("accepted", True, False, False)[0], "draft")
        self.assertEqual(decide_sample_status("draft", True, True, False), ("draft", []))

    def test_token_savings_are_measured_or_absent(self):
        self.assertIsNone(CoTGenerator.measure_token_savings("a b c", None))
        self.assertEqual(CoTGenerator.measure_token_savings("w " * 50, "w " * 100), 50)
        self.assertEqual(CoTGenerator.measure_token_savings("w " * 120, "w " * 100), -20)
        self.assertIsNone(CoTGenerator(llm_client=None).generate_standard_cot("q?", "ctx"))
        meta = CoTGenerator(llm_client=None).generate_naive_cot("q?", "claim.", "ctx.", return_meta=True)
        self.assertTrue(meta["synthetic"])

    def test_dpo_only_from_accepted_real_negatives(self):
        with tempfile.TemporaryDirectory() as d:
            db = DatasetManager(db_path=str(Path(d) / "t.db"))
            real_naive = ("Hmm, so I guess I should start by recalling what I know about this, and honestly "
                          "there are many things here, maybe the first idea is right, maybe not, let me go "
                          "around it a few times before I settle on the point that it holds.")
            synth = CoTGenerator(llm_client=None).generate_naive_cot("Why x?", "X holds.", "Ctx one. Ctx two.")

            def mk(status, naive):
                return DatasetSample(sample_id="", doc_name="d", question="Why does X hold?",
                                     reasoning="Because A requires B and B implies X, X holds.",
                                     answer="X holds because A requires B.", naive_cot=naive, status=status)
            db.save_sample(mk("accepted", real_naive))
            db.save_sample(mk("rejected", real_naive))
            db.save_sample(mk("draft", real_naive))
            db.save_sample(mk("accepted", synth))
            out = Path(d) / "dpo.jsonl"
            self.assertEqual(db.export_dpo_pairs(str(out)), 1)
            self.assertEqual(db.export_dpo_pairs(str(out), only_accepted=False, skip_synthetic_rejected=False), 4)


if __name__ == "__main__":
    unittest.main()
