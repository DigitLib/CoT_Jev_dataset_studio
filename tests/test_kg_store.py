"""Named knowledge-graph store (core/kg_store.py) and its session helpers (gui/_kg_session.py)."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tests  # noqa: F401  (redirects all runtime data to a temp dir; see tests/__init__.py)

from core import kg_store
from core.kg_builder import KnowledgeGraphBuilder
from gui._kg_session import KG_ID_KEY, reset_graph, restore_graph, save_current
from gui._paths import DISCOVERED_KEY, publish_discovered_paths


def _kg(doc_names=("paper.pdf",)):
    kg = KnowledgeGraphBuilder()
    for i, d in enumerate(doc_names):
        kg.graph.add_edge(f"A{i}", f"B{i}", predicate="requires", doc_name=d)
        kg.graph.add_edge(f"B{i}", f"C{i}", predicate="implies", doc_name=d)
    return kg


class TestKgIds(unittest.TestCase):
    def test_single_and_multi_document_ids(self):
        self.assertEqual(kg_store.kg_id_for(["World Development Report 2026.pdf"]), "world-development-report-2026")
        a = kg_store.kg_id_for(["b.pdf", "a.pdf", "c.pdf"])
        self.assertEqual(a, kg_store.kg_id_for(["c.pdf", "a.pdf", "b.pdf"]), "order independent")
        self.assertTrue(a.startswith("a+2-"))
        self.assertNotEqual(a, kg_store.kg_id_for(["a.pdf", "b.pdf"]))
        self.assertEqual(kg_store.kg_id_for([]), "untitled-graph")

    def test_sanitize_never_yields_a_path(self):
        for bad in ("../../etc/passwd", "..\\..\\x", "/abs/name.json", "a/b", "index", ""):
            s = kg_store.sanitize_kg_id(bad)
            self.assertNotIn("/", s)
            self.assertNotIn("\\", s)
            self.assertFalse(s.startswith("."))
            self.assertNotEqual(s, "index")
            self.assertTrue(s)


class TestKgStore(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="kgstore-"))

    def test_save_index_load_delete(self):
        kg = _kg(["paper.pdf"])
        kg_store.save_graph(kg, "paper", chunks=[], communities=[], kg_dir=self.dir)
        rows = kg_store.list_graphs(self.dir)
        self.assertEqual([r["kg_id"] for r in rows], ["paper"])
        self.assertEqual(rows[0]["documents"], ["paper.pdf"])
        self.assertEqual(rows[0]["num_edges"], 2)
        self.assertEqual(kg_store.last_active(self.dir)["kg_id"], "paper")

        kg2 = KnowledgeGraphBuilder()
        restored = kg_store.load_graph(kg2, "paper", kg_dir=self.dir)
        self.assertEqual(restored["kg_id"], "paper")
        self.assertEqual(kg2.graph.number_of_edges(), 2)

        self.assertTrue(kg_store.delete_graph("paper", kg_dir=self.dir))
        self.assertFalse((self.dir / "paper.json").exists())
        self.assertEqual(kg_store.list_graphs(self.dir), [])
        self.assertIsNone(kg_store.last_active(self.dir))

    def test_unique_id_never_overwrites(self):
        kg_store.save_graph(_kg(), "paper", kg_dir=self.dir)
        self.assertEqual(kg_store.unique_kg_id("paper", self.dir), "paper-2")
        self.assertEqual(kg_store.unique_kg_id("other", self.dir), "other")

    def test_legacy_latest_json_registered_without_modification(self):
        legacy = self.dir / "latest.json"
        _kg(["old.pdf"]).save(str(legacy))
        before = hashlib.sha256(legacy.read_bytes()).hexdigest()
        (self.dir / "before_concept_repair.json").write_bytes(legacy.read_bytes())  # legacy backup: not a graph
        rows = kg_store.list_graphs(self.dir)
        self.assertEqual([r["kg_id"] for r in rows], ["latest"])
        self.assertEqual(rows[0]["documents"], ["old.pdf"])
        kg_store.load_graph(KnowledgeGraphBuilder(), "latest", kg_dir=self.dir)
        self.assertEqual(hashlib.sha256(legacy.read_bytes()).hexdigest(), before, "latest.json must stay byte-identical")

    def test_backup_before_repair(self):
        kg_store.save_graph(_kg(), "paper", kg_dir=self.dir)
        b = kg_store.backup_before_repair("paper", kg_dir=self.dir)
        self.assertEqual(b.name, "paper.before_repair.json")
        self.assertEqual([r["kg_id"] for r in kg_store.list_graphs(self.dir)], ["paper"], "backups are not listed")


class TestKgIdProvenance(unittest.TestCase):
    def test_stamp_kg_id(self):
        import json
        self.assertEqual(json.loads(kg_store.stamp_kg_id('{"a": 1}', "paper")), {"a": 1, "kg_id": "paper"})
        self.assertEqual(json.loads(kg_store.stamp_kg_id('{"kg_id": "orig"}', "paper"))["kg_id"], "orig")
        self.assertEqual(kg_store.stamp_kg_id('{"a": 1}', None), '{"a": 1}')
        self.assertEqual(kg_store.stamp_kg_id("not json", "paper"), "not json")

    def test_manager_stamps_new_samples(self):
        import json
        from core.dataset_manager import DatasetManager, DatasetSample
        m = DatasetManager(db_path=str(Path(tempfile.mkdtemp()) / "t.db"))
        m.kg_id = "paper"
        m.save_sample(DatasetSample(sample_id="S-1", doc_name="d", question="q", reasoning="r", answer="a"))
        self.assertEqual(json.loads(m.get_samples()[0].provenance_json)["kg_id"], "paper")


class TestKgSession(unittest.TestCase):
    def test_reset_clears_graph_and_derived_state(self):
        kg = _kg()
        state = {"communities": [1], "selected_path": {"x": 1}, KG_ID_KEY: "paper", "kg_built": True,
                 "chunks": ["c"]}
        publish_discovered_paths(state, kg, [{"path_repr": "p"}], {})
        reset_graph(state, kg)
        self.assertEqual(kg.graph.number_of_nodes(), 0)
        for k in ("communities", "selected_path", KG_ID_KEY, DISCOVERED_KEY):
            self.assertNotIn(k, state)
        self.assertFalse(state["kg_built"])
        self.assertEqual(state["chunks"], ["c"], "library kept unless asked")
        reset_graph(state, kg, clear_chunks=True)
        self.assertEqual(state["chunks"], [])

    def test_restore_does_not_touch_library_by_default(self):
        import config
        kg = _kg(["paper.pdf"])
        state = {}
        kid = save_current(state, kg, chunks=[])
        self.assertEqual(kid, "paper")
        self.assertTrue((Path(config.KG_DIR) / "paper.json").exists())
        state = {"chunks": ["mine"]}
        restore_graph(state, KnowledgeGraphBuilder(), kid)
        self.assertEqual(state["chunks"], ["mine"])
        self.assertEqual(state[KG_ID_KEY], "paper")
        kg_store.delete_graph(kid)


if __name__ == "__main__":
    unittest.main()
