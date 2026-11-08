"""
Studio UI state: review-tab state handling and the knowledge-graph paths shared between tabs.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tests  # noqa: F401  (redirects all runtime data to a temp dir; see tests/__init__.py)

from gui._paths import clear_discovered_paths, get_jev_paths, graph_signature, publish_discovered_paths


try:
    import streamlit as st
    from gui.tab_review import _regenerate_cot_only, _build_candidate_sample
    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None
    _regenerate_cot_only = None
    _build_candidate_sample = None
    STREAMLIT_AVAILABLE = False


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Streamlit not installed in this environment")
class TestTabReviewState(unittest.TestCase):
    def setUp(self):
        # Clear Streamlit session state before each test
        for k in list(st.session_state.keys()):
            del st.session_state[k]

    def test_regenerate_cot_only_updates_sample_and_version(self):
        sample = {
            "question": "What is Gentzen natural deduction?",
            "reasoning": "Old reasoning",
            "answer": "Old answer",
            "clarity_score": 0.5,
            "target_claim": "Gentzen deduction",
            "logic_tree_text": "A |- A",
            "context_text": "Context paragraph",
        }
        st.session_state.current_sample = sample
        st.session_state.card_version = 1

        mock_cot_gen = MagicMock()
        mock_cot_gen.build_cot_from_logic_tree.return_value = {
            "question": "What is Gentzen natural deduction?",
            "reasoning": "New rigorous deductive reasoning trace.",
            "answer": "A formal proof calculus established by Gerhard Gentzen.",
            "clarity_score": 0.98,
            "is_fallback": False,
        }

        with patch("streamlit.rerun") as mock_rerun, patch("streamlit.spinner"):
            _regenerate_cot_only(sample, mock_cot_gen, model="test-model")
            mock_rerun.assert_called_once()

        # Check sample was updated
        self.assertEqual(sample["reasoning"], "New rigorous deductive reasoning trace.")
        self.assertEqual(sample["answer"], "A formal proof calculus established by Gerhard Gentzen.")
        self.assertEqual(sample["clarity_score"], 0.98)

        # Check session_state.card_version was incremented
        self.assertEqual(st.session_state.card_version, 2)
        self.assertIn("review_notice", st.session_state)
        self.assertEqual(st.session_state["review_notice"]["type"], "toast")
        self.assertEqual(st.session_state.current_sample["reasoning"], "New rigorous deductive reasoning trace.")

    def test_regenerate_cot_only_handles_fallback_warning(self):
        sample = {
            "question": "What is Occam razor?",
            "reasoning": "Old reasoning",
            "answer": "Old answer",
            "clarity_score": 0.5,
        }
        st.session_state.current_sample = sample
        st.session_state.card_version = 0

        mock_cot_gen = MagicMock()
        mock_cot_gen.build_cot_from_logic_tree.return_value = {
            "reasoning": "Fallback reasoning",
            "answer": "Fallback answer",
            "clarity_score": 0.8,
            "is_fallback": True,
            "error": "Connection timed out",
        }

        with patch("streamlit.rerun"), patch("streamlit.spinner"):
            _regenerate_cot_only(sample, mock_cot_gen, model="test-model")

        self.assertEqual(st.session_state.card_version, 1)
        self.assertEqual(st.session_state["review_notice"]["type"], "warning")
        self.assertIn("Connection timed out", st.session_state["review_notice"]["message"])

    def test_path_change_clears_old_sample_and_increments_card_version(self):
        """Verify that when active_preview_path_repr changes, current_sample is reset and card_version bumped."""
        st.session_state.card_version = 0
        st.session_state.active_preview_path_repr = "(A) --> (B)"
        st.session_state.current_sample = {
            "question": "Old question",
            "reasoning": "Old reasoning",
            "provenance": {"path_repr": "(A) --> (B)"}
        }

        # Simulate user changing selectbox to path 2
        new_path_repr = "(X) --> (Y)"
        if new_path_repr != st.session_state.active_preview_path_repr:
            st.session_state.active_preview_path_repr = new_path_repr
            st.session_state.card_version = st.session_state.get("card_version", 0) + 1
            old_sample = st.session_state.get("current_sample")
            if old_sample and old_sample.get("provenance", {}).get("path_repr") != new_path_repr:
                st.session_state.current_sample = None

        self.assertEqual(st.session_state.card_version, 1)
        self.assertEqual(st.session_state.active_preview_path_repr, "(X) --> (Y)")
        self.assertIsNone(st.session_state.current_sample)


class FakeKG:
    def __init__(self, n_nodes=5):
        self.graph = nx.MultiDiGraph()
        for i in range(n_nodes):
            self.graph.add_node(f"N{i}")
        for i in range(n_nodes - 1):
            self.graph.add_edge(f"N{i}", f"N{i+1}")
        self.calls = []

    def find_multihop_paths(self, **kw):
        self.calls.append(kw)
        if kw.get("min_hops") == 2:
            return [{"path_repr": f"rediscovered-{j}", "hops": 2} for j in range(min(kw["limit"], 7))]
        return []


class TestSharedPaths(unittest.TestCase):
    def test_uses_published_paths_without_searching(self):
        kg, state = FakeKG(), {}
        published = [{"path_repr": "A -> B -> C", "hops": 2}, {"path_repr": "B -> C -> D", "hops": 2}]
        publish_discovered_paths(state, kg, published, {"limit": 300, "scope_doc_name": None})
        paths, source = get_jev_paths(state, kg)
        self.assertEqual(paths, published)
        self.assertIn("Knowledge Graph tab", source)
        self.assertEqual(kg.calls, [], "must not run its own search")

    def test_stale_signature_is_not_reused(self):
        kg, state = FakeKG(), {}
        publish_discovered_paths(state, kg, [{"path_repr": "old"}], {})
        kg.graph.add_node("NEW")  # graph changed after discovery
        paths, _ = get_jev_paths(state, kg)
        self.assertEqual(paths, [])
        self.assertEqual(kg.calls, [])

    def test_rediscover_honours_scope_limit_and_entailment_and_caches(self):
        kg = FakeKG()
        state = {"kg_path_limit": 300, "jev_require_entailment": True, "intra_doc_only": True}
        publish_discovered_paths(state, kg, [], {"scope_doc_name": "paper.pdf"})
        paths, source = get_jev_paths(state, kg, rediscover=True)
        self.assertEqual(len(paths), 7)
        call = kg.calls[0]
        self.assertEqual(call["scope_doc_name"], "paper.pdf")
        self.assertTrue(call["require_entailment"])
        self.assertEqual(call["limit"], 300)
        self.assertFalse(call["verify_with_gentzen"])
        self.assertIn("re-discovered", source)
        get_jev_paths(state, kg, rediscover=True)
        self.assertEqual(len(kg.calls), 1, "second call must come from the cache")

    def test_no_graph(self):
        self.assertIsNone(graph_signature(None))
        self.assertEqual(get_jev_paths({}, FakeKG(n_nodes=1))[0], [])

    def test_clear(self):
        kg, state = FakeKG(), {}
        publish_discovered_paths(state, kg, [{"path_repr": "x"}], {})
        clear_discovered_paths(state)
        self.assertEqual(get_jev_paths(state, kg)[0], [])


if __name__ == "__main__":
    unittest.main()
