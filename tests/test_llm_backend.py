"""
LLM backend: OpenAI-compatible client, reasoning-trace adapters and the local vLLM model manager.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tests  # noqa: F401  (redirects all runtime data to a temp dir; see tests/__init__.py)

from core.batch_generator import BatchDatasetGenerator
from core.chunker import Chunk
from core.dataset_manager import DatasetManager
from core.kg_builder import KnowledgeGraphBuilder
from llm_backend import vllm_server as vs
from llm_backend.adapters import ReasoningTraceAdapter
from llm_backend.client import LocalLLMClient
from nanogentzen.logical_architect import LogicalArchitect


def _model(root: Path, name: str, config=None, sidecar=None, modeling=False) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config or {"model_type": "x"}))
    if sidecar is not None:
        (d / vs.SIDECAR_NAME).write_text(json.dumps(sidecar))
    if modeling:
        (d / "modeling_x.py").write_text("# custom code")
    return d


class TestLLMClient(unittest.TestCase):

    def test_client_parse_response_content(self):
        client = LocalLLMClient(base_url="http://localhost:8000/v1")
        # Test standard content
        resp1 = {"choices": [{"message": {"content": "Direct response content"}}]}
        self.assertEqual(client._parse_response_content(resp1), "Direct response content")

        # Test reasoning tokens (e.g. DeepSeek-R1 / Qwen3.5 reasoning_content)
        resp2 = {
            "choices": [{
                "message": {
                    "reasoning_content": "Internal logic chain",
                    "content": "Final synthesis"
                }
            }]
        }
        parsed2 = client._parse_response_content(resp2)
        self.assertIn("<think>\nInternal logic chain\n</think>", parsed2)
        self.assertIn("Final synthesis", parsed2)

    def test_async_generate_batch(self):
        client = LocalLLMClient(base_url="http://localhost:8000/v1")
        prompts = ["Question 1", "Question 2", "Question 3"]

        with patch.object(client, "async_generate") as mock_async_gen:
            async def _fake_gen(prompt, **kwargs):
                return f"Answer for {prompt}"
            mock_async_gen.side_effect = _fake_gen

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    client.async_generate_batch(prompts, concurrency=2)
                )
            finally:
                loop.close()

            self.assertEqual(len(results), 3)
            self.assertEqual(results[0], "Answer for Question 1")
            self.assertEqual(results[1], "Answer for Question 2")
            self.assertEqual(results[2], "Answer for Question 3")

    def test_generate_concurrent_batch_threads(self):
        client = LocalLLMClient(base_url="http://localhost:8000/v1")
        prompts = [f"Prompt {i}" for i in range(5)]

        with patch.object(client, "generate") as mock_gen:
            mock_gen.side_effect = lambda prompt, **kwargs: f"Echo: {prompt}"
            results = client.generate_concurrent_batch(prompts, max_workers=3)

            self.assertEqual(len(results), 5)
            for i, res in enumerate(results):
                self.assertEqual(res, f"Echo: Prompt {i}")

            # Verify batch_generate alias
            alias_results = client.batch_generate(prompts, max_workers=3)
            self.assertEqual(alias_results, results)

    def test_kg_builder_worker_resilience(self):
        client = MagicMock(spec=LocalLLMClient)
        client.health_check.return_value = True
        client.generate.return_value = "[]"

        builder = KnowledgeGraphBuilder(llm_client=client)
        chunks = [
            Chunk(
                chunk_id=f"err_chunk_{i}",
                doc_name="test_doc.pdf",
                page_number=i,
                section_title="Concepts",
                text=f"Entropy implies Information {i}.",
                chunk_type="paragraph"
            )
            for i in range(4)
        ]

        # Simulate worker error on one item
        original_extract = builder._extract_propositions_for_chunk
        def _flaky_extract(chunk, model, known_entities=None, client=None):
            if chunk.chunk_id == "err_chunk_2":
                raise RuntimeError("Simulated network timeout")
            return original_extract(chunk, model, known_entities, client)

        with patch.object(builder, "_extract_propositions_for_chunk", side_effect=_flaky_extract):
            # Should not raise exception
            builder.build_from_chunks(chunks, concurrency=2)

    def test_kg_builder_concurrent(self):
        client = MagicMock(spec=LocalLLMClient)
        client.health_check.return_value = False  # Triggers fast heuristic extraction

        builder = KnowledgeGraphBuilder(llm_client=client)
        chunks = [
            Chunk(
                chunk_id=f"c_{i}",
                doc_name="test_doc.pdf",
                page_number=i,
                section_title="Concepts",
                text=f"Entropy implies Information. Information requires Energy. Item {i} connects them.",
                chunk_type="paragraph"
            )
            for i in range(10)
        ]

        # Test parallel build with 4 workers
        progress_records = []
        def progress_cb(curr, total, msg):
            progress_records.append((curr, total))

        builder.build_from_chunks(chunks, concurrency=4, progress_callback=progress_cb)

        self.assertGreater(builder.graph.number_of_nodes(), 0)
        self.assertGreater(builder.graph.number_of_edges(), 0)
        self.assertEqual(len(progress_records), 10)
        self.assertEqual(progress_records[-1][0], 10)

    def test_batch_generator_concurrent_execution(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_clearcot.db"
            db_manager = DatasetManager(db_path=str(db_path))

            mock_q_gen = MagicMock()
            def _gen_chunk(chunk, model=None):
                i = chunk.chunk_id
                return {
                    "question": f"How does energy enable physical work and transformations in system {i}?",
                    "target_claim": f"Energy enables physical work and state transformation in system {i}.",
                    "seed_answer": f"Energy enables physical work and state transformation in system {i}.",
                    "key_premises": [f"Energy enables physical work and state transformation in system {i}."],
                    "provenance": {"doc_name": "test.txt", "pages": [1]},
                    "context_text": f"Sample text context for {i}"
                }
            mock_q_gen.generate_from_chunk.side_effect = _gen_chunk

            mock_architect = MagicMock(spec=LogicalArchitect)
            mock_architect.generate_best_logic_tree.return_value = {
                "is_verified": True,
                "sequent": "Energy ⊢ Work",
                "proof_tree": {"rule": "Ax", "conclusion": "Work", "score": 1.0, "children": []},
                "logic_tree_text": "- Step 1: Energy enables work",
                "provability_score": 1.0,
                "depth": 1,
                "node_count": 1
            }

            mock_cot_gen = MagicMock()
            def _build_cot(question, target_claim, logic_tree_text, context_text="", model=None, **kwargs):
                return {
                    "question": question,
                    "reasoning": f"Since energy is defined as capacity to do work, therefore {target_claim}",
                    "answer": target_claim,
                    "clarity_score": 0.95,
                    # a real LLM trace that passed the grounding check: the batch gate accepts these
                    "is_grounded": True,
                    "is_fallback": False,
                    "grounding_score": 0.9,
                }
            mock_cot_gen.build_cot_from_logic_tree.side_effect = _build_cot

            batch_gen = BatchDatasetGenerator(
                q_generator=mock_q_gen,
                architect=mock_architect,
                cot_generator=mock_cot_gen,
                db_manager=db_manager
            )

            chunks = [
                Chunk(
                    chunk_id=f"chk_{i}",
                    doc_name="test.txt",
                    page_number=1,
                    section_title="Physics",
                    text=f"Energy enables work and transformation in physical systems {i}.",
                    chunk_type="paragraph"
                )
                for i in range(6)
            ]

            progress_calls = []
            def progress_cb(curr, total, sample_dict, msg):
                progress_calls.append(curr)

            # Generate batch with 3 workers
            res = batch_gen.generate_batch(
                source_mode="chunks",
                kg_paths=[],
                chunks=chunks,
                model="test-model",
                max_samples=6,
                concurrency=3,
                progress_callback=progress_cb
            )

            self.assertEqual(res["generated_count"], 6)
            self.assertEqual(res["verified_count"], 6)
            self.assertEqual(res["verification_rate"], 1.0)
            self.assertGreater(len(progress_calls), 0)

            # Check database persistence
            stats = db_manager.get_stats()
            self.assertEqual(stats["total_samples"], 6)
            self.assertEqual(stats["accepted_count"], 6)

    def test_vllm_server_manager(self):
        from llm_backend.vllm_server import is_vllm_installed, is_vllm_running, get_active_vllm_pid, stop_vllm_server
        # Verify function calls run without raising exceptions
        installed = is_vllm_installed()
        self.assertIsInstance(installed, bool)

        running = is_vllm_running(base_url="http://127.0.0.1:59999/v1")
        self.assertFalse(running)

        # Active PID when not started
        pid = get_active_vllm_pid()
        self.assertTrue(pid is None or isinstance(pid, int))

        stopped = stop_vllm_server(port=59999)
        self.assertIsInstance(stopped, bool)


class TestReasoningTraceAdapter(unittest.TestCase):
    def test_parse_chat_response_openai_reasoning(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "42 is the answer.",
                        "reasoning_content": "Calculating life, universe, and everything...",
                    }
                }
            ]
        }
        res = ReasoningTraceAdapter.parse_chat_response(payload)
        self.assertIn("<think>\nCalculating life, universe, and everything...\n</think>", res)
        self.assertIn("42 is the answer.", res)

    def test_parse_chat_response_open_think_tag(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "First let's check premise A.\n</think>\nThe answer is positive.",
                    }
                }
            ]
        }
        res = ReasoningTraceAdapter.parse_chat_response(payload)
        self.assertTrue(res.startswith("<think>"))
        self.assertIn("</think>", res)

    def test_extract_trace_and_answer(self):
        text = "<think>\nStep 1: P implies Q.\nStep 2: P holds.\n</think>\n<answer>Therefore Q holds.</answer>"
        trace, ans = ReasoningTraceAdapter.extract_trace_and_answer(text)
        self.assertEqual(trace, "Step 1: P implies Q.\nStep 2: P holds.")
        self.assertEqual(ans, "Therefore Q holds.")

    def test_extract_trace_reasoning_tag(self):
        text = "<reasoning>\nStep 1: Check rules.\n</reasoning>\n<answer>Verified.</answer>"
        trace, ans = ReasoningTraceAdapter.extract_trace_and_answer(text)
        self.assertEqual(trace, "Step 1: Check rules.")
        self.assertEqual(ans, "Verified.")


class TestVllmModels(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="vllm-models-"))

    def test_default_model_is_local_occam_if_present(self):
        """The default model is the local Occam-2B-CCoT folder (vllm_model/ or the Studio root) if present."""
        from config import VLLM_MODELS_DIR
        default_m = vs.get_default_model()
        for d in (VLLM_MODELS_DIR / "Occam-2B-CCoT", vs.LOCAL_OCCAM_DIR):
            if (d / "config.json").exists():
                self.assertEqual(default_m, str(d))
                break
        else:
            self.assertEqual(default_m, "Sagicc/Occam-2B-CCoT")

    def test_discovery_requires_config_json(self):
        _model(self.root, "Occam-2B-CCoT")
        (self.root / "not-a-model").mkdir()
        (self.root / "README.md").write_text("x")
        names = [m["name"] for m in vs.discover_local_models(self.root, include_legacy=False)]
        self.assertEqual(names, ["Occam-2B-CCoT"])

    def test_discovery_follows_symlinked_folders(self):
        real = _model(self.root / "store", "Real")
        (self.root / "models").mkdir()
        os.symlink(real, self.root / "models" / "Linked")
        names = [m["name"] for m in vs.discover_local_models(self.root / "models", include_legacy=False)]
        self.assertEqual(names, ["Linked"])

    def test_sidecar_and_remote_code_detection(self):
        _model(self.root, "Spark", config={"auto_map": {"AutoModel": "modeling_spark.Spark"}},
               sidecar={"max_model_len": 8192, "unknown_key": 1})
        _model(self.root, "Plain")
        m = {x["name"]: x for x in vs.discover_local_models(self.root, include_legacy=False)}
        self.assertTrue(m["Spark"]["settings"]["trust_remote_code"])
        self.assertEqual(m["Spark"]["settings"]["max_model_len"], 8192)
        self.assertNotIn("unknown_key", m["Spark"]["settings"])
        self.assertFalse(m["Plain"]["settings"]["trust_remote_code"])

    def test_resolve_name_path_and_hf_id(self):
        d = _model(self.root, "Occam-2B-CCoT")
        self.assertEqual(vs.resolve_model("Occam-2B-CCoT", self.root)["path"], str(d))
        self.assertEqual(vs.resolve_model(str(d), self.root)["name"], "Occam-2B-CCoT")
        hf = vs.resolve_model("org/some-model", self.root)
        self.assertEqual((hf["name"], hf["path"]), ("org/some-model", "org/some-model"))

    def test_command_uses_short_served_name_and_safe_host(self):
        d = _model(self.root, "Spark-X2.6-4B-vllm", sidecar={"trust_remote_code": True})
        info = vs.resolve_model("Spark-X2.6-4B-vllm", self.root)
        cmd = vs.build_vllm_command(info, trust_remote_code=info["settings"]["trust_remote_code"])
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "Spark-X2.6-4B-vllm")
        self.assertEqual(cmd[cmd.index("--model") + 1], str(d))
        self.assertIn("--trust-remote-code", cmd)
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")
        plain = vs.build_vllm_command(vs.resolve_model("org/m", self.root))
        self.assertNotIn("--trust-remote-code", plain)

    def test_stale_pid_file_never_names_a_foreign_process(self):
        vs.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        vs.PID_FILE.write_text(str(os.getpid()))  # a live process that is not vLLM
        self.assertIsNone(vs.get_active_vllm_pid())
        self.assertFalse(vs.PID_FILE.exists())


if __name__ == "__main__":
    unittest.main()
