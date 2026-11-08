"""
Batch generation: stopping, saving of in-flight samples and progress.
"""

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tests  # noqa: F401  (redirects all runtime data to a temp dir; see tests/__init__.py)

from core.batch_generator import BatchDatasetGenerator
from core.dataset_manager import DatasetManager, DatasetSample


class SlowBatch(BatchDatasetGenerator):
    def __init__(self, db, delay=0.2):
        super().__init__(q_generator=None, architect=None, cot_generator=None, db_manager=db)
        self.delay, self.started = delay, 0
        self._lock = threading.Lock()

    def _assemble_seeds(self, mode, kg_paths, chunks):
        return [{"type": "chunk", "i": i} for i in range(40)]

    def _synthesize_seed(self, seed, chunks, model, status, seen_questions, dedup_lock):
        with self._lock:
            self.started += 1
        time.sleep(self.delay)
        s = DatasetSample(sample_id=f"S-{seed['i']:03d}", doc_name="d.pdf", question=f"q{seed['i']}",
                          reasoning="r", answer="a", gentzen_verified=True)
        return s, {"sample_id": s.sample_id}, True, 0.9


class TestBatchGeneration(unittest.TestCase):
    def setUp(self):
        self.db = DatasetManager(db_path=str(Path(tempfile.mkdtemp()) / "b.db"))

    def test_request_stop_cancels_queue_and_keeps_inflight(self):
        g = SlowBatch(self.db)
        threading.Timer(0.3, g.request_stop).start()
        t0 = time.time()
        res = g.generate_batch("chunks", [], [], model="m", concurrency=4)
        self.assertLess(time.time() - t0, 2.0, "40 seeds x 0.2s / 4 workers would take ~2s without stopping")
        self.assertTrue(res["stopped"])
        self.assertLess(g.started, 40)
        self.assertEqual(len(self.db.get_samples()), res["generated_count"], "every finished sample is saved")
        self.assertGreater(res["generated_count"], 0)

    def test_interrupted_caller_does_not_wait_for_queue(self):
        g = SlowBatch(self.db)

        class Interrupt(Exception):
            pass

        def cb(done, total, sample, msg):
            if done >= 2:
                raise Interrupt()   # what Streamlit does to a running script when a button is clicked

        t0 = time.time()
        with self.assertRaises(Interrupt):
            g.generate_batch("chunks", [], [], model="m", concurrency=4, progress_callback=cb)
        self.assertLess(time.time() - t0, 1.0)
        self.assertTrue(g.stop_event.is_set())
        time.sleep(0.5)  # in-flight workers finish in the background and still save
        self.assertLess(g.started, 40)
        self.assertGreaterEqual(len(self.db.get_samples()), 2)

    def test_sequential_and_full_run(self):
        g = SlowBatch(self.db, delay=0.0)
        res = g.generate_batch("chunks", [], [], model="m", concurrency=1)
        self.assertEqual(res["generated_count"], 40)
        self.assertFalse(res["stopped"])
        g2 = SlowBatch(DatasetManager(db_path=str(Path(tempfile.mkdtemp()) / "c.db")), delay=0.0)
        self.assertEqual(g2.generate_batch("chunks", [], [], model="m", concurrency=4)["generated_count"], 40)


if __name__ == "__main__":
    unittest.main()
