"""
Redirect all runtime data to a throw-away directory before any other test module imports `config`.

`python -m unittest discover -s tests` imports test modules as top-level modules, so tests/__init__.py is NOT
executed in that mode; unittest discovery imports modules in sorted order, so this module (test_000_...) runs
first. tests/__init__.py does the same for `python -m unittest tests.test_x`.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_DATA = (Path(__file__).resolve().parent.parent / "data").resolve()

if not os.environ.get("DTSTUDIO_DATA_DIR"):
    os.environ["DTSTUDIO_DATA_DIR"] = tempfile.mkdtemp(prefix="dtstudio-tests-")

if "config" in sys.modules and Path(sys.modules["config"].DATA_DIR).resolve() == _REPO_DATA:
    raise RuntimeError("config was imported before the test data directory was redirected; "
                       "refusing to run tests against the real data/ folder")


class TestDataIsolation(unittest.TestCase):
    def test_config_points_at_temp_data_dir(self):
        import config
        self.assertNotEqual(Path(config.DATA_DIR).resolve(), _REPO_DATA)
        self.assertNotEqual(Path(config.KG_DIR).resolve(), _REPO_DATA / "kg")
        self.assertNotEqual(Path(config.DB_PATH).resolve(), _REPO_DATA / "clearcot.db")
