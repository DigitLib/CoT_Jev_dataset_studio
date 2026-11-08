"""Unit tests for nanoGentzen ClearCoT Studio.

All runtime data (SQLite DB, knowledge graphs, exports, uploads) is redirected to a throw-away
directory, so running the suite can never modify the real `data/` folder.
"""

import os
import tempfile

if not os.environ.get("DTSTUDIO_DATA_DIR"):
    os.environ["DTSTUDIO_DATA_DIR"] = tempfile.mkdtemp(prefix="dtstudio-tests-")
