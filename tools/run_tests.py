"""
run_tests.py - run the whole test suite with a readable summary.

    python tools/run_tests.py            # all tests
    python tools/run_tests.py -k v4      # only test modules whose name contains "v4"

Failures caused by a missing package are listed separately with the pip command that fixes them, so they
are not confused with real failures.
"""
import argparse
import io
import re
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Tests must never touch the real data/ folder (DB, knowledge graphs, exports)
import os
import tempfile
os.environ.setdefault("DTSTUDIO_DATA_DIR", tempfile.mkdtemp(prefix="dtstudio-tests-"))

PIP = {"pyarrow": "pyarrow", "fastparquet": "pyarrow", "torch": "torch", "networkx": "networkx",
       "streamlit": "streamlit", "pymupdf": "pymupdf", "fitz": "pymupdf", "docx": "python-docx",
       "pandas": "pandas", "httpx": "httpx", "transformers": "transformers", "safetensors": "safetensors"}


def missing_package(text: str):
    m = re.search(r"No module named '([\w.]+)'", text) or re.search(r"tried using: '(\w+)'", text)
    if m:
        return m.group(1).split(".")[0]
    if "safetensors_v3" in text and "heuristic" in text:
        return "torch"          # v3 tests that need the torch model
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="substring of test module names to run")
    a = ap.parse_args()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        if a.k in f.stem:
            suite.addTests(loader.loadTestsFromName(f"tests.{f.stem}"))
    stream = io.StringIO()
    t0 = time.time()
    res = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    env, real = {}, []
    for test, tb in res.failures + res.errors:
        pkg = missing_package(tb)
        if pkg:
            env.setdefault(pkg, []).append(test.id())
        else:
            real.append((test.id(), tb))
    print(f"Ran {res.testsRun} tests in {time.time() - t0:.1f}s: {res.testsRun - len(res.failures) - len(res.errors) - len(res.skipped)} passed, "
          f"{len(real)} failed, {sum(map(len, env.values()))} need a missing package, {len(res.skipped)} skipped")
    for pkg, tests in env.items():
        print(f"\n  missing '{pkg}' (pip install {PIP.get(pkg, pkg)}): {len(tests)} test(s)")
        for t in tests:
            print(f"    {t}")
    if res.skipped:
        print("\n  skipped:")
        for test, why in res.skipped:
            print(f"    {test.id()}: {why}")
    for tid, tb in real:
        print(f"\n  FAILED {tid}\n" + "\n".join("    " + l for l in tb.strip().splitlines()[-12:]))
    sys.exit(1 if real or env else 0)


if __name__ == "__main__":
    main()
