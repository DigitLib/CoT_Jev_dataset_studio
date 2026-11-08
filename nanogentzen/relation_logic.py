"""
relation_logic.py - compatibility alias for nanogentzen.cot.relation_logic (ClearCoT pipeline only).

The code lives in nanogentzen/cot/relation_logic.py. This name is kept so existing imports keep working;
importing it yields the very same module object.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("nanogentzen.cot.relation_logic")
