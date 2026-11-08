"""
architect_jev.py - compatibility alias for nanogentzen.jev.architect (Jev pipeline only).

The code lives in nanogentzen/jev/architect.py. This name is kept so existing imports keep working;
importing it yields the very same module object.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("nanogentzen.jev.architect")
