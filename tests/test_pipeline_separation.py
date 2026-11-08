"""
The ClearCoT (nanogentzen/cot/) and Jev (nanogentzen/jev/) pipelines are independent: neither
imports the other; both use only the shared modules.
"""

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tests  # noqa: F401  (redirects all runtime data to a temp dir; see tests/__init__.py)


PKG = ROOT / "nanogentzen"


def _imported_modules(path: Path):
    """Absolute dotted names imported by a module inside the nanogentzen package (relative imports resolved)."""
    rel = path.relative_to(ROOT).with_suffix("")
    pkg_parts = list(rel.parts[:-1])
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg_parts[: len(pkg_parts) - node.level + 1]
                mod = ".".join(base + ([node.module] if node.module else []))
                out.add(mod)
                if not node.module:                       # "from .. import verify"
                    out |= {f"{mod}.{a.name}" for a in node.names}
            else:
                out.add(node.module or "")
    return out


class TestNoCrossImports(unittest.TestCase):
    def _check(self, own: str, other: str):
        for f in sorted((PKG / own).glob("*.py")):
            bad = {m for m in _imported_modules(f) if m == f"nanogentzen.{other}" or m.startswith(f"nanogentzen.{other}.")}
            # the compatibility modules re-export both front ends; using them would re-couple the pipelines
            bad |= {m for m in _imported_modules(f)
                    if m in ("nanogentzen.parser", "nanogentzen.architect_cot", "nanogentzen.architect_jev",
                             "nanogentzen.logical_architect", "nanogentzen.relation_logic")}
            self.assertFalse(bad, f"{f.relative_to(ROOT)} imports {sorted(bad)}")

    def test_cot_does_not_import_jev(self):
        self._check("cot", "jev")

    def test_jev_does_not_import_cot(self):
        self._check("jev", "cot")

    def test_generators_use_their_own_front_end(self):
        jev = _imported_modules(ROOT / "core" / "jev_generator.py")
        self.assertFalse({m for m in jev if m.startswith("nanogentzen.cot") or m == "nanogentzen.parser"}, jev)
        for f in ("cot_generator.py", "batch_generator.py"):
            mods = _imported_modules(ROOT / "core" / f)
            self.assertFalse({m for m in mods if m.startswith("nanogentzen.jev")}, (f, mods))

    def test_compat_aliases_are_the_same_modules(self):
        import nanogentzen.architect_cot as a_cot
        import nanogentzen.architect_jev as a_jev
        import nanogentzen.relation_logic as rl
        import nanogentzen.cot.architect as cot_arch
        import nanogentzen.jev.architect as jev_arch
        import nanogentzen.cot.relation_logic as cot_rl
        from nanogentzen import parser
        self.assertIs(a_cot, cot_arch)
        self.assertIs(a_jev, jev_arch)
        self.assertIs(rl, cot_rl)
        for name in ("parse_sequent", "text_to_sequent", "path_to_sequent", "ground_state_generic",
                     "select_relevant_gamma", "clean_atom_name", "FormulaParser"):
            self.assertTrue(hasattr(parser, name), name)


if __name__ == "__main__":
    unittest.main()
