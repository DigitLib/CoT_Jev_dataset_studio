"""
vendor_v4_engine.py - copy a trained nanoGentzen v4 model folder into the Studio.

    python tools/vendor_v4_engine.py --src v41/v41/model            # code -> nanogentzen_v4/, weights -> model/
    python tools/vendor_v4_engine.py --src ../train_v4/runs/v4/model --report-dir ../train_v4/runs/v4

Why a separate package: the v4 engine files import each other as top-level modules (`import kernel`,
`from nanogentzen.parser import ...`) and the server re-points sys.modules["nanogentzen.*"] at them.
Inside the Studio, which has its own v3 `nanogentzen` package, that would silently swap kernels.
This tool rewrites every such import to a package-relative one, removes the sys.modules aliasing,
makes torch / fastapi optional (a NumPy backend is provided in numpy_model.py), and stops with an
error if the upstream code no longer matches what it expects, instead of producing a half-patched copy.
"""
import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_FILES = ["kernel", "parser", "search", "certificates", "checker", "modeling_nanogentzen",
              "configuration_nanogentzen", "tokenization_nanogentzen", "domain_registry", "jev_server"]
WEIGHT_FILES = ["model.safetensors", "config.json", "vocab.json", "tokenizer_config.json", "special_tokens_map.json"]
REPORT_FILES = ["report.json", "gates.json", "after_train_results.json"]
HEADER = "# Vendored from the nanoGentzen v4 model folder by tools/vendor_v4_engine.py. Do not edit by hand.\n"


def _sub(text: str, old: str, new: str, fname: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"[vendor] {fname}: expected {count} occurrence(s) of:\n{old}\n... found {n}. "
                         f"Upstream code changed; update tools/vendor_v4_engine.py.")
    return text.replace(old, new)


def _span(text: str, start: str, end: str, new: str, fname: str) -> str:
    i = text.find(start)
    j = text.find(end, i)
    if i < 0 or j < 0:
        raise SystemExit(f"[vendor] {fname}: span markers not found ({start[:40]!r} .. {end[:40]!r}).")
    return text[:i] + new + text[j + len(end):]


def patch_kernel(t, f):
    t = _sub(t, '_sys.modules.setdefault("kernel", _sys.modules[__name__])\n'
                '_sys.modules.setdefault("nanogentzen.kernel", _sys.modules[__name__])\n', "", f)
    t = _sub(t, "        try:\n            from nanogentzen.parser import parse_symbolic_sequent as parser_fn\n"
                "        except ImportError:\n            try:\n                from parser import parse_symbolic_sequent as parser_fn\n"
                "            except ImportError:\n                return False\n",
             "        from .parser import parse_symbolic_sequent as parser_fn\n", f)
    t = _sub(t, "    try:\n        from nanogentzen.parser import parse_symbolic_sequent as parse\n"
                "    except ImportError:\n        from parser import parse_symbolic_sequent as parse\n",
             "    from .parser import parse_symbolic_sequent as parse\n", f)
    t = _sub(t, "        try:\n            import certificates as _c\n        except ImportError:\n"
                "            from nanogentzen import certificates as _c\n",
             "        from . import certificates as _c\n", f)
    return t


def patch_certificates(t, f):
    t = _sub(t, "try:\n    from kernel import BOT, And, Bot, Imp, Not, Or, Sequent, Var, g4ip_prove, normalize_formula\n"
                "    import checker as _checker\nexcept ImportError:\n"
                "    from nanogentzen.kernel import BOT, And, Bot, Imp, Not, Or, Sequent, Var, g4ip_prove, normalize_formula\n"
                "    import nanogentzen.checker as _checker\n",
             "from .kernel import BOT, And, Bot, Imp, Not, Or, Sequent, Var, g4ip_prove, normalize_formula\n"
             "from . import checker as _checker\n", f)
    t = _sub(t, "    try:\n        from nanogentzen.parser import parse_symbolic_sequent\n    except ImportError:\n"
                "        from parser import parse_symbolic_sequent\n",
             "    from .parser import parse_symbolic_sequent\n", f)
    return t


def patch_domain_registry(t, f):
    t = _sub(t, "try:\n    from kernel import And, Formula, Imp, Not, Var\nexcept ImportError:\n    try:\n"
                "        from nanogentzen.kernel import And, Formula, Imp, Not, Var\n    except ImportError:\n"
                "        from .kernel import And, Formula, Imp, Not, Var\n",
             "from .kernel import And, Formula, Imp, Not, Var\n", f)
    t = _sub(t, "try:\n    from parser import FormulaParser\nexcept ImportError:\n    try:\n"
                "        from nanogentzen.parser import FormulaParser\n    except ImportError:\n"
                "        from .parser import FormulaParser\n",
             "from .parser import FormulaParser\n", f)
    return t


def patch_parser(t, f):
    t = _sub(t, "try:\n    from kernel import BOT, And, Bot, Formula, Imp, Not, Or, Sequent, Var\nexcept ImportError:\n"
                "    try:\n        from nanogentzen.kernel import BOT, And, Bot, Formula, Imp, Not, Or, Sequent, Var\n"
                "    except ImportError:\n        from .kernel import BOT, And, Bot, Formula, Imp, Not, Or, Sequent, Var\n",
             "from .kernel import BOT, And, Bot, Formula, Imp, Not, Or, Sequent, Var\n", f)
    t = _span(t, "try:\n    from domain_registry import (",
              "        from .domain_registry import (\n            REGISTRY,\n            register_domain,\n"
              "            benchmark_detect_workflow,\n            compile_benchmark_structured_decision,\n"
              "            BENCHMARK_CALIBRATED_DISTRIBUTIONS,\n        )\n",
              "from .domain_registry import (\n    REGISTRY,\n    register_domain,\n    benchmark_detect_workflow,\n"
              "    compile_benchmark_structured_decision,\n    BENCHMARK_CALIBRATED_DISTRIBUTIONS,\n)\n", f)
    return t


def patch_search(t, f):
    t = _span(t, "try:\n    import torch\n", "        from .tokenization_nanogentzen import LogicTokenizer\n",
              "try:\n    import torch\nexcept ImportError:  # the search itself does not need torch; the model does\n"
              "    torch = None\n\n"
              "from .kernel import (CHOICE_LEFT_RULES, RULES, Sequent, _first_invertible, apply_rule, applicable_moves,\n"
              "                     canonicalize_sequent, find_kripke_countermodel, normalize_sequent, rule_label)\n"
              "from . import kernel as _kernel\nfrom .backend import as_input, to_numpy\n", f)
    t = _sub(t, "        x = torch.tensor([padded], dtype=torch.long, device=self.device)\n"
                "        rule_logits, pivot_logits, constructive_val, _ = self.model(input_ids=x)\n"
                "        return (F.softmax(rule_logits[0], dim=-1), F.softmax(pivot_logits[0], dim=-1),\n"
                "                self.model._last_epistemic_probs[0], float(constructive_val[0].item()))\n",
             "        x = as_input([padded], self.device, self.model)\n"
             "        rp, pp, ep, val = self.model.predict_epistemic(x)   # probabilities, NumPy or torch backend\n"
             "        return to_numpy(rp)[0], to_numpy(pp)[0], to_numpy(ep)[0], float(to_numpy(val)[0])\n", f)
    t = _sub(t, "        n_piv = pivot_probs.size(-1)\n", "        n_piv = int(pivot_probs.shape[-1])\n", f)
    return t


def patch_modeling(t, f):
    t = _sub(t, "try:\n    from .configuration_nanogentzen import GentzenConfig, PolicyValueConfig\nexcept ImportError:\n"
                "    try:\n        from configuration_nanogentzen import GentzenConfig, PolicyValueConfig\n"
                "    except ImportError:\n        from nanogentzen.configuration_nanogentzen import GentzenConfig, PolicyValueConfig\n",
             "from .configuration_nanogentzen import GentzenConfig, PolicyValueConfig\n", f)
    t = _sub(t, "            try:\n                import jev_server\n            except ImportError:\n"
                "                from . import jev_server\n            try:\n                import tokenization_nanogentzen\n"
                "            except ImportError:\n                from . import tokenization_nanogentzen\n",
             "            from . import jev_server\n            from . import tokenization_nanogentzen\n", f)
    return t


JEV_HEADER = '''try:
    import torch
except ImportError:          # NumPy backend (numpy_model.py) needs no torch
    torch = None
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:          # in-process use inside the Studio needs no web server
    class HTTPException(Exception):
        def __init__(self, status_code: int = 400, detail: Any = None):
            super().__init__(detail)
            self.status_code, self.detail = status_code, detail

    class FastAPI:
        def __init__(self, *a, **k):
            pass

        def add_middleware(self, *a, **k):
            pass

        def get(self, *a, **k):
            return lambda fn: fn

        post = get

    CORSMiddleware = None
from pydantic import BaseModel, Field

from . import certificates as _certs
from . import parser as parser_module
from .backend import as_input, to_numpy, no_grad as _no_grad
from .kernel import (
    BOT,
    CHOICE_LEFT_RULES,
    applicable_moves,
    find_kripke_countermodel,
    g4ip_prove,
    model_view_sequent,
    normalize_sequent,
    And,
    Formula,
    Imp,
    Not,
    Or,
    RULES,
    Sequent,
    Var,
    apply_rule,
    canonicalize_sequent,
    extract_kripke_countermodel,
    extract_relevance_witnesses,
    proof_tree_to_curry_howard,
    verify_proof_tree_strict,
)
from .parser import (
    compile_to_formula,
    ground_state_generic,
    parse_symbolic_sequent,
    detect_workflow,
    compile_structured_decision,
    get_calibrated_distribution,
)
from .search import NeuralProofSearch
from .tokenization_nanogentzen import LogicTokenizer

try:
    from .modeling_nanogentzen import GentzenConfig, GentzenPolicyValueModel
except ImportError:          # torch missing: callers must pass `model=` (e.g. NumpyGentzenModel)
    GentzenConfig = GentzenPolicyValueModel = None

_HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("NANOGENTZEN_DIR", str(_HERE.parent / "model"))).resolve()
'''


def patch_jev_server(t, f):
    t = _span(t, "import torch\nimport torch.nn.functional as F\nfrom fastapi",
              "    from nanogentzen.tokenizer import LogicTokenizer\n", JEV_HEADER, f)
    t = _sub(t, "    @torch.no_grad()\n", "    @_no_grad\n", f, count=2)
    t = _sub(t, "        self.device = device or os.getenv(\"DEVICE\") or (\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
             "        self.device = device or os.getenv(\"DEVICE\") or (\n"
             "            \"cuda\" if (torch is not None and torch.cuda.is_available()) else \"cpu\")\n", f)
    t = _sub(t, "                from tokenization_nanogentzen import tokenizer_for_config\n            except ImportError:\n"
                "                from nanogentzen.tokenization_nanogentzen import tokenizer_for_config\n",
             "                from .tokenization_nanogentzen import tokenizer_for_config\n            except ImportError:\n"
             "                raise\n", f)
    t = _sub(t, "                eval_seq = sys.modules[canonicalize_sequent.__module__].model_view_sequent(seq)[0]   # v4 checkpoints\n",
             "                eval_seq = model_view_sequent(seq)[0]   # v4 checkpoints\n", f)
    t = _sub(t, "            inp = torch.tensor(batch_padded, dtype=torch.long, device=self.device)\n",
             "            inp = as_input(batch_padded, self.device, self.model)\n", f)
    t = _sub(t, "            ep_cpu = ep_probs.cpu()\n            val_cpu = val_score.cpu()\n",
             "            ep_cpu = to_numpy(ep_probs)\n            val_cpu = to_numpy(val_score)\n", f)
    return t


PATCHES = {"kernel": patch_kernel, "certificates": patch_certificates, "domain_registry": patch_domain_registry,
           "parser": patch_parser, "search": patch_search, "modeling_nanogentzen": patch_modeling,
           "jev_server": patch_jev_server}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="v4 model folder (has config.json, model.safetensors, kernel.py ...)")
    ap.add_argument("--report-dir", default=None, help="folder with report.json / gates.json (default: parent of --src)")
    ap.add_argument("--code-out", default=str(ROOT / "nanogentzen_v4"))
    ap.add_argument("--model-out", default=str(ROOT / "model"))
    ap.add_argument("--code-only", action="store_true")
    a = ap.parse_args()
    src, code_out, model_out = Path(a.src).resolve(), Path(a.code_out), Path(a.model_out)
    code_out.mkdir(parents=True, exist_ok=True)
    for name in CODE_FILES:
        text = (src / f"{name}.py").read_text(encoding="utf-8")
        if name in PATCHES:
            text = PATCHES[name](text, name)
        leftover = re.findall(r"^\s*(?:from|import) (?:nanogentzen\b|kernel\b|parser\b|search\b|certificates\b|checker\b)", text, re.M)
        if leftover:
            raise SystemExit(f"[vendor] {name}.py still has top-level-style imports: {leftover}")
        (code_out / f"{name}.py").write_text(HEADER + text, encoding="utf-8")
        print(f"[vendor] code  {name}.py")
    if not a.code_only:
        model_out.mkdir(parents=True, exist_ok=True)
        for fn in WEIGHT_FILES:
            if (src / fn).exists():
                shutil.copy2(src / fn, model_out / fn)
                print(f"[vendor] model {fn}")
        rep = Path(a.report_dir) if a.report_dir else src.parent
        for fn in REPORT_FILES:
            if (rep / fn).exists():
                shutil.copy2(rep / fn, model_out / fn)
                print(f"[vendor] report {fn}")
    print("[vendor] done. Re-apply the Studio safety patches to the engine "
          "(this tool does not carry them), then run: python tools/run_tests.py")


if __name__ == "__main__":
    main()
