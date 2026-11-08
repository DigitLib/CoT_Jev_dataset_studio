"""
engine_v4.py - nanoGentzen v4 engine for the Studio.

Wraps the vendored v4 package (nanogentzen_v4/) behind the interface the Studio already uses for v3
(evaluate_sequent, reload, backend, is_safetensors_loaded, device ...) and adds what v4 offers:

    prove_sequent(seq)   kernel-decided proof / refutation of a v3 Sequent, with a Curry-Howard certificate
                         checked by an independent type checker, relevance + minimal witnesses, a checked
                         Kripke countermodel, missing facts and a premise-consistency check
    decide(request)      the v4 /v1/decide API in-process (noul / choice / score questions)
    system_one(request)  the v4 /v1/systemone estimate (no proof)

Backends: "torch" (GPU/CPU), "numpy" (no torch needed), "symbolic" (no weights: fixed-order G4ip).
The network only orders proof search and gives the System 1 estimate; every verdict comes from the kernel,
so all three backends return the same verdicts.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_V4_DIR = PROJECT_DIR / "model"

# identifiers the v4 formula parser reads as operators or falsum (see nanogentzen_v4/parser.py)
_V4_RESERVED = {"and", "or", "not", "iff", "implies", "therefore", "falsum"}
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# v3 atom names that mean falsum (nanogentzen/kernel.py RESERVED_FALSUM)
_V3_FALSUM = {"0", "FALSUM", "false", "BOT", "_|_", "⊥"}


def read_model_config(model_path: Optional[str]) -> Tuple[Optional[Path], Dict[str, Any]]:
    """(model dir, config.json contents) for a model folder or a model.safetensors path."""
    if not model_path:
        return None, {}
    p = Path(model_path)
    d = p if p.is_dir() else p.parent
    cfg_path = d / "config.json"
    if not cfg_path.exists():
        return d, {}
    try:
        return d, json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:  # unreadable config: treat as unknown
        logger.warning(f"Cannot read {cfg_path}: {e}")
        return d, {}


def is_v4_config(cfg: Dict[str, Any]) -> bool:
    return int(cfg.get("format_version", 3) or 3) >= 4 or cfg.get("input_format") == "normalized_canonical"


def pretty_sequent(s: str) -> str:
    """v4 ASCII sequent -> the Unicode form the Studio shows (and cot_generator parses: goal after '⊢')."""
    if not s:
        return s
    if "|-" in s:
        g, d = s.split("|-", 1)
        g, d = g.strip(), d.strip()
        g = "∅" if g in ("", "0") else _pretty_formula(g)
        d = "⊥" if d in ("", "0") else _pretty_formula(d)
        return f"{g} ⊢ {d}"
    return _pretty_formula(s)


def _pretty_formula(s: str) -> str:
    return (s.replace("=>", "→").replace(" & ", " ∧ ").replace(" | ", " ∨ ").replace("~", "¬"))


class _SymbolicModel:
    """Stand-in model when no weights are available: the engine runs with the network switched off."""

    accepts_numpy = True

    def __init__(self, cfg: Dict[str, Any]):
        base = {"block_size": 256, "max_antecedents": 32, "tokenizer_type": "word",
                "input_format": "normalized_canonical", "s1_temperature": 1.0, "search_confidence_threshold": 0.0,
                "rule_classes": ["R_OR_1", "R_OR_2", "L_IMP_IMP", "L_IMP_NOT", "L_NOT_IMP", "L_NOT_NOT"],
                "num_epistemic_classes": 4}
        base.update(cfg or {})
        self.config = SimpleNamespace(**base)

    def to(self, *_a, **_k):
        return self

    def eval(self):
        return self

    def predict_epistemic(self, input_ids):  # never called with network=False; defined for safety
        n = len(input_ids)
        e = np.tile(np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32), (n, 1))
        return (np.full((n, 6), 1 / 6, np.float32), np.full((n, 32), 1 / 32, np.float32), e, e[:, 2] + 0.5 * e[:, 1])


class NanoGentzenV4Engine:
    version = 4

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        backend: str = "auto",          # auto | torch | numpy | symbolic
        search_budget: int = 64,
        max_depth: int = 8,
        network: bool = True,
    ):
        self.requested_device = device
        self.requested_backend = backend
        self.search_budget = int(search_budget)
        self.max_depth = int(max_depth)
        self.network = bool(network)
        self._load(model_path)

    # ------------------------------------------------------------------ loading
    def _load(self, model_path: Optional[str]):
        from nanogentzen_v4 import jev_server as J

        self._J = J
        mdir, cfg = read_model_config(model_path or str(DEFAULT_V4_DIR))
        if mdir is None or not cfg:
            mdir, cfg = read_model_config(str(DEFAULT_V4_DIR))
        self.model_dir = mdir
        self.model_path = mdir / "model.safetensors" if mdir else None
        self.model_config = cfg
        self.report = self._read_json(mdir, "report.json")
        self.gates = self._read_json(mdir, "gates.json")
        if cfg and not is_v4_config(cfg):
            raise ValueError(f"{mdir} holds a v3 checkpoint; load it with NanoGentzenEngine (nanogentzen.engine).")

        model, backend, device = None, "symbolic", "cpu"
        has_weights = bool(mdir and (mdir / "model.safetensors").exists() and cfg)
        want = self.requested_backend
        if has_weights and want in ("auto", "torch"):
            try:
                import torch
                from nanogentzen_v4.modeling_nanogentzen import GentzenPolicyValueModel

                device = ("cuda" if torch.cuda.is_available() else "cpu") if self.requested_device in ("auto", None) \
                    else self.requested_device
                model = GentzenPolicyValueModel.from_pretrained(str(mdir), local_files_only=True).to(device).eval()
                backend = "torch"
            except Exception as e:
                if want == "torch":
                    raise
                logger.info(f"nanoGentzen v4: torch backend unavailable ({type(e).__name__}: {e}); using NumPy.")
                model = None
        if model is None and has_weights and want in ("auto", "numpy", "torch"):
            from nanogentzen_v4.numpy_model import NumpyGentzenModel

            model, backend, device = NumpyGentzenModel(str(mdir)), "numpy", "cpu"
        if model is None:
            model, backend, device = _SymbolicModel(cfg), "symbolic", "cpu"

        self.core = J.ProductionGentzenEngine(base_dir=mdir or DEFAULT_V4_DIR, model=model, device=device)
        self.model = model
        self.tokenizer = self.core.tokenizer
        self.backend = f"v4_{backend}"
        self.device = self.target_device = device
        self.is_safetensors_loaded = backend in ("torch", "numpy")
        self.is_onnx_loaded = self.is_safetensors_loaded  # v3 compatibility flag
        if backend == "symbolic":
            self.network = False
        logger.info(f"nanoGentzen v4 loaded: backend={backend} device={device} dir={mdir}")

    @staticmethod
    def _read_json(d: Optional[Path], name: str) -> Dict[str, Any]:
        try:
            return json.loads((d / name).read_text(encoding="utf-8")) if d and (d / name).exists() else {}
        except Exception:
            return {}

    def reload(self, model_path: str):
        self._load(model_path)

    def status(self) -> Dict[str, Any]:
        """Short description for the sidebar and dataset metadata."""
        g = self.gates.get("gates", {}) if isinstance(self.gates, dict) else {}
        return {
            "engine": "nanoGentzen-v4",
            "backend": self.backend,
            "device": self.device,
            "network": self.network,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "s1_temperature": getattr(self.model.config, "s1_temperature", None),
            "search_confidence_threshold": getattr(self.model.config, "search_confidence_threshold", None),
            "best_epoch": self.report.get("best_epoch"),
            "gates": {k: v.get("pass") for k, v in g.items()},
            "all_must_pass": self.gates.get("all_must_pass"),
        }

    def metadata(self) -> Dict[str, Any]:
        s = self.status()
        return {"engine": s["engine"], "backend": s["backend"], "network": s["network"],
                "s1_temperature": s["s1_temperature"], "best_epoch": s["best_epoch"]}

    # ------------------------------------------------------------------ v3 <-> v4 conversion
    def to_v4_formula(self, f: Any, names: Dict[str, str]):
        from nanogentzen_v4.kernel import BOT, And, Imp, Not, Or, Var

        if f is None:
            return None
        cls = type(f).__name__
        if hasattr(f, "inner") and cls == "Not":
            return Not(self.to_v4_formula(f.inner, names))
        if hasattr(f, "left") and hasattr(f, "right"):
            ctor = {"And": And, "Or": Or, "Imp": Imp}.get(cls)
            if ctor is None:
                raise TypeError(f"unsupported formula node {cls}")
            return ctor(self.to_v4_formula(f.left, names), self.to_v4_formula(f.right, names))
        if hasattr(f, "name"):
            name = str(f.name)
            if name in _V3_FALSUM:
                return BOT
            return Var(self._safe_name(name, names))
        if cls == "Bot":
            return BOT
        raise TypeError(f"unsupported formula node {cls}")

    _RENAME_PREFIX = "rn__"

    @staticmethod
    def _safe_name(name: str, names: Dict[str, str]) -> str:
        """
        v4-safe identifier for `name`. Valid identifiers pass through unchanged, unless they start with the
        rename prefix: every renamed atom starts with it, so a renamed atom can never coincide with an original
        atom ("x y" and a real atom "a_x_y" used to collapse into one variable).
        """
        prefix = NanoGentzenV4Engine._RENAME_PREFIX
        if (_IDENT.match(name) and name.lower() not in _V4_RESERVED and name != "FALSUM"
                and not name.startswith(prefix)):
            return name
        if name in names:
            return names[name]
        base = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or "x"
        new = f"{prefix}{base}"
        while new in names.values():
            new += "_"
        names[name] = new
        return new

    def to_v4_sequent(self, seq: Any):
        """v3 Sequent (or v4 Sequent, or sequent text) -> (v4 Sequent, {v4 name: original name})."""
        from nanogentzen_v4.kernel import Sequent as V4Sequent
        from nanogentzen_v4.parser import parse_symbolic_sequent

        if isinstance(seq, V4Sequent):
            return seq, {}
        if isinstance(seq, str):
            s = parse_symbolic_sequent(seq)
            if s is None:
                raise ValueError(f"cannot parse sequent {seq!r}")
            return s, {}
        names: Dict[str, str] = {}
        gamma = tuple(self.to_v4_formula(g, names) for g in seq.gamma)
        delta = tuple(self.to_v4_formula(d, names) for d in seq.delta[:1])
        return V4Sequent(gamma, delta), {v: k for k, v in names.items()}

    @staticmethod
    def restore_names(obj: Any, rev: Dict[str, str]) -> Any:
        """Put original atom names back into strings / nested results (only when a name had to be changed)."""
        if not rev:
            return obj
        pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(rev, key=len, reverse=True)) + r")\b")
        if isinstance(obj, str):
            return pat.sub(lambda m: rev[m.group(0)], obj)
        if isinstance(obj, list):
            return [NanoGentzenV4Engine.restore_names(x, rev) for x in obj]
        if isinstance(obj, dict):
            return {k: NanoGentzenV4Engine.restore_names(v, rev) for k, v in obj.items()}
        return obj

    # ------------------------------------------------------------------ System 1
    def evaluate_sequent(self, seq: Any) -> Dict[str, Any]:
        """
        v3-compatible System 1 call. Returns rule_probs (6 G4ip choice moves), pivot_probs (32 slots),
        epistemic_probs [false, undecided, true, inconsistent], provability_value, epistemic, confidence, mode.
        """
        v4seq, _ = self.to_v4_sequent(seq)
        s1 = self.core.system1_evaluate(v4seq, network=self.network)
        ep = s1.get("epistemic", {})
        ep_arr = np.array([ep.get("false", 0.0), ep.get("undecided", 0.0), ep.get("true", 0.0),
                           ep.get("inconsistent", 0.0)], dtype=np.float32)
        rule_probs = np.zeros(6, np.float32)
        pivot_probs = np.zeros(32, np.float32)
        if self.network and self.backend != "v4_symbolic":
            ev = self.core.searcher.evaluate_sequent(v4seq)
            if ev is not None:
                rule_probs, pivot_probs = np.asarray(ev[0], np.float32), np.asarray(ev[1], np.float32)
        return {
            "rule_probs": rule_probs,
            "pivot_probs": pivot_probs,
            "epistemic_probs": ep_arr,
            "epistemic": ep,
            "provability_value": float(np.clip(s1.get("confidence", 0.5), 0.0, 1.0)),
            "confidence": s1.get("confidence"),
            "window_overflow": bool(s1.get("window_overflow", False)),
            "mode": self.backend,
        }

    # ------------------------------------------------------------------ System 2
    def prove_sequent(
        self,
        seq: Any,
        budget: Optional[int] = None,
        max_depth: Optional[int] = None,
        decide_negation: bool = True,
        check_consistency: bool = True,
        want_missing_facts: bool = True,
    ) -> Dict[str, Any]:
        """
        Decide `seq` with the v4 kernel. Status is one of
          PROVEN_TRUE               goal proven (tree verified, certificate normally checked)
          PROVEN_FALSE              the negation of the goal is proven
          INCONSISTENT_PREMISES     premises prove falsum (any proof would be ex falso); not usable as data
          INTUITIONISTIC_UNDECIDED  neither goal nor negation provable; Kripke countermodel + missing facts
          SEARCH_BUDGET_EXHAUSTED   the time-capped complete prover gave up (rare)
        """
        from nanogentzen_v4.kernel import Not, Sequent as V4Sequent, normalize_sequent

        v4seq, rev = self.to_v4_sequent(seq)
        v4seq = normalize_sequent(v4seq)
        budget = int(budget or self.search_budget)
        max_depth = int(max_depth or self.max_depth)
        out: Dict[str, Any] = {"sequent": v4seq.to_str(), "engine": "nanoGentzen-v4", "backend": self.backend}

        s1 = self.core.system1_evaluate(v4seq, network=self.network)
        out["system1"] = s1
        conflict = self.core._conflict(list(v4seq.gamma)) if (check_consistency and v4seq.gamma) else None
        if conflict:
            out.update(status="INCONSISTENT_PREMISES", proved=False, conflicting_premises=conflict,
                       proof_tree=None, certificate=None, curry_howard_term="", countermodel=None,
                       relevance=self._empty_relevance(v4seq))
            return self.restore_names(out, rev)

        ok, tree, cm, rel, term, info = self.core.system2_prove(v4seq, max_depth, budget, network=self.network)
        out["search"] = {"positive": info}
        status = "PROVEN_TRUE" if ok else ("SEARCH_BUDGET_EXHAUSTED" if info.get("exhausted") else "INTUITIONISTIC_UNDECIDED")
        refutation = None
        if not ok and decide_negation and v4seq.delta:
            neg_seq = V4Sequent(v4seq.gamma, (Not(v4seq.delta[0]),))
            ok_n, tree_n, _cm_n, rel_n, term_n, info_n = self.core.system2_prove(neg_seq, max_depth, budget,
                                                                                network=self.network)
            out["search"]["negative"] = info_n
            if ok_n:
                status = "PROVEN_FALSE"
                refutation = {"sequent": normalize_sequent(neg_seq).to_str(), "proof_tree": tree_n,
                              "curry_howard_term": term_n if term_n != "nil" else "",
                              "certificate": info_n.get("certificate"), "relevance": rel_n}
        cert = info.get("certificate") if ok else None
        out.update(
            status=status,
            proved=bool(ok),
            proof_tree=tree if ok else None,
            certificate=cert,
            certificate_checked=bool(cert and cert.get("checked")),
            certificate_error=info.get("certificate_error"),
            curry_howard_term=(term if ok and term != "nil" else ""),
            relevance=rel if ok else self._empty_relevance(v4seq),
            countermodel=cm if status == "INTUITIONISTIC_UNDECIDED" else None,
            refutation=refutation,
        )
        if status in ("INTUITIONISTIC_UNDECIDED", "SEARCH_BUDGET_EXHAUSTED") and want_missing_facts and v4seq.delta:
            try:
                out["missing_facts"] = self.core._missing_facts(v4seq.delta[0], list(v4seq.gamma), [])
            except Exception as e:  # diagnosis only; never fail the call for it
                logger.debug(f"missing_facts failed: {e}")
                out["missing_facts"] = []
        return self.restore_names(out, rev)

    @staticmethod
    def _empty_relevance(v4seq) -> Dict[str, Any]:
        return {"relevant_witnesses": [], "unused_premises": [f.to_str() for f in v4seq.gamma],
                "minimal_witnesses": [], "is_strictly_relevant": False, "relevance_ratio": 0.0}

    # ------------------------------------------------------------------ decide API
    def decide(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req = dict(request)
        req.setdefault("search_budget", self.search_budget)
        req.setdefault("max_depth", self.max_depth)
        if not self.network:
            req["network"] = False
        req.setdefault("model", "nanoGentzen-v4")
        try:
            return self.core.decide(req)
        except Exception as e:  # HTTPException from the server code -> plain ValueError for the Studio
            detail = getattr(e, "detail", None) or str(e)
            raise ValueError(str(detail)) from e

    def system_one(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req = dict(request)
        req.setdefault("model", "nanoGentzen-v4")
        if not self.network:
            req["network"] = False
        try:
            return self.core.system_one(req)
        except Exception as e:
            raise ValueError(str(getattr(e, "detail", None) or e)) from e

    # ------------------------------------------------------------------ helpers for proof trees
    @staticmethod
    def iter_postorder(tree: Optional[dict]):
        if not tree:
            return
        for b in tree.get("branches", []) or []:
            yield from NanoGentzenV4Engine.iter_postorder(b)
        yield tree

    @staticmethod
    def tree_depth(tree: Optional[dict]) -> int:
        if not tree:
            return 0
        bs = tree.get("branches", []) or []
        return 1 + (max(NanoGentzenV4Engine.tree_depth(b) for b in bs) if bs else 0)

    @staticmethod
    def tree_size(tree: Optional[dict]) -> int:
        if not tree:
            return 0
        return 1 + sum(NanoGentzenV4Engine.tree_size(b) for b in tree.get("branches", []) or [])


def is_v4_engine(engine: Any) -> bool:
    return getattr(engine, "version", 3) >= 4 and hasattr(engine, "prove_sequent")
