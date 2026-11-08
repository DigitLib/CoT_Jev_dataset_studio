"""
engine.py (nanoGentzen-v3 Neural Inference Engine)
Runs native PyTorch / SafeTensors nanoGentzen-v3:
  - 4.93M parameters
  - 32 Span Pointer pivots
  - 3-class Saul Kripke Epistemic Head [P(False), P(Undecided), P(True)]
  - Constructive provability value scoring
  - Deterministic symbolic heuristic fallback when running without weights
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .kernel import (
    Sequent,
    RULE_NAMES,
    And,
    Or,
    Imp,
    Not,
    Atom,
    Var,
    canonicalize_sequent,
)

logger = logging.getLogger(__name__)


class NanoGentzenEngine:
    """
    Authoritative neural inference engine for nanoGentzen (v4).
    Directly runs the Dyckhoff G4ip kernel with Curry-Howard certificate generation,
    Kripke countermodel extraction, and System 1 heuristic valuation.
    """

    version = 4

    def __new__(cls, model_path: Optional[str] = None, device: str = "auto", **kwargs):
        from .engine_v4 import NanoGentzenV4Engine

        return NanoGentzenV4Engine(model_path=model_path, device=device, **kwargs)

    def _resolve_model_path(self, model_path: Optional[str]) -> Optional[Path]:
        if model_path:
            p = Path(model_path)
            if p.exists():
                return p

        # Check default paths
        base_dir = Path(__file__).resolve().parent.parent / "model"
        candidates = [
            base_dir / "model.safetensors",
            base_dir,
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _try_init_safetensors(self, path: Path) -> bool:
        if not HAS_TORCH:
            return False

        model_dir = path if path.is_dir() else path.parent
        sft_path = path if path.is_file() and path.suffix == ".safetensors" else (model_dir / "model.safetensors")

        if not sft_path.exists():
            return False

        try:
            from .modeling_nanogentzen import GentzenPolicyValueModel
            from .tokenization_nanogentzen import LogicTokenizerHF

            target_device = (
                ("cuda" if torch.cuda.is_available() else "cpu")
                if self.device in ("auto", None)
                else self.device
            )

            logger.info(f"Loading nanoGentzen-v3 SafeTensors model on {target_device} from {model_dir}")
            self.model = GentzenPolicyValueModel.from_pretrained(str(model_dir)).to(target_device).eval()

            vocab_path = model_dir / "vocab.json"
            if not vocab_path.exists():
                vocab_path = Path(__file__).resolve().parent / "vocab.json"
            self.tokenizer = LogicTokenizerHF(str(vocab_path) if vocab_path.exists() else None)

            self.target_device = target_device
            self.device = target_device
            self.is_safetensors_loaded = True
            self.is_onnx_loaded = True  # Backward compatibility flag
            self.backend = "safetensors_v3"
            logger.info("nanoGentzen-v3 SafeTensors engine loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load SafeTensors model: {e}")
            self.model = None
            self.tokenizer = None
            return False

    def reload(self, model_path: str):
        """Reload session with a new model path."""
        p = Path(model_path)
        self.model_path = p
        self.model = None
        self.tokenizer = None
        self.is_safetensors_loaded = False
        self.is_onnx_loaded = False
        self.backend = "heuristic"

        if p.exists():
            self._try_init_safetensors(p)

    def evaluate_sequent(self, seq: Sequent) -> Dict[str, Any]:
        """
        Runs neural forward pass on a single sequent.
        Returns:
            - rule_probs: np.ndarray [11]
            - pivot_probs: np.ndarray [32]
            - epistemic_probs: Optional[np.ndarray] [3] (False, Undecided, True)
            - provability_value: float in [0.0, 1.0]
            - mode: str ('safetensors_v3', 'symbolic_heuristic')
        """
        # 1. Native SafeTensors v3 Execution (32 pivots + 3-class Kripke head)
        if self.backend == "safetensors_v3" and self.model is not None and self.tokenizer is not None:
            try:
                eval_seq, _, _ = canonicalize_sequent(seq)
                encoded = self.tokenizer.encode(eval_seq.to_str())[: self.model.config.block_size]
                padded = encoded + [self.tokenizer.pad_id] * (self.model.config.block_size - len(encoded))

                x = torch.tensor([padded], dtype=torch.long, device=self.target_device)
                with torch.no_grad():
                    rule_logits, pivot_logits, constructive_val, _ = self.model(x)
                    rule_probs = F.softmax(rule_logits[0], dim=-1).cpu().numpy()
                    pivot_probs = F.softmax(pivot_logits[0], dim=-1).cpu().numpy()

                    epistemic_probs = None
                    if hasattr(self.model, "_last_epistemic_probs") and self.model._last_epistemic_probs is not None:
                        epistemic_probs = self.model._last_epistemic_probs[0].cpu().numpy()

                    val_score = float(constructive_val[0].item())

                return {
                    "rule_probs": rule_probs,
                    "pivot_probs": pivot_probs,
                    "epistemic_probs": epistemic_probs,
                    "provability_value": float(np.clip(val_score, 0.0, 1.0)),
                    "mode": "safetensors_v3",
                }
            except Exception as e:
                logger.error(f"SafeTensors neural inference failed: {e}. Falling back to heuristic.")

        # 2. Symbolic Heuristic Fallback
        return self._heuristic_evaluate(seq)

    def _heuristic_evaluate(self, seq: Sequent) -> Dict[str, Any]:
        """Smart symbolic heuristic that estimates rule priorities when running without weights."""
        rule_probs = np.zeros(len(RULE_NAMES), dtype=np.float32)
        pivot_probs = np.zeros(32, dtype=np.float32)
        gamma = seq.antecedents
        delta = seq.succedent

        if delta is not None and (delta in gamma or seq.is_axiom()):
            rule_probs[0] = 0.95  # Axiom
            return {
                "rule_probs": rule_probs,
                "pivot_probs": pivot_probs,
                "epistemic_probs": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                "provability_value": 1.0,
                "mode": "symbolic_heuristic",
            }

        if isinstance(delta, Imp):
            rule_probs[1] = 0.85  # Imp_R
        elif isinstance(delta, And):
            rule_probs[3] = 0.85  # And_R
        elif isinstance(delta, Or):
            rule_probs[6] = 0.50  # Or_R1
            rule_probs[7] = 0.50  # Or_R2
        elif isinstance(delta, Not):
            rule_probs[9] = 0.80  # Not_R

        for idx, form in enumerate(gamma[:32]):
            pivot_probs[idx] = 1.0 / max(1, len(gamma))
            if isinstance(form, Imp):
                rule_probs[2] = max(rule_probs[2], 0.70)
            elif isinstance(form, And):
                rule_probs[4] = max(rule_probs[4], 0.65)
            elif isinstance(form, Or):
                rule_probs[8] = max(rule_probs[8], 0.60)
            elif isinstance(form, Not):
                rule_probs[10] = max(rule_probs[10], 0.60)

        total_r = np.sum(rule_probs)
        if total_r > 0:
            rule_probs /= total_r
        else:
            rule_probs[0] = 1.0

        total_p = np.sum(pivot_probs)
        if total_p > 0:
            pivot_probs /= total_p

        val = 0.85 if any(f == delta for f in gamma) or isinstance(delta, (Imp, And)) else 0.50

        return {
            "rule_probs": rule_probs,
            "pivot_probs": pivot_probs,
            "epistemic_probs": np.array([0.1, 0.4, 0.5], dtype=np.float32),
            "provability_value": val,
            "mode": "symbolic_heuristic",
        }


# Backward-compatible alias
NanoGentzenONNXEngine = NanoGentzenEngine


# ---------------------------------------------------------------------------
# Engine factory: picks the v3 or v4 engine from the checkpoint's config.json
# ---------------------------------------------------------------------------
def detect_model_version(model_path: Optional[str]) -> int:
    """
    3 for a checkpoint whose config.json is not a nanoGentzen v4 config, else 4. A missing checkpoint counts as
    v4: the v4 engine is the only engine, and it reports missing weights itself.
    """
    from .engine_v4 import read_model_config, is_v4_config, DEFAULT_V4_DIR

    target = model_path or str(DEFAULT_V4_DIR)
    _, cfg = read_model_config(target)
    if cfg and not is_v4_config(cfg):
        return 3
    return 4


def load_engine(model_path: Optional[str] = None, device: str = "auto", **v4_options):
    """
    Returns the nanoGentzen v4 engine. v3 checkpoints are no longer supported (the Studio's verification
    guarantees rely on the v4 kernel and certificate checker), so a v3 checkpoint raises a clear error.
    v4_options (backend, search_budget, max_depth, network) are passed to the v4 engine.
    """
    if detect_model_version(model_path) < 4:
        raise ValueError(f"{model_path} is a nanoGentzen v3 checkpoint. v3 is no longer supported; "
                         "use a nanoGentzen v4 model (see docs/nanogentzen_v4.md).")
    from .engine_v4 import NanoGentzenV4Engine

    return NanoGentzenV4Engine(model_path=model_path, device=device, **v4_options)

