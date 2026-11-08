"""
Configuration module for 100% Local nanoGentzen ClearCoT & Jev Decision Studio.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import logging
import httpx

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
# DTSTUDIO_DATA_DIR relocates all runtime data (DB, uploads, exports, KG); the test suite points it at a temp dir.
DATA_DIR = Path(os.environ["DTSTUDIO_DATA_DIR"]).expanduser().resolve() if os.environ.get("DTSTUDIO_DATA_DIR") \
    else BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
EXPORTS_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "clearcot.db"
# Local vLLM models: one sub-folder per model (llm_backend/vllm_server.py); can live on another disk
VLLM_MODELS_DIR = Path(os.environ["DTSTUDIO_VLLM_MODELS_DIR"]).expanduser().resolve() \
    if os.environ.get("DTSTUDIO_VLLM_MODELS_DIR") else BASE_DIR / "vllm_model"
DEFAULT_VLLM_MODEL = os.environ.get("DTSTUDIO_DEFAULT_VLLM_MODEL", "Occam-2B-CCoT")
KG_DIR = DATA_DIR / "kg"                     # saved knowledge graphs, one <kg_id>.json per topic (core/kg_store.py)
KG_LATEST_PATH = KG_DIR / "latest.json"       # legacy single-slot file of older versions; still listed and loadable
# What happens to the last active graph on startup:
#   "ask"  - offer [Restore] / [Start empty] (default; avoids silently mixing an old topic into new work)
#   "off"  - start with an empty graph
#   "last" - restore it automatically (previous behaviour)
KG_AUTOLOAD = os.environ.get("DTSTUDIO_KG_AUTOLOAD", "ask").strip().lower()
if KG_AUTOLOAD in ("1", "true", "yes"):
    KG_AUTOLOAD = "last"
elif KG_AUTOLOAD in ("0", "false", "no"):
    KG_AUTOLOAD = "off"
elif KG_AUTOLOAD not in ("ask", "off", "last"):
    KG_AUTOLOAD = "ask"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_V4_DIR = BASE_DIR / "model"   # nanoGentzen v4 canonical model directory
DEFAULT_MODEL_PATH = MODEL_DIR / "model.safetensors"
DEFAULT_SAFETENSORS_PATH = DEFAULT_MODEL_PATH
DEFAULT_ONNX_PATH = DEFAULT_MODEL_PATH  # Backward compatibility alias

@dataclass
class LLMBackendConfig:
    backend_type: str = "ollama"  # "ollama", "lm_studio", "vllm", "custom"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "not-needed"
    available_models: List[str] = field(default_factory=list)
    selected_model: Optional[str] = None  # Text Generation Model
    selected_embedding_model: Optional[str] = None  # Text Embedding Model
    separate_embedding_endpoint: bool = False
    embedding_backend_type: str = "ollama"
    embedding_base_url: str = "http://localhost:11434/v1"
    timeout_seconds: float = 120.0
    model_name: Optional[str] = None
    concurrency: int = 4

    @staticmethod
    def get_default_concurrency(backend_name: str) -> int:
        """Return recommended concurrency for given inference engine."""
        b = (backend_name or "").lower()
        if "vllm" in b:
            return 8
        elif "lm" in b or "studio" in b:
            return 1
        elif "ollama" in b:
            return 1
        return 4

    def __post_init__(self):
        if self.model_name and not self.selected_model:
            self.selected_model = self.model_name
        elif self.selected_model and not self.model_name:
            self.model_name = self.selected_model

    @property
    def models(self) -> List[str]:
        """Alias for available_models."""
        return self.available_models

    @models.setter
    def models(self, value: List[str]) -> None:
        self.available_models = value

    def list_models_from_backend(self, url: Optional[str] = None) -> List[str]:
        """
        Fetch available models list from the backend endpoint (/models).
        """
        target_url = (url or self.base_url).rstrip("/")
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            with httpx.Client(timeout=min(self.timeout_seconds, 5.0)) as client:
                res = client.get(f"{target_url}/models", headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    models = [
                        item["id"] for item in data.get("data", []) if "id" in item
                    ]
                    if not url or url == self.base_url:
                        self.available_models = models
                    return models
        except Exception as e:
            logger.warning(f"Failed to fetch models from backend at {target_url}: {e}")
        return self.available_models

    @staticmethod
    def is_embedding_model(name: str) -> bool:
        """Check if model identifier indicates an embedding model."""
        name_lower = name.lower()
        return any(
            token in name_lower
            for token in ("embed", "bge", "nomic", "arctic", "minilm", "e5", "sentence")
        )


@dataclass
class GentzenConfig:
    model_path: str = str(DEFAULT_MODEL_PATH)
    onnx_path: str = str(DEFAULT_ONNX_PATH)
    device: str = "auto"  # "auto", "cuda", "cpu"
    max_search_depth: int = 8
    max_nodes: int = 500
    beam_width: int = 6
    prune_threshold: float = 0.15
    max_antecedents: int = 32
    # nanoGentzen v4
    backend: str = "auto"               # auto | torch | numpy | symbolic
    search_budget: int = 64             # nodes for network-guided G4ip search (complete prover backs it up)
    network: bool = True                # False = fixed-order search; verdicts are the same
    allow_state_completion: bool = True  # Jev: add minimal missing facts for target_decision (recorded)
    label_smoothing: float = 0.0        # Jev: mass moved off a kernel-proven label

@dataclass
class CoTConfig:
    min_reasoning_tokens: int = 50
    max_reasoning_tokens: int = 250
    enforce_qra_tags: bool = True
