# Vendored from the nanoGentzen v4 model folder by tools/vendor_v4_engine.py. Do not edit by hand.
"""
configuration_nanogentzen.py (nanoGentzen-v3 Kripke + Jev)
Hugging Face PretrainedConfig class for nanoGentzen with Span Pointer Pivot Head
and 3-Class Intuitionistic Epistemic Value Head (Saul Kripke 1965 semantics).
"""
from typing import List, Optional
try:
    from transformers import PretrainedConfig
except ImportError:
    # Minimal fallback if transformers is not yet installed
    class PretrainedConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class GentzenConfig(PretrainedConfig):
    model_type = "nanogentzen"

    def __init__(
        self,
        vocab_size: int = 95,
        block_size: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        n_embd: int = 256,
        num_rules: int = 11,
        max_antecedents: int = 32,
        use_pointer_pivot: bool = True,
        pointer_dim: int = 64,
        num_epistemic_classes: int = 3,
        comma_token_id: int = 88,
        turnstile_token_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.num_rules = num_rules
        self.max_antecedents = max_antecedents
        self.use_pointer_pivot = use_pointer_pivot
        self.pointer_dim = pointer_dim
        self.num_epistemic_classes = num_epistemic_classes
        self.comma_token_id = comma_token_id
        self.turnstile_token_ids = turnstile_token_ids if turnstile_token_ids is not None else [5, 6, 7]


# Backward compatibility alias
PolicyValueConfig = GentzenConfig

