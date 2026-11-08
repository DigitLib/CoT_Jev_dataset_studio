# Vendored from the nanoGentzen v4 model folder by tools/vendor_v4_engine.py. Do not edit by hand.
"""
modeling_nanogentzen.py (nanoGentzen-v3 Kripke + Jev)
Hugging Face PreTrainedModel for Gentzen Policy-Value Transformer featuring:
  1. Vectorized Boundary-Aware Span Pointer Pivot Head (up to 32 antecedents)
  2. 3-Class Intuitionistic Epistemic Value Head: [P(False), P(Undecided), P(True)]
  3. Constructive Provability Metric: V = P(True) + 0.5 * P(Undecided)
"""
import math
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import PreTrainedModel
except ImportError:
    class PreTrainedModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def post_init(self):
            if hasattr(self, "_init_weights"):
                self.apply(self._init_weights)

from .configuration_nanogentzen import GentzenConfig, PolicyValueConfig


class BidirectionalBlock(nn.Module):
    def __init__(self, config: GentzenConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd, num_heads=config.n_head, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(
            norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class GentzenPolicyValueModel(PreTrainedModel):
    config_class = GentzenConfig
    _keys_to_ignore_on_load_unexpected = [
        r"^W_q\.",
        r"^W_k\.",
        r"^pivot_bias$",
        r"^epistemic_value_head\.",
        r"^value_head\.",
        r"^policy_pivot_head\.",
    ]

    def __init__(self, config: GentzenConfig):
        super().__init__(config)
        self.config = config
        pad_id = getattr(config, "pad_token_id", 0) or 0
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(
            torch.empty(1, config.block_size + 1, config.n_embd)
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.n_embd))
        self.blocks = nn.ModuleList(
            [BidirectionalBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)

        # 1. Rule Policy Head (11 Gentzen LI rules)
        self.policy_rule_head = nn.Linear(config.n_embd, config.num_rules, bias=False)

        # 2. Span Pointer Antecedent Pivot Head (max_antecedents = 32, pointer_dim = 64)
        ptr_dim = getattr(config, "pointer_dim", 64)
        max_ant = getattr(config, "max_antecedents", 32)
        self.pointer_q = nn.Linear(config.n_embd, ptr_dim, bias=False)
        self.pointer_k = nn.Linear(config.n_embd, ptr_dim, bias=False)
        self.pointer_pos_bias = nn.Parameter(torch.zeros(1, max_ant))

        if not getattr(config, "use_pointer_pivot", True):
            self.policy_pivot_head = nn.Linear(config.n_embd, max_ant, bias=False)

        # 3. 3-Class Kripke Epistemic Value Head: [P(False), P(Undecided), P(True)]
        num_epistemic = getattr(config, "num_epistemic_classes", 3)
        self.epistemic_head = nn.Sequential(
            nn.Linear(config.n_embd, 128),
            nn.GELU(),
            nn.Linear(128, num_epistemic),
        )

        self._last_epistemic_probs: Optional[torch.Tensor] = None
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.post_init()

    def _init_weights(self, module: nn.Module):
        """Initializes linear, embedding, and normalization weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if module.out_proj.weight is not None:
                nn.init.normal_(module.out_proj.weight, mean=0.0, std=0.02)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)

    # Non-registering property aliases for backward API compatibility (avoids shared-tensor state_dict duplication)
    @property
    def W_q(self) -> nn.Linear:
        return self.pointer_q

    @property
    def W_k(self) -> nn.Linear:
        return self.pointer_k

    @property
    def pivot_bias(self) -> nn.Parameter:
        return self.pointer_pos_bias

    @property
    def epistemic_value_head(self) -> nn.Sequential:
        return self.epistemic_head

    def load_v2_compatible_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        Loads a v2 or v3 state_dict cleanly. Remaps legacy alias keys if needed and
        transfers the 6-layer Transformer backbone, rule head, and first value MLP layer.
        """
        own_state = self.state_dict()
        filtered = {}
        alias_map = {
            "W_q.weight": "pointer_q.weight",
            "W_k.weight": "pointer_k.weight",
            "pivot_bias": "pointer_pos_bias",
            "epistemic_value_head.0.weight": "epistemic_head.0.weight",
            "epistemic_value_head.0.bias": "epistemic_head.0.bias",
            "epistemic_value_head.2.weight": "epistemic_head.2.weight",
            "epistemic_value_head.2.bias": "epistemic_head.2.bias",
        }

        for k, v in state_dict.items():
            target_k = alias_map.get(k, k)
            if target_k in own_state and own_state[target_k].shape == v.shape:
                filtered[target_k] = v

        # Transfer v2 value_head.0 into epistemic_head.0 if epistemic_head is not in state_dict
        if (
            "epistemic_head.0.weight" not in filtered
            and "value_head.0.weight" in state_dict
        ):
            filtered["epistemic_head.0.weight"] = state_dict["value_head.0.weight"].clone()
            filtered["epistemic_head.0.bias"] = state_dict["value_head.0.bias"].clone()
        self.load_state_dict(filtered, strict=False)

    def _compute_span_pointer_logits(
        self,
        input_ids: torch.Tensor,
        seq_h: torch.Tensor,
        cls_rep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized Boundary-Aware Span Pointer Network over comma-delimited antecedents in Gamma:
        Splits tokens before turnstile ('|-') by comma tokens (','), combines mean and
        first/last boundary representations for each antecedent span h_i (preserving formula
        directionality), and computes scaled dot-product attention across all 32 slots.
        """
        B, T, D = seq_h.shape
        max_ant = getattr(self.config, "max_antecedents", 32)
        pad_id = getattr(self.config, "pad_token_id", 0) or 0
        space_id = getattr(self.config, "space_token_id", 24)
        comma_id = getattr(self.config, "comma_token_id", 88)
        turnstile_ids = getattr(self.config, "turnstile_token_ids", [5, 6, 7])

        is_turnstile = torch.zeros_like(input_ids, dtype=torch.bool)
        for tid in turnstile_ids:
            is_turnstile = is_turnstile | (input_ids == tid)

        after_turnstile = is_turnstile.cumsum(dim=1) > 0
        before_turnstile = ~after_turnstile
        not_pad_or_space = (input_ids != pad_id) & (input_ids != space_id)

        # Succedent (Delta) representation: mean pool semantic tokens after turnstile + CLS
        is_delta = after_turnstile & (~is_turnstile) & not_pad_or_space
        delta_mask = is_delta.unsqueeze(-1).to(seq_h.dtype)
        delta_count = delta_mask.sum(dim=1).clamp(min=1.0)
        delta_mean = (seq_h * delta_mask).sum(dim=1) / delta_count
        h_goal = 0.5 * cls_rep + 0.5 * delta_mean

        # Antecedent (Gamma) span indexing via cumulative comma count before turnstile
        is_comma = (input_ids == comma_id) & before_turnstile
        span_idx = is_comma.long().cumsum(dim=1)
        is_ant_tok = (
            before_turnstile
            & (~is_comma)
            & not_pad_or_space
            & (span_idx < max_ant)
        )

        one_hot_bool = F.one_hot(span_idx.clamp(0, max_ant - 1), num_classes=max_ant).bool()
        one_hot_bool = one_hot_bool & is_ant_tok.unsqueeze(-1)
        one_hot = one_hot_bool.to(seq_h.dtype)

        span_counts = one_hot.sum(dim=1).unsqueeze(-1)  # (B, max_ant, 1)
        span_valid = span_counts.squeeze(-1) > 0        # (B, max_ant)
        span_valid[:, 0] = True  # Ensure at least slot 0 valid so softmax never sees all -inf

        span_mean = torch.bmm(one_hot.transpose(1, 2), seq_h) / span_counts.clamp(min=1.0)

        # Boundary-aware span representation: combine mean with first & last semantic tokens in span
        tok_pos = torch.arange(T, device=input_ids.device).view(1, T, 1)
        first_idx = torch.where(one_hot_bool, tok_pos, T).amin(dim=1).clamp(0, max(0, T - 1))
        last_idx = torch.where(one_hot_bool, tok_pos, 0).amax(dim=1).clamp(0, max(0, T - 1))

        span_first = seq_h.gather(1, first_idx.unsqueeze(-1).expand(-1, -1, D))
        span_last = seq_h.gather(1, last_idx.unsqueeze(-1).expand(-1, -1, D))
        valid_mask_f = (span_counts > 0).to(seq_h.dtype)
        span_h = (0.5 * span_mean + 0.25 * span_first + 0.25 * span_last) * valid_mask_f

        ptr_dim = getattr(self.config, "pointer_dim", 64)
        q = self.pointer_q(h_goal).unsqueeze(1)  # (B, 1, ptr_dim)
        k = self.pointer_k(span_h)               # (B, max_ant, ptr_dim)
        ptr_logits = (q * k).sum(dim=-1) / math.sqrt(ptr_dim)
        ptr_logits = ptr_logits + self.pointer_pos_bias[:, :max_ant]

        # Mask non-existent antecedent slots with -1e4
        ptr_logits = ptr_logits.masked_fill(~span_valid, -1e4)
        return ptr_logits

    def forward(
        self,
        input_ids: Any,
        targets_rule: Optional[torch.Tensor] = None,
        targets_pivot: Optional[torch.Tensor] = None,
        targets_epistemic: Optional[torch.Tensor] = None,
        targets_value: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        elif not isinstance(input_ids, torch.Tensor) and hasattr(input_ids, "__getitem__") and "input_ids" in input_ids:
            input_ids = input_ids["input_ids"]

        B, T = input_ids.size()
        tok_embeddings = self.tok_emb(input_ids)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, tok_embeddings), dim=1)
        pos = self.pos_emb[:, : T + 1, :]
        x = x + pos

        pad_id = getattr(self.config, "pad_token_id", 0) or 0
        pad_mask = input_ids == pad_id
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=input_ids.device)
        full_pad_mask = torch.cat((cls_mask, pad_mask), dim=1)

        for block in self.blocks:
            x = block(x, key_padding_mask=full_pad_mask)
        x = self.ln_f(x)

        cls_rep = x[:, 0, :]
        seq_h = x[:, 1:, :]

        # 1. Rule Policy Logits (11 rules)
        rule_logits = self.policy_rule_head(cls_rep)

        # 2. Span Pointer Pivot Logits (32 antecedents)
        if getattr(self.config, "use_pointer_pivot", True):
            pivot_logits = self._compute_span_pointer_logits(input_ids, seq_h, cls_rep)
        else:
            pivot_logits = self.policy_pivot_head(cls_rep)

        # 3. 3-Class Epistemic Value Head: [P(False), P(Undecided), P(True)]
        epistemic_logits = self.epistemic_head(cls_rep)
        probs = F.softmax(epistemic_logits, dim=-1)
        self._last_epistemic_probs = probs.detach()

        # Constructive Provability Score: V = P(True) + 0.5 * P(Undecided)
        constructive_value = probs[:, 2] + 0.5 * probs[:, 1]

        # Harmonize target_epistemic vs target_value
        ep_target = targets_epistemic
        if ep_target is None and targets_value is not None:
            # Map float value targets [0.0, 0.5, 1.0] to long class targets [0, 1, 2]
            ep_target = (targets_value * 2.0).round().long().clamp(0, 2)

        loss = None
        if (
            targets_rule is not None
            and targets_pivot is not None
            and ep_target is not None
        ):
            valid_r = targets_rule != -100
            loss_rule = (
                F.cross_entropy(rule_logits[valid_r], targets_rule[valid_r])
                if valid_r.any()
                else rule_logits.sum() * 0.0
            )

            valid_p = (targets_pivot != -100) & (targets_pivot < pivot_logits.size(-1))
            loss_pivot = (
                F.cross_entropy(pivot_logits[valid_p], targets_pivot[valid_p])
                if valid_p.any()
                else pivot_logits.sum() * 0.0
            )

            loss_ep = F.cross_entropy(epistemic_logits, ep_target)
            loss = loss_rule + loss_pivot + 0.5 * loss_ep

        return rule_logits, pivot_logits, constructive_value, loss

    @torch.no_grad()
    def predict_epistemic(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-pass System 1 inference returning:
          - rule_probs: (B, 11)
          - pivot_probs: (B, max_antecedents)
          - epistemic_probs: (B, 3) -> [P(False), P(Undecided), P(True)]
          - constructive_value: (B,) scalar in [0, 1]
        """
        rule_logits, pivot_logits, constructive_val, _ = self.forward(input_ids)
        rule_probs = F.softmax(rule_logits, dim=-1)
        pivot_probs = F.softmax(pivot_logits, dim=-1)
        epistemic_probs = self._last_epistemic_probs
        return rule_probs, pivot_probs, epistemic_probs, constructive_val

    def _get_jev_engine(self):
        if getattr(self, "_jev_engine", None) is None:
            import sys
            from pathlib import Path
            cand_paths = [
                Path(__file__).resolve().parent,
                Path(getattr(self.config, "_name_or_path", ".")).resolve(),
            ]
            base_dir = next((p for p in cand_paths if (p / "vocab.json").exists()), cand_paths[0])
            if str(base_dir) not in sys.path:
                sys.path.insert(0, str(base_dir))

            from . import jev_server
            from . import tokenization_nanogentzen

            device = str(next(self.parameters()).device)
            vocab_path = base_dir / "vocab.json"
            tok = tokenization_nanogentzen.LogicTokenizerHF(
                vocab_file=str(vocab_path) if vocab_path.exists() else None
            )
            self._jev_engine = jev_server.ProductionGentzenEngine(
                base_dir=base_dir, model=self, tokenizer=tok, device=device
            )
        return self._jev_engine

    def decide(self, payload: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        System-2 Verified Gentzen Proof Search + Curry-Howard Certificates + IR Relevance Tracking.
        """
        req = payload if payload is not None else kwargs
        return self._get_jev_engine().decide(req)

    def system_one(self, payload: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        System-1 Fast Epistemic Decision Gate (< 3ms).
        """
        req = payload if payload is not None else kwargs
        return self._get_jev_engine().system_one(req)


# Backward compatibility aliases
GentzenPolicyValueNet = GentzenPolicyValueModel
