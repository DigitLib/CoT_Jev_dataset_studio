"""
numpy_model.py - torch-free forward pass of the nanoGentzen v4 network.

Same computation as modeling_nanogentzen.GentzenPolicyValueModel (CLS token, 6 bidirectional pre-LN
blocks with exact GELU, rule head, span-pointer pivot head, 4-class epistemic head), read straight from
model.safetensors. The model handoff reports PyTorch vs NumPy max |diff| 5.3e-6 on the saved weights.

It lets the Studio run the v4 model on machines without torch (slower: roughly 5-20 ms per sequent on
CPU, versus ~2 ms batched on GPU). Verdicts never depend on it: the kernel decides every answer.
"""
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np

try:
    from scipy.special import erf as _erf
except ImportError:  # Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    def _erf(x):
        x = np.asarray(x)
        s = np.sign(x)
        a = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * a)
        y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                   + 0.254829592) * t * np.exp(-a * a)
        return s * y

_DTYPES = {"F32": np.float32, "F16": np.float16, "F64": np.float64}


def load_safetensors(path: Path) -> Dict[str, np.ndarray]:
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        buf = raw[base + s:base + e]
        if v["dtype"] == "BF16":
            u16 = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
            arr = u16.view(np.float32)
        elif v["dtype"] in _DTYPES:
            arr = np.frombuffer(buf, dtype=_DTYPES[v["dtype"]])
        else:
            raise ValueError(f"unsupported dtype {v['dtype']} for {k}")
        out[k] = arr.reshape(v["shape"]).astype(np.float32)
    return out


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class NumpyGentzenModel:
    """Drop-in for GentzenPolicyValueModel where the engine needs it (config, predict_epistemic, to, eval)."""

    accepts_numpy = True

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.config = SimpleNamespace(**cfg)
        self.c = cfg
        self.w = load_safetensors(model_dir / "model.safetensors")
        self.n_params = int(sum(a.size for a in self.w.values()))

    # torch-style no-ops so the engine can treat both backends alike
    def to(self, *_a, **_k):
        return self

    def eval(self):
        return self

    def _ln(self, x, p):
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / np.sqrt(v + 1e-5) * self.w[p + ".weight"] + self.w[p + ".bias"]

    def forward_one(self, ids) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unpadded token ids -> (rule_logits, pivot_logits, epistemic_probs)."""
        W, c = self.w, self.c
        ids = np.asarray(ids, dtype=np.int64)
        T, D, H = len(ids), c["n_embd"], c["n_head"]
        hd = D // H
        x = np.concatenate([W["cls_token"][0], W["tok_emb.weight"][ids]], 0) + W["pos_emb"][0, :T + 1]
        mask = np.concatenate([[False], ids == c.get("pad_token_id", 0)])
        for layer in range(c["n_layer"]):
            p = f"blocks.{layer}."
            nx = self._ln(x, p + "ln_1")
            qkv = nx @ W[p + "attn.in_proj_weight"].T + W[p + "attn.in_proj_bias"]
            q, k, v = np.split(qkv, 3, -1)
            q, k, v = (z.reshape(-1, H, hd).transpose(1, 0, 2) for z in (q, k, v))
            a = q @ k.transpose(0, 2, 1) / math.sqrt(hd)
            a[:, :, mask] = -1e30
            a = _softmax(a)
            o = (a @ v).transpose(1, 0, 2).reshape(-1, D) @ W[p + "attn.out_proj.weight"].T + W[p + "attn.out_proj.bias"]
            x = x + o
            h = _gelu(self._ln(x, p + "ln_2") @ W[p + "mlp.0.weight"].T + W.get(p + "mlp.0.bias", 0.0))
            x = x + h @ W[p + "mlp.2.weight"].T + W.get(p + "mlp.2.bias", 0.0)
        x = self._ln(x, "ln_f")
        cls, sh = x[0], x[1:]
        rule = cls @ W["policy_rule_head.weight"].T + W.get("policy_rule_head.bias", 0.0)
        e = _gelu(cls @ W["epistemic_head.0.weight"].T + W["epistemic_head.0.bias"])
        e = e @ W["epistemic_head.2.weight"].T + W["epistemic_head.2.bias"]
        ep = _softmax(e)
        # span pointer over premise slots (commas before the turnstile separate premises)
        maxa = c.get("max_antecedents", 32)
        ts = np.isin(ids, c.get("turnstile_token_ids", [5]))
        after = np.cumsum(ts) > 0
        before = ~after
        nps = (ids != (c.get("pad_token_id", 0) or 0)) & (ids != c.get("space_token_id", 3))
        isd = after & ~ts & nps
        dm = isd.astype(float)[:, None]
        dmean = (sh * dm).sum(0) / max(dm.sum(), 1.0)
        hg = 0.5 * cls + 0.5 * dmean
        isc = (ids == c.get("comma_token_id", 12)) & before
        sidx = np.cumsum(isc)
        isa = before & ~isc & nps & (sidx < maxa)
        oh = np.zeros((T, maxa), bool)
        oh[np.arange(T), np.clip(sidx, 0, maxa - 1)] = True
        oh &= isa[:, None]
        cnt = oh.sum(0)
        valid = cnt > 0
        valid[0] = True
        smean = (oh.T.astype(float) @ sh) / np.maximum(cnt, 1)[:, None]
        pos = np.arange(T)[:, None]
        fi = np.clip(np.where(oh, pos, T).min(0), 0, T - 1)
        li = np.clip(np.where(oh, pos, 0).max(0), 0, T - 1)
        span = (0.5 * smean + 0.25 * sh[fi] + 0.25 * sh[li]) * (cnt > 0)[:, None]
        qv = hg @ W["pointer_q.weight"].T
        kv = span @ W["pointer_k.weight"].T
        pl = (kv @ qv) / math.sqrt(c.get("pointer_dim", 64)) + W["pointer_pos_bias"][0]
        pl[~valid] = -1e4
        return rule, pl, ep

    def predict_epistemic(self, input_ids: Any):
        """(rule_probs, pivot_probs, epistemic_probs, constructive_value), same contract as the torch model."""
        rows = np.asarray(input_ids)
        pad = self.c.get("pad_token_id", 0)
        R, P, E = [], [], []
        for row in rows:
            n = int((row != pad).sum())
            r, p, e = self.forward_one(row[:n])
            R.append(_softmax(r))
            P.append(_softmax(p))
            E.append(e)
        E = np.asarray(E)
        return np.asarray(R), np.asarray(P), E, E[:, 2] + 0.5 * E[:, 1]
