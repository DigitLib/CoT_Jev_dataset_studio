# Vendored from the nanoGentzen v4 model folder by tools/vendor_v4_engine.py. Do not edit by hand.
"""
tokenization_nanogentzen.py
Hugging Face PreTrainedTokenizer wrapper and standalone LogicTokenizer for nanoGentzen symbolic expressions.
"""
import json
import os
from typing import Any, Dict, List, Optional

try:
    from transformers import PreTrainedTokenizer
except ImportError:
    class PreTrainedTokenizer:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class LogicTokenizerHF(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token: str = "<UNK>",
        pad_token: str = "<PAD>",
        cls_token: str = "<CLS>",
        sep_token: str = "<SEP>",
        eos_token: str = "<EOS>",
        **kwargs,
    ):
        if vocab_file is None or not os.path.exists(vocab_file):
            current_dir = os.path.dirname(__file__)
            candidate = os.path.join(current_dir, "vocab.json")
            if os.path.exists(candidate):
                vocab_file = candidate
            else:
                parent_candidate = os.path.join(os.path.dirname(current_dir), "vocab.json")
                if os.path.exists(parent_candidate):
                    vocab_file = parent_candidate
                else:
                    vocab_file = candidate

        if os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                self.encoder: Dict[str, int] = {token: idx for idx, token in enumerate(raw)}
            else:
                self.encoder: Dict[str, int] = {str(k): int(v) for k, v in raw.items()}
        else:
            self.encoder = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3, "<EOS>": 4}

        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}
        self.special_tokens_list = [
            "<PAD>", "<UNK>", "<CLS>", "<SEP>", "<EOS>",
            "|-", "⟶", "⊢", "=>", "&", "|", "~", "0",
            "AXIOM", "R_IMP", "L_IMP", "R_AND", "L_AND",
            "R_OR_1", "R_OR_2", "L_OR", "R_NOT", "L_NOT", "L_CONTR",
        ]

        try:
            super().__init__(
                unk_token=unk_token,
                pad_token=pad_token,
                cls_token=cls_token,
                sep_token=sep_token,
                eos_token=eos_token,
                **kwargs,
            )
        except Exception:
            self.unk_token = unk_token
            self.pad_token = pad_token
            self.cls_token = cls_token
            self.sep_token = sep_token
            self.eos_token = eos_token

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    @property
    def pad_id(self) -> int:
        return self.encoder.get("<PAD>", 0)

    @property
    def turnstile_id(self) -> int:
        return self.encoder.get("|-", self.encoder.get("⊢", 5))

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.encoder)

    @property
    def vocab(self) -> Dict[str, int]:
        return dict(self.encoder)

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            matched = False
            for st in self.special_tokens_list:
                if text.startswith(st, i):
                    tokens.append(st)
                    i += len(st)
                    matched = True
                    break
            if not matched:
                tokens.append(text[i])
                i += 1
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.encoder.get(str(self.unk_token), 1))

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, str(self.unk_token))

    def encode(self, text: str, **kwargs) -> List[int]:
        tokens = self._tokenize(text)
        return [self._convert_token_to_id(t) for t in tokens]

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return "".join([self._convert_id_to_token(t) for t in token_ids if t != self.pad_id])

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, indent=2, ensure_ascii=False)
        return (vocab_file,)


# Backward compatibility alias
LogicTokenizer = LogicTokenizerHF


# =====================================================================
# v4 word-level tokenizer: one token per connective, bracket, comma and canonical variable
# =====================================================================
import re as _re

WORD_VOCAB = (["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<EOS>", "|-", "=>", "&", "|", "~", "(", ")", ",", "0",
               "P", "Q", "R", "S"] + [f"v{i}" for i in range(64)])
_WORD_RE = _re.compile(r"\|-|=>|[&|~(),]|[A-Za-z_][A-Za-z0-9_]*|0")


class LogicWordTokenizer:
    """Tokenizer of v4 checkpoints. Input is Sequent.model_str() of a canonicalised sequent."""

    def __init__(self, vocab_file: Optional[str] = None, **kwargs):
        if vocab_file and os.path.exists(vocab_file):
            raw = json.load(open(vocab_file, encoding="utf-8"))
            toks = raw if isinstance(raw, list) else [t for t, _ in sorted(raw.items(), key=lambda kv: kv[1])]
        else:
            toks = list(WORD_VOCAB)
        self.encoder: Dict[str, int] = {t: i for i, t in enumerate(toks)}
        self.decoder: Dict[int, str] = {i: t for t, i in self.encoder.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    @property
    def pad_id(self) -> int:
        return self.encoder["<PAD>"]

    @property
    def turnstile_id(self) -> int:
        return self.encoder["|-"]

    @property
    def comma_id(self) -> int:
        return self.encoder[","]

    def tokenize(self, text: str) -> List[str]:
        return _WORD_RE.findall(text)

    def encode(self, text: str, **kwargs) -> List[int]:
        unk = self.encoder["<UNK>"]
        return [self.encoder.get(t, unk) for t in self.tokenize(text)]

    def decode(self, ids: List[int], **kwargs) -> str:
        return " ".join(self.decoder.get(i, "<UNK>") for i in ids if i != self.pad_id)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        path = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({t: i for t, i in self.encoder.items()}, f, indent=1, ensure_ascii=False)
        return (path,)


def tokenizer_for_config(config, vocab_file: Optional[str] = None):
    """The tokenizer a checkpoint was trained with."""
    if getattr(config, "tokenizer_type", "char") == "word":
        return LogicWordTokenizer(vocab_file=vocab_file)
    return LogicTokenizerHF(vocab_file=vocab_file)
