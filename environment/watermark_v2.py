# environment/watermark_v2.py
# Watermarks v2 (Run 16): every type changes the output distribution in a way a detector
# can in principle learn.  What was wrong with v1 (see EXPERIMENT_RESULTS.md, Q1):
#   positional : constant shift of all logits         -> softmax no-op, undetectable
#   token      : fresh unkeyed random green list/step  -> pure noise, no stable signature
#   syntactic  : +1.5 on ids 100-199, 17 tokens        -> far too weak
#   semantic   : +1.0 on ids 1010-4009                 -> weak, only an oracle sees it
#
# v2 design: one FIXED, secret-key-seeded, mutually DISJOINT token set per type, so each type
# has its own stable lexical signature.  The detector is not given the key; it has to learn it.
#
#   token      : fixed random `token_frac` of the whole vocabulary            (+delta)
#   syntactic  : function words and punctuation (the, of, and, ",", "." ...)  (+delta)
#   semantic   : fixed random `semantic_frac` of whole-word (content) tokens  (+delta)
#   positional : a different fixed word set, with a bias that RAMPS UP linearly across the
#                watermark window (position-dependent, so it is no longer a constant shift)
#
# bias = delta[type] * strength * intensity  (positional additionally * (step_in_window+1)/window)
# `strength` is one global knob; experiments/gate_watermarks.py picks it before any RL.

import random
from typing import Dict, List, Sequence, Tuple

import torch

FUNCTION_WORDS = [
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "for", "with", "as", "on", "by",
    "at", "from", "it", "an", "be", "are", "this", "which", "or", "had", "have", "not", "but",
    "his", "he", "were", "has", "their", "its", "they", "been", "who", "one", "all", "also",
    "more", "after", "than", "other", "into", "when", "there", "would", "can",
]
PUNCT = [",", ".", ";", ":"]

DEFAULT_DELTAS = {"token": 2.0, "syntactic": 1.2, "semantic": 2.0, "positional": 2.5}


class WatermarkEngine:
    def __init__(
        self,
        vocab: Sequence[str],          # index -> GPT-2 BPE token string ("Ġthe", ",", ...)
        vocab_size: int,               # size of the logits dimension
        key: int = 20260101,
        strength: float = 1.0,
        deltas: Dict[str, float] = None,
        token_frac: float = 0.20,
        semantic_frac: float = 0.10,
        positional_frac: float = 0.10,
    ):
        self.vocab_size = vocab_size
        self.strength = float(strength)
        self.deltas = dict(DEFAULT_DELTAS)
        if deltas:
            self.deltas.update(deltas)

        tok2id = {t: i for i, t in enumerate(vocab)}
        func = []
        for w in FUNCTION_WORDS:
            if "Ġ" + w in tok2id:
                func.append(tok2id["Ġ" + w])
        for p in PUNCT:
            if p in tok2id:
                func.append(tok2id[p])
        func = sorted(set(i for i in func if i < vocab_size))
        func_set = set(func)

        words = [i for i, t in enumerate(vocab)
                 if i < vocab_size and i not in func_set
                 and len(t) >= 4 and t[0] == "Ġ" and t[1:].isalpha()]
        rng = random.Random(key)
        rng.shuffle(words)
        k_sem = int(len(words) * semantic_frac)
        k_pos = int(len(words) * positional_frac)
        semantic = sorted(words[:k_sem])
        positional = sorted(words[k_sem:k_sem + k_pos])

        used = func_set | set(semantic) | set(positional)
        rest = [i for i in range(vocab_size) if i not in used]
        rng.shuffle(rest)
        token = sorted(rest[:int(token_frac * vocab_size)])

        self.sets: Dict[str, List[int]] = {
            "token": token, "syntactic": func, "semantic": semantic, "positional": positional,
        }
        self._masks: Dict[Tuple[str, str], torch.Tensor] = {}

    def set_sizes(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.sets.items()}

    def _mask(self, wm_type: str, device, dtype) -> torch.Tensor:
        key = (wm_type, str(device))
        m = self._masks.get(key)
        if m is None:
            m = torch.zeros(self.vocab_size, device=device)
            m[torch.tensor(self.sets[wm_type], dtype=torch.long, device=device)] = 1.0
            self._masks[key] = m
        return m.to(dtype)

    def bias(self, wm_type: str, rel_pos: int, window_len: int, intensity: float) -> float:
        """Bias added to the type's token set at this step of the window."""
        if wm_type not in self.sets:
            raise ValueError(f"Unknown watermark type: {wm_type}")
        d = self.deltas[wm_type] * self.strength * float(intensity)
        if wm_type == "positional":
            d *= (min(max(rel_pos, 0), window_len - 1) + 1) / max(1, window_len)
        return d

    def apply(self, wm_type: str, scores: torch.Tensor, rel_pos: int, window_len: int,
              intensity: float) -> Tuple[torch.Tensor, Dict]:
        d = self.bias(wm_type, rel_pos, window_len, intensity)
        out = scores + d * self._mask(wm_type, scores.device, scores.dtype)
        return out, {"type": wm_type, "bias": d, "set_size": len(self.sets[wm_type])}
