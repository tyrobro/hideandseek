# environment/watermark_types.py
# All strategies now have hard intensity caps to prevent logit explosion
# which was the root cause of PPL runaway in Run 01.

import torch
import random
from typing import Tuple, Dict

# ── Hard caps per strategy ────────────────────────────────────────────────────
# These are the maximum bias magnitudes allowed regardless of what the
# policy head requests. Prevents Agent A from destroying text quality.
MAX_INTENSITY = {
    "token"      : 1.5,
    "syntactic"  : 1.0,   # was uncapped — biggest PPL offender in Run 01
    "semantic"   : 1.0,
    "positional" : 1.0,   # base bias also reduced from 3.0 → 1.5
}


def apply_token_watermark(
    logits: torch.Tensor,
    green_list_ratio: float = 0.25,
    bias: float = 2.0,
    intensity: float = 1.0,
    seed: int = None,
) -> Tuple[torch.Tensor, Dict]:
    intensity = min(intensity, MAX_INTENSITY["token"])   # hard cap

    if seed is not None:
        random.seed(seed)

    vocab_size = logits.shape[-1]
    green_size = int(vocab_size * green_list_ratio)
    green_list = random.sample(range(vocab_size), green_size)
    logits[:, green_list] += bias * intensity

    return logits, {"type": "token", "green_list": green_list}


def apply_syntactic_watermark(
    logits: torch.Tensor,
    pos_bias_tokens,
    intensity: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    intensity = min(intensity, MAX_INTENSITY["syntactic"])   # hard cap

    if pos_bias_tokens:
        logits[:, pos_bias_tokens] += 1.5 * intensity

    return logits, {"type": "syntactic", "biased_tokens": pos_bias_tokens}


def apply_semantic_watermark(
    logits: torch.Tensor,
    synonym_map: Dict,
    intensity: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    intensity = min(intensity, MAX_INTENSITY["semantic"])   # hard cap

    preferred = list(synonym_map.values())
    for orig, pref in synonym_map.items():
        logits[:, pref] += 1.0 * intensity

    return logits, {"type": "semantic", "preferred_tokens": preferred}


def apply_positional_watermark(
    logits: torch.Tensor,
    position: int,
    total_length: int,
    bias: float = 1.5,      # was 3.0 in Run 01 — halved
    intensity: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    intensity = min(intensity, MAX_INTENSITY["positional"])   # hard cap

    span_size = max(5, total_length // 5)
    start     = min(position, total_length - span_size)
    end       = start + span_size
    logits   += bias * intensity

    return logits, {"type": "positional", "span": (start, end)}


WATERMARK_STRATEGIES = {
    "token"      : apply_token_watermark,
    "syntactic"  : apply_syntactic_watermark,
    "semantic"   : apply_semantic_watermark,
    "positional" : apply_positional_watermark,
}