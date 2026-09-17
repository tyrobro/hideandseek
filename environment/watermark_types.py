# environment/watermark_types.py
# Defines the four watermarking strategies Agent A can choose from.
# Each function takes logits + parameters and returns (modified_logits, ground_truth_marker)
# ground_truth_marker tells the environment WHERE and WHAT the watermark is.

import torch
import random
from typing import Tuple, List, Dict, Any


def apply_token_watermark(
    logits: torch.Tensor,
    green_list_ratio: float = 0.25,
    bias: float = 2.0,
    intensity: float = 1.0,
    seed: int = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    KGW-style green/red list watermark.
    Partitions the vocabulary and biases logits toward 'green' tokens.

    Returns:
        logits: modified logits
        marker: {"type": "token", "green_list": [...token ids...]}
    """
    if seed is not None:
        random.seed(seed)

    vocab_size = logits.shape[-1]
    green_size = int(vocab_size * green_list_ratio)
    green_list = random.sample(range(vocab_size), green_size)
    logits[:, green_list] += bias * intensity

    return logits, {"type": "token", "green_list": green_list}


def apply_syntactic_watermark(
    logits: torch.Tensor,
    pos_bias_tokens: List[int],
    intensity: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    """
    Biases generation toward tokens that produce specific syntactic patterns
    (e.g., passive voice, subordinate clauses).
    pos_bias_tokens: token ids associated with the target syntactic pattern.

    Returns:
        logits: modified logits
        marker: {"type": "syntactic", "biased_tokens": [...]}
    """
    if pos_bias_tokens:
        logits[:, pos_bias_tokens] += 1.5 * intensity

    return logits, {"type": "syntactic", "biased_tokens": pos_bias_tokens}


def apply_semantic_watermark(
    logits: torch.Tensor,
    synonym_map: Dict[int, int],
    intensity: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    """
    Biases generation toward specific synonym choices to embed a
    meaning-preserving but statistically detectable semantic shift.
    synonym_map: {original_token_id: preferred_synonym_token_id}

    Returns:
        logits: modified logits
        marker: {"type": "semantic", "preferred_tokens": [...]}
    """
    preferred = list(synonym_map.values())
    for orig, pref in synonym_map.items():
        logits[:, pref] += 1.0 * intensity

    return logits, {"type": "semantic", "preferred_tokens": preferred}


def apply_positional_watermark(
    logits: torch.Tensor,
    position: int,
    total_length: int,
    bias: float = 3.0,
    intensity: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    """
    Concentrates the watermark in a specific positional window of the text.
    The actual bias is strong within the window, zero outside.
    Caller is responsible for only calling this at the right generation step.

    Returns:
        logits: modified logits (biased if within window)
        marker: {"type": "positional", "span": (start, end)}
    """
    span_size = max(5, total_length // 5)
    start = min(position, total_length - span_size)
    end = start + span_size
    logits += bias * intensity   # caller only invokes this within window

    return logits, {"type": "positional", "span": (start, end)}


# ── Strategy dispatcher ───────────────────────────────────────────────────────
# Agent A's policy head indexes into this dict by strategy name.

WATERMARK_STRATEGIES = {
    "token"      : apply_token_watermark,
    "syntactic"  : apply_syntactic_watermark,
    "semantic"   : apply_semantic_watermark,
    "positional" : apply_positional_watermark,
}