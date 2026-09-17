# environment/watermark_types.py
# Each strategy takes (logits, tokens, intensity) and returns modified logits

import torch
import random

def apply_token_watermark(logits, green_list_ratio=0.5, bias=2.0, intensity=1.0):
    """
    KGW-style: partition vocab into green/red lists,
    bias logits toward green tokens.
    """
    vocab_size = logits.shape[-1]
    green_size = int(vocab_size * green_list_ratio)
    green_list = random.sample(range(vocab_size), green_size)
    logits[:, green_list] += bias * intensity
    return logits, green_list   # return green_list as ground truth marker

def apply_syntactic_watermark(logits, pos_bias_tokens, intensity=1.0):
    """
    Bias toward specific POS-associated tokens
    (e.g., prefer passive voice constructions).
    pos_bias_tokens: list of token ids associated with target syntactic pattern
    """
    logits[:, pos_bias_tokens] += 1.5 * intensity
    return logits, pos_bias_tokens

def apply_semantic_watermark(logits, synonym_map, intensity=1.0):
    """
    Bias toward specific synonym choices to embed semantic shift.
    synonym_map: dict {original_token_id: preferred_synonym_id}
    """
    for orig, preferred in synonym_map.items():
        logits[:, preferred] += 1.0 * intensity
    return logits, list(synonym_map.values())

def apply_positional_watermark(logits, position, total_length, intensity=1.0):
    """
    Only apply watermark bias at a specific position window.
    Returns a mask of which positions are watermarked.
    """
    # The actual bias is applied externally at the right position
    # This just computes the target span
    span_size = max(5, total_length // 5)
    start = min(position, total_length - span_size)
    end = start + span_size
    return logits, (start, end)

# Strategy dispatcher
WATERMARK_STRATEGIES = {
    "token":      apply_token_watermark,
    "syntactic":  apply_syntactic_watermark,
    "semantic":   apply_semantic_watermark,
    "positional": apply_positional_watermark,
}