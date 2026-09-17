# rewards/reward_functions.py
# Standalone reward computations — imported by the environment and train loop.

import torch
from typing import Tuple


def compute_perplexity(model, tokenizer, text: str, device: str = "cpu") -> float:
    """
    Compute perplexity of text under the base LM.
    Used as Agent A's fluency penalty — high PPL = watermark hurt quality.
    """
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    input_ids = encodings["input_ids"]

    with torch.no_grad():
        outputs = model(**encodings, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()


def compute_iou(
    pred_span: Tuple[int, int],
    true_span: Tuple[int, int],
) -> float:
    """
    Token-level Intersection over Union between predicted and true watermark spans.
    Both spans are (start_token_idx, end_token_idx) inclusive.
    Returns a float in [0, 1].
    """
    pred_start, pred_end = pred_span
    true_start, true_end = true_span

    intersection = max(0, min(pred_end, true_end) - max(pred_start, true_start))
    union = max(1, max(pred_end, true_end) - min(pred_start, true_start))

    return intersection / union


def compute_type_accuracy(pred_type: str, true_type: str) -> float:
    return 1.0 if pred_type == true_type else 0.0


def compute_agent_rewards(
    true_type: str,
    true_span: Tuple[int, int],
    pred_type: str,
    pred_span: Tuple[int, int],
    perplexity: float,
    novelty_bonus: float,
    config: dict,
) -> Tuple[float, float]:
    """
    Core reward computation for both agents.

    R_B =  alpha * IoU + beta * type_correct
    R_A = -R_B - gamma_fluency * fluency_penalty + eta_novelty * novelty_bonus
    """
    iou          = compute_iou(pred_span, true_span)
    type_correct = compute_type_accuracy(pred_type, true_type)

    r_b = config["alpha"] * iou + config["beta"] * type_correct

    fluency_penalty = max(0.0, perplexity - config["ppl_threshold"]) / config["ppl_threshold"]

    r_a = (
        -r_b
        - config["gamma_fluency"] * fluency_penalty
        + config["eta_novelty"]   * novelty_bonus
    )

    return r_a, r_b