# rewards/reward_functions.py
# Run 02: added reward clipping to prevent catastrophic policy updates

import torch
from typing import Tuple


def compute_perplexity(model, tokenizer, text: str, device: str = "cpu") -> float:
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    input_ids = encodings["input_ids"]

    with torch.no_grad():
        outputs = model(**encodings, labels=input_ids)
        loss    = outputs.loss

    ppl = torch.exp(loss).item()

    # Cap returned PPL to prevent inf/nan from propagating
    return min(ppl, 10000.0)


def compute_iou(
    pred_span: Tuple[int, int],
    true_span: Tuple[int, int],
) -> float:
    pred_start, pred_end = pred_span
    true_start, true_end = true_span

    intersection = max(0, min(pred_end, true_end) - max(pred_start, true_start))
    union        = max(1, max(pred_end, true_end) - min(pred_start, true_start))

    return intersection / union


def compute_type_accuracy(pred_type: str, true_type: str) -> float:
    return 1.0 if pred_type == true_type else 0.0


def compute_agent_rewards(
    true_type     : str,
    true_span     : Tuple[int, int],
    pred_type     : str,
    pred_span     : Tuple[int, int],
    perplexity    : float,
    novelty_bonus : float,
    config        : dict,
) -> Tuple[float, float]:

    iou          = compute_iou(pred_span, true_span)
    type_correct = compute_type_accuracy(pred_type, true_type)

    r_b = config["alpha"] * iou + config["beta"] * type_correct

    fluency_penalty = max(0.0, perplexity - config["ppl_threshold"]) / config["ppl_threshold"]

    r_a = (
        -r_b
        - config["gamma_fluency"] * fluency_penalty
        + config["eta_novelty"]   * novelty_bonus
    )

    # ── Reward clipping — prevents extreme values from blowing up policy ──
    r_a = max(min(r_a, config["reward_clip_a_max"]), config["reward_clip_a_min"])
    r_b = max(min(r_b, config["reward_clip_b_max"]), config["reward_clip_b_min"])

    return r_a, r_b