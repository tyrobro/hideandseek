# environment/watermark_env.py

import torch
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from environment.watermark_types import WATERMARK_STRATEGIES

@dataclass
class Episode:
    """Holds all info about one game episode."""
    prompt: str
    watermarked_text: str
    watermark_type: str          # ground truth type
    watermark_span: Tuple[int, int]  # ground truth (start, end) token indices
    intensity: float
    agent_a_logprobs: torch.Tensor   # for PPO
    agent_b_logprobs: torch.Tensor   # for PPO
    reward_a: float
    reward_b: float


class WatermarkEnv:
    """
    The zero-sum RL environment for Hide&Seek.

    Flow per episode:
      1. Sample prompt
      2. Agent A picks strategy (type, intensity, position)
      3. Agent A generates watermarked text — env logs ground truth
      4. Agent B receives only the text
      5. Agent B predicts (type, span)
      6. Env computes rewards and returns them to both agents
    """

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.strategy_history = []     # for novelty bonus computation

    def reset(self, prompt: str) -> Dict:
        """Start a new episode with a fresh prompt."""
        return {"prompt": prompt, "step": 0}

    def compute_iou(
        self,
        pred_span: Tuple[int, int],
        true_span: Tuple[int, int]
    ) -> float:
        """
        Intersection over Union between predicted and true watermark spans.
        Both spans are (start, end) token index pairs.
        """
        pred_start, pred_end = pred_span
        true_start, true_end = true_span

        intersection_start = max(pred_start, true_start)
        intersection_end = min(pred_end, true_end)
        intersection = max(0, intersection_end - intersection_start)

        union_start = min(pred_start, true_start)
        union_end = max(pred_end, true_end)
        union = max(1, union_end - union_start)   # avoid div by zero

        return intersection / union

    def compute_novelty_bonus(self, chosen_type: str) -> float:
        """
        Reward Agent A for using strategies it hasn't used recently.
        Looks back over last 20 episodes.
        """
        recent = self.strategy_history[-20:]
        if not recent:
            return 0.2
        frequency = recent.count(chosen_type) / len(recent)
        return max(0.0, 0.2 - frequency)   # less used = higher bonus

    def compute_rewards(
        self,
        true_type: str,
        true_span: Tuple[int, int],
        pred_type: str,
        pred_span: Tuple[int, int],
        perplexity: float,
        chosen_type: str,
    ) -> Tuple[float, float]:
        """
        Compute R_A and R_B for one episode.

        R_B = alpha * IoU + beta * type_correct
        R_A = -R_B - gamma * fluency_penalty + eta * novelty_bonus
        """
        cfg = self.config

        iou = self.compute_iou(pred_span, true_span)
        type_correct = float(pred_type == true_type)

        # Agent B reward
        r_b = cfg["alpha"] * iou + cfg["beta"] * type_correct

        # Fluency penalty for Agent A
        fluency_penalty = max(0.0, perplexity - cfg["ppl_threshold"]) / cfg["ppl_threshold"]

        # Novelty bonus for Agent A
        novelty = self.compute_novelty_bonus(chosen_type)

        # Agent A reward (zero-sum core, plus constraints)
        r_a = -r_b \
              - cfg["gamma_fluency"] * fluency_penalty \
              + cfg["eta_novelty"] * novelty

        return r_a, r_b

    def step(
        self,
        watermarked_text: str,
        true_type: str,
        true_span: Tuple[int, int],
        pred_type: str,
        pred_span: Tuple[int, int],
        perplexity: float,
        chosen_type: str,
    ) -> Tuple[float, float, Dict]:
        """
        Execute one environment step.
        Returns (reward_a, reward_b, info_dict)
        """
        r_a, r_b = self.compute_rewards(
            true_type, true_span,
            pred_type, pred_span,
            perplexity, chosen_type
        )

        # Log strategy for novelty tracking
        self.strategy_history.append(chosen_type)

        info = {
            "iou": self.compute_iou(pred_span, true_span),
            "type_correct": pred_type == true_type,
            "perplexity": perplexity,
            "reward_a": r_a,
            "reward_b": r_b,
            "chosen_type": chosen_type,
        }

        return r_a, r_b, info