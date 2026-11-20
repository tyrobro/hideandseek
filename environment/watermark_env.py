# environment/watermark_env.py
# The core zero-sum RL environment for Hide&Seek.
# Manages one episode: prompt → watermarked text → detection → rewards.

from typing import Tuple, Dict
from rewards.reward_functions import compute_iou, compute_agent_rewards


class WatermarkEnv:
    """
    Zero-sum game environment.

    Episode flow:
      1. reset(prompt)                          → fresh episode state
      2. Agent A generates watermarked text     → env logs ground truth
      3. Agent B receives only the text         → predicts (type, span)
      4. step(...)                              → returns (r_a, r_b, info)
    """

    def __init__(self, config: dict):
        self.config           = config
        self.strategy_history = []   # tracks Agent A's choices for novelty
        self.episode_count    = 0
        self.position_history = []
    # ── Episode management ────────────────────────────────────────────────

    def reset(self, prompt: str) -> Dict:
        self.current_prompt = prompt
        self.episode_count += 1
        return {"prompt": prompt, "episode": self.episode_count}

    # ── Novelty bonus ─────────────────────────────────────────────────────

    def compute_novelty_bonus(self, chosen_type: str) -> float:
        window = self.config["novelty_window"]
        recent = self.strategy_history[-window:]
        if not recent:
            return 1.0
        frequency = recent.count(chosen_type) / len(recent)
        # Penalty when overused: negative when frequency > 0.5
        return (1.0 - 2.0 * frequency)

    # ── Environment step ──────────────────────────────────────────────────

    def step(
        self,
        watermarked_text : str,
        true_type        : str,
        true_span        : Tuple[int, int],   # (start_token, end_token) ground truth
        pred_type        : str,
        pred_span        : Tuple[int, int],   # Agent B's prediction
        perplexity       : float,
        chosen_type      : str,    
        position_frac    : float = None,
    ) -> Tuple[float, float, Dict]:
        """
        Compute rewards and return episode info.

        Returns:
            r_a   : Agent A's scalar reward
            r_b   : Agent B's scalar reward
            info  : logging dict
        """
        novelty_bonus = 0.0 if self.config.get("force_strategy") else self.compute_novelty_bonus(chosen_type)
        
        r_a, r_b = compute_agent_rewards(
            true_type     = true_type,
            true_span     = true_span,
            pred_type     = pred_type,
            pred_span     = pred_span,
            perplexity    = perplexity,
            novelty_bonus = novelty_bonus,
            config        = self.config,
        )

        # Log strategy for future novelty computation
        self.strategy_history.append(chosen_type)
        if self.config.get("legacy_history_bug"):
            # Reproduces the Run 08-10 bug for experiment Q4: spans interleaved with strategies,
            # so the last 20 entries hold at most 10 strategies (floor never fires, novelty halved)
            self.strategy_history.append(true_span)
        
        info = {
            "episode"      : self.episode_count,
            "iou"          : compute_iou(pred_span, true_span),
            "type_correct" : pred_type == true_type,
            "perplexity"   : perplexity,
            "novelty_bonus": novelty_bonus,
            "reward_a"     : r_a,
            "reward_b"     : r_b,
            "true_type"    : true_type,
            "pred_type"    : pred_type,
            "true_span"    : true_span,
            "pred_span"    : pred_span,
            "chosen_type"  : chosen_type,
            "position_frac": position_frac,
        }

        return r_a, r_b, info