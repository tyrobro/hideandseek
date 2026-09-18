# agents/agent_a.py
# Agent A: The Watermarker
# GPT-2 backbone + a lightweight policy head that selects
# (watermark_type, intensity, position) before each generation.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Tuple, Dict, List
from environment.watermark_types import WATERMARK_STRATEGIES
from config import CONFIG


class WatermarkPolicyHead(nn.Module):
    """
    Small MLP that sits on top of GPT-2's hidden states.
    Takes the mean-pooled last hidden state of the prompt
    and outputs a probability distribution over:
      - watermark type  (4 classes)
      - intensity       (3 levels: low / medium / high)
      - position        (3 options: early / middle / late)
    """

    INTENSITY_VALUES  = [0.5, 1.0, 2.0]       # maps index → actual intensity
    POSITION_FRACTIONS = [0.1, 0.45, 0.75]    # maps index → fraction of seq length

    def __init__(self, hidden_size: int, num_types: int = 4):
        super().__init__()
        self.type_head      = nn.Linear(hidden_size, num_types)   # 4-way softmax
        self.intensity_head = nn.Linear(hidden_size, 3)           # 3-way softmax
        self.position_head  = nn.Linear(hidden_size, 3)           # 3-way softmax

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        hidden: (batch, seq_len, hidden_size)
        Returns logits for each decision head.
        """
        pooled = hidden.mean(dim=1)   # (batch, hidden_size)
        return {
            "type_logits"     : self.type_head(pooled),
            "intensity_logits": self.intensity_head(pooled),
            "position_logits" : self.position_head(pooled),
        }


class AgentA(nn.Module):
    """
    The Watermarker agent.

    Responsibilities:
      1. Encode the prompt through GPT-2
      2. Policy head selects (type, intensity, position)
      3. Generate token-by-token, applying the chosen
         watermark strategy at the right position window
      4. Return: generated text, ground-truth watermark metadata,
         and log-probabilities needed for PPO
    """

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__()
        self.config    = config
        self.device    = device
        self.type_list = config["watermark_types"]   # ["token","syntactic","semantic","positional"]

        # ── Backbone ──────────────────────────────────────────────────────
        self.tokenizer = GPT2Tokenizer.from_pretrained(config["base_model"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model     = GPT2LMHeadModel.from_pretrained(
            config["base_model"],
            output_hidden_states=True,
        ).to(device)

        # ── Policy head ───────────────────────────────────────────────────
        hidden_size  = self.model.config.hidden_size
        self.policy  = WatermarkPolicyHead(hidden_size, config["num_watermark_types"]).to(device)

        # ── Dummy context tokens for syntactic/semantic strategies ────────
        # In a full implementation these would come from a POS tagger /
        # synonym database. Here we use fixed vocab subsets as placeholders
        # so the system is end-to-end runnable from day one.
        vocab_size = self.tokenizer.vocab_size
        self.syntactic_bias_tokens = list(range(100, 200))          # placeholder
        self.semantic_synonym_map  = {i: i+1 for i in range(50)}   # placeholder

    # ── Strategy selection ────────────────────────────────────────────────

    def select_strategy(
        self, prompt_hidden: torch.Tensor
    ) -> Tuple[str, float, float, Dict]:
        """
        Run the policy head and sample a strategy.

        Returns:
            chosen_type      : e.g. "token"
            intensity        : float
            position_frac    : fraction along sequence where watermark starts
            log_probs        : dict of log-probs for PPO update
        """
        head_out = self.policy(prompt_hidden)

        # Sample from each head (not argmax — we need exploration)
        type_dist      = torch.distributions.Categorical(logits=head_out["type_logits"])
        intensity_dist = torch.distributions.Categorical(logits=head_out["intensity_logits"])
        position_dist  = torch.distributions.Categorical(logits=head_out["position_logits"])

        type_idx      = type_dist.sample()
        intensity_idx = intensity_dist.sample()
        position_idx  = position_dist.sample()

        chosen_type   = self.type_list[type_idx.item()]
        intensity     = WatermarkPolicyHead.INTENSITY_VALUES[intensity_idx.item()]
        position_frac = WatermarkPolicyHead.POSITION_FRACTIONS[position_idx.item()]

        log_probs = {
            "type"     : type_dist.log_prob(type_idx),
            "intensity": intensity_dist.log_prob(intensity_idx),
            "position" : position_dist.log_prob(position_idx),
        }

        return chosen_type, intensity, position_frac, log_probs

    # ── Watermarked generation ────────────────────────────────────────────

    def generate(self, prompt: str) -> Tuple[str, str, Tuple[int, int], Dict]:
        inputs     = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(self.device)
        input_ids  = inputs["input_ids"]
        max_new    = self.config["max_gen_length"]

        # Policy head needs grad — but run it first, then free the hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        prompt_hidden = outputs.hidden_states[-1].detach()

        # Select strategy — this part needs grad for PPO
        chosen_type, intensity, position_frac, strategy_logprobs = \
            self.select_strategy(prompt_hidden)

        # Watermark span
        wm_start  = int(position_frac * max_new)
        wm_size   = max(5, max_new // 5)
        wm_end    = min(wm_start + wm_size, max_new - 1)
        true_span = (wm_start, wm_end)

        # Token generation — NO gradients here, saves enormous memory
        generated_ids = input_ids.clone()

        with torch.no_grad():
            for step in range(max_new):
                out    = self.model(input_ids=generated_ids)
                logits = out.logits[:, -1, :].clone()

                in_window = wm_start <= step <= wm_end
                if in_window:
                    logits, _ = self._apply_strategy(
                        chosen_type, logits, wm_start, max_new, intensity
                    )

                probs    = torch.nn.functional.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_tok], dim=-1)

                if next_tok.item() == self.tokenizer.eos_token_id:
                    break

                # Free intermediate tensors aggressively
                del out, logits, probs

        new_ids = generated_ids[:, input_ids.shape[1]:]
        text    = self.tokenizer.decode(new_ids.squeeze(), skip_special_tokens=True)

        # Clean up
        del generated_ids, prompt_hidden
        torch.cuda.empty_cache()

        # Token logprobs are not needed since generation is no_grad
        # PPO only updates the policy head via strategy_logprobs
        all_logprobs = {"strategy": strategy_logprobs, "tokens": None}

        return text, chosen_type, true_span, all_logprobs

    # ── Internal strategy dispatcher ──────────────────────────────────────

    def _apply_strategy(
        self,
        chosen_type : str,
        logits      : torch.Tensor,
        position    : int,
        total_len   : int,
        intensity   : float,
    ) -> Tuple[torch.Tensor, Dict]:
        """Route to the correct watermarking function."""

        if chosen_type == "token":
            return WATERMARK_STRATEGIES["token"](
                logits, intensity=intensity
            )
        elif chosen_type == "syntactic":
            return WATERMARK_STRATEGIES["syntactic"](
                logits,
                pos_bias_tokens=self.syntactic_bias_tokens,
                intensity=intensity,
            )
        elif chosen_type == "semantic":
            return WATERMARK_STRATEGIES["semantic"](
                logits,
                synonym_map=self.semantic_synonym_map,
                intensity=intensity,
            )
        elif chosen_type == "positional":
            return WATERMARK_STRATEGIES["positional"](
                logits,
                position=position,
                total_length=total_len,
                intensity=intensity,
            )
        else:
            raise ValueError(f"Unknown watermark type: {chosen_type}")

    # ── PPO update ────────────────────────────────────────────────────────

    def ppo_loss(
    self,
    old_logprobs : Dict,
    new_logprobs : Dict,
    reward       : torch.Tensor,
) -> torch.Tensor:
        clip_eps = self.config["clip_epsilon"]
        losses   = []

        for key in ["type", "intensity", "position"]:
            old_lp  = old_logprobs["strategy"][key].detach()   # always detach old
            new_lp  = new_logprobs["strategy"][key]
            ratio   = torch.exp(new_lp - old_lp)
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            obj     = torch.min(ratio * reward, clipped * reward)
            losses.append(-obj)

        return torch.stack(losses).mean()