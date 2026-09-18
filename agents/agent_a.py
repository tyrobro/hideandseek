# agents/agent_a.py
# Agent A: The Watermarker
# GPT-2 backbone + a lightweight policy head that selects
# (watermark_type, intensity, position) before each generation.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessor
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

class WatermarkLogitsProcessor(LogitsProcessor):
    """
    Plugs into HuggingFace's generate() pipeline.
    Applies the chosen watermark strategy only within the designated span window.
    This replaces the manual token-by-token loop entirely.
    """

    def __init__(self, strategy, wm_start, wm_end, intensity, agent, max_new):
        self.strategy  = strategy
        self.wm_start  = wm_start
        self.wm_end    = wm_end
        self.intensity = intensity
        self.agent     = agent     # reference to AgentA for _apply_strategy
        self.max_new   = max_new
        self.step      = 0         # tracks generation step

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # Only apply watermark within the designated window
        if self.wm_start <= self.step <= self.wm_end:
            scores, _ = self.agent._apply_strategy(
                self.strategy,
                scores,
                self.step,
                self.max_new,
                self.intensity,
            )
        self.step += 1
        return scores

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

    def select_strategy(self, prompt_hidden: torch.Tensor) -> Tuple:
        head_out = self.policy(prompt_hidden)

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
            "type"      : type_dist.log_prob(type_idx).detach(),
            "intensity" : intensity_dist.log_prob(intensity_idx).detach(),
            "position"  : position_dist.log_prob(position_idx).detach(),
        }

        # Also store the action indices so PPO can recompute logprob of same action
        action_indices = {
            "type"      : type_idx.detach(),
            "intensity" : intensity_idx.detach(),
            "position"  : position_idx.detach(),
        }

        return chosen_type, intensity, position_frac, log_probs, action_indices
    # ── Watermarked generation ────────────────────────────────────────────

    def generate(self, prompt: str) -> Tuple[str, str, Tuple[int, int], Dict]:
        inputs    = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=64,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        max_new   = self.config["max_gen_length"]

        # ── Step 1: Policy head selects strategy (needs grad) ────────────
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        prompt_hidden = outputs.hidden_states[-1].detach()
        del outputs
        torch.cuda.empty_cache()

        chosen_type, intensity, position_frac, strategy_logprobs, action_indices = \
            self.select_strategy(prompt_hidden)
        del prompt_hidden
        torch.cuda.empty_cache()

        # ── Step 2: Determine watermark span ─────────────────────────────
        wm_start  = int(position_frac * max_new)
        wm_size   = max(5, max_new // 5)
        wm_end    = min(wm_start + wm_size, max_new - 1)
        true_span = (wm_start, wm_end)

        # ── Step 3: Apply logit processor for watermark ───────────────────
        # Instead of manual loop, use HF's logits_processor hook
        processor = WatermarkLogitsProcessor(
            strategy    = chosen_type,
            wm_start    = wm_start,
            wm_end      = wm_end,
            intensity   = intensity,
            agent       = self,
            max_new     = max_new,
        )

        # ── Step 4: HF generate with KV-cache (memory efficient) ─────────
        with torch.no_grad():
            generated = self.model.generate(
                input_ids         = input_ids,
                max_new_tokens    = max_new,
                do_sample         = True,
                temperature       = 1.0,
                logits_processor  = [processor],
                pad_token_id      = self.tokenizer.eos_token_id,
                use_cache         = True,   # KV-cache: huge memory saving
            )

        new_ids = generated[:, input_ids.shape[1]:]
        text    = self.tokenizer.decode(new_ids.squeeze(), skip_special_tokens=True)

        del generated, new_ids, input_ids
        torch.cuda.empty_cache()

        all_logprobs = {
            "strategy"       : strategy_logprobs,
            "action_indices" : action_indices,
            "tokens"         : None,
        }

        return text, chosen_type, true_span, all_logprobs
    # ── Internal strategy dispatcher ──────────────────────────────────────

    def recompute_strategy_logprobs(self, prompt: str) -> Dict:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=64,
        ).to(self.device)

        # This forward pass NEEDS grad — it's what PPO differentiates
        outputs       = self.model(**inputs, output_hidden_states=True)
        prompt_hidden = outputs.hidden_states[-1]

        head_out = self.policy(prompt_hidden)

        type_dist      = torch.distributions.Categorical(logits=head_out["type_logits"])
        intensity_dist = torch.distributions.Categorical(logits=head_out["intensity_logits"])
        position_dist  = torch.distributions.Categorical(logits=head_out["position_logits"])

        return {
            "type_logits"      : head_out["type_logits"],
            "intensity_logits" : head_out["intensity_logits"],
            "position_logits"  : head_out["position_logits"],
            "type_dist"        : type_dist,
            "intensity_dist"   : intensity_dist,
            "position_dist"    : position_dist,
        }
    
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
    old_logprobs    : Dict,
    fresh_head_out  : Dict,
    action_indices  : Dict,
    reward          : torch.Tensor,
) -> torch.Tensor:
        """
        PPO loss using fresh logprobs recomputed this epoch
        vs old logprobs stored from the original rollout.
        """
        clip_eps = self.config["clip_epsilon"]
        losses   = []

        key_to_dist = {
            "type"      : fresh_head_out["type_dist"],
            "intensity" : fresh_head_out["intensity_dist"],
            "position"  : fresh_head_out["position_dist"],
        }

        for key in ["type", "intensity", "position"]:
            old_lp  = old_logprobs["strategy"][key]           # detached scalar
            new_lp  = key_to_dist[key].log_prob(action_indices[key])
            ratio   = torch.exp(new_lp - old_lp)
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            obj     = torch.min(ratio * reward, clipped * reward)
            losses.append(-obj)

        return torch.stack(losses).mean()