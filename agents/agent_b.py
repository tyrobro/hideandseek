# agents/agent_b.py
# Agent B: The Detector
# DistilBERT backbone with two output heads:
#   1. Classification head  → watermark type (4-class)
#   2. Span prediction head → (start, end) token indices (like SQuAD QA)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizerFast
from typing import Tuple, Dict
from config import CONFIG


class AgentB(nn.Module):
    """
    The Detector agent.

    Given only the watermarked text (no metadata), predicts:
      - What TYPE of watermark was used  (classification head)
      - WHERE in the text the watermark is  (span head, start + end logits)

    Output is used to compute IoU reward against Agent A's ground truth.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__()
        self.config    = config
        self.device    = device
        self.type_list = config["watermark_types"]

        # ── Backbone ──────────────────────────────────────────────────────
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            config["detector_model"]
        )
        self.encoder   = DistilBertModel.from_pretrained(
            config["detector_model"]
        ).to(device)

        hidden_size = self.encoder.config.hidden_size   # 768 for DistilBERT

        # ── Head 1: Watermark type classification ─────────────────────────
        self.type_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, config["num_watermark_types"]),
        ).to(device)

        # ── Head 2: Span prediction (start + end logits per token) ────────
        # Identical to SQuAD-style QA heads
        self.span_start_head = nn.Linear(hidden_size, 1).to(device)
        self.span_end_head   = nn.Linear(hidden_size, 1).to(device)

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, text: str) -> Dict:
        """
        Encode text and run both heads.

        Returns:
            type_logits  : (1, num_types)
            start_logits : (1, seq_len)
            end_logits   : (1, seq_len)
            num_tokens   : int — needed to clamp span predictions
        """
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        hidden = self.encoder(**encoding).last_hidden_state  # (1, seq_len, 768)

        # Classification: pool [CLS] token
        attention_mask = encoding["attention_mask"].unsqueeze(-1).float()
        mean_hidden    = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        type_logits    = self.type_head(mean_hidden)
        # Span: per-token start/end logits
        start_logits = self.span_start_head(hidden).squeeze(-1)  # (1, seq_len)
        end_logits   = self.span_end_head(hidden).squeeze(-1)    # (1, seq_len)

        return {
            "type_logits"  : type_logits,
            "start_logits" : start_logits,
            "end_logits"   : end_logits,
            "num_tokens"   : hidden.shape[1],
        }

    # ── Detection (inference) ─────────────────────────────────────────────
    @torch.no_grad()
    def detect(self, text: str) -> Tuple[str, Tuple[int, int], Dict]:
        """
        Run detection on a piece of text.

        Returns:
            pred_type  : predicted watermark type string
            pred_span  : (start_token, end_token) predicted indices
            log_probs  : dict of log-probs for PPO update
        """
        out = self.forward(text)

        # ── Type prediction ──────────────────────────────────────────────
        type_dist  = torch.distributions.Categorical(logits=out["type_logits"])
        type_idx   = type_dist.sample()
        pred_type  = self.type_list[type_idx.item()]
        type_lp    = type_dist.log_prob(type_idx)

        # ── Span prediction ──────────────────────────────────────────────
        # Sample start, then sample end conditioned on end >= start
        start_dist = torch.distributions.Categorical(logits=out["start_logits"])
        start_idx  = start_dist.sample()
        start_lp   = start_dist.log_prob(start_idx)

        # Mask end logits so end >= start
        seq_len    = out["num_tokens"]
        end_logits = out["end_logits"].clone()
        end_logits[:, :start_idx.item()] = -1e9   # mask positions before start

        end_dist   = torch.distributions.Categorical(logits=end_logits)
        end_idx    = end_dist.sample()
        end_lp     = end_dist.log_prob(end_idx)

        pred_span  = (start_idx.item(), end_idx.item())

        log_probs = {
            "type" : type_lp,
            "start": start_lp,
            "end"  : end_lp,
            "type_logits" : out["type_logits"],
        }

        return pred_type, pred_span, log_probs

    def evaluate(self, text: str, pred_type: str, pred_span: Tuple[int, int]) -> Dict:
        """Log-prob of the actions B actually took, under the current weights."""
        out   = self.forward(text)
        start, end = pred_span
        type_lp  = F.log_softmax(out["type_logits"], dim=-1)[0, self.type_list.index(pred_type)]
        start_lp = F.log_softmax(out["start_logits"], dim=-1)[0, start]
        end_logits = out["end_logits"].clone()
        end_logits[:, :start] = -1e9
        end_lp   = F.log_softmax(end_logits, dim=-1)[0, end]
        probs    = F.softmax(out["type_logits"], dim=-1)
        entropy  = -(probs * torch.log(probs + 1e-9)).sum()
        return {"type": type_lp, "start": start_lp, "end": end_lp, "type_entropy": entropy}
    # ── PPO update ────────────────────────────────────────────────────────

    def ppo_loss(self, old_logprobs: Dict, new_logprobs: Dict,
                 advantage: torch.Tensor, entropy_coef: float = 0.0) -> torch.Tensor:
        clip_eps = self.config["clip_epsilon"]
        losses = []
        for key in ["type", "start", "end"]:
            ratio   = torch.exp(new_logprobs[key] - old_logprobs[key].detach())
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            losses.append(-torch.min(ratio * advantage, clipped * advantage))
        loss = torch.stack(losses).mean()
        if entropy_coef > 0:
            loss = loss - entropy_coef * new_logprobs["type_entropy"]
        return loss