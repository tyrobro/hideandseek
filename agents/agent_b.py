# agents/agent_b.py
# Agent B: The Detector
# DistilBERT backbone with two output heads:
#   1. Classification head  → watermark type (4-class)
#   2. Span prediction head → (start, end) token indices (like SQuAD QA)

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizerFast
from typing import Tuple, Dict
from config import CONFIG


class RunningStandardiser(nn.Module):
    """Per-feature standardisation with running statistics.
    Q2 (EXPERIMENT_RESULTS.md): raw pooled features of different texts are nearly identical
    (cosine ~0.74-0.89); RL on them collapses to one class. Statistics are tracked during B's
    warm-up and then frozen, so rollouts and PPO updates see identical normalisation."""

    def __init__(self, dim: int, enabled: bool = True, eps: float = 1e-5):
        super().__init__()
        self.enabled, self.eps, self.track = enabled, eps, False
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def _update(self, x: torch.Tensor):
        x = x.detach().float().reshape(-1, x.shape[-1])
        n = float(self.count) + 1.0
        m = max(1.0 / n, 0.01)                     # running mean/var, EMA floor 0.01
        self.mean.mul_(1 - m).add_(x.mean(0) * m)
        self.var.mul_(1 - m).add_(((x - self.mean) ** 2).mean(0) * m)
        self.count.fill_(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if self.track:
            self._update(x)
        return (x - self.mean.to(x.dtype)) / torch.sqrt(self.var.to(x.dtype) + self.eps)


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
        # eager attention: the fused SDPA kernels are the usual suspect for illegal
        # memory accesses when a model lives on a non-default GPU (cuda:1) and is
        # run with grad enabled. Sequences are ~100 tokens, so the cost is negligible.
        self.encoder   = DistilBertModel.from_pretrained(
            config["detector_model"], attn_implementation="eager"
        ).to(device)

        hidden_size = self.encoder.config.hidden_size   # 768 for DistilBERT

        # ── Run 15 (Path 4): stronger detector ────────────────────────────
        #   unfreeze_encoder_b : fine-tune DistilBERT instead of using frozen features
        #   b_pooling          : "mean" (old) or "cls_mean_max" — max-pooling over tokens
        #                        keeps a rare, strongly-biased token from being averaged away
        self.train_encoder = bool(config.get("unfreeze_encoder_b", False))
        self.pooling       = config.get("b_pooling", "mean")
        for p in self.encoder.parameters():
            p.requires_grad_(self.train_encoder)
        pooled_dim = hidden_size * (3 if self.pooling == "cls_mean_max" else 1)

        self.norm = RunningStandardiser(pooled_dim, enabled=bool(config.get("b_standardise", False))).to(device)

        # ── Head 1: Watermark type classification ─────────────────────────
        self.type_head = nn.Sequential(
            nn.Linear(pooled_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, config["num_watermark_types"]),
        ).to(device)

        # ── Head 2: Span prediction (start + end logits per token) ────────
        # Identical to SQuAD-style QA heads
        self.span_start_head = nn.Linear(hidden_size, 1).to(device)
        self.span_end_head   = nn.Linear(hidden_size, 1).to(device)

        # The encoder is frozen and the type head has Dropout. In train mode the
        # rollout (detect) and the update (evaluate) see different dropout masks, so
        # the PPO ratio is noisy even at the first epoch. Keep B deterministic.
        self.eval()

    def head_parameters(self):
        return (list(self.type_head.parameters()) +
                list(self.span_start_head.parameters()) +
                list(self.span_end_head.parameters()))

    def encoder_parameters(self):
        return list(self.encoder.parameters()) if self.train_encoder else []

    def trainable_parameters(self):
        return self.head_parameters() + self.encoder_parameters()

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """hidden (1, L, H) -> pooled (1, H) or (1, 3H)."""
        if self.pooling != "cls_mean_max" or hidden.shape[1] <= 2:
            m = attention_mask.unsqueeze(-1).float()
            mean = (hidden * m).sum(dim=1) / m.sum(dim=1)
            return mean if self.pooling != "cls_mean_max" else torch.cat([hidden[:, 0], mean, mean], dim=-1)
        content = hidden[:, 1:-1]                     # drop [CLS] and [SEP]
        return torch.cat([hidden[:, 0], content.mean(dim=1), content.max(dim=1).values], dim=-1)

    def char_span_to_token_span(self, text: str, char_span: Tuple[int, int]) -> Tuple[int, int]:
        """Map a character span in `text` to B's wordpiece indices (end exclusive).
        Special tokens have zero-width offsets and are never part of the span."""
        enc  = self.tokenizer(text, truncation=True, max_length=512,
                              return_offsets_mapping=True)
        cs, ce = char_span
        idx = [i for i, (a, b) in enumerate(enc["offset_mapping"])
               if b > a and a < ce and b > cs]
        if not idx:
            return (0, 0)
        return (idx[0], idx[-1] + 1)

    # ── Forward pass ──────────────────────────────────────────────────────

    def _device_ctx(self):
        """Make B's GPU the *current* device while B runs (A is on cuda:0)."""
        d = str(self.device)
        return torch.cuda.device(d) if d.startswith("cuda") else contextlib.nullcontext()

    def forward(self, text: str) -> Dict:
        with self._device_ctx():
            return self._forward_impl(text)

    def _forward_impl(self, text: str) -> Dict:
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
        
        ctx = contextlib.nullcontext() if self.train_encoder else torch.no_grad()
        with ctx:
            hidden = self.encoder(**encoding).last_hidden_state  # (1, seq_len, 768)

        # Classification: pooled sequence features
        pooled      = self.norm(self._pool(hidden, encoding["attention_mask"]))
        type_logits = self.type_head(pooled)
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
        return {"type": type_lp, "start": start_lp, "end": end_lp, "type_entropy": entropy,
                "type_logits": out["type_logits"], "start_logits": out["start_logits"],
                "end_logits": out["end_logits"]}

    def supervised_loss(self, ev: Dict, true_type: str, true_span: Tuple[int, int]) -> torch.Tensor:
        """Cross-entropy of the heads against the known label (used with the PPO term; see Q2).
        true_span is in B's wordpiece space with an EXCLUSIVE end (char_span_to_token_span)."""
        dev = ev["type_logits"].device
        L = ev["start_logits"].shape[-1]
        s = min(max(int(true_span[0]), 0), L - 1)
        e = min(max(int(true_span[1]) - 1, s), L - 1)          # inclusive end index
        t = torch.tensor([self.type_list.index(true_type)], device=dev)
        ce_t = F.cross_entropy(ev["type_logits"], t)
        ce_s = F.cross_entropy(ev["start_logits"], torch.tensor([s], device=dev))
        ce_e = F.cross_entropy(ev["end_logits"], torch.tensor([e], device=dev))
        return ce_t + 0.5 * (ce_s + ce_e)

    @torch.no_grad()
    def detect_greedy(self, text: str):
        """Deterministic prediction for evaluation. Span end is EXCLUSIVE (same convention as the
        true span and compute_iou)."""
        out = self.forward(text)
        pred_type = self.type_list[int(out["type_logits"].argmax(-1))]
        s = int(out["start_logits"].argmax(-1))
        el = out["end_logits"].clone()
        el[:, :s] = -1e9
        e = int(el.argmax(-1))
        return pred_type, (s, e + 1), out["type_logits"].softmax(-1)[0].tolist()
    # ── PPO update ────────────────────────────────────────────────────────

    def ppo_loss(self, old_logprobs: Dict, new_logprobs: Dict,
                 adv_type: torch.Tensor, adv_span: torch.Tensor,
                 entropy_coef: float = 0.0) -> torch.Tensor:
        clip_eps = self.config["clip_epsilon"]
        losses = []
        for key, adv in (("type", adv_type), ("start", adv_span), ("end", adv_span)):
            ratio   = torch.exp(new_logprobs[key] - old_logprobs[key].detach())
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            losses.append(-torch.min(ratio * adv, clipped * adv))
        loss = torch.stack(losses).mean()
        if entropy_coef > 0:
            loss = loss - entropy_coef * new_logprobs["type_entropy"]
        return loss