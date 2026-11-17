# train.py
# Main PPO training loop for Hide&Seek.
# Ties together the environment, Agent A, Agent B, and reward computation.

import os
import glob
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from rewards.reward_functions import compute_perplexity
from environment.watermark_env import WatermarkEnv
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from config import CONFIG
from collections import deque


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.frozen_a = bool(config.get("freeze_agent_a"))

        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        os.makedirs(os.path.dirname(config["log_path"]), exist_ok=True)

        # ── Assign one GPU per agent ───────────────────────────────────────
        if torch.cuda.device_count() >= 2:
            self.device_a = "cuda:0"   # Agent A (GPT-2 watermarker)
            self.device_b = "cuda:1"   # Agent B (DistilBERT detector)
            print(f"Dual GPU mode: Agent A → {self.device_a} | Agent B → {self.device_b}")
        elif torch.cuda.device_count() == 1:
            self.device_a = "cuda:0"
            self.device_b = "cuda:0"
            print("Single GPU mode: both agents on cuda:0")
        else:
            self.device_a = "cpu"
            self.device_b = "cpu"
            print("CPU mode")

        # ── Agents ────────────────────────────────────────────────────────
        self.agent_a = AgentA(config, self.device_a)
        self.agent_b = AgentB(config, self.device_b)

        # ── Environment ───────────────────────────────────────────────────
        self.env = WatermarkEnv(config)
        self.agent_a._strategy_history_ref = self.env.strategy_history
        self.agent_a._position_history_ref = self.env.position_history

        # ── Optimizers ────────────────────────────────────────────────────
        # Run 14: if A is frozen, switch off its grads entirely.
        # Otherwise the backbone gets its own (much smaller) learning rate.
        if self.frozen_a:
            for p in list(self.agent_a.model.parameters()) + list(self.agent_a.policy.parameters()):
                p.requires_grad_(False)

        self.opt_a = torch.optim.Adam([
            {"params": self.agent_a.policy.parameters(), "lr": config["lr_agent_a"]},
            {"params": self.agent_a.model.parameters(),  "lr": config.get("lr_backbone_a", 1e-6)},
        ])
        b_groups = [{"params": self.agent_b.head_parameters(), "lr": config["lr_agent_b"]}]
        if self.agent_b.train_encoder:
            b_groups.append({"params": self.agent_b.encoder_parameters(),
                             "lr": config.get("lr_encoder_b", 2e-5)})
        self.opt_b = torch.optim.Adam(b_groups)
        # Mixed precision — cuts VRAM ~40-50% with negligible quality loss
        self.scaler_a = torch.amp.GradScaler(device='cuda')
        self.scaler_b = torch.amp.GradScaler(device='cuda')

        # ── Dataset ───────────────────────────────────────────────────────
        print("Loading dataset...")
        dataset      = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        self.prompts = [
            x["text"] for x in dataset
            if len(x["text"].strip()) > 100
        ]
        print(f"Loaded {len(self.prompts)} prompts.")

        self.log_rows    = []
        self.b_diag      = {}
        self._b_buffer   = []        # B rollouts waiting for a batched update
        self.n_no_wm     = 0         # episodes skipped: generation ended before the window
        self.type_hist   = deque(maxlen=50)

        # ── Resume from checkpoint if one exists ──────────────────────────
        ckpts = sorted(glob.glob(os.path.join(config["checkpoint_dir"], "ep*.pt")))
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location="cpu")
            self.agent_a.policy.load_state_dict(ckpt["agent_a_policy"])
            if "agent_a_model" in ckpt:
                self.agent_a.model.load_state_dict(ckpt["agent_a_model"])
            self.agent_b.type_head.load_state_dict(ckpt["agent_b_heads"]["type"])
            self.agent_b.span_start_head.load_state_dict(ckpt["agent_b_heads"]["start"])
            self.agent_b.span_end_head.load_state_dict(ckpt["agent_b_heads"]["end"])
            if "agent_b_encoder" in ckpt and self.agent_b.train_encoder:
                self.agent_b.encoder.load_state_dict(ckpt["agent_b_encoder"])
            for name, opt in (("opt_a", self.opt_a), ("opt_b", self.opt_b)):
                if name in ckpt:
                    try:
                        opt.load_state_dict(ckpt[name])
                    except Exception as e:
                        print(f"Could not restore {name} ({e}); continuing with fresh optimizer state")
            self.start_episode = ckpt["episode"] + 1
            self.env.strategy_history = ckpt.get("strategy_history", [])
            self.env.position_history = ckpt.get("position_history", [])
            self.agent_a._strategy_history_ref = self.env.strategy_history
            self.agent_a._position_history_ref = self.env.position_history
            self.log_rows = ckpt.get("log_rows", [])
            print(f"Resumed from ep {ckpt['episode']} with {len(self.log_rows)} log rows "
                  f"(backbone restored: {'agent_a_model' in ckpt})")
        else:
            self.start_episode = 1

    # ── Main training loop ────────────────────────────────────────────────

    def run(self):
        print(f"\nStarting Hide&Seek training for {self.config['num_episodes']} episodes...\n")
        print("Config in effect:", {k: self.config.get(k) for k in [
            "force_strategy", "random_policy", "freeze_agent_a", "ppl_brake",
            "min_unique_ratio", "lr_agent_a", "lr_backbone_a", "lr_agent_b",
            "entropy_coef_b", "b_update_batch", "unfreeze_encoder_b", "lr_encoder_b", "b_pooling", "log_every", "save_every"]})

        for episode in tqdm(range(self.start_episode, self.config["num_episodes"] + 1)):
            torch.cuda.empty_cache()

            # 1. Sample a prompt
            idx    = torch.randint(0, len(self.prompts), (1,)).item()
            prompt = self.prompts[idx][:100]

            # 2. Reset environment
            self.env.reset(prompt)

            # 3. Agent A generates watermarked text
            text, chosen_type, true_span, logprobs_a = self.agent_a.generate(prompt)

            if not text.strip():
                continue   # skip degenerate generations
            if not logprobs_a.get("wm_in_text", True):
                self.n_no_wm += 1   # generation ended before the watermark window:
                continue            # no watermark in the text, label would be noise

            toks = self.agent_a.tokenizer.encode(text)
            uniq = len(set(toks)) / max(1, len(toks))

            # 4. Compute fluency (perplexity) of generated text
            perplexity = compute_perplexity(
                self.agent_a.model,
                self.agent_a.tokenizer,
                text,
                device=self.device_a,
            )
            self.agent_a.update_ppl_tracker(perplexity)

            # Brake: high PPL, or repetition (low unique-token ratio)
            braked = (perplexity > self.config["ppl_brake"]
                      or uniq < self.config.get("min_unique_ratio", 0.0))
            if braked:
                if not self.frozen_a:
                    self._corrective_step()
                if episode % self.config["log_every"] == 0:
                    print(f"[Ep {episode}] BRAKED — PPL {perplexity:.1f}, unique {uniq:.2f}"
                          f"{'' if not self.frozen_a else ' (A frozen, no corrective step)'}")
                    self.log_rows.append({
                        "episode": episode, "iou": None, "type_correct": None,
                        "perplexity": perplexity, "uniq_ratio": uniq, "novelty_bonus": None,
                        "reward_a": None, "reward_b": None, "true_type": None,
                        "pred_type": None, "true_span": None, "pred_span": None,
                        "chosen_type": None, "braked": True,
                    })
                    pd.DataFrame(self.log_rows).to_csv(self.config["log_path"], index=False)
                continue

            # 5. Agent B detects watermark
            pred_type, pred_span, logprobs_b = self.agent_b.detect(text)

            # Ground-truth span in B's own (wordpiece) index space
            true_span_gpt2 = true_span
            true_span      = self.agent_b.char_span_to_token_span(text, logprobs_a["char_span"])

            # 6. Environment step → rewards
            r_a, r_b, info = self.env.step(
                watermarked_text = text,
                true_type        = chosen_type,
                true_span        = true_span,
                pred_type        = pred_type,
                pred_span        = pred_span,
                perplexity       = perplexity,
                chosen_type      = chosen_type,
                position_frac    = logprobs_a.get("position_frac"),
            )

            self.type_hist.append(float(info["type_correct"]))
            info["intensity_idx"]   = int(logprobs_a["action_indices"]["intensity"])
            info["true_span_gpt2"]  = true_span_gpt2
            info["uniq_ratio"]    = uniq

            # 7. PPO updates
            if not self.frozen_a:
                self._update_agent_a(prompt, text, logprobs_a, r_a)
            self._b_buffer.append((text, pred_type, pred_span, logprobs_b,
                                   float(info["type_correct"]), float(info["iou"])))
            if len(self._b_buffer) >= self.config.get("b_update_batch", 8):
                self._update_agent_b_batch()

            # 8. Logging
            if episode % self.config["log_every"] == 0:
                self._log(episode, info)

            # 9. Checkpoint
            if episode % self.config["save_every"] == 0:
                self._save(episode)

        print(f"\nTraining complete. Episodes skipped (no watermark in text): {self.n_no_wm}")
        self._save_logs()

    # ── PPO update helpers ────────────────────────────────────────────────

    def _anchor_text(self) -> str:
        """Real WikiText passage used to anchor fluency / KL (not A's own samples,
        which reward repetition)."""
        i = torch.randint(0, len(self.prompts), (1,)).item()
        return self.prompts[i][:600]

    def _corrective_step(self):
        anchor = self._anchor_text()
        loss = (self.config.get("fluency_loss_coef", 0.1) * self.agent_a.compute_fluency_loss(anchor)
              + self.config.get("kl_coef", 0.05) * self.agent_a.compute_kl_to_ref(anchor))
        self.opt_a.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent_a.model.parameters(), 1.0)
        self.opt_a.step()

    def _update_agent_a(self, prompt: str, text: str, old_logprobs: dict, reward: float):
        reward_tensor  = torch.tensor(reward, dtype=torch.float32).to(self.device_a)
        action_indices = old_logprobs["action_indices"]

        fluency_coef = self.config.get("fluency_loss_coef", 0.1)
        kl_coef      = self.config.get("kl_coef", 0.05)
        anchor       = self._anchor_text()   # one real-text anchor per episode

        for _ in range(self.config["ppo_epochs"]):
            with torch.amp.autocast(device_type='cuda'):
                fresh_head_out = self.agent_a.recompute_strategy_logprobs(prompt)
                policy_loss = self.agent_a.ppo_loss(
                    old_logprobs   = old_logprobs,
                    fresh_head_out = fresh_head_out,
                    action_indices = action_indices,
                    reward         = reward_tensor,
                )
                pos_dist = torch.distributions.Categorical(logits=fresh_head_out["position_logits"])
                if self.config["entropy_coef_position"] > 0:
                    policy_loss = policy_loss - self.config["entropy_coef_position"] * pos_dist.entropy().mean()

                fluency_loss = self.agent_a.compute_fluency_loss(anchor)
                kl_loss      = self.agent_a.compute_kl_to_ref(anchor)

                loss = policy_loss + fluency_coef * fluency_loss + kl_coef * kl_loss

            self.opt_a.zero_grad()
            self.scaler_a.scale(loss).backward()
            self.scaler_a.unscale_(self.opt_a)
            torch.nn.utils.clip_grad_norm_(
                list(self.agent_a.policy.parameters()) + list(self.agent_a.model.parameters()),
                1.0,
            )
            self.scaler_a.step(self.opt_a)
            self.scaler_a.update()
            torch.cuda.empty_cache()

    def _update_agent_b_batch(self):
        """One PPO update for B on a batch of episodes.
        Single-episode updates with a 0/1 reward are very noisy; with no signal they
        drift into a one-class collapse. Advantages are normalised across the batch."""
        buf, self._b_buffer = self._b_buffer, []
        dev = self.device_b

        def norm(r):
            r = torch.tensor(r, dtype=torch.float32, device=dev)
            # std floored so near-constant rewards (e.g. IoU all ~0) don't get amplified
            return (r - r.mean()) / r.std(unbiased=False).clamp(min=0.1)

        adv_type = norm([b[4] for b in buf])
        adv_span = norm([b[5] for b in buf])
        entropy_coef = self.config.get("entropy_coef_b", 0.01)
        head_params = self.agent_b.trainable_parameters()   # heads (+ encoder if unfrozen)

        for _ in range(self.config["ppo_epochs"]):
            with torch.amp.autocast(device_type='cuda'):
                losses, ents = [], []
                for k, (text, p_type, p_span, old_lp, _, _) in enumerate(buf):
                    new_lp = self.agent_b.evaluate(text, p_type, p_span)
                    losses.append(self.agent_b.ppo_loss(old_lp, new_lp, adv_type[k], adv_span[k],
                                                        entropy_coef))
                    ents.append(float(new_lp["type_entropy"]))
                loss = torch.stack(losses).mean()
            self.opt_b.zero_grad()
            self.scaler_b.scale(loss).backward()
            self.scaler_b.unscale_(self.opt_b)
            grad_norm = torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            self.scaler_b.step(self.opt_b)
            self.scaler_b.update()
            torch.cuda.empty_cache()

        self.b_diag = {"b_grad_norm": float(grad_norm),
                       "b_type_entropy": sum(ents) / len(ents),
                       "b_adv_type": float(adv_type.abs().mean()),
                       "b_adv_span": float(adv_span.abs().mean())}

    # ── Logging & checkpointing ───────────────────────────────────────────

    def _log(self, episode: int, info: dict):
        row = {"episode": episode, "braked": False, "position_frac": info.get("position_frac"), **info}
        row.update(self.b_diag)
        row["type_acc_win"] = sum(self.type_hist) / max(1, len(self.type_hist))
        self.log_rows.append(row)
        pd.DataFrame(self.log_rows).to_csv(self.config["log_path"], index=False)

        print(
            f"[Ep {episode:>6}] "
            f"IoU={info['iou']:.3f} | "
            f"TypeCorrect={info['type_correct']} | "
            f"PPL={info['perplexity']:.1f} | "
            f"Uniq={info.get('uniq_ratio', float('nan')):.2f} | "
            f"R_A={info['reward_a']:+.3f} | "
            f"R_B={info['reward_b']:+.3f} | "
            f"Strategy={info['chosen_type']} | "
            f"TypeAcc50={row['type_acc_win']:.2f} | "
            f"B_H={self.b_diag.get('b_type_entropy', float('nan')):.2f}"
        )

    def _save(self, episode: int):
        for f in glob.glob(os.path.join(self.config["checkpoint_dir"], "ep*.pt")):
            os.remove(f)

        path = os.path.join(self.config["checkpoint_dir"], f"ep{episode}.pt")
        payload = {
            "episode"        : episode,
            "agent_a_policy" : self.agent_a.policy.state_dict(),
            "agent_b_heads"  : {
                "type"  : self.agent_b.type_head.state_dict(),
                "start" : self.agent_b.span_start_head.state_dict(),
                "end"   : self.agent_b.span_end_head.state_dict(),
            },
            "opt_a"          : self.opt_a.state_dict(),
            "opt_b"          : self.opt_b.state_dict(),
            "strategy_history": self.env.strategy_history,
            "position_history": self.env.position_history,
            "log_rows"       : self.log_rows,
        }
        if self.agent_b.train_encoder:
            payload["agent_b_encoder"] = self.agent_b.encoder.state_dict()
        # Backbone only when it is being trained (frozen = unchanged pretrained GPT-2)
        if not self.frozen_a:
            payload["agent_a_model"] = self.agent_a.model.state_dict()
        torch.save(payload, path)
        print(f"Checkpoint saved → {path}")

    def _save_logs(self):
        df = pd.DataFrame(self.log_rows)
        df.to_csv(self.config["log_path"], index=False)
        print(f"Training log saved → {self.config['log_path']}")


if __name__ == "__main__":
    trainer = Trainer(CONFIG)
    trainer.run()