# train.py
# Main PPO training loop for Hide&Seek.
# Ties together the environment, Agent A, Agent B, and reward computation.

import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from rewards.reward_functions import compute_perplexity
from environment.watermark_env import WatermarkEnv
from agents.agent_a import AgentA
from agents.agent_b import AgentB
from config import CONFIG


class Trainer:
    def __init__(self, config: dict):
        self.config = config

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

        # ── Optimizers ────────────────────────────────────────────────────
        self.opt_a = torch.optim.Adam(
            list(self.agent_a.policy.parameters()) +
            list(self.agent_a.model.parameters()),
            lr=config["lr_agent_a"],
        )
        self.opt_b = torch.optim.Adam(
            list(self.agent_b.type_head.parameters()) +
            list(self.agent_b.span_start_head.parameters()) +
            list(self.agent_b.span_end_head.parameters()),
            lr=config["lr_agent_b"],
        )
# Mixed precision — cuts VRAM ~40-50% with negligible quality loss
        self.scaler_a = torch.cuda.amp.GradScaler()
        self.scaler_b = torch.cuda.amp.GradScaler()

        # ── Dataset ───────────────────────────────────────────────────────
        print("Loading dataset...")
        dataset      = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        self.prompts = [
            x["text"] for x in dataset
            if len(x["text"].strip()) > 100
        ]
        print(f"Loaded {len(self.prompts)} prompts.")

        self.log_rows = []
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
    # ── Main training loop ────────────────────────────────────────────────

    def run(self):
        print(f"\nStarting Hide&Seek training for {self.config['num_episodes']} episodes...\n")

        for episode in tqdm(range(1, self.config["num_episodes"] + 1)):
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

            # 4. Compute fluency (perplexity) of generated text
            perplexity = compute_perplexity(
                self.agent_a.model,
                self.agent_a.tokenizer,
                text,
                device=self.device_a,
            )

            # 5. Agent B detects watermark
            pred_type, pred_span, logprobs_b = self.agent_b.detect(text)

            # 6. Environment step → rewards
            r_a, r_b, info = self.env.step(
                watermarked_text = text,
                true_type        = chosen_type,
                true_span        = true_span,
                pred_type        = pred_type,
                pred_span        = pred_span,
                perplexity       = perplexity,
                chosen_type      = chosen_type,
            )

            # 7. PPO updates
            self._update_agent_a(prompt, logprobs_a, r_a)
            self._update_agent_b(pred_type, pred_span, text, r_b)

            # 8. Logging
            if episode % self.config["log_every"] == 0:
                self._log(episode, info)

            # 9. Checkpoint
            if episode % self.config["save_every"] == 0:
                self._save(episode)

        print("\nTraining complete.")
        self._save_logs()

    # ── PPO update helpers ────────────────────────────────────────────────

    def _update_agent_a(self, prompt: str, old_logprobs: dict, reward: float):
        reward_tensor  = torch.tensor(reward, dtype=torch.float32).to(self.device_a)
        action_indices = old_logprobs["action_indices"]

        for _ in range(self.config["ppo_epochs"]):
            with torch.amp.autocast(device_type = 'cuda'):
                fresh_head_out = self.agent_a.recompute_strategy_logprobs(prompt)
                loss = self.agent_a.ppo_loss(
                    old_logprobs   = old_logprobs,
                    fresh_head_out = fresh_head_out,
                    action_indices = action_indices,
                    reward         = reward_tensor,
                )
            self.opt_a.zero_grad()
            self.scaler_a.scale(loss).backward()
            self.scaler_a.unscale_(self.opt_a)
            torch.nn.utils.clip_grad_norm_(
                list(self.agent_a.policy.parameters()) +
                list(self.agent_a.model.parameters()),
                1.0,
            )
            self.scaler_a.step(self.opt_a)
            self.scaler_a.update()
            torch.cuda.empty_cache()

    def _update_agent_b(self, pred_type, pred_span, text, reward: float):
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device_b)

        for _ in range(self.config["ppo_epochs"]):
            with torch.amp.autocast(device_type = 'cuda'):
                _, _, new_logprobs = self.agent_b.detect(text)
                loss = self.agent_b.ppo_loss(new_logprobs, new_logprobs, reward_tensor)
            self.opt_b.zero_grad()
            self.scaler_b.scale(loss).backward()
            self.scaler_b.unscale_(self.opt_b)
            torch.nn.utils.clip_grad_norm_(
                list(self.agent_b.type_head.parameters()) +
                list(self.agent_b.span_start_head.parameters()) +
                list(self.agent_b.span_end_head.parameters()),
                1.0,
            )
            self.scaler_b.step(self.opt_b)
            self.scaler_b.update()
            torch.cuda.empty_cache()

    # ── Logging & checkpointing ───────────────────────────────────────────

    def _log(self, episode: int, info: dict):
        row = {"episode": episode, **info}
        self.log_rows.append(row)
        print(
            f"[Ep {episode:>6}] "
            f"IoU={info['iou']:.3f} | "
            f"TypeCorrect={info['type_correct']} | "
            f"PPL={info['perplexity']:.1f} | "
            f"R_A={info['reward_a']:+.3f} | "
            f"R_B={info['reward_b']:+.3f} | "
            f"Strategy={info['chosen_type']}"
        )

    def _save(self, episode: int):
        path = os.path.join(self.config["checkpoint_dir"], f"ep{episode}.pt")
        torch.save({
            "episode"       : episode,
            "agent_a_policy": self.agent_a.policy.state_dict(),
            "agent_b_heads" : {
                "type" : self.agent_b.type_head.state_dict(),
                "start": self.agent_b.span_start_head.state_dict(),
                "end"  : self.agent_b.span_end_head.state_dict(),
            },
            "opt_a"         : self.opt_a.state_dict(),
            "opt_b"         : self.opt_b.state_dict(),
        }, path)
        print(f"Checkpoint saved → {path}")

    def _save_logs(self):
        df = pd.DataFrame(self.log_rows)
        df.to_csv("training_log.csv", index=False)
        print("Training log saved → training_log.csv")


if __name__ == "__main__":
    trainer = Trainer(CONFIG)
    trainer.run()