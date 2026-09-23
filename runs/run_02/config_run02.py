# config.py
# Central configuration for Hide&Seek
# Override specific keys in the notebook for Colab/Kaggle constraints

CONFIG = {
    # ── Models ──────────────────────────────────────────────────────────
    "base_model"      : "gpt2",                    # Agent A backbone
    "detector_model"  : "distilbert-base-uncased", # Agent B backbone
    # On free Colab swap base_model → "distilgpt2" to save VRAM

    # ── Watermark strategies ─────────────────────────────────────────────
    "watermark_types"     : ["token", "syntactic", "semantic", "positional"],
    "num_watermark_types" : 4,

    # ── Text generation ──────────────────────────────────────────────────
    "max_gen_length" : 80,
    "min_gen_length" : 40,

    # ── RL / PPO ─────────────────────────────────────────────────────────
    "ppo_epochs"   : 2,
    "batch_size"   : 8,      # drop to 4 on free Colab
    "gamma"        : 0.99,
    "clip_epsilon" : 0.2,
    "lr_agent_a"   : 1e-5,
    "lr_agent_b"   : 1e-4,

    # ── Reward weights ───────────────────────────────────────────────────
    "alpha"         : 1.0,   # IoU weight
    "beta"          : 0.5,   # type classification weight
    "gamma_fluency" : 1.5,   # fluency penalty weight (Agent A)
    "eta_novelty"   : 0.2,   # novelty bonus weight  (Agent A)

    # ── Fluency ──────────────────────────────────────────────────────────
    "ppl_threshold" : 30.0,  # perplexity above this → penalty

    # ── Reward clipping ──────────────────────────────────────────────────
    "reward_clip_a_min" : -5.0,
    "reward_clip_a_max" :  2.0,
    "reward_clip_b_min" : -2.0,
    "reward_clip_b_max" :  2.0,

    # ── PPL emergency brake ──────────────────────────────────────────────
    "ppl_brake" : 500.0,       # skip PPO update if PPL exceeds this


    # ── Training loop ────────────────────────────────────────────────────
    "num_episodes" : 10000,
    "log_every"    : 100,
    "save_every"   : 500,
    "checkpoint_dir": "./checkpoints",

    # ── Novelty tracking ─────────────────────────────────────────────────
    "novelty_window": 20,    # how many past episodes to look back
}