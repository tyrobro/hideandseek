CONFIG = {
    "base_model"      : "gpt2",
    "detector_model"  : "distilbert-base-uncased",
    "watermark_types"     : ["token", "syntactic", "semantic", "positional"],
    "num_watermark_types" : 4,
    "max_gen_length" : 80,
    "min_gen_length" : 40,
    "ppo_epochs"   : 2,
    "batch_size"   : 8,
    "gamma"        : 2.0,
    "clip_epsilon" : 0.2,
    "lr_agent_a"   : 1e-5,
    "lr_agent_b"   : 3e-4,   # B now updates once per b_update_batch episodes
    "alpha"         : 2.0,
    "beta"          : 0.3,
    "gamma_fluency" : 0.3,
    "eta_novelty"   : 0.5,
    "stealth_bonus" : 1.5,
    "ppl_threshold" : 200.0,
    "reward_clip_a_min" : -3.0,
    "reward_clip_a_max" :  3.0,
    "reward_clip_b_min" : -2.0,
    "reward_clip_b_max" :  2.0,
    "ppl_brake"     : 1e9,        # Run 16: backbone frozen -> a brake can only skip learning, never helps
    "num_episodes"  : 6000,    # 1500 warm-up + 4500 game episodes
    "log_every"     : 25,
    "save_every"    : 500,
    "checkpoint_dir": "runs/run_16/checkpoints",
    "log_path"      : "runs/run_16/results/training_log.csv",
    "plot_path"     : "runs/run_16/results/training_curves.png",
    "novelty_window": 20,
    "entropy_coef_position": 0.0,
    "fluency_loss_coef": 0.1,
    "kl_coef": 0.05,
    "entropy_coef_b": 0.05,
    "b_update_batch": 8,
    "unfreeze_encoder_b": True,
    "lr_encoder_b": 2e-5,
    "b_pooling": "cls_mean_max",
    "random_policy": False, 
    "freeze_agent_a": False,
    "lr_backbone_a": 0.0,      # unused while freeze_backbone_a is True
    "min_unique_ratio": 0.0,
    "b_mode": "model",            # "random" = random guesser instead of Agent B (A-side experiments)
    "legacy_history_bug": False,  # True reproduces the Run 08-10 strategy_history pollution

    # ── Run 16 (everything learned in Runs 01-15 + the Q1-Q5 experiments) ─────────────
    "wm_version"        : "v2",   # watermarks v2: fixed disjoint token sets (environment/watermark_v2.py)
    "wm_strength"       : 4.0,    # gate result: 0.69 4-class acc, PPL ratio <= 1.41 (1.0: 0.28, 2.0: 0.43, 3.0: 0.55 not decisive)
    "freeze_backbone_a" : True,   # Q3: unanchored backbone training -> word salad; policy head only (C1)
    "b_standardise"     : True,   # Q2: raw pooled features are near-identical; standardise (stats frozen after warm-up)
    "b_warmup_episodes" : 1500,   # A uniform-random + frozen, B supervised only
    "b_sup_coef"        : 1.0,    # supervised CE kept alongside PPO after warm-up
    "warmup_min_acc"    : 0.40,   # abort if B cannot beat this (chance 0.25) on the true labels
    "final_eval_episodes": 400,   # held-out evaluation at the end (A random, B greedy)
    "sample_every"      : 500,    # append a generated sample to samples.txt
}