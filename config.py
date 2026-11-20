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
    "ppl_brake"     : 250.0,
    "num_episodes"  : 10000,   # was 10000 — resume continues ep10001→20000
    "log_every"     : 50,
    "save_every"    : 500,
    "checkpoint_dir": "runs/run_15/checkpoints",   # was run_05
    "log_path"      : "runs/run_15/results/training_log.csv",   # was run_05
    "plot_path"     : "runs/run_15/results/training_curves.png",   # was run_05
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
    "lr_backbone_a": 1e-6, 
    "min_unique_ratio": 0.0,
    "b_mode": "model",            # "random" = random guesser instead of Agent B (A-side experiments)
    "legacy_history_bug": False,  # True reproduces the Run 08-10 strategy_history pollution
}