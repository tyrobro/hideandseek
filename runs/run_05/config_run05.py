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
    "lr_agent_b"   : 1e-4,
    "alpha"         : 1.0,
    "beta"          : 0.5,
    "gamma_fluency" : 0.3,   # was 3.0
    "eta_novelty"   : 0.2,
    "stealth_bonus" : 1.5,   # NEW
    "ppl_threshold" : 200.0, # was 20.0
    "reward_clip_a_min" : -3.0,
    "reward_clip_a_max" :  3.0,
    "reward_clip_b_min" : -2.0,
    "reward_clip_b_max" :  2.0,
    "ppl_brake"     : 500.0, # was 100.0
    "num_episodes"  : 10000,
    "log_every"     : 100,
    "save_every"    : 500,
    "checkpoint_dir": "runs/run_05/checkpoints",
    "log_path"      : "runs/run_05/results/training_log.csv",
    "plot_path"     : "runs/run_05/results/training_curves.png",
    "novelty_window": 20,
}