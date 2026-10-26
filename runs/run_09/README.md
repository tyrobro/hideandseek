Markdown · run09_readme.md

# Run 09 — Fix Semantic PPL Bomb + Clean Retrain

## Summary

Run 09 fixes the catastrophic perplexity explosion introduced in Run 08 by replacing the broken semantic synonym map, reverts PPL thresholds to safer values, and retrains cleanly from episode 0 to 10,000. Position head diversity (introduced in Run 08) is retained. The run reveals a new bug: the novelty penalty is silent after \~ep 1950 due to a reference-vs-reassignment issue in the checkpoint resume block, allowing syntactic strategy to collapse to 87%.

---

## Changes from Run 08

### 1. Fixed Semantic Synonym Map (`agents/agent_a.py`)

**Root cause of Run 08 PPL catastrophe:** The placeholder synonym map used token IDs 0–49, which are special tokens (`[PAD]`, `[UNK]`, `[CLS]`, etc.). Substituting these into generated text corrupts the logit distribution, causing perplexity to spike to 10,000+.

```python
# Run 08 (BROKEN) — uses special token range:
self.semantic_synonym_map = {i: i+1 for i in range(50)}

# Run 09 (FIXED) — uses real word token range:
self.semantic_synonym_map = {i: i + 10 for i in range(1000, 4000)}
```

### 2. Reverted PPL Thresholds (`config.py`)

Tighter thresholds in Run 08 caused the PPL brake to fire immediately once semantic corrupted text. Reverted to more permissive values:

| Parameter | Run 08 | Run 09 |
| --- | --- | --- |
| `ppl_threshold` | 150.0 | 200.0 |
| `ppl_brake` | 300.0 | 500.0 |
| `gamma_fluency` | 0.5 | 0.3 |

### 3. Disabled Entropy Regularisation (`config.py`)

Position entropy bonus was pushing the policy toward semantic (highest-entropy strategy), compounding the PPL issue. Disabled for Run 09:

```python
"entropy_coef_position": 0.0,  # was 0.01
```

### 4. Clean Retrain from Episode 0

Run 09 first attempt had duplicate rows (ep 0–2000 appeared twice) because the resume block picked up a stale checkpoint from Run 08. The log was deleted and training restarted cleanly from ep 0 to 10,000.

---

## Config (Run 09)

```python
CONFIG = {
    "base_model": "gpt2",
    "detector_model": "distilbert-base-uncased",
    "watermark_types": ["token", "syntactic", "semantic", "positional"],
    "num_watermark_types": 4,
    "max_gen_length": 80,
    "min_gen_length": 40,
    "ppo_epochs": 2,
    "batch_size": 8,
    "gamma": 2.0,
    "clip_epsilon": 0.2,
    "lr_agent_a": 1e-5,
    "lr_agent_b": 1e-4,
    "alpha": 2.0,
    "beta": 0.3,
    "gamma_fluency": 0.3,
    "eta_novelty": 0.2,
    "stealth_bonus": 1.5,
    "ppl_threshold": 200.0,
    "reward_clip_a_min": -3.0,
    "reward_clip_a_max": 3.0,
    "reward_clip_b_min": -2.0,
    "reward_clip_b_max": 2.0,
    "ppl_brake": 500.0,
    "num_episodes": 10000,
    "log_every": 50,
    "save_every": 500,
    "checkpoint_dir": "runs/run_09/checkpoints",
    "log_path": "runs/run_09/results/training_log.csv",
    "plot_path": "runs/run_09/results/training_curves.png",
    "novelty_window": 20,
    "entropy_coef_position": 0.0,
}
```

---

## Results

| Metric | Value |
| --- | --- |
| Episodes | 10,000 |
| Final syntactic share | 87.1% |
| Final semantic share | 4.8% |
| Final positional share | 4.8% |
| Final token share | 3.2% |
| PPL (typical range) | 10–200 |
| PPL spikes | Occasional, up to \~2000 |
| Peak IoU | 0.69 |
| Stealth bonus fires | Frequently (+1.5) |

### Position Head Diversity

Unlike Run 08 where position was stuck at `(8, 24)`, Run 09 shows genuine diversity across all three position buckets:

- `position_frac = 0.1` (early span)
- `position_frac = 0.45` (mid span)
- `position_frac = 0.75` (late span)

The position floor (introduced in Run 08) is working correctly.

---

## Bugs Discovered

### Silent Novelty Penalty (Critical)

`novelty_bonus` is exactly `0.0` from episode \~1950 through episode 10,000. At 87% syntactic frequency, the correct value should be:

```
eta_novelty * (1 - 2 * frequency) = 0.2 * (1 - 2*0.87) = -0.148
```

**Root cause:** The checkpoint resume block reassigns `self.env.strategy_history` to a new list object loaded from the checkpoint file. However, `self.agent_a._strategy_history_ref` still points to the old empty list created at `__init__`. The floor and novelty penalty both read from this stale empty reference, so they see an empty history and never fire.

```python
# Resume block (BROKEN):
self.env.strategy_history = ckpt.get("strategy_history", [])  # new list object
# agent_a._strategy_history_ref → still points to old empty list ❌

# Fix (Run 10):
self.env.strategy_history = ckpt.get("strategy_history", [])
self.agent_a._strategy_history_ref = self.env.strategy_history  # rewire ✓
```

This bug means the hard exploration floor also never fires after a resume, allowing syntactic to collapse to 87% unchecked.

---

## Why Syntactic Dominates

Even without the novelty bug, syntactic has a genuine reward advantage:

1. **Low PPL naturally** — syntactic transformations (POS-tag swaps, determiners) preserve fluency better than token substitution or semantic replacements, so the PPL brake fires less often.
2. **DistilBERT struggles** — syntactic watermarks are subtle; Agent B's IoU on syntactic detections is lower than on token watermarks, so Agent A earns the stealth bonus (+1.5) more reliably with syntactic.
3. **Penalty too weak** — even if novelty worked, `eta_novelty=0.2` produces a maximum penalty of `-0.2` against a stealth bonus of `+1.5`. Net reward is still positive.

---

## What Run 10 Will Fix

1. **Rewire `_strategy_history_ref` in resume block** — floor and novelty penalty will now see actual history after checkpoint resume.
2. **Increase `eta_novelty` to 0.5** — penalty becomes `-0.5` at 100% syntactic, stronger signal against collapse.

---

## Retained from Run 08

- Position head with `WatermarkPolicyHead.POSITION_FRACTIONS = [0.1, 0.45, 0.75]`
- Position history tracking and floor logic in `select_strategy`
- `_position_history_ref` wired in `train.py __init__`
- `position_frac` saved to checkpoint and logged per episode