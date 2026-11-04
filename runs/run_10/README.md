Markdown · run10_readme.md

# Run 10 — Resume Fix + Stronger Novelty Penalty

## Summary

Run 10 applies two targeted fixes from Run 09's post-mortem: rewiring `_strategy_history_ref` after checkpoint resume so the exploration floor and novelty penalty see the real history, and increasing `eta_novelty` from 0.2 to 0.5. Despite these fixes, syntactic dominance worsens to 97.5% (from 87.1% in Run 09), and IoU collapses to near zero earlier. The run confirms the system has reached a degenerate Nash equilibrium, not a training bug.

---

## Changes from Run 09

### 1. Resume Block Rewire (`train.py`)

In Run 09, the checkpoint resume block replaced `self.env.strategy_history` with a new list object from the checkpoint, but `self.agent_a._strategy_history_ref` still pointed to the old empty list created at `__init__`. Both the exploration floor and novelty penalty read from this stale reference, so they saw an empty history and never fired after any resume.

```python
# After restoring from checkpoint — added in Run 10:
self.env.strategy_history = ckpt.get("strategy_history", [])
self.env.position_history = ckpt.get("position_history", [])
self.agent_a._strategy_history_ref = self.env.strategy_history  # rewire ✓
self.agent_a._position_history_ref = self.env.position_history  # rewire ✓
```

### 2. Stronger Novelty Penalty (`config.py`)

```python
"eta_novelty": 0.5,   # was 0.2
```

At 100% syntactic frequency, penalty = `0.5 × (1 − 2×1.0)` = **−0.5** (was −0.2).

---

## Config (Run 10)

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
    "eta_novelty": 0.5,
    "stealth_bonus": 1.5,
    "ppl_threshold": 200.0,
    "reward_clip_a_min": -3.0,
    "reward_clip_a_max": 3.0,
    "reward_clip_b_min": -2.0,
    "reward_clip_b_max": 2.0,
    "ppl_brake": 500.0,
    "num_episodes": 10000,
    "log_every": 50,
    "save_every": 1000,
    "checkpoint_dir": "runs/run_10/checkpoints",
    "log_path": "runs/run_10/results/training_log.csv",
    "plot_path": "runs/run_10/results/training_curves.png",
    "novelty_window": 20,
    "entropy_coef_position": 0.0,
}
```

---

## Results

| Metric | Run 09 | Run 10 |
| --- | --- | --- |
| Episodes | 10,000 | 10,000 |
| Syntactic share | 87.1% | **97.5%** |
| Positional share | 4.8% | 1.2% |
| Token share | 3.2% | 0.6% |
| Semantic share | 4.8% | 0.6% |
| IoU collapse point | \~ep 6000 | \~ep 5000 |

Syntactic dominance worsened despite both fixes. IoU collapses to near zero \~1000 episodes earlier than Run 09, meaning Agent B stopped detecting anything sooner.

---

## Why the Fixes Didn't Help

The resume block rewire only matters when resuming mid-run. For a fresh run from ep 0, `_strategy_history_ref` is wired correctly in `__init__` from the start — the floor *was* active throughout. It still failed to prevent collapse.

The reward math explains why the floor is insufficient:

| Component | Value |
| --- | --- |
| Stealth bonus (syntactic fools B) | +1.5 |
| Novelty penalty at 100% syntactic | −0.5 |
| **Net reward** | **+1.0** |

The floor forces a random non-syntactic strategy when freq > 70%, but Agent A immediately snaps back to syntactic the next episode. The floor cannot make syntactic *suboptimal* — it can only briefly interrupt it.

---

## Research Finding: Degenerate Nash Equilibrium

Runs 09 and 10 together confirm the system has converged to a **degenerate Nash equilibrium**:

- Agent A plays syntactic almost exclusively
- Agent B cannot detect syntactic watermarks (IoU → 0)
- Neither agent benefits from deviating unilaterally

This is the expected outcome of a zero-sum game where one strategy dominates because the opponent has not learned to counter it. The system is behaving correctly as a game-theoretic system — Agent A found the dominant strategy and is exploiting it rationally.

### Why Can't Agent B Detect Syntactic Watermarks?

Two structural reasons:

**1. Data starvation.** At 97.5% syntactic frequency, Agent B sees token/semantic/positional watermarks in only \~2.5% of episodes. DistilBERT cannot learn span boundaries or type classification from this signal — gradient updates for non-syntactic strategies are drowned out by the dominant class. B never gets enough exposure to the other strategies to build a counter.

**2. Syntactic watermarks are architecturally harder for DistilBERT.** Syntactic transformations (POS-tag swaps, determiner changes) spread the signal across grammatical structure rather than substituting a discrete token. DistilBERT's SQuAD-style span prediction head is designed to find answer-like token spans, not diffuse grammatical perturbations. There may be no sharp boundary for the head to anchor on.

### What This Means for the Research

The collapse is not a failure — it is a finding. Agent A has identified that syntactic is the dominant strategy against DistilBERT-class detectors. The open question for Run 11 is *why*:

**Option A — Force diversity (engineering):** Use a balanced replay buffer, curriculum forcing, or type-conditioned rewards to ensure Agent B sees all four strategies equally. This prevents collapse but is engineering the outcome rather than observing it.

**Option B — Study the equilibrium (research):** Give Agent B a syntactic-only curriculum to test whether it *can* learn to detect syntactic watermarks when given sufficient signal, or whether DistilBERT is structurally incapable. This answers the more interesting research question.

---

## Retained from Run 09

- Fixed semantic synonym map (`{i: i + 10 for i in range(1000, 4000)}`)
- Reverted PPL thresholds (`ppl_threshold=200`, `ppl_brake=500`)
- Disabled entropy regularisation (`entropy_coef_position=0.0`)
- Position head with three position buckets `[0.1, 0.45, 0.75]`
- Position history floor logic