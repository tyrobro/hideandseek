# Run 04 — Hide&Seek Training Run

**Status:** Complete (10,000 episodes)  
**Date:** October 2026  
**Key change from Run 02:** Fixed entropy_loss tensor shape bug (`.unsqueeze(0)`); entropy weight raised to 0.15; mean pooling on Agent B type head; type loss weighted 3×.

---

## Config

| Parameter | Value |
|---|---|
| `base_model` | gpt2 |
| `detector_model` | distilbert-base-uncased |
| `max_gen_length` | 80 |
| `min_gen_length` | 40 |
| `ppo_epochs` | 2 |
| `lr_agent_a` | 1e-5 |
| `lr_agent_b` | 1e-4 |
| `alpha` (IoU weight) | 1.0 |
| `beta` (type weight) | 0.5 |
| `gamma_fluency` | 3.0 |
| `eta_novelty` | 0.2 |
| `ppl_threshold` | 20.0 |
| `ppl_brake` | 100.0 |
| `reward_clip_a` | [-3.0, 1.0] |
| `reward_clip_b` | [-1.0, 2.0] |
| `num_episodes` | 10,000 |

---

## Results Summary

| Metric | Value |
|---|---|
| Final PPL (ep 10000) | ~147 |
| Mean PPL (all episodes) | ~100 |
| Agent A positive reward rate | 3.5% (7/200 logged) |
| Agent A mean reward | -4.33 |
| Type classification accuracy | 20.5% |
| Mean IoU | ~0.13 |
| Agent A strategy distribution | token 32%, positional 28%, semantic 23%, syntactic 18% |

---

## What Worked

- Training ran to completion without crashes (entropy shape bug fixed).
- Agent B type predictions are **distributed across all 4 classes** — no collapse to a single class (entropy bonus working as intended).
- Agent B type accuracy (20.5%) is above random chance (25% would be random), though only marginally so.
- Early episodes (0–700) showed healthy PPL in range 5–25 with Agent A occasionally getting positive reward.

---

## What Failed

### 1. PPL explosion — the core problem

PPL crossed the `ppl_brake` threshold of 100 for the first time at episode ~2500 and **never recovered**, sitting 100–270 for the second half of training. The PPL-aware intensity scaling introduced here was supposed to prevent this with a negative feedback loop, but it was **not effective** for two reasons:

- **The intensity scale only affects watermark strength**, not the underlying cause of PPL growth. Agent A's policy head itself shifts toward token distributions that produce incoherent text, and reducing watermark intensity doesn't undo that.
- **The PPL brake (skip update if PPL > 100) made things worse.** When PPL is high, Agent A gets *no gradient update at all*. By episode 5000, 50% of episodes were above the brake threshold — meaning A was getting no learning signal for half its episodes, while the bad policy was still generating high-PPL text. This is a **dead zone**: high PPL → skip update → no correction → PPL stays high.

### 2. Agent A never learns to hide

Agent A received a positive reward in only 7 of 200 logged episodes (3.5%), all in the first 550 episodes while PPL was still low. After PPL crosses ~30, the `gamma_fluency=3.0` penalty dominates: even with `iou=0` and wrong type prediction (perfect stealth), `r_a = -3.0 * max(0, (ppl-20)/20)` overwhelms the tiny novelty bonus (max 0.038).

**Root cause:** The reward function makes PPL the dominant term. Agent A cannot trade off watermark strength against detectability — it is always punished for PPL regardless of how well it hides.

### 3. Agent B span prediction is not learning

Even when type is correct (20.5% of episodes), mean IoU is only 0.13. Agent B is predicting random-looking spans that occasionally overlap by chance. The span prediction heads (start/end) need stronger supervision or a pretraining phase.

---

## Key Diagnostic Numbers

```
PPL progression:
  ep 50–700:   5–25   (healthy)
  ep 750:      52     (first spike)
  ep 2000:     74     (above comfortable range)
  ep 2500+:    100+   (brake triggering)
  ep 5400:     205    (worst spike)
  ep 10000:    147    (end of run)

Episodes where PPL > 100 (A update skipped): 101/200 = 50.5%
Agent B pred distribution: syntactic 30%, positional 28%, token 24%, semantic 19%
True type distribution:    token 32%, positional 28%, semantic 23%, syntactic 18%
```

---

## Planned Changes for Run 05

Based on this analysis, Run 05 addresses the reward design and brake logic:

1. **Remove the PPL brake** — replace with a soft PPL penalty that always provides gradient signal, so Agent A always learns. A hard skip is counterproductive.
2. **Restructure Agent A's reward** — decouple PPL penalty from the stealth reward. Agent A should get a clear positive signal for fooling Agent B *independently* of fluency. Fluency is a constraint, not the objective.
3. **Increase `ppl_threshold`** — raising from 20 to 40 to match GPT-2's natural distribution on WikiText (~25–35 on random completions). A threshold of 20 is too tight for GPT-2.
4. **Reduce `gamma_fluency`** — from 3.0 to 1.0 so the PPL penalty doesn't dominate the reward.
5. **Pre-train Agent B span heads** on a few hundred labelled examples before the adversarial loop begins.
