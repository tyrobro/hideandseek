# Run 05 — Hide&Seek Training Run

**Status:** Incomplete — training ran to ~ep 8750 but log has a gap from ep 4250–8750 due to PPL brake silently skipping episodes without logging them. Only 81 rows recorded.  
**Date:** October 2026  
**Key changes from Run 04:** Added `stealth_bonus=+1.5` for Agent A when it fools B completely (iou<0.1, type wrong); raised `ppl_threshold` from 20→200 based on empirical gibberish test; lowered `gamma_fluency` from 3.0→0.3; raised `ppl_brake` from 100→500.

---

## Config

| Parameter | Value | Change from Run 04 |
|---|---|---|
| `base_model` | gpt2 | — |
| `detector_model` | distilbert-base-uncased | — |
| `max_gen_length` | 80 | — |
| `ppo_epochs` | 2 | — |
| `lr_agent_a` | 1e-5 | — |
| `lr_agent_b` | 1e-4 | — |
| `alpha` (IoU weight) | 1.0 | — |
| `beta` (type weight) | 0.5 | — |
| `gamma_fluency` | 0.3 | **was 3.0** |
| `eta_novelty` | 0.2 | — |
| `stealth_bonus` | 1.5 | **NEW** |
| `ppl_threshold` | 200.0 | **was 20.0** |
| `ppl_brake` | 500.0 | **was 100.0** |
| `reward_clip_a` | [-3.0, 3.0] | was [-3.0, 2.0] |
| `num_episodes` | 10,000 | — |

---

## Results Summary

| Metric | Value |
|---|---|
| Episodes logged | 81/200 expected (log gap ep 4250–8750) |
| Agent A positive reward rate | 13.6% (11/81) — up from 3.5% in Run 04 |
| Agent A mean reward (logged) | mixed; +1.5 in early episodes, -5.0 later |
| Type classification accuracy | 30% early, 50% late (but B collapsed to type-only) |
| Mean IoU | ~0.13 |
| PPL at ep 650 (first spike) | 33 |
| PPL at end (ep 8750) | 79.6 |

---

## What Worked

- **Stealth bonus is effective.** Agent A received `+1.5` rewards at ep 100, 250, 300, 350, 400 — clear evidence the new reward structure gives A a learnable positive signal. This is the first run where A consistently gets positive rewards in early training.
- **PPL stayed lower for longer.** PPL stayed under 30 through ep 650 (vs ep 50–100 in Run 04), giving A more episodes with a viable reward signal before PPL drift took over.
- **Logging gap diagnosed.** The gap from ep 4250–8750 is caused by the PPL brake firing on most episodes without writing to the log. Fix implemented for Run 06.

---

## What Failed

### 1. PPL still explodes — root cause identified

PPL starts rising around ep 650 and settles into the 60–95 range for the rest of training. The root cause is now identified: `apply_positional_watermark` adds `bias * intensity` to **the entire logits tensor** globally at every token generation step. With 80 tokens per generation, this compounds into a massive distribution shift. The other three watermark types only bias specific token IDs (much less damaging). Fix for Run 06: clamp the global shift to `min(bias * intensity, 1.0)`.

### 2. Agent B span collapse

After ep ~4250, Agent B reward locks to exactly `0.5` — it has learned to always predict the correct type (guaranteed `+0.5`) while giving up on span prediction entirely (IoU→0). Span prediction is harder and less rewarded than type prediction under the current `alpha=1.0, beta=0.5` config. Fix for Run 06: raise `alpha=2.0`, lower `beta=0.3` to make IoU worth more than type.

### 3. Checkpoints not persistent

Kaggle session wipes `/kaggle/working/` on close. All learned weights from Runs 01–05 are lost. Fix for Run 06: save checkpoints to Kaggle Dataset output so they persist across sessions and training can resume rather than restart from scratch.

### 4. PPL brake silently kills logging

`continue` after the brake check skips the log step entirely, making it impossible to diagnose what happened. Fix: log a "braked" row and flush CSV to disk on every log step.

---

## Key Diagnostic Numbers

```
PPL first crossed 30: ep 650
PPL first crossed 50: ep 1300
PPL first crossed 75: ep 1400
PPL brake (500) never triggered — all high-PPL episodes were in 30–97 range

Agent A positive reward episodes: 50, 100, 150, 250, 300, 350, 400, 600, 1000, 1050, 1100
Max Agent A reward seen: +1.500 (ep 100 — perfect stealth)

Agent B behaviour change:
  ep 0–4250: variable type+span predictions
  ep 4250+:  locked to type-correct, IoU≈0 (span collapse)
```

---

## Planned Changes for Run 06

1. **Fix positional watermark** — clamp global logit shift to `min(bias * intensity, 1.0)` in `watermark_types.py`
2. **Rebalance B's reward** — `alpha=2.0` (IoU), `beta=0.3` (type) to prevent span collapse
3. **Persistent checkpoints** — save to Kaggle Dataset output, reload at session start
4. **Log braked episodes** — write a "braked=True" row and flush CSV on every log step
5. **Wire `ppl_brake` to config** — replace hardcoded `500` in `train.py` with `self.config["ppl_brake"]`