# Run 13 — B-Only Detectability Probe (Did Not Run as Designed)

## Intent

Run 12 fixed Agent B's update path (B's type accuracy went from 25% to 85–100%), but with `force_strategy="syntactic"` the label was constant, so that accuracy measured the prior, not detection. Run 13 was meant to test whether B can detect watermarks at all:

- Strategy sampled uniformly at random (`random_policy=True`, `force_strategy=None`)
- Agent A frozen (`freeze_agent_a=True`)
- PPL brake disabled (`ppl_brake=1e9`)
- B trained with per-head unit rewards (type: 0/1, span: IoU), EMA baselines (decay 0.98), entropy bonus 0.01

## What Actually Ran

The probe configuration did not take effect. Evidence from the logs:

| Intended | Observed |
|---|---|
| Random strategy | `chosen_type = syntactic` in all 200 logged rows |
| A frozen | Position choice and intensity shifted over training; A's text quality drifted (see below) |
| Random policy | Intensity and position were not uniform later in the run |

Most likely cause: the notebook override was not applied (or `config.py` was loaded after it), so `force_strategy="syntactic"` stayed on and `freeze_agent_a` / `random_policy` were not set.

Run facts: 10,000 episodes, no resume, no brake, log every 50 episodes (200 rows), checkpoint every 500, about 1.3–1.4 s/it, 3h46m total.

Because the strategy was constant, **this run is a repeat of Run 12's setting with B's fixed update path, not a detectability probe.**

## Results

### Agent B: constant-label learning, then a span shortcut

| Episodes | Type acc | Median B type entropy | Mean IoU | B span start == 0 |
|---|---|---|---|---|
| 1–2.5k | 0.98 | 0.11 | 0.164 | 4% |
| 2.5–5k | 0.96 | 0.15 | 0.160 | 4% |
| 5–7.5k | 0.96 | 0.08 | 0.189 | 16% |
| 7.5–10k | 0.98 | 0.13 | 0.237 | 86% |

- TypeAcc50 reached 0.88 by episode 100 and 1.0 by episode 150 (Run 12 only reached 0.8 around episode 250). The fixed update path works and learns faster.
- With a constant label, 96–98% accuracy only shows B learned the prior. It says nothing about detecting the watermark.
- IoU rises from 0.16 to 0.24 not through localisation but because B converges on a span starting at token 0 (86% of predictions in the last quarter). That reproduces the shortcut seen in Run 12 (a near-full-text span gives IoU of roughly 0.2 regardless of the true position).
- B's type entropy fell to about 0.001 around episode 7000.
- The four type misses (episodes 4400, 4450, 5200, 6350) coincided with large gradient norms (6–12) and A's +1.5 stealth bonus.

### Agent A: partial backbone drift toward repetition

| Episodes | Mean unique-token ratio | Median PPL |
|---|---|---|
| 1–2.5k | 0.71 | 21 |
| 2.5–5k | 0.73 | 23 |
| 5–7.5k | 0.35 | 33 |
| 7.5–10k | 0.35 | 24 |

- The unique-token ratio halves around episode 5000, while median PPL looks normal. PPL hides repetition.
- Near-degenerate samples appeared: episode 5350 (unique ratio 0.075, PPL 3.5) and episode 6800 (unique ratio 0.14, PPL 3.2). PPL spikes also occurred (up to 168 at episode 5600).
- The final generated sample was coherent English, so the drift was partial, not a full collapse as in Run 12.
- Policy head behaviour: position choice moved from roughly uniform to 0.1 about 73% of the time after episode 5000; intensity moved from mostly 0 early to mostly 1–2 later.
- Overall PPL range in the console log: min 3.2, median 23.6, max 168.1.

## Conclusions

1. **No detectability information was obtained.** The strategy label was constant throughout.
2. **B's update path is confirmed working** (fast learning to the constant label, low entropy). The remaining failure is B's span shortcut, not its optimiser.
3. **A's backbone degrades when trained.** It is slower than in Run 12, but the unique-token ratio shows it, and PPL, NLL and KL on A's own samples do not.
4. **The experiment-launch process needs a check.** The overrides did not reach the run, so config values should be printed at the start of every run.

## Known Issues in This Run's Code

- `type_hist.append(...)` ran twice per episode, so `type_acc_win` covers about 25 episodes, not 50, and moves in steps of 0.04.
- `_save` does not store `agent_a.model`, so a resume would restore the policy head onto a fresh GPT-2.
- A's optimiser covers all of GPT-2 at lr 1e-5, which is likely too high for the backbone.

## Next

- Rerun the B-only probe with the configuration actually applied (print config values at launch; 200-episode smoke test first).
- Run 14: repetition brake on unique-token ratio, lower backbone lr, real-text anchor for fluency/KL, backbone and optimiser checkpointing, and alignment of `true_span` to DistilBERT indices.