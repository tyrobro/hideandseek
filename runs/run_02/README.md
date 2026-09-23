# Run 02 — Fluency Stabilisation

## Changes from Run 01
- `gamma_fluency`: 0.3 → 1.5 (stronger fluency penalty)
- `ppl_threshold`: 50.0 → 30.0 (tighter acceptable PPL range)
- `ppo_epochs`: 4 → 2 (reduce overfitting per episode)
- `ppl_brake`: added at 500.0 (skip PPO if PPL exceeds threshold)
- Reward clipping added: R_A ∈ [-5, +2], R_B ∈ [-2, +2]
- Logit bias hard caps added per strategy:
  - token: max intensity 1.5
  - syntactic: max intensity 1.0 (was biggest PPL offender in Run 01)
  - semantic: max intensity 1.0
  - positional: max intensity 1.0, base bias reduced 3.0 → 1.5
- `max_gen_length`: 80, `min_gen_length`: 40 (unchanged from Run 01)

## Config
- base_model: gpt2
- detector_model: distilbert-base-uncased
- max_gen_length: 80
- min_gen_length: 40
- ppo_epochs: 2
- gamma_fluency: 1.5
- ppl_threshold: 30.0
- ppl_brake: 500.0
- reward_clip_a: [-5.0, +2.0]
- reward_clip_b: [-2.0, +2.0]
- episodes: 10,000

## Results

### Strategy Distribution (Agent A)
| Strategy   | Frequency |
|------------|-----------|
| syntactic  | 0.307     |
| token      | 0.251     |
| semantic   | 0.236     |
| positional | 0.206     |

No mode collapse — all four strategies used. Novelty bonus working correctly.

### Key Metrics
- PPL: Started ~8 (ep50), climbed to ~300 peak, stabilised ~100-200 late training
- PPL improvement vs Run 01: ceiling dropped from ~7000 → ~300 (23x improvement)
- IoU peaks: 0.882 (ep1550), 0.764 (ep2400), 0.695 (ep8250)
- Agent A reward: stabilised at -5.0 floor (clipping working), occasional +0.0 when fooling B
- Agent B reward: oscillating 0.0–1.125, showing genuine detection learning

## What Went Right
- PPL explosion completely eliminated — reward clipping and logit caps worked
- Adversarial dynamic confirmed and active throughout all 10,000 episodes
- Strategy distribution healthy and diverse
- Reward clipping at -5.0 successfully prevented catastrophic policy updates

## What Went Wrong

### Critical: Agent B Type Prediction Collapse
- From episode ~2000 onwards, Agent B predicts "positional" for almost
  every episode regardless of what Agent A actually used
- Root cause: "positional" gives enough type_correct hits (~20% of time)
  to get partial reward without actually learning to classify
- No entropy regularisation on the type head allowed this collapse

### PPL Still Too High
- Average PPL in second half of training: ~100-200
- Target is hard 50
- Root cause: Agent A's policy head selecting intensity_idx=2 (value=2.0)
  too frequently — even with caps, this pushes logits hard enough to
  degrade fluency beyond threshold
- INTENSITY_VALUES = [0.5, 1.0, 2.0] — top value too aggressive

## What to Fix in Run 03
- Add entropy bonus to Agent B type head loss to prevent prediction collapse
- Reduce Agent A intensity range: [0.5, 1.0, 2.0] → [0.3, 0.6, 1.0]
- Increase gamma_fluency: 1.5 → 2.0
- Tighten ppl_threshold: 30.0 → 25.0
- Lower ppl_brake: 500.0 → 200.0 (brake much earlier)
- Add B prediction distribution to live logging for early collapse detection