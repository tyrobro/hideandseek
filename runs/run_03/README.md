# Run 03 — True Entropy Fix + Intensity Reduction

## Changes from Run 02
- Agent B: entropy bonus added to type head PPO loss (weight 0.05)
- Agent A: INTENSITY_VALUES reduced [0.5, 1.0, 2.0] → [0.3, 0.6, 1.0]
- gamma_fluency: 1.5 → 2.0
- ppl_threshold: 30.0 → 25.0
- ppl_brake: 500.0 → 200.0
- Note: Run 02 mislabelling fixed — this is the genuine first run with
  these changes applied correctly

## Config
- base_model: gpt2
- max_gen_length: 80, min_gen_length: 40
- ppo_epochs: 2
- gamma_fluency: 2.0
- ppl_threshold: 25.0
- ppl_brake: 200.0
- reward_clip_a: [-3.0, +1.0]
- reward_clip_b: [-1.0, +2.0]
- INTENSITY_VALUES: [0.3, 0.6, 1.0]
- episodes: 10,000

## Results

### Strategy Distribution (Agent A)
| Strategy   | Frequency |
|------------|-----------|
| token      | 0.335     |
| syntactic  | 0.310     |
| semantic   | 0.240     |
| positional | 0.115     |

### Key Metrics
- PPL ep1-1500: mostly under 50 — best early performance across all runs
- PPL ep1500+: climbs to 100-300, same explosion pattern as Run 02
- Agent B diversity ep1-1500: excellent — predicts all 4 types correctly
- Agent B ep2000+: collapses to "positional" again
- IoU peak: 0.714 (ep4650), 0.563 (ep6150)
- Agent A reward: hits -5.0 clip constantly from ep2350+

## Critical Finding
Early training (ep1-1500) looks exactly like the target behaviour:
- PPL under 50
- Agent B genuinely diverse
- Both agents actually fighting

Root cause of failure: PPL gradually climbs from ep750 onwards. Once PPL
consistently exceeds ~60-80, generated text becomes near-incoherent and
Agent B cannot extract meaningful signal from it, causing it to collapse
to the most statistically safe prediction ("positional"). The two failure
modes (PPL explosion + Agent B collapse) are the same bug — PPL causes B's
collapse, not two independent problems.

## What to Fix in Run 04
- The intensity policy needs a HARD constraint during generation, not just
  a cap on the values available. Agent A's policy learns to always pick
  intensity_idx=2 (1.0) which is still too high for many strategies.
- Add PPL-aware intensity scaling: if recent PPL trend is rising, force
  Agent A to use lower intensity regardless of what its policy selects.
- Strengthen entropy bonus: 0.05 → 0.15
- Change Agent B type head from [CLS] to mean pooling
- Weight type loss 3x over span loss in Agent B's PPO update
- Lower ppl_brake: 200 → 100
- Increase gamma_fluency: 2.0 → 3.0