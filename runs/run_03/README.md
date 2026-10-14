# Run 03

## Background
Previous "Run 03" accidentally used Run 02 code due to Kaggle not pulling
latest git commits. This is the genuine first execution of Run 03 code.

## Changes from Run 02
- Agent B: entropy bonus added to type head PPO loss (weight 0.05)
- Agent A: INTENSITY_VALUES reduced [0.5, 1.0, 2.0] → [0.3, 0.6, 1.0]
- gamma_fluency: 1.5 → 2.0
- ppl_threshold: 30.0 → 25.0
- ppl_brake: 500.0 → 200.0
- Agent B type head: still [CLS] pooling (mean pooling is Run 04)
- PPL-aware intensity scaling: NOT present (Run 04 only)

## Config
- base_model: gpt2
- max_gen_length: 80, min_gen_length: 40
- ppo_epochs: 2
- gamma_fluency: 2.0
- ppl_threshold: 25.0
- ppl_brake: 200.0
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
- PPL ep1-800: consistently under 50 — best early performance yet
- PPL onset of explosion: ep950 (62.6) → gradual climb to ep1700 (126)
- Agent B diversity: genuine until ep~1900, then collapses to positional
- R_A clip (-5.0) hit from ep1700 onwards constantly
- IoU peaks: 0.714 (ep1300), 0.706 (ep1550), 0.695 (ep1800)

## What Went Right
- Entropy bonus confirmed working — Agent B diverse for first 1500+ episodes
- Reduced intensity values confirmed working — PPL healthy for first 800 episodes
- Both improvements together buy ~1500 clean episodes vs ~100 in Run 01

## Root Cause of Failure
Missing PPL-aware intensity scaling (get_ppl_intensity_scale).
Without it, Agent A's policy gradually learns to select intensity_idx=2
(value=1.0) more frequently. Even at capped 1.0, repeated application
across 80 tokens accumulates enough logit distortion to push PPL past 60.
Once PPL > 60, Agent B can no longer read meaningful signal from the text
and collapses to positional guessing. The two failure modes remain one bug.

## What Run 04 Adds
- get_ppl_intensity_scale: negative feedback loop — rising PPL automatically
  reduces intensity, should keep PPL stable for all 10,000 episodes
- Agent B mean pooling: full sequence context for type classification
- Agent B type loss weighted 3x over span loss
- Entropy bonus strengthened: 0.05 → 0.15
- PPL brake tightened: 200 → 100
- gamma_fluency: 2.0 → 3.0
- ppl_threshold: 25.0 → 20.0