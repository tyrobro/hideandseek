# Run 01 — Baseline

## Config
- base_model: gpt2
- max_gen_length: 50
- ppo_epochs: 4
- gamma_fluency: 0.3
- ppl_threshold: 50.0
- reward clip: none
- episodes: 10,000

## Observations
- Strategy distribution healthy — no mode collapse (positional 30%, syntactic 27%, semantic 24%, token 18%)
- Adversarial dynamic confirmed — both agents responding to each other
- CRITICAL ISSUE: Perplexity exploded from ~1 at ep50 to ~7000 at ep6850
- Root cause: logit bias uncapped, fluency penalty too weak (0.3) to compete with IoU loss
- Agent A reward drifted to -43 at worst — policy destabilized by extreme PPL

## What to Fix in Run 02
- Cap logit bias in watermark strategies
- Increase gamma_fluency: 0.3 → 1.5
- Tighten ppl_threshold: 50 → 30
- Add emergency PPL brake (skip PPO if PPL > 500)
- Clip rewards to [-5, +2]
- Reduce ppo_epochs: 4 → 2