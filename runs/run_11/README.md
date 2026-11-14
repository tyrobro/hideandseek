# Run 11 — Forced-Syntactic Probe with Fluency + KL Constraints

## Summary

Run 11 forced Agent A to play `syntactic` on every episode (`force_strategy="syntactic"`) to test whether Agent B can learn to detect syntactic watermarks when given 100% exposure. Agent A's backbone remains trainable, now constrained by an explicit fluency loss and a KL penalty to a frozen reference GPT-2.

**Fluency:** stable for all 10,000 episodes (mean PPL 23.6, max 91.7, 0/200 sampled episodes braked).

**Detection:** Agent B performed at chance throughout. Type accuracy was 25.0% even though the true type was always `syntactic` (a constant predictor would score 100%), and span IoU (0.113) was indistinguishable from a shuffled-prediction null (0.117). The run therefore does **not** test whether syntactic watermarks are hard to detect, because B showed no evidence of learning at all. The cause is unverified; the leading suspect is Agent B's PPO update path (see Next Steps).

---

## Run History

| Attempt | Outcome |
|---|---|
| 1 | PPL collapsed at ep ~3000. The brake fired on every episode from ep 3000 to 10000 with no recovery. Unusable. This attempt used stale settings (`ppl_brake=250`, `gamma_fluency=0.5`, `entropy_coef_position=0.01`) and had no fluency constraint on the trainable backbone. |
| 2 | Completed 10,000 episodes but produced an empty log and no checkpoints. Perplexities in the tracker were normal, yet no PPO update, logging or checkpoint code ran. Void. |
| 3 | This run. Clean start, 10,000 episodes, valid log. |

Attempt 1 changed several settings at once with the fluency constraints, so the improvement in attempt 3 cannot be attributed to a single change.

---

## Changes from Run 10

| Change | Value |
|---|---|
| `force_strategy` | `"syntactic"` |
| Backbone fluency loss (`compute_fluency_loss`, mean token NLL on generated text) | `fluency_loss_coef = 0.1` |
| KL penalty to frozen reference GPT-2 (`compute_kl_to_ref`, per-token KL(current ‖ ref)) | `kl_coef = 0.05` |
| Corrective fluency/KL step on braked episodes instead of skipping them | enabled |
| `entropy_coef_position` | `0.0` |
| `gamma_fluency` / `ppl_brake` | `0.3` / `500.0` |
| `eta_novelty` | `0.5` |
| `log_every` / `save_every` | `50` / `1000` |
| Output paths | `runs/run_11/...` |

The reference model is a frozen copy of the original GPT-2 loaded in `AgentA.__init__`. Agent A's optimizer still includes the backbone parameters.

---

## Results

All statistics are computed from the 200 logged rows (one episode every 50), not from all 10,000 episodes.

### Fluency

| Metric | Value |
|---|---|
| Braked rows | 0 / 200 (95% upper bound ≈ 1.5%) |
| PPL mean / median | 23.6 / 22.3 |
| PPL p95 / max / min | 36.8 / 91.7 / 8.1 |
| PPL trend vs episode | none (Spearman ρ = −0.005, p = 0.95) |

### Agent B

| Metric | Value |
|---|---|
| True type | `syntactic` in 200/200 rows |
| Type accuracy | **25.0%** (50/200); chance = 25%, constant-predictor baseline = 100% |
| Predicted type distribution | semantic 26%, syntactic 25%, positional 24.5%, token 24.5% |
| Mean IoU | 0.113 |
| IoU under shuffled predictions (null) | mean 0.117, 95th percentile 0.134 |
| Rows with IoU > 0 / IoU ≥ 0.5 | 36.5% / 5.5% |
| Predicted span: mean length / centre | 20.8 tokens (true = 16) / 49.5 ± 18.9 |
| Mean reward B | 0.301 |

| Training third | Type accuracy | Mean IoU |
|---|---|---|
| ep 1–3333 | 19.7% | 0.119 |
| ep 3334–6667 | 26.9% | 0.122 |
| ep 6668–10000 | 28.4% | 0.098 |

Neither trend is significant (type accuracy ρ = 0.086, p = 0.23; IoU ρ = −0.038, p = 0.60).

### Agent A

| Metric | Value |
|---|---|
| Mean reward A | 0.471 |
| Stealth bonus fired (reward ≥ 1.4) | 51% of sampled rows |
| Position `0.1` / `0.45` / `0.75` | 74.5% / 12.5% / 13.0% |

The position distribution sits close to the 70% hard-floor threshold, which is consistent with the floor capping early-position dominance, though this was not tested.

---

## Cross-Run Comparison of Agent B

| Run | True syntactic share (usable rows) | Type accuracy | Constant-predictor baseline |
|---|---|---|---|
| 09 | 87% | 25.8% | 87% |
| 10 | 98% | 24.5% | 98% |
| 11 | 100% | 25.0% | 100% |

Every other training log checked shows the same pattern: type accuracy near 25% with uniformly spread predictions, regardless of Agent A's strategy mix.

---

## Interpretation

**Established**
- The fluency + KL constraints (together with the config reverts) kept perplexity stable for 10,000 episodes in forced-syntactic mode.
- Agent B did not learn type or span detection in this run, and the same pattern appears in earlier runs.

**Not established**
- That syntactic watermarks are intrinsically hard for DistilBERT. With B at chance, this run cannot distinguish "hard to detect" from "B is not learning".
- The Run 09/10 explanations (degenerate equilibrium, data starvation, architectural limits) are not supported by this evidence. Those runs show the same chance-level B on every strategy mix, so Agent A's drift toward syntactic may reflect cost (e.g. fluency) rather than evasion of a competent detector. This is a hypothesis.
- Agent A's stealth rewards in this run are largely a by-product of a chance-level detector.

**Suspected cause (unverified).** In `train.py`, `_update_agent_b` re-runs `agent_b.detect(text)` and passes `new_logprobs` as both the old and new log-probs to `ppo_loss`. It ignores the action B actually took and the rollout log-probs. Depending on `agent_b.detect` and `ppo_loss`, this could credit rewards to a different sampled action than the one that earned them, or yield a PPO ratio identically 1 with no useful gradient. Confirmation requires inspecting `agents/agent_b.py`.

---

## Limitations

- 200 logged episodes (every 50th), each a single episode rather than a window average.
- Single seed.
- Attempt 1 differs from attempt 3 in several settings, so the cause of the PPL recovery cannot be isolated.

---

## Next Steps (Run 12)

1. Inspect `agent_b.detect` and `agent_b.ppo_loss`; store the sampled action and its rollout log-prob and use them in the update.
2. Log B diagnostics: type-head entropy, gradient norm of B's heads, and running type accuracy.
3. Sanity-check B in isolation before any adversarial run: with forced-syntactic text, B should reach near-100% type accuracy quickly.
4. Log windowed means (e.g. mean over each 50-episode window) instead of single episodes.
5. Only after B demonstrably learns, return to the strategy-diversity and detectability questions.