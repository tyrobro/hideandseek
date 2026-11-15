# Run 12 — Forced-Syntactic Run with Agent B's Update Fixed

## Summary

Run 12 repeated the Run 11 setup (Agent A forced to `syntactic` on every episode, trainable backbone) after fixing Agent B's PPO update path. B now learns: type accuracy rises from ~55% in the first 1000 episodes to 90–100% from ep 5000, and the entropy of B's type head falls from 0.84 to about 0.2 (Run 11: 25.0% type accuracy, uniform predictions, no trend).

Two problems limit what the run can show:

1. **The result is not evidence of watermark detection.** With a constant label, type accuracy measures only whether B learns the prior. B's span predictions converge on a shortcut (start at index 0, end near the end of the text) that yields IoU ≈ 0.2 whatever the true watermark position.
2. **Agent A's backbone degenerated into repetition**, visible from ~ep 5000 and severe by ~ep 8400. The perplexity-based safeguards could not detect it because repeated text has low perplexity.

---

## Changes from Run 11

### Agent B update path (`agents/agent_b.py`, `train.py`)

Run 11's B update re-called `detect(text)`, which samples a **new** action, and passed the new log-probs as both old and new to `ppo_loss`. The reward earned by B's first sample was credited to an independent second sample, so the expected policy gradient was zero. In addition, the entropy bonus sat inside the `for key` loop of `ppo_loss`, so it was added three times; it was the only systematic gradient on B's type head and pushes the type distribution toward uniform, matching the ~25/25/25/25 prediction split seen in every earlier log.

| Change | Detail |
|---|---|
| New `AgentB.evaluate(text, pred_type, pred_span)` | log-prob of the action B actually took, under the current weights |
| `ppo_loss` rewritten | uses stored rollout log-probs (old) vs `evaluate` (new), takes an advantage, entropy applied once |
| Reward baseline | EMA, decay 0.98; `advantage = r_b − baseline` |
| Entropy bonus | `entropy_coef_b = 0.01` |
| `detect` under `torch.no_grad()`; encoder forward under `no_grad` | encoder is frozen (not in `opt_b`) |
| New log columns | `b_grad_norm`, `b_type_entropy`, `b_advantage`, `type_acc_win` (50-episode window) |

### Environment fixes (`environment/watermark_env.py`)

- Removed `self.strategy_history.append(true_span)`. The history interleaved strategy strings with span tuples, so the last 20 entries held at most 10 of any strategy: the 0.7 hard floor could never fire, and the novelty frequency capped at 0.5, making the bonus 0.0 at full overuse and never negative (observed: `novelty_bonus` minimum 0.0 in the Run 09, 10 and 11 logs).
- `compute_novelty_bonus` now returns the raw `1 − 2·frequency`; `eta_novelty` was previously applied twice (effective strength eta²).
- Novelty is skipped when `force_strategy` is set, so these fixes do not change Run 12's reward relative to Run 11.

Because the strategy is forced, neither the strategy floor nor the novelty bonus is exercised in this run. Their effect on strategy collapse remains untested.

### Unchanged from Run 11

`force_strategy="syntactic"`, trainable backbone with `fluency_loss_coef=0.1` and `kl_coef=0.05` (fluency/KL measured on Agent A's own generated text), corrective fluency/KL step on braked episodes, `gamma_fluency`, `ppl_brake`, `entropy_coef_position=0.0`, 10,000 episodes, `log_every=50`.

---

## Results

All statistics use the 200 logged rows (one episode every 50), in bins of 1000 episodes (20 rows each).

| Episodes | Type acc | Mean IoU | PPL median | Rows PPL < 8 | Rows pred start = 0 | B type entropy | Mean reward A |
|---|---|---|---|---|---|---|---|
| 1–1000 | 0.55 | 0.166 | 19.4 | 0% | 0% | 0.84 | −0.12 |
| 1001–2000 | 0.85 | 0.106 | 20.0 | 0% | 10% | 0.77 | −0.32 |
| 2001–3000 | 0.75 | 0.151 | 22.2 | 0% | 0% | 0.76 | −0.30 |
| 3001–4000 | 0.80 | 0.138 | 20.2 | 0% | 10% | 0.65 | −0.37 |
| 4001–5000 | 0.85 | 0.248 | 21.7 | 0% | 40% | 0.49 | −0.68 |
| 5001–6000 | 0.95 | 0.268 | 16.2 | 35% | 75% | 0.39 | −0.75 |
| 6001–7000 | 0.85 | 0.225 | 17.6 | 10% | 100% | 0.46 | −0.63 |
| 7001–8000 | 1.00 | 0.254 | 16.7 | 15% | 100% | 0.36 | −0.81 |
| 8001–9000 | 1.00 | 0.010 | 6.0 | 55% | 100% | 0.31 | −0.32 |
| 9001–10000 | 0.90 | 0.059 | 14.8 | 30% | 95% | 0.21 | −0.24 |

Overall type accuracy 85%; `type_acc_win` reached ≥ 0.9 at ep 300 and was 0.98 at ep 10000.

### Agent B

- **Type head learned the constant label.** Accuracy and entropy move monotonically toward "always syntactic". This is the expected behaviour of a working update on a constant label and says nothing about detecting watermark content.
- **Span head converged on a shortcut.** The predicted span starts at index 0 in 100% of sampled rows from ep 6000 to 9000, and the end sits near the end of the text, regardless of where the watermark is. A span of (0, ~80) against a 16-token true span gives IoU ≈ 16/80 = 0.2 for any position. Example, ep 7650: true (60, 76), predicted (0, 80), IoU 0.20, reward B 0.70.
- **IoU rise is not localisation.** Mean IoU rises from ~0.14 to ~0.25 between ep 3000 and 8000, in step with the shift to start = 0; 45% of true spans are the early (8, 24) window, which also favours starting at 0.
- **Span collapse after ep 8000.** Predicted spans become tiny, e.g. (0, 1), IoU falls to ~0.01 and B's reward falls to the type-only 0.3, while type accuracy stays at 100%.

### Agent A

- Perplexity is stable (~16–22 median) until ep 5000. No row before ep 5000 has PPL < 8; the first appears at ep 5150. 18 sampled rows have PPL < 5 (ep 5150, 5200, 5450, 6000, 6100, 7050, 8400, 8650–9050, 9200, 9250, 9400), 10 of them consecutive from 8650.
- B's predicted span ends reach 205–228 in 9450–10000 (previously ≤ 127). A DistilBERT sequence of 200+ wordpieces from an 80-token generation implies a repeated multi-wordpiece token.
- A sample from the final model (`generate("The history of the")`) is the single token " practition" repeated almost throughout. The tracker's last ten perplexities were 16–68, so this output is not flagged by the perplexity threshold.
- Mean reward A is about −0.7 in 4000–8000. In Run 11, where B was at chance, it averaged +0.47.

The episode texts themselves were not logged; apart from the final sample, repetition at earlier episodes is inferred from the proxies above.

---

## Interpretation

**Established**
- The B update fix works. In the same forced setting, B's type accuracy went from 25.0% (Run 11) to 85–100%, with falling entropy and non-zero gradients.
- Every earlier run (Runs 01–11) therefore used an untrained Agent B; their results describe Agent A against a random detector.
- Agent A's backbone collapsed into repetition in this run, and the perplexity and KL safeguards did not detect it.

**Not established**
- Whether B can detect syntactic (or any) watermark content. A constant label cannot test it, and the span head's gains are explained by a position/full-text shortcut.
- The cause of the collapse. Candidate explanations, all untested:
  1. The fluency loss is mean NLL on A's own generations, which repetition minimises.
  2. The KL term is measured on those same generations; the reference model also predicts repeated tokens once a pattern is established, and the per-token mean dilutes the cost of starting the repetition.
  3. Once B learns, A's rewards turn consistently negative (no baseline), and that gradient flows into the trainable backbone through the prompt hidden states.
  4. Possible reset of the backbone at the resume (see Run History).
- Run 11 was stable while B was at chance and A's rewards were positive, which fits explanation 3, but the runs also differ in other ways.

---

## Revisions to Earlier Interpretations

These supersede explanations written in the Run 09 and Run 10 READMEs, which are left as written.

- "Syntactic watermarks are hard for DistilBERT / degenerate Nash equilibrium": unsupported. B never learned in those runs (type accuracy ≈ 25% in Runs 09–11 regardless of strategy mix), so no inference about B's ability on syntactic watermarks can be drawn.
- "The novelty penalty was silent because the resume block left a stale history reference": incorrect. Run 10 started fresh and showed the same zeros. The cause is the polluted `strategy_history` and the doubled `eta` described above.
- The regression from Run 07 (57% syntactic, hard floor working) to Runs 09–10 (87% / 97.5%) is consistent with the history-pollution bug disabling the floor, but this has not been tested; Runs 11 and 12 force the strategy.

---

## Limitations

- 200 sampled single episodes, not window averages; one seed.
- Run interrupted and resumed once; effect on the backbone unresolved.
- Constant-label setting; no mixed-strategy test of detection yet.
- Span indices: `true_span` is in GPT-2 token space while B's span is in DistilBERT wordpiece space over the decoded text, so IoU is only approximately meaningful, especially late in the text.
- B's encoder is frozen; B is an MLP on frozen DistilBERT features.

---

## Next Steps (Run 13)

1. **Stabilise Agent A's backbone:** compute the fluency and KL terms on real WikiText text instead of A's own generations; lower the backbone learning rate (1e-6); log a repetition metric (unique-token ratio) and brake on it; save the backbone and optimizer in checkpoints.
2. **B-only detectability probe:** freeze Agent A and sample strategy, position and intensity uniformly at random, so B must distinguish four types (chance 25%) with no position prior. Compare per-type accuracy to chance and IoU to the trivial full-text baseline (≈ 0.2) and to a shuffled-prediction null.
3. Later: align `true_span` to DistilBERT token indices via character offsets, and test whether the strategy floor and novelty fixes prevent collapse in a free-play (unforced) run.