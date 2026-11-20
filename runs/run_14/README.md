# Run 14 — B-Only Detectability Probe (Random Strategy, A Frozen)

## Purpose

Runs 12–13 forced `syntactic`, so B's type accuracy only reflected the prior. Run 14 is the intended probe: strategy sampled uniformly, Agent A frozen, so any above-chance type accuracy would have to come from detecting the watermark.

## Configuration

| Setting | Value |
|---|---|
| `force_strategy` | None (key removed) |
| `random_policy` | True |
| `freeze_agent_a` | True (backbone and policy head grads off) |
| `ppl_brake` / `min_unique_ratio` | 1e9 / 0.0 (disabled) |
| B update | per-head unit rewards (type 0/1, span IoU), EMA baselines (0.98), `entropy_coef_b` 0.01, `lr_agent_b` 1e-4, `ppo_epochs` 2 <!-- VERIFY --> |
| Episodes / logging | 10,000 episodes, 400 logged rows (every 25th) <!-- VERIFY log_every, save_every --> |
| Fresh run directory | yes, no resume |

Changes since Run 13: forced strategy removed, config printed at launch, duplicate `type_hist.append` removed, real-text anchor for A's fluency/KL, separate backbone lr (1e-6), repetition brake, backbone/optimiser checkpointing. The A-side changes were inactive because A was frozen.

## Validity checks

- Strategy counts: token 112, positional 106, semantic 98, syntactic 84 (near uniform).
- A stable: unique-token ratio 0.71 in every quarter (min 0.34, none < 0.3); median PPL ~22 flat (max 137.8); intensity and position choices near uniform.
- No brakes, no resume.

## Results

### Type detection

| Episodes | Type acc | B type entropy (median) | Pred "semantic" share |
|---|---|---|---|
| 1–2.5k | 0.22 | 1.12 | 36% |
| 2.5–5k | 0.23 | 0.56 | 82% |
| 5–7.5k | 0.29 | 0.61 | 84% |
| 7.5–10k | 0.25 | 0.56 | 83% |

- Overall accuracy 24.8% (99/400) against 25% chance; binomial test p = 0.57. Trend over episodes: Spearman p = 0.64.
- Recall by true type: semantic 0.71, syntactic 0.13, positional 0.09, token 0.07 (macro recall 0.253).
- Confusion: B predicts "semantic" for 285 of 400 episodes. Semantic's recall reflects the collapse, not detection.

### Span detection

- Real mean IoU 0.172 against a shuffled-prediction null of 0.159 (95th percentile 0.171): at the null.
- Predicted span length grows from 20.5 to 41.0 tokens; IoU rises 0.155 → 0.192 by widening spans.
- IoU is ~0.10 at position 0.1 and ~0.21 at 0.45 / 0.75: span-geometry artefact, not localisation.
- `true_span` (GPT-2 token indices) and `pred_span` (DistilBERT wordpiece indices) are still not aligned.

## Conclusions

1. With a frozen DistilBERT encoder, mean-pooled features and per-episode REINFORCE updates, B does not detect watermark type above chance in 10,000 episodes.
2. B's entropy collapse onto one class is expected under noisy 0/1 rewards with no signal.
3. Span IoU is at the shuffled null.
4. This does not show the watermarks are undetectable in principle. Two explanations remain: the feature representation (frozen, mean-pooled) washes out the signal, or the RL signal is too noisy to learn from. They cannot be separated without a supervised test.
5. A is stable when frozen, which confirms the Run 13 drift came from training A's backbone.

## Next

- Offline supervised detectability probe (no RL): generate labelled samples with frozen A and train (a) frozen-mean-pool MLP, (b) fine-tuned DistilBERT, (c) a TF-IDF/n-gram baseline.
- Do not resume joint A-vs-B training until B beats chance in that probe.
- Align `true_span` to DistilBERT indices before trusting IoU.