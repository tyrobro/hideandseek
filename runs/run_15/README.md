# Run 15 — B-Only Probe with Fine-Tuned Encoder and Aligned Spans (Path 4)

## Purpose

Run 14 left B at chance (24.8%) with a frozen DistilBERT, mean-pooled features and per-episode updates. Run 15 tested whether a stronger B changes that, and applied three fixes:

- Fine-tuned DistilBERT (`unfreeze_encoder_b`), CLS + mean + max pooling (`b_pooling = cls_mean_max`)
- Batched B updates (`b_update_batch` = 8) with batch-normalised advantages
- True spans mapped into B's wordpiece index space via character offsets; episodes with no watermark in the text are skipped

Setup is otherwise Run 14's: random strategies, A frozen, no brake.

## Configuration

| Setting | Value |
|---|---|
| `random_policy` / `freeze_agent_a` | True / True |
| `ppl_brake` / `min_unique_ratio` | 1e9 / 0.0 |
| `unfreeze_encoder_b` / `lr_encoder_b` / `lr_agent_b` | True / 2e-5 / 3e-4 <!-- VERIFY --> |
| `b_pooling` / `b_update_batch` | cls_mean_max / 8 |
| `entropy_coef_b` | <!-- VERIFY: 0.01 or 0.05 (an earlier smoke-test printout showed 0.01) --> |
| Episodes / logging | 10,000 episodes, log every 25 (393 of 400 rows present; the rest are skipped episodes) <!-- VERIFY: n_no_wm --> |
| B attention / device | eager attention, B on cuda:1 with cuda:1 as the current device (after a CUDA illegal-memory-access crash in the first smoke test) |

## Validity checks

- Strategy counts: syntactic 110, semantic 104, token 100, positional 79 (no significant skew).
- A stable throughout: unique-token ratio 0.69–0.73, median PPL ~22, max 86.
- True spans now average 16.5 wordpiece tokens (the watermark window), in the same index space as B's predictions.

## Results

### Type detection

| Episodes | Type acc | Median B type entropy | Median grad norm (clip 1.0) |
|---|---|---|---|
| 1–2.5k | 0.18 | 0.029 | 3.9 |
| 2.5–5k | 0.20 | 0.010 | 5.3 |
| 5–7.5k | 0.21 | 0.001 | 6.8 |
| 7.5–10k | 0.23 | 0.000 | 5.7 |

- B predicted "positional" for 388 of 393 episodes (98.7%); its type entropy fell from 1.16 to ~0.01 within about 300 episodes.
- Overall accuracy 20.4%, which equals positional's share of the samples (79/393); recall: positional 1.00, semantic 0.00, syntactic 0.00, token 0.01. Macro recall 0.25. Accuracy is below chance (binomial test against 25%, p = 0.99); trend over episodes p = 0.22.

### Span detection

- Real IoU 0.248 against a shuffled-prediction null of 0.227 (95th percentile 0.236); the excess is geometry, not localisation:
  - Predicted spans average 38–44 tokens against a true length of 16.5.
  - IoU by true position: 0.047 (pos 0.1), 0.335 (0.45), 0.356 (0.75).
  - Mean predicted start moves only 31.0 → 33.6 as the true start moves 8.9 → 59.8.
- Centre-of-span Spearman correlation is 0.23 overall, but is as high for positional (0.31) and token (0.34) watermarks, which leave no signal in the text, so it cannot be detection.
- IoU is not comparable with Run 14 (misaligned indices).

## Conclusions

1. A stronger encoder and pooling did not help: B collapsed onto one class before any signal could be learned, so Run 15 says nothing about whether semantic or syntactic watermarks are detectable.
2. The collapse is faster than in Run 14 (≈300 episodes vs. ≈2,500), consistent with the higher head learning rate plus encoder fine-tuning on a noisy 0/1 reward.
3. Span prediction is a geometry shortcut (a long span over the middle of the text), not localisation.
4. The pattern across Runs 12–15 is that policy-gradient training on a type reward drives B to a single class, and the entropy bonus used is too weak to stop it.

## Next

- Retrospective on Runs 1–15: causes, evidence and lessons.
- Analysis of how the watermarks are generated and how B tries to locate them (see the Run 14 finding: positional is a constant logit shift and token uses an unkeyed random green list).