# Experiment Suite: Answers to the Five Open Questions (after Run 15)

This document records the controlled experiments run after Run 15 to answer the five open questions left by the Run 01–15 retrospective (`RETROSPECTIVE.md`). Nothing here changes `train.py`; all experiments live in `experiments/` and write to `data/` and `runs/exp_*/`. Earlier run READMEs are not rewritten with hindsight.

Each result is single-seed unless stated. Where a claim is not established, it says so.

## Summary

| # | Question | Status | One-line answer |
|---|---|---|---|
| Q1 | Can the semantic / syntactic watermarks be detected at all (supervised, no RL)? | Answered | No, not above chance as implemented. Only the semantic watermark is visible, and only to an oracle that knows the exact token set. |
| Q2 | Why does B collapse under random labels; does supervision or a smaller lr fix it? | Answered, with a caveat | RL on a signal-free reward over near-identical raw features collapses B to one class (~episode 500). Supervision and standardised features both prevent it. |
| Q3 | What caused the PPL explosions in Runs 02–10? | Partly answered | Training the backbone without an anchor degrades text into word salad; the anchor prevents it. The size of the old PPL explosions did not reproduce. |
| Q4 | Does the strategy floor + history fix prevent A's strategy collapse in an unforced run? | Answered | Yes. With the legacy history bug, A loses half its strategy repertoire; with the fix, all four are used. |
| Q5 | Does Run 01's PPL ≈ 1 at episode 50 indicate an early degenerate state? | Answered | No. It came from rare repetitive samples, not from short texts. |

Overall: the watermarks as implemented carry no signal B can learn, so B's collapse is RL noise. The A-side instabilities came from fixable bugs (history pollution, an unanchored backbone).

## Setup

- Dataset: labelled generations (token / syntactic / semantic / positional / none) produced with the real `WatermarkLogitsProcessor` through HF `generate` (`experiments/gen_dataset.py`). About 6100 rows were generated in total (3000 and 3041 per part, balanced classes). A generation cell was launched twice, which is harmless: different seed offsets.
- Watermark parameters as in the repo: window of 17 steps out of 80 generated tokens; intensities {0.3, 0.6, 1.0}.
- A-side runs use `experiments/run_cfg.py` with overrides, 3000 episodes each, `b_mode = "random"` (random type and span, no B update), no PPL brake, no checkpoints. Each condition ran in its own process.
- Kaggle free tier, 2×T4.

## Q1: Detectability of the watermarks

**Method.** Oracle statistics (AUROC with the true token sets), GBM and TF-IDF cross-validation, frozen-feature logistic probes, a fine-tuned DistilBERT, a planted-signal control and a shuffled-label control (`analyse_oracles.py`, `probe_classifiers.py`).

**Result.**
- Real labels stay at chance for the learned probes; the planted-signal control is learnable, so the pipeline can detect a signal when one exists.
- The semantic watermark (token ids 1010–4009, +1.0 bias) is visible to an oracle that knows the id range, but not to a learned model.
- The syntactic watermark (ids 100–199) is too weak to see at these intensities.
- The positional watermark adds a constant to all logits, which softmax ignores, so it has no effect on the output distribution.
- The token watermark draws a fresh random 25% green list every step with no key, so there is no consistent signal across steps.

**Caveat.** This is about the watermarks as implemented in this repository, not about watermarking in general.

## Q2: Why B collapses, and what fixes it

**Method.** `exp2_b_learning.py` trains the type head on cached features with three label sets (real, planted, shuffled) and three learners (supervised, batched RL, single-episode RL), over learning rates and entropy coefficients. Two feature setups: standardised, all pooling (CLS + mean + max), and raw, mean pooling, no standardisation. Chance accuracy is 0.25.

**Result.**

| Setup | Real | Shuffled | Planted |
|---|---|---|---|
| Standardised, all pooling | acc ≈ 0.25, `max_share` 0.30–0.43, no collapse | same | supervised 0.84, RL 0.78–0.80 |
| Raw, mean pooling, no standardise | RL: `max_share` 0.77–1.00, entropy 0.006–1.3, collapse from about episode 500 | RL (batched): `max_share` 1.00, entropy ≈ 0.007, collapse at episode 504 | supervised 0.50–0.59, RL 0.35–0.55, collapses start |
| Raw, supervised | no collapse (`max_share` 0.42–0.53, entropy ≈ 1.37) | no collapse | no collapse |

The raw features of random texts have mean pairwise cosine similarity 0.74 (0.89 with all three pooling blocks), so the inputs are nearly identical.

**Interpretation.** With no learnable signal, RL updates are noise on near-identical inputs, and the policy drifts to a single class. The shuffled-label control collapses just like the real labels, which shows the collapse does not need any structure in the data. Supervised training, and standardising the inputs, both prevent it.

**Caveats.**
- The raw test changed standardisation and pooling together, so their individual contributions are not separated.
- Runs 14–15 used a fine-tuned encoder, not frozen features. This reproduces the mechanism but does not prove it was the cause there.
- Only `--quick` settings were used for the raw test (lr 1e-4 and 3e-4).

## Q3: PPL explosions (Runs 02–10)

**Method.** Four 3000-episode A-side runs with a random B (no B update):

| Condition | Overrides |
|---|---|
| C1 policy only | `lr_backbone_a = 0` |
| C2 legacy backbone | `lr_backbone_a = 1e-5`, `fluency_loss_coef = 0`, `kl_coef = 0` |
| C3 anchored | `lr_backbone_a = 1e-6`, `fluency_loss_coef = 0.1`, `kl_coef = 0.05` |
| C4 legacy history | `lr_backbone_a = 0`, `legacy_history_bug = True` |

**Result (last 500-episode window; per-window values in `summarise_a_runs.py` output).**

| Run | PPL median / p90 | Unique ratio | Samples |
|---|---|---|---|
| C1 | 23.5 / 36.9 | 0.70 | coherent |
| C2 | 28.2 / 44.4 | 0.69 | word salad ("imagination… gravity… he could that all there is still") |
| C3 | 20.6 / 27.4 | 0.70 | coherent |
| C4 | 21.5 / 33.0 | 0.71 | coherent |

**Interpretation.**
- Training the backbone without an anchor is the only change that produced gibberish (C2 vs C1).
- The anchor (small backbone lr plus fluency and KL terms) kept the text coherent with flat PPL (C3).
- The logged PPL and the unique-token ratio did not flag the degradation in C2. Samples must be read, or a separate quality metric added.
- C3 stops choosing the semantic strategy (share 0.00 after the first window). The semantic bias (+1.0 over thousands of ids) probably conflicts with the fluency penalty, so the anchor trades one strategy for stability. This is a hypothesis; it was not isolated.

**Not answered.** The very large PPL values of Runs 02–10 did not reproduce in 3000 episodes at these settings. Remaining candidates: longer training, a higher effective lr, the uncapped-bias effects in Runs 01–05 (not covered here), or how PPL was computed. It was not verified which model scores the logged PPL.

## Q4: Strategy collapse

**Result (strategy shares and `max_share`, 500-episode windows).**

| Run | Strategies used | `max_share` per window |
|---|---|---|
| C1 | all four | 0.50, 0.50, 0.35, 0.40, 0.45, 0.37 |
| C4 (legacy history bug) | positional and syntactic only; `semantic` and `token` never chosen | 0.80, 0.70, 0.68, 0.58, 0.63, 0.60 |

An earlier C4 attempt, aborted after about 550 episodes, had collapsed onto `token` with a share of 1.00 in its second window. The dominant strategy differs between runs, which points at the bug and not at a strategy being better.

**Interpretation.** The hard exploration floor alone does not prevent collapse: under the history bug it allows `max_share` up to 0.80 and a lost half of the repertoire. With the fix (C1), the repertoire stays intact.

**Caveat.** One completed seed per condition, plus the partial C4 run.

## Q5: Run 01 PPL ≈ 1

**Result.** The low PPL values came from rare repetitive samples, not from short texts. The earlier hypothesis that short generations explained it was refuted by the PPL-vs-length analysis in `analyse_oracles.py`. PPL and NLL do not detect repetition; the unique-token ratio does.

## Implications for the project

1. Any further RL on B is pointless until the watermarks carry a learnable signal. Redesign candidates: a keyed token list, a position-dependent positional watermark, a stronger bias.
2. If B is trained again, standardise features and prefer supervised or supervised-aided updates before RL.
3. A's backbone should not be trained without a fluency/KL anchor; quality checks must read samples or use a repetition-aware metric, not PPL alone.
4. Log the strategy repertoire (`max_share`, unused strategies) as a standard health check.

## Open items

- Why the Run 02–10 PPL values were so large (see Q3).
- Whether standardisation or pooling matters more for the B collapse (Q2).
- Effects of the uncapped bias in Runs 01–05.
- Multiple seeds for the A-side conditions.

## Reproducing

```bash
git pull
python experiments/gen_dataset.py ...                 # see README_EXPERIMENTS.md for arguments
python experiments/analyse_oracles.py ...
python experiments/probe_classifiers.py ...
python experiments/exp2_b_learning.py --data_dir data                       # standardised
python experiments/exp2_b_learning.py --data_dir data --no_standardise --pool mean --quick --out data/exp2_raw_mean.csv
python experiments/run_cfg.py C1_policy_only '{"lr_backbone_a": 0.0}'
python experiments/run_cfg.py C2_legacy_backbone '{"lr_backbone_a": 1e-5, "fluency_loss_coef": 0.0, "kl_coef": 0.0}'
python experiments/run_cfg.py C3_anchored '{"lr_backbone_a": 1e-6, "fluency_loss_coef": 0.1, "kl_coef": 0.05}'
python experiments/run_cfg.py C4_legacy_history '{"lr_backbone_a": 0.0, "legacy_history_bug": true}'
python experiments/summarise_a_runs.py
```

Run each `run_cfg.py` condition in its own process; the runs save no checkpoints, so an interrupted run restarts from zero.