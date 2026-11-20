# Answering the five open questions — experiment guide

Copy the `experiments/` folder into the repo root, and replace `train.py`, `config.py` and `environment/watermark_env.py` with the delivered versions (they add `b_mode`, `legacy_history_bug` and nothing else). Everything below runs from the repo root on Kaggle. None of these scripts has been run (this workspace has no GPU, torch or Hugging Face access), so expect to fix small things on the smoke test.

| Question | How it is answered | Needs GPU | Time |
|---|---|---|---|
| Q1 Can semantic/syntactic be detected at all? | Step 1 → 2 → 3 | gen + probe | ~2 h |
| Q2 Why does B collapse, does supervision / lower lr fix it? | Step 4 (uses Step 3 features) | no | ~15 min |
| Q3 What caused the PPL explosions? | Step 2 (watermark cost) + Step 5 (backbone conditions) | Step 5 | ~4 h |
| Q4 Does the floor + history fix prevent A's collapse? | Step 5 (conditions C1 vs C4) | yes | same runs |
| Q5 Run 01's PPL ≈ 1 at episode 50 | Answered below from the repo; Step 2 adds the length check | no | — |

## Q5 — answered now (from the repo, no new run)

- Run 01's log has only **two** rows with PPL below 8: episode 50 (1.18) and episode 100 (2.34). Episode 150 onward is 8–7000, with 3 of 200 rows below 15 in total. So it was **not a persistent degenerate state**; it was two early episodes.
- In Run 01 only A's policy head was in the optimiser (the backbone entered the optimiser in the "fixed" commits before Run 02), so the backbone cannot have drifted by episode 50.
- Run 01 generated with a **manual token loop** (full-softmax sampling, up to 128 tokens, `prompt[:200]`); Hugging Face `generate` replaced it in commit `3ff4711`, before Run 02. PPL regimes of Run 01 and later runs are not directly comparable.
- Most likely explanation (**Likely, not verified**): very short generations. PPL is computed on the generated text alone, and a text of a few tokens can have a PPL near 1–2 (`analyse_oracles.py` section Q5 tabulates PPL against generated length and EOS-early share on the real dataset, which will confirm or reject this). The log has no text or length column, so this cannot be settled from Run 01 itself.

## Step 1 — dataset with the real watermark code (frozen A)

```
python experiments/gen_dataset.py --n 5000 --out data/wm_dataset.jsonl            # as trained (default sampling)
python experiments/gen_dataset.py --n 2500 --out data/wm_dataset_nok.jsonl --top_k 0 --seed 7
```
The first line prints the model's default `top_k`/`top_p`. If it is 50 (the usual default), top-k filtering is applied **after** the watermark bias and may truncate weak biases (syntactic +1.5 on rarely-top-50 token ids); the second file generates with top-k off to test that. Two processes can run in parallel with `CUDA_VISIBLE_DEVICES=0` / `1` and different `--out`.

## Step 2 — does the watermark leave a trace? (CPU)

```
python experiments/analyse_oracles.py data/wm_dataset.jsonl
python experiments/analyse_oracles.py data/wm_dataset_nok.jsonl
```
Reads the tables as follows:

| Result | Meaning |
|---|---|
| Syntactic / semantic oracle AUROC vs 'none' clearly above 0.5 (e.g. > 0.7 at intensity 1.0) | The watermark exists in the token statistics; a detector could find it |
| Same AUROC ≈ 0.5, but higher in the `nok` file | Top-k truncation was hiding the bias |
| Positional AUROC ≈ 0.5 and KS p-values not small | Confirms positional is a no-op |
| Token AUROC ≈ 0.5 (maybe a small logppl effect) | Unkeyed noise: no signature, at most a fluency footprint |
| Oracle-feature GBM accuracy well above 0.20 | There is learnable signal in simple features; the confusion matrix shows which classes |
| A. PPL table: all labels ≈ 'none' at every intensity | The watermarks themselves do not wreck fluency (key evidence for Q3) |

## Step 3 — supervised B (Q1)

```
python experiments/probe_classifiers.py --data data/wm_dataset.jsonl --out_dir data
```
Prints validation accuracy and per-class recall for (a) frozen DistilBERT features with logistic regression (mean-pool vs CLS+mean+max; 5-class and 4-class), (b) a **planted-signal control**, (c) fine-tuned DistilBERT.

| Result | Meaning |
|---|---|
| Control (planted) near 100% | The feature pipeline can detect a strong signal; a failure elsewhere is about the watermarks |
| Frozen ≈ chance, fine-tuned clearly above | The frozen features are the bottleneck (supports Path 4's premise) |
| All ≈ chance except semantic/syntactic recall above 0.25 | Only the token-set watermarks are detectable; expected ceiling ≈ 0.75 for 4-class |
| All at chance incl. fine-tuned, while Step 2 oracles show signal | The signal is too weak for DistilBERT's tokenisation; consider stronger bias |
| Everything at chance incl. Step 2 | No usable signal as implemented; watermarks need redesign before any RL |

This step also writes `features_real.npz` and `features_planted.npz` for Step 4.

## Step 4 — why does B collapse, and does supervision fix it? (Q2, no generation)

```
python experiments/exp2_b_learning.py --data_dir data
```
Trains B's type head on cached features under three label sets (real, planted, shuffled) and three learners (supervised, RL batched as in Run 15, RL single-episode as in Runs 12–14), with lr ∈ {1e-5, 1e-4, 3e-4}, entropy ∈ {0.01, 0.05}, 3 seeds. Output columns: `acc` (argmax), `acc_sampled`, `max_share` (share of the most-predicted class), `first_collapse_ep`.

| Result | Meaning |
|---|---|
| RL reaches high accuracy on **planted**, but collapses (`max_share` > 0.9) on **shuffled** | Collapse is what RL does when no signal exists; it is not a bug. Cure is real signal or a supervised/auxiliary loss |
| RL fails on **planted** too, supervised succeeds | The RL update/hyper-parameters are the problem; look at the lr/entropy rows for a working setting |
| Collapse only at lr ≥ 1e-4 | The learning rate is the main driver (Runs 14–15 used 1e-4 and 3e-4) |
| Supervised on **real** labels matches Step 3 | Consistent with the probe; the RL gap is the extra cost of RL |

## Step 5 — A-side: PPL explosions and strategy collapse (Q3, Q4)

B is replaced by a random guesser (the situation of Runs 01–11), A is trained unforced for 3000 episodes per condition. Run each in its own process:

```
python experiments/run_cfg.py C1_policy_only   '{"lr_backbone_a": 0.0}'
python experiments/run_cfg.py C2_legacy_backbone '{"lr_backbone_a": 1e-5, "fluency_loss_coef": 0.0, "kl_coef": 0.0}'
python experiments/run_cfg.py C3_anchored      '{"lr_backbone_a": 1e-6, "fluency_loss_coef": 0.1, "kl_coef": 0.05}'
python experiments/run_cfg.py C4_legacy_history '{"lr_backbone_a": 0.0, "legacy_history_bug": true}'
python experiments/summarise_a_runs.py
```
Run two at a time with `CUDA_VISIBLE_DEVICES=0` / `1` prefixes to halve the time. Each run also writes `runs/exp_<name>/samples.txt` (gibberish vs repetition is visible there).

| Comparison | Reads as |
|---|---|
| **C2** PPL (median/p90) climbs and `uniq` stays normal or collapses, **C1** stays flat | Backbone training by the RL reward is the cause of the PPL explosions (Q3 answered) |
| C2 and C1 both climb | The watermarks or the policy head drive the explosion; revisit with the Step 2 PPL table |
| **C3** flat PPL **and** flat `uniq` | The real-text anchor + small backbone lr is sufficient to keep A stable |
| **C1** `max_share` stays < ~0.7, **C4** `max_share` → 0.85–1.0 and `nov_min` ≈ 0 | The floor + history fix prevents collapse; the old history pollution reproduces it (Q4 answered) |
| C1 also collapses | The floor is not enough on its own: the reward (stealth bonus against an untrained B) drives A to the cheapest strategy |
| `nov_min` in C4 stays at ≥ 0 and in C1 goes negative | The novelty term works only in C1 (the Run 09–11 signature was `nov_min = 0.0`) |

Limits: 3000 episodes, one seed per condition; Runs 08–10 had 10,000 episodes and a brake. Treat C2-vs-C1 and C4-vs-C1 as strong tests of direction, not of exact magnitudes.

## Order and budget

1. Step 1 (gen) then Step 2 (30 min of waiting for data, minutes of analysis).
2. Step 3 (probe, ~30–40 min) → Step 4 (minutes).
3. Step 5 can run in parallel with Steps 1–3 on the second GPU.
4. Send back the console output of Steps 2–4 and `summarise_a_runs.py` plus the `samples.txt` files; I'll turn them into the answers and the README.
