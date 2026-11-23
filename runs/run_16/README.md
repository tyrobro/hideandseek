# Run 16: the consolidated run

Everything learned in Runs 01–15 and the Q1–Q5 experiments (`EXPERIMENT_RESULTS.md`, `RETROSPECTIVE.md`, `WATERMARK_ANALYSIS.md`), applied at once. This README is written **before** the run: the design and the success criteria are fixed here, the results section is filled in afterwards and earlier READMEs are not edited with hindsight.

## What changed relative to Run 15

| Area | Run 15 | Run 16 | Reason |
|---|---|---|---|
| Watermarks | v1: no-op positional, unkeyed token noise, weak fixed-id syntactic/semantic | **v2**: four fixed, disjoint token sets; positional ramps over the window (`environment/watermark_v2.py`) | Q1: v1 carried no learnable signal |
| Watermark strength | fixed biases | one global `wm_strength`, **chosen by `experiments/gate_watermarks.py` on real GPT-2 text before the run** | detectability and fluency cost must be measured, not assumed |
| Agent A backbone | trainable (lr 1e-6, fluency + KL) | **frozen**; only the policy head learns | Q3: unanchored backbone → word salad; anchored kept text but dropped a strategy; policy-only (C1) was stable with all four strategies |
| Strategy history | fixed in Run 14 | fixed, plus `max_share20` logged every row | Q4: the history bug lost half the repertoire |
| B features | raw pooled | running per-feature standardisation (statistics frozen after warm-up) | Q2: raw features nearly identical (cosine 0.74–0.89); RL collapsed |
| B training | pure PPO | **1500-episode supervised warm-up** (A uniform-random, not updated), then PPO **plus** a supervised cross-entropy term | Q2: RL on a weak reward collapses; supervision did not |
| Span convention | B end index inclusive vs exclusive true span | prediction converted to exclusive end | off-by-one in Runs 12–15 |
| Safety gate | none | warm-up gate: abort at episode 1501 if B's greedy type accuracy on the true labels is below `warmup_min_acc` (0.40) | a run that cannot learn even with labels only reproduces Runs 14–15 |
| Evaluation | training-time curves only | 400-episode held-out evaluation (A random, B greedy): per-class recall, confusion, accuracy per intensity, shuffled-chance, IoU vs full-text and best-fixed-window baselines | training-time accuracy against a moving opponent is not a measurement |
| PPL brake | 250 | off | with a frozen backbone a brake can only skip learning |

B is no longer a purely RL agent: the supervised term is a deliberate departure, justified by Q2, and results must be described that way.

## Pre-registered success criteria (proposals; change them before the run, not after)

Gate (before training, `gate_watermarks.py`):
1. 4-class supervised TF-IDF accuracy ≥ 0.50 (chance 0.25) at the chosen strength.
2. Median PPL of every watermark class ≤ 2.5 × the unwatermarked median.

Warm-up check (episode 1500):
3. Greedy type accuracy on the last 200 warm-up episodes ≥ 0.40.

Final held-out evaluation:
4. Type accuracy ≥ 0.45 **and** at least 0.15 above the shuffled-prediction chance.
5. Recall ≥ 0.25 for every class (no class ignored).
6. Localisation claimed only if IoU (model) exceeds the best fixed-window baseline by ≥ 0.05.
7. Accuracy at intensity 1.0 higher than at 0.3 (the stealth knob works).

A-side health (last 1000 episodes of the game phase):
8. All four strategies chosen ≥ 5% of the time; mean `max_share20` < 0.6.
9. Median PPL < 2 × the gate's unwatermarked median; unique-token ratio median > 0.5; samples read and coherent.

How to read outcomes:
- Gate fails → no strength works. Do not train; the watermark design needs changing.
- Gate passes, warm-up check fails → the encoder cannot learn from the labels; check features and learning rates before anything else.
- 4–5 pass, 8–9 fail → B works but A's policy degenerates; look at the reward balance.
- All pass → the game produces a detector that beats chance and a watermarker that stays stealthy and diverse.

## Gate result (real GPT-2 text, 300 per class, TF-IDF logistic, 5-fold CV; chance 0.25)

| wm_strength | 4-class acc | acc at intensity 1.0 / 0.3 | PPL ratio vs none | verdict |
|---|---|---|---|---|
| 1.0 | 0.279 | 0.30 / 0.26 | 0.96–0.97 | fail: too weak |
| 2.0 | 0.430 | 0.58 / 0.32 | 1.02–1.18 | fail: below 0.50 |
| 3.0 | 0.554 | 0.75 / 0.38 | not computed (NaN bug in the gate script, fixed afterwards); medians 22–28 vs about 22 | inconclusive, would likely pass |
| 4.0 | 0.687 | 0.85 / 0.46 | 1.07–1.41 | pass |

Chosen: **4.0**, because B gets less data in the warm-up than the gate's classifier did, so the margin matters more than subtlety. At 4.0 the unique-token ratio is unchanged (about 0.70–0.74), and the watermark does land: in-window share of each class's own token set is 0.43–0.76, against 0.04–0.36 in unwatermarked text. Intensity 0.3 stays much harder to detect (0.46) than 1.0 (0.85), so the stealth knob exists.

## How to run

See `experiments/README_EXPERIMENTS.md` and the Run 16 notebook cells: pull; run the gate; set `wm_strength` in `config.py`; smoke test; launch `python train.py` as a background process; poll; read `final_eval.json`, `samples.txt` and the log.

## Results

Single seed. Held-out evaluation: 400 fresh episodes, A playing uniformly at random (all intensities, all strategies equally), B greedy. The training log has one row per 25 episodes (60 warm-up rows, 179 game rows), so per-window numbers from it are noisy.

### Held-out evaluation (`final_eval.json`)

| Metric | Value | Reference |
|---|---|---|
| Type accuracy | **0.560** | chance 0.25; shuffled-prediction chance 0.242 |
| Recall: token / semantic / positional | 0.889 / 0.611 / 0.813 | |
| Recall: **syntactic** | **0.000** | 110 syntactic samples: 84 predicted positional, 25 token, 1 semantic |
| Accuracy at intensity 0.3 / 0.6 / 1.0 | 0.399 / 0.617 / 0.674 | |
| IoU (model) | **0.599** | full-text span 0.217; best fixed window 0.316 |

Confusion matrix (rows = true, columns = predicted: token, syntactic, semantic, positional):

| | token | syntactic | semantic | positional |
|---|---|---|---|---|
| token | 88 | 0 | 1 | 10 |
| syntactic | 25 | 0 | 1 | 84 |
| semantic | 22 | 0 | 58 | 15 |
| positional | 16 | 0 | 2 | 78 |

### Agent A during the game phase (179 logged rows)

- Median PPL 22–29 per window (maximum 80); unique-token ratio median about 0.72; mean `max_share20` about 0.44.
- Strategy shares over the game: positional 38%, semantic 33%, token 29%, **syntactic 0%** (zero of 179 rows, including the first 20 rows after the warm-up ended).
- Intensity: about two thirds of recent choices are the lowest intensity (0.3).
- Samples (`samples.txt`) are mostly coherent. Two show repetition: episode 1500 ("Great Great Great…", during the uniform warm-up) and episode 3500 (unique ratio 0.55).

### Verdict against the pre-registered criteria

| # | Criterion | Result |
|---|---|---|
| 1 | Gate accuracy ≥ 0.50 | pass (0.687) |
| 2 | Gate PPL ratio ≤ 2.5 | pass (≤ 1.41) |
| 3 | Warm-up accuracy ≥ 0.40 | pass (the run did not abort; exact value in `data/run16.log`) |
| 4 | Type accuracy ≥ 0.45 and ≥ 0.15 above shuffled chance | pass (0.560; +0.318) |
| 5 | Recall ≥ 0.25 for every class | **fail** (syntactic 0.000) |
| 6 | IoU exceeds best fixed window by ≥ 0.05 | pass (+0.283) |
| 7 | Accuracy at intensity 1.0 above 0.3 | pass (0.674 vs 0.399) |
| 8 | All four strategies ≥ 5%; mean `max_share20` < 0.6 | **partly fail** (max_share20 passes; syntactic is 0%) |
| 9 | PPL < 2× plain text; unique ratio > 0.5; samples coherent | pass, with two weak samples |

### What this shows

- B now detects three of the four watermark types well above chance and localises the window well above the best fixed-window guess (0.60 vs 0.32). Runs 12–15 never produced a detector above chance. The sampled-prediction accuracy during the game (about 0.5) agrees with the greedy held-out figure.
- The stealth trade-off works as designed: B is more accurate at higher intensity, and A moved toward low intensity.
- A frozen backbone kept fluency and diversity stable throughout, as predicted by the C1 experiment.

### What it does not show, and the open problem

Syntactic is neither detected nor played. The two failures probably share one cause, but the cause is not established.

- **Signal exists.** At the same strength the gate's bag-of-words classifier recognised syntactic 69% of the time, so the watermark is detectable in principle.
- **A never played it.** Zero syntactic episodes in 179 logged game rows, starting from the first window. If A had been choosing uniformly, the chance of none in the first 20 logged rows would be about 0.3%, which points at A's initial type-head distribution (the head is untrained during the warm-up, has no entropy bonus, and the exploration floor only fires above 70% share) and not at a learned preference. Not verified.
- **B then never saw it again.** For 4500 episodes B's training stream contained no syntactic examples, so its syntactic logit was only ever pushed down. Whether B had learned syntactic at the end of the warm-up and then forgot it, or never learned it, cannot be separated: no checkpoint from the end of the warm-up was kept.

Other caveats: one seed; 400 evaluation episodes (about ±0.05 on overall accuracy); B is not purely RL (supervised warm-up and supervised term); the intensity-0.3 accuracy of 0.40 still beats chance, so low intensity is not yet fully stealthy.

### Candidate fixes if another run is made (not applied)

1. Initialise A's type head to uniform and add an entropy bonus on it, so no strategy starts at about 0%.
2. Keep about 25% of game episodes as uniform-random "probe" episodes used for B's supervised loss only, so B cannot forget a class A stops playing.
3. Save a checkpoint at the end of the warm-up and report per-class recall at that point.