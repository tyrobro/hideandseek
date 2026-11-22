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
| B training | pure PPO | **800-episode supervised warm-up** (A uniform-random, not updated), then PPO **plus** a supervised cross-entropy term | Q2: RL on a weak reward collapses; supervision did not |
| Span convention | B end index inclusive vs exclusive true span | prediction converted to exclusive end | off-by-one in Runs 12–15 |
| Safety gate | none | warm-up gate: abort at episode 801 if B's greedy type accuracy on the true labels is below `warmup_min_acc` (0.40) | a run that cannot learn even with labels only reproduces Runs 14–15 |
| Evaluation | training-time curves only | 400-episode held-out evaluation (A random, B greedy): per-class recall, confusion, accuracy per intensity, shuffled-chance, IoU vs full-text and best-fixed-window baselines | training-time accuracy against a moving opponent is not a measurement |
| PPL brake | 250 | off | with a frozen backbone a brake can only skip learning |

B is no longer a purely RL agent: the supervised term is a deliberate departure, justified by Q2, and results must be described that way.

## Pre-registered success criteria (proposals; change them before the run, not after)

Gate (before training, `gate_watermarks.py`):
1. 4-class supervised TF-IDF accuracy ≥ 0.50 (chance 0.25) at the chosen strength.
2. Median PPL of every watermark class ≤ 2.5 × the unwatermarked median.

Warm-up check (episode 800):
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

## How to run

See `experiments/README_EXPERIMENTS.md` and the Run 16 notebook cells: pull; run the gate; set `wm_strength` in `config.py`; smoke test; launch `python train.py` as a background process; poll; read `final_eval.json`, `samples.txt` and the log.

## Results

*(to be filled in after the run: gate table, warm-up accuracy, final evaluation JSON, A-side windows, samples, and a verdict against criteria 1–9, with caveats.)*
