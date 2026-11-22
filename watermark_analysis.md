# Watermark Analysis: how each technique works, how it changed, and how B tries to find it

Scope: the four watermark strategies as implemented in this repository through Run 15 (`environment/watermark_types.py`, "v1"), how they evolved across Runs 01–15, how Agent B locates them, and why Run 16 replaces them (`environment/watermark_v2.py`). Mechanisms are read from the code; detectability claims come from the Q1 experiments in `EXPERIMENT_RESULTS.md`. Nothing here is a claim about watermarking in general.

## 1. How a watermark gets into the text

1. Agent A's policy head picks a strategy (token / syntactic / semantic / positional), an intensity in {0.3, 0.6, 1.0} and a position fraction in {0.1, 0.45, 0.75}.
2. The fraction becomes `wm_start = int(frac × 80)`, so the window starts at generated token **8, 36 or 60** and lasts 17 steps (`wm_end = wm_start + 16`).
3. `WatermarkLogitsProcessor` (a Hugging Face logits processor) adds the strategy's bias to GPT-2's logits **only inside that window**. Outside it, sampling is plain GPT-2 (no top-k/top-p in the training configuration).
4. The label handed to B is the strategy plus the character span of the window.

Consequence: at most 17 of ~80 tokens are touched, and the window can only start at three places.

## 2. The four techniques in v1 (Runs 01–15)

| Technique | What it does to the logits | Effect on the text | Detectable? |
|---|---|---|---|
| **Positional** | adds `min(bias × intensity, 1.0)` to **every** logit | none: softmax is invariant to a constant shift (checked numerically, max probability change ≈ 3e-19) | No. Output is distributed exactly like unwatermarked text. |
| **Token** | draws a fresh random 25% of the vocabulary at **every step** and adds `2 × intensity`, no key or seed | pushes mass to a different random token set each step, i.e. noise | No stable signature; a detector cannot recompute the list. |
| **Syntactic** | adds `1.5 × intensity` to token ids 100–199 | slight over-use of 100 fixed ids for 17 tokens | Practically no: expected extra hits in 17 tokens are well below one. |
| **Semantic** | adds `1.0 × intensity` to ids 1010–4009 (about 3000 ids) | mild, consistent preference for a fixed band | Visible only to an oracle that knows the id range; not learned by any model tested. |

Two further points:
- "Syntactic" and "semantic" are names only. Both are fixed token-id bonuses with no linguistic content.
- The positional strategy carried a span in its metadata, but the logit change was a no-op, so it existed as a label only.

## 3. Evolution across runs

| Stage | Change | What happened |
|---|---|---|
| Run 01 | Uncapped biases; positional base 3.0 | PPL rose from about 1 to about 7000; A's reward fell to −43 |
| Run 02 | Per-strategy intensity caps (token 1.5, others 1.0); positional base 3.0 → 1.5 | PPL ceiling about 300 |
| Run 04 | PPL-aware intensity scaling | changes watermark strength only |
| Runs 05–06 | Positional shift clamped to ≤ 1.0, on the belief that a global shift exploded PPL | wrong diagnosis (softmax no-op); the clamp changed nothing |
| Run 08 | Placeholder semantic map `{i: i+1}` on ids 0–49 | PPL about 10,000; the map was blamed, which is unverified and probably wrong |
| Run 09 | Semantic map moved to ids 1000–4000 | syntactic dominated A's choices; real cause was history pollution |
| Runs 14–15 | Current v1 | frozen A, all strategies and intensities: median PPL 22–23, so the capped watermarks do not by themselves wreck the text |

The pattern: biases were made weaker and safer over time, but nobody checked whether the output distribution still carried a signal until the Q1 experiment. It did not.

## 4. How Agent B tries to locate them

- **Architecture:** a DistilBERT encoder over the generated text; a type head (4 classes) on pooled features; start and end heads over per-token states for the span.
- **Reward:** type correct (0/1) plus IoU between predicted and true span. A is rewarded for the opposite, including a stealth bonus when B is wrong and IoU < 0.1.
- **Span alignment:** the true span is in GPT-2 token indices, B predicts DistilBERT wordpieces. Until Run 14 these were compared directly; from Run 15 the true span is mapped through character offsets.
- **An off-by-one found while preparing Run 16:** B's sampled end index is inclusive, but the true span and `compute_iou` use an exclusive end. For Runs 12–15 the predicted span was therefore one wordpiece short. The effect on IoU is small but systematic. Run 16 converts the prediction to an exclusive end.

What B could actually exploit in v1:
- **The type:** nothing (Q1). Accuracy was at or below chance once labels varied.
- **The location:** the window starts at only three places, so a predictor that learns that prior beats the full-text baseline (IoU about 0.2) without reading the text. Logged spans from Run 15 were geometry (roughly the whole text), not evidence-based localisation.

## 5. Why v1 could not work, and what v2 changes

The problem was never B's architecture or learning rate; there was no signal to learn. v2 (Run 16) gives every type its own stable lexical signature:

| Technique | v2 definition | Why it is learnable |
|---|---|---|
| **Token** | a fixed, key-seeded random 20% of the vocabulary gets `+2.0 × strength × intensity` | stable set, large mass shift |
| **Syntactic** | function words and punctuation ("the", "of", "and", ",", ".") get `+1.2 × …` | a real syntactic class: function-word frequency rises |
| **Semantic** | a fixed, key-seeded 10% of whole-word (content) tokens get `+2.0 × …` | stable content-word signature |
| **Positional** | a different fixed 10% word set with a bias that **ramps linearly across the window** (`+2.5 × … × step/window`) | no longer a constant shift; the effect depends on position |

All four sets are mutually disjoint. B is not given the key; it has to learn the sets from data, which is the point of the game.

What v2 does *not* guarantee: real GPT-2 distributions are peaked and context-dependent, which shrinks every effect compared with a simulation. That is why `experiments/gate_watermarks.py` measures detectability and fluency cost on real generations, and chooses the strength, before any training. A simple i.i.d. simulation (`tests/sim_detectability.py`, a proxy only) gave 4-class accuracy of 0.34, 0.59 and 0.72 at strengths 1, 2 and 3 (chance 0.25), with low intensity much harder to detect than high.

## 6. Design implications

1. Verify detectability before any RL: if a supervised probe does not beat chance, RL cannot either.
2. Treat location as its own task and report it against the best fixed-window baseline, so a real gain is separable from the position prior.
3. Intensity is the stealth knob: low intensity is harder to detect and cheaper in fluency. That trade-off is what gives the A-versus-B game something to play for.