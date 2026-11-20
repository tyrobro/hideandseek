# Hide&Seek — Retrospective, Runs 01–15

Sources: the run READMEs, the training logs for Runs 11–15, and the git history (the code as it was in each commit). Each conclusion is labelled **Confirmed** (checked against code or data), **Likely** (fits the evidence, not directly tested) or **Untested**. Earlier READMEs are left as written; this document supersedes their explanations where it says so.

---

## 1. The short version

After 15 runs, the main result is that **the experiment as built could not answer its research question**, and for reasons that were hidden by the metrics we were watching. Four independent problems stacked up:

1. **Agent B was never trained correctly until Run 12.** From the first commit, B's update re-sampled a new action and used it as both the old and new policy, so the expected gradient was zero. Runs 01–11 are therefore "Agent A against an untrained detector". **Confirmed** (code).
2. **Agent A's backbone was trained by the RL reward**, from the Run 02-era code onward, with no anchor to real text. This is the likely driver of the perplexity explosions (Runs 02–10) and of the repetition collapse (Runs 12–13). **Likely**; supported by Runs 14–15, where A was frozen and PPL stayed flat.
3. **Two of the four watermark types are undetectable by construction.** Positional adds a constant to every logit (a softmax no-op); token re-draws an unkeyed random green list at every step. Syntactic and semantic are weak biases on fixed token-id sets. **Confirmed** (code, plus a numerical check of the no-op).
4. **Our instruments misled us repeatedly**: perplexity cannot see repetition, accuracy against a constant label measures the prior, and IoU against a near-full-text span is ≈0.2 for free. **Confirmed**.

Even after B's update was fixed, B collapsed onto a single class whenever the label varied (Run 14: 83% "semantic"; Run 15: 98.7% "positional" within ~300 episodes). **Confirmed** (logs); the cause is **Untested**.

---

## 2. Run by run

| Run | What changed | What happened | Explanation given then | Verdict now |
|---|---|---|---|---|
| 01 | Baseline (policy head only in A's optimiser; B update already broken) | PPL ~1 → ~7000; A reward to −43 | Uncapped logit bias, weak fluency penalty | **Likely right.** The backbone was not being trained here, so the drift had to come from the biases. PPL ≈ 1 at episode 50 is itself abnormal (healthy GPT-2 sampling is ~20) and was never investigated |
| 02 | Bias caps, brake 500, reward clipping, γ_fluency 1.5 | PPL ceiling 300; B predicts "positional" from ep ~2000 | No entropy regularisation on B's type head | **Revised.** B's update had zero expected gradient, so drifting to one class is what a noise-driven update does. A's backbone was in the optimiser by this code version |
| 03 | Entropy bonus 0.05, lower intensities (note: an earlier "Run 03" accidentally ran Run 02 code because Kaggle had not pulled) | Clean PPL for ~800 episodes, then climb; B diverse until ep ~1900 then collapses to positional | Missing PPL-aware intensity scaling | **Unverified.** Fits backbone drift as well |
| 04 | PPL-aware intensity scale, mean pooling, 3× type loss, entropy 0.15 | PPL > 100 from ep ~2500; brake skipped 50% of updates; B type acc 20.5%; A positive reward 3.5% | Intensity scale only changes watermark strength; brake creates a dead zone | **Dead-zone point confirmed** (a braked A makes no update and cannot recover). The README calls 20.5% "above random chance (25%)"; it is below |
| 05 | Stealth bonus, threshold 200, γ_fluency 0.3 | A positive reward 13.6%; log gap (brake wrote no rows) | **Positional watermark adds a global shift every step and compounds into PPL explosion** | **Refuted.** A constant shift of every logit does not change the softmax (checked numerically: max probability change ≈ 3e-19). The real PPL driver is unidentified; backbone drift is the leading candidate |
| 06 | Clamp positional shift; α 2.0 / β 0.3; resume support | A plays syntactic 98%; IoU ≈ 0; B 15% | Exploration too weak; novelty only removes upside | **Half right.** The clamp changed nothing in practice. The stealth bonus fires whenever B fails, and B was untrained, so any cheap strategy wins |
| 07 | Hard floor (>70% → uniform), temperature annealing, signed novelty | Syntactic 56.5%; `true_span` stuck at (8, 24); PPL 100–300 | Position head collapse; PPL from exploration | **Floor worked here**, before position-history tracking was added to the same list |
| 08 | Position floor, position entropy, brake 300, γ 0.5 | PPL → 10,000, brake on ~95% of episodes, training effectively stopped after ep ~2100 | Placeholder semantic map `{i: i+1}` (ids 0–49) "corrupts logits" | **Unverified, probably wrong.** Under that map semantic adds at most +1 to 50 tokens, and GPT-2 ids 0–49 are ordinary byte-level pieces, not special tokens (vocab not checked here: no network). A braked A cannot recover, so the damage was done before ~2100 |
| 09 | New semantic map (ids 1000–4000 → +10), thresholds reverted, entropy off | Syntactic 87%; `novelty_bonus` exactly 0.0 from ep ~1950 | Resume block left a stale history reference | **Refuted** (Run 10 started fresh and showed the same zeros). Real cause: `strategy_history` was interleaved with span tuples, so the last 20 entries held at most 10 strategies and the floor never fired; η was applied twice (η²) |
| 10 | η 0.5, rewired references | Syntactic 97.5% | **Degenerate Nash equilibrium; B structurally cannot detect syntactic** | **Unsupported.** B was untrained, so nothing can be inferred about its ability |
| 11 | `force_strategy="syntactic"`, fluency loss + KL, corrective step | PPL stable; B at exactly 25%, uniform predictions, IoU at the shuffled null | B's update path suspected | **Confirmed in Run 12** |
| 12 | B update fixed (`evaluate`, stored rollout log-probs, EMA baseline, entropy once) | B type acc 25% → 85–100%; B span converges on (0, ~80), IoU ≈ 0.2; A's backbone collapses into repetition (final sample: one token repeated) | Four untested hypotheses for the collapse | **Fix confirmed.** Result says nothing about detection (constant label) |
| 13 | Intended B-only probe (random strategy, frozen A) | Did not take effect: all 200 rows `syntactic`, A still updating; partial repetition drift (unique-token ratio 0.71 → 0.35 after ep ~5000) | Override not applied | **Confirmed.** Identical to Run 12 in effect |
| 14 | Valid B-only probe (random strategy, A frozen, aligned launch) | B 24.8% = chance; predictions collapse to "semantic" (83%); IoU at null; A stable (median PPL 22, unique ratio 0.71) | B cannot detect with frozen features | **Result confirmed; cause revised** — see §3, problem 3 |
| 15 | Path 4: fine-tuned DistilBERT, CLS/mean/max pooling, batched updates, aligned spans | B predicts "positional" 98.7% (entropy 1.16 → 0.01 in ~300 episodes); accuracy 20.4% = positional's share; spans are geometry, not localisation | Stronger B might detect | **Not supported.** B collapsed before any signal could be learned |

---

## 3. Root causes, in order of impact

### Problem 1 — B's update was broken for Runs 01–11 (Confirmed)
From the first commit, `_update_agent_b` called `detect(text)` again, taking a freshly sampled action, and passed its log-probs as both old and new to a PPO loss, with a raw non-negative reward and no baseline. The reward earned by B's first sample was credited to an independent second sample. Expected gradient: zero. Behaviour: noise-driven drift to one class (Runs 02, 03, 05), or, once an entropy bonus was added (and applied three times inside the loop), a pull toward uniform ~25/25/25/25 (Runs 04, 09–11). Fixed in Run 12.

**Consequence:** every explanation in the Run 09 and Run 10 READMEs about B's inability to detect syntactic watermarks, the Nash-equilibrium framing, and "data starvation" is unsupported.

### Problem 2 — A's backbone drifts when trained (Likely)
`agent_a.model` entered the optimiser in the "fixed" commits between the first run and Run 02 and stayed there. Its gradient comes from the RL reward through the prompt hidden states, plus later a fluency loss. Two opposite failure modes appeared:
- reward-only pressure with no real-text anchor: gibberish and rising PPL (Runs 02–10; Run 08 reached ~10,000);
- fluency and KL measured on A's *own* samples: repetition, which has *low* perplexity (Runs 12–13, unique-token ratio falling to 0.075–0.14 in degenerate samples).

**Evidence for it:** in Runs 14–15 A was frozen and ran all four strategies at all intensities {0.3, 0.6, 1.0}: median PPL 22–23, max 138 and 86, unique-token ratio flat at ~0.71. So the watermarks at these capped intensities do not by themselves wreck the text.
**Caveat:** this does not isolate the mechanism, and the Run 01 explosion happened with the backbone out of the optimiser.

### Problem 3 — The watermarks as implemented (Confirmed from code)
| Type | Implementation | Detectability |
|---|---|---|
| Positional | `logits += min(bias·intensity, 1.0)` on all logits | **None**: softmax is invariant to a constant shift. The text is distributed exactly like unwatermarked text |
| Token | `random.sample` of a fresh 25% of the vocabulary at every step, +2·intensity, no seed or key | **No consistent signature**: a different random green list each step is pure noise |
| Syntactic | token ids 100–199, +1.5·intensity (cap 1.0) | Weak, fixed token set |
| Semantic | token ids 1010–4009, +1.0·intensity | Weak, fixed token set |

The bias is applied only inside a 17-step window of an 80-token generation. So even a perfect detector can separate at most semantic and syntactic from the rest, and must guess between positional and token: **a ceiling of about 75% accuracy** (0.25 + 0.25 + 0.5 × 0.5). Runs 12–15 never got near any detection signal because of this and Problem 4.

The Run 05 diagnosis (global positional shift explodes PPL) was wrong for this reason, and the Run 06 clamp did nothing.

### Problem 4 — B collapses to one class whenever the label varies (Confirmed in logs; cause Untested)
With the update fixed and a constant label, B learns the prior in ~150 episodes (Runs 12–13). With random labels it fails: Run 14 (frozen features, lr 1e-4) drifted from 36% to 83% "semantic"; Run 15 (fine-tuned encoder, head lr 3e-4) reached 98.7% "positional" within ~300 episodes. Candidate explanations: a 0/1 reward with one sample per update and no true gradient signal pushes the policy toward whichever class gets lucky; Adam amplifies noise; the entropy bonus is far smaller than the policy-gradient term; the learning rates were raised. None was tested in isolation.

### Problem 5 — Instruments that misled us (Confirmed)
- **Perplexity** is blind to repetition, and so are mean-NLL and KL measured on A's own samples.
- **Type accuracy with a constant label** equals the prior (Runs 11–13).
- **IoU** is ≈ 0.2 for any near-full-text span, and ≈ 0.17–0.25 for a long span in the middle of the text. Run 14's IoU was 0.172 against a shuffled null of 0.159; Run 15's 0.248 against 0.227, which is geometry.
- **Misaligned indices**: `true_span` was in GPT-2 token indices, B's span in DistilBERT wordpiece indices, so IoU was only approximately meaningful until Run 15.
- **Centre-of-span correlation** (Run 15: 0.23) is equally high for the positional and token types, which leave no signal, so it is not evidence of detection.
- **Single-episode logging** every 50 episodes; one seed; no episode texts saved.

### Problem 6 — Strategy collapse (Likely)
A's policy fell to 98% syntactic (Run 06), 87% (Run 09) and 97.5% (Run 10). The stealth bonus (+1.5) fires whenever B fails; with an untrained B it fired for almost anything, so A simply picked the cheapest strategy. The history-pollution bug (Run 09/10) also disabled the floor and halved the novelty frequency. The Run 07 result (56.5% syntactic with the floor working, before position tracking joined the same history) is consistent with this. **Untested**, because Runs 11–15 force or randomise the strategy.

### Process problems (Confirmed)
Kaggle running stale code (Run 03), silent brake with no log rows (Run 05), interrupted resumes (Runs 09, 12), the notebook overrides not reaching the run (Run 13), a multi-GPU illegal-memory-access crash (Run 15), and a README written from an earlier hypothesis that the data later contradicted (the stale-reference story for Run 09, which was my incorrect diagnosis).

---

## 4. What we learned

1. A game result is only meaningful if each agent is verifiably learning. Test each agent in isolation (does B reach ~100% on a constant label? does A with a fixed B do something sensible?) before running them together.
2. Check what each watermark actually does to the output distribution before asking a detector to find it. A one-line test (compare token distributions with and without the watermark) would have caught the no-op in Run 01.
3. Never rely on a single fluency metric. Track repetition (unique-token ratio) next to PPL, and anchor any fluency or KL term on real text.
4. Every metric needs its trivial baseline: prior accuracy, full-text IoU, shuffled-prediction null.
5. Do not train a language-model backbone with an RL reward and no anchor; if the backbone is trainable, keep its learning rate very small and measure its drift.
6. A brake that skips updates removes the only force that can pull a degraded policy back.
7. Print the effective config at launch, run in the same process, and use a fresh run directory.
8. Single-episode logs hide trends; log window means and a few raw samples.
9. Explanations should be written as hypotheses until a run isolates them. Four READMEs (05, 08, 09, 10) contained confident causal stories that later evidence contradicted.

---

## 5. Still open

- Can semantic or syntactic watermarks, as implemented, be detected at all (supervised, no RL)? This is the key missing measurement.
- Why does B collapse under random labels, and does supervision (or a much smaller learning rate) fix it?
- What actually caused the PPL explosions in Runs 02–10 (backbone drift is the leading candidate)?
- Does the strategy floor, with the history fix, prevent A's collapse in an unforced run?
- Does Run 01's PPL ≈ 1 at episode 50 point to an early degenerate state?