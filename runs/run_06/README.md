# Run 06 — Hide&Seek Training Run

**Status:** Complete
**Date:** October 2026  
**Key changes from Run 05:** Fixed positional watermark PPL explosion (clamped global logit shift); rebalanced reward alpha=2.0/beta=0.3 to prevent span collapse; fixed logging (log_rows saved to checkpoint, braked rows written to CSV); added checkpoint resume logic.

---

## Config

| Parameter | Value | Change from Run 05 |
|---|---|---|
| `alpha` (IoU weight) | 2.0 | **was 1.0** |
| `beta` (type weight) | 0.3 | **was 0.5** |
| `gamma_fluency` | 0.3 | — |
| `stealth_bonus` | 1.5 | — |
| `ppl_threshold` | 200.0 | — |
| `ppl_brake` | 500.0 | — |
| `num_episodes` | 20,000 | **was 10,000** |

---

## Results Summary

| Metric | Value |
|---|---|
| Episodes logged | 150 (ep 20050–30000, every 50) |
| Agent A positive reward rate | ~85% — stealth bonus fires almost every episode |
| Agent A strategy distribution | syntactic 98%, positional 2% |
| Mean IoU | ~0.000 — B never localises the span |
| Type accuracy (B) | ~15% |
| PPL range | 3–461, median ~20 |

---

## What Worked

- **Logging fixed.** All episodes from ep 20050 onward logged correctly to CSV and flushed to disk incrementally. No more log gaps.
- **Checkpoint resume works.** Training resumed from ep10000 seamlessly; log_rows restored from checkpoint.
- **PPL is healthy.** With the positional logit shift clamped, PPL stays in the 3–100 range for the vast majority of episodes. Only occasional spikes above 200 (4 episodes in 10k logged).
- **Stealth bonus dominates correctly.** Agent A receives +1.5 on ~85% of episodes — the reward signal is learnable and consistent.

---

## What Failed

### 1. Agent A collapsed entirely to `syntactic`

Agent A plays `syntactic` on 98% of episodes. This is a degenerate equilibrium: `syntactic` watermarking doesn't affect fluency meaningfully (PPL stays low), and Agent B cannot detect it (IoU=0 consistently), so the stealth bonus fires every time. A has found the path of least resistance and stays there.

The novelty bonus fails to prevent this because it only removes upside (bonus → 0 when frequency → 1.0) but doesn't add downside. A rational agent ignores a reward that goes from +0.2 to 0 when it's already sitting at +1.5.

**Root cause:** The exploration mechanism is too weak. No gradient pressure forces A away from a strategy that already maximises reward.

### 2. IoU is permanently zero

Agent B never overlaps the true span. B has learned to predict type (15% accuracy) while completely ignoring span localisation — the same span collapse seen in Run 05, now more severe. With alpha=2.0, span is worth more in theory, but B can't improve span prediction when A always uses syntactic watermarking which produces no detectable spatial signal.

### 3. The two failures compound

A collapses to syntactic → B can't detect syntactic → stealth bonus fires → A has no reason to change → B has no signal to learn from → both agents are stuck. The zero-sum game has degenerated into a non-game.

---

## Planned Changes for Run 07

1. **Hard exploration floor in Agent A** — if one strategy appears >70% of last 20 episodes, force a uniform random sample. Breaks the syntactic lock immediately.
2. **Temperature annealing on type head** — start at temperature=2.0 (near-uniform sampling), anneal to 1.0 over 5000 episodes. Encourages exploration early before policy hardens.
3. **Novelty penalty (not just lost bonus)** — change novelty term to `eta * (1 - 2*frequency)`: positive when rare, zero at 50%, negative when overused. A now pays a cost for staying at syntactic.