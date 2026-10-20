# Run 07 — Exploration Recovery via Hard Floor & Temperature Annealing

## Objective
Break the degenerate equilibrium from Run 06 where Agent A collapsed entirely to the `syntactic`
strategy (~98% frequency), exploiting the stealth bonus without any counter-pressure from Agent B.

## Changes from Run 06

### `agents/agent_a.py` — `select_strategy`
- **Hard exploration floor**: when any single strategy exceeds 70% of the last 20 episodes,
  force a uniform random strategy sample instead of sampling from the policy logits.
- **Temperature annealing**: type-head logits divided by `temp = max(1.0, 2.0 - n_episodes/5000)`,
  decaying from 2.0 (high entropy) to 1.0 (policy-driven) over the first 5000 episodes.
- **Strategy history wired**: `_strategy_history_ref` passed from `Trainer.__init__` into
  `select_strategy` so the floor can see the live history.

### `environment/watermark_env.py` — `compute_novelty_bonus`
- Changed from a one-sided bonus (`eta * (1 - frequency)`) to a **signed penalty**:
  `eta * (1 - 2 * frequency)` — positive when underused, negative when overused.
  Removes the asymmetry that made over-specialisation cost-free.

## Results

### Strategy Distribution (episodes 1–10 000)
| Strategy   | Frequency |
|------------|-----------|
| syntactic  | 56.5 %    |
| token      | 16.5 %    |
| positional | 14.0 %    |
| semantic   | 13.0 %    |

Syntactic dominance reduced from ~98 % → ~57 %. All four strategies are now sampled.

### Key Metrics
- **IoU**: episodic spikes up to ~0.81; median remains near 0, no sustained learning by Agent B.
- **Agent A reward**: no longer locked at +1.5; fluctuates between −2.0 and +1.5 as Agent B
  occasionally catches watermarks.
- **Agent B reward**: transient spikes (max ~1.85) with no sustained upward trend — B has not
  converged on a stable detection policy yet.
- **Perplexity**: rises from ~15 (ep 1–500) to 100–300 (ep 1000+), indicating fluency degrades
  as Agent A explores non-syntactic strategies under PPO pressure.

## Failure Modes Identified

### 1. Position Head Collapse
`true_span` is `(8, 24)` on virtually every episode across all 10 000 steps regardless of
strategy or intensity. The position head is not learning — the watermark always lands at the
same token window. This makes positional diversity superficial: the strategy label changes but
the actual span does not.

### 2. PPL Degradation Under Exploration
As Agent A moves away from syntactic (which produces fluent-sounding output at low PPL), the
other strategies drive PPL into the 100–300 range. The fluency penalty (`gamma_fluency`) is
not strong enough to counteract this.

### 3. No Sustained Agent B Learning
Agent B reward spikes are high-IoU single episodes, not a learning trend. B is not accumulating
a detection policy — it fires occasionally by chance rather than by learned generalisation.

## Next Steps (Run 08)
1. **Position novelty tracking**: apply the same hard-floor and novelty-penalty logic to the
   position head — track which token window is overused and penalise/force diversity.
2. **Entropy regularisation on position head**: add an entropy bonus term to the PPO loss for
   the position logits to prevent mode collapse independent of the history floor.
3. **PPL cap tightening**: lower `ppl_threshold` or increase `gamma_fluency` to prevent
   fluency collapse under exploration.
4. **Agent B curriculum**: consider giving B more signal — e.g. logging span-only IoU separately
   from type accuracy to diagnose which head is failing first.