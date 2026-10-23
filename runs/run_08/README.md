# Run 08 — Position Diversity & Entropy Regularisation

## Objective
Fix the position head collapse from Run 07 (span always `(8, 24)`) and add
entropy regularisation to the position logits to prevent mode collapse.

## Changes from Run 07

### `agents/agent_a.py`
- **Position floor**: same hard-floor logic as type strategy — if one
  `position_frac` value exceeds 70% of last 20 episodes, force uniform sample.
- **`position_frac` returned**: added to `all_logprobs` dict so it can be
  logged and passed through to the environment.
- **Position history append**: `_position_history_ref.append(position_frac)`
  called after span computation in `generate()`.

### `environment/watermark_env.py`
- **`position_frac` parameter** added to `step()` signature and logged in
  info dict.
- **`position_history` append**: `self.position_history.append(true_span)`
  in `step()`.

### `train.py`
- **`_position_history_ref` wired** at line 45.
- **Position history save/resume** in `_save()` and resume block.
- **Entropy bonus** on position logits in `_update_agent_a`:
  `loss = loss - entropy_coef_position * pos_dist.entropy().mean()`
- **`position_frac` logged** in `_log` row.

### `config.py`
- `entropy_coef_position`: 0.01
- `gamma_fluency`: 0.5 (was 0.3)
- `ppl_threshold`: 150.0 (was 200.0)
- `ppl_brake`: 300.0 (was 500.0)

## Results

### Strategy Distribution (episodes 1–2100, before brake lockout)
| Strategy   | Frequency |
|------------|-----------|
| semantic   | ~65 %     |
| positional | ~20 %     |
| syntactic  | ~10 %     |
| token      | ~5 %      |

### Key Metrics
- **IoU**: active only in ep 1–2100; spikes up to 0.59, mean ~0.15 — best
  span localisation seen so far across all runs.
- **Agent A reward**: fluctuating between −1.0 and +1.5 in active phase,
  indicating genuine competition.
- **Agent B reward**: spikes up to 1.18 — B occasionally catching watermarks.
- **Perplexity**: catastrophic explosion from ep ~2000 onwards (values
  reaching 10 000), triggering the PPL brake on ~95% of all subsequent
  episodes. Effectively no training after ep 2100.

## Failure Mode: PPL Catastrophe

The tighter `ppl_brake=300` combined with the entropy bonus caused a
runaway feedback loop:

1. Entropy bonus forces position head to explore → non-syntactic strategies
   sampled more.
2. Semantic strategy uses a placeholder synonym map that produces incoherent
   token sequences → PPL explodes.
3. `ppl_brake=300` fires immediately on the high-PPL episodes.
4. No PPO updates while braked → agents frozen.
5. Semantic collapse locks in (novelty bonus = 0 since `eta_novelty` is not
   penalising the frozen distribution).
6. PPL stays at 10 000 for remainder of run.

The semantic watermark strategy is fundamentally broken — the placeholder
`{i: i+1 for i in range(50)}` synonym map is a no-op that corrupts logits.

## Next Steps (Run 09)
1. **Fix or disable semantic strategy**: replace the placeholder synonym map
   with a real one, or remove `semantic` from `watermark_types` until it is
   implemented properly. This is the single highest-priority fix.
2. **Restore PPL brake**: revert `ppl_brake` to 500.0 and `ppl_threshold`
   to 200.0 — the tighter values offer no benefit while the semantic
   strategy is broken.
3. **Remove entropy bonus**: it is pushing the policy toward the broken
   semantic strategy. Remove until semantic is fixed.
4. **Keep position floor and history**: these worked — position diversity
   improved (three distinct `position_frac` values observed: 0.1, 0.45,
   0.75).