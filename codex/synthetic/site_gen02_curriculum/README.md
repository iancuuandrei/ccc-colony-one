# GEN02 tactical training curriculum

This is a **training candidate awaiting independent qualification**, separate from the frozen `holdout_v1` test cohort. It contains 15 replay-derived positions and 40 explicitly constructed positions from 20 teacher-authored perturbations, across 12 source-game groups. Constructed positions have valid engine state and legal moves; reachability from the standard opening is **not proven**.

`proposals.json` is the original teacher-authored scenario design and factual explanation record. `sources/` preserves the four input NPZ files with asserted hashes. `states.jsonl` and `packet.json` contain sparse exact states, readable pieces and legal moves, complete first-action classifications, winning witnesses and teacher explanations. `certified_states.npz` contains observations and all numerical targets. `MANIFEST.json` freezes their identities and the exact engine/oracle revisions before the separate baseline evaluation.

## Target contract

- Nine exact current-turn wins have value +1. Three positions where every current-turn continuation admits an enemy terminal win before the next own turn have value -1.
- The other 43 full-game values are NaN with `value_valid=false`. Bounded safety is never a positive game-outcome label.
- Fifty-two rows have a policy target: uniform mass over **all** certified winning first actions, or over **all** bounded-safe first actions when no current-turn win exists. A safe first action with two actions remaining means **at least one** safe second action exists; it does not endorse every completion.
- The three forced-loss positions have no policy preference: zero policy target and `policy_valid=false`. All 4096-action masks include the complete legal set. Excluded legal actions receive zero target mass.
- `q_valid=false` throughout. No baseline search value is presented as an exact teacher outcome. All emitted safety classifications are complete; none hit the search cap. Safety is intentionally not evaluated for rows already certified to win this turn.
- Keep each `site-game-<id>` group together with its replay rows in other corpora. Rotations and counterfactual descendants cannot cross the game-level split. Deduplicate across sources before training.

## Baseline evidence

`evaluation_gen02.json` uses only GEN02 SHA `579d3ec01da7328dc706003b8dcc5eba357bb74902252187e45633ca0595a9a0`, the supplied exact MCTS at 100 simulations, one CPU process and one ORT thread. All frozen rows were retained regardless of results.

| Cohort | States | Missed current-turn wins | Avoidable next-turn losses | Already forced losses |
|---|---:|---:|---:|---:|
| Actual observed | 15 | 5 | 9 | 1 |
| Constructed | 40 | 0 | 10 | 2 |
| Total | 55 | 5 | 19 | 3 |

These deliberately selected training cases do not estimate held-out strength or a full-game win rate. `failure_certificates.json` replays every demonstrated error's terminal witness using the real `env.step`; for unsafe first actions with two moves remaining, it covers every legal second action and supplies one bounded-safe alternative.

## What the contrast pairs teach

1. In actual game 8079, Trace C3-C2 retains a guard but loses: Daemon C5-C2 relocates while capturing, then shoots Kernel B3. Trace C3-C4 is uniquely safe because the adjacent Daemon attack shoots without relocating. Moving the enemy Daemon from C5 to C6 flips the target: C3-C2 becomes uniquely safe. Guard count alone cannot express this distinction.
2. In actual game 7525, Ice E5-G7 spends the final action sacrificing an enemy guard while the attacker stays E5, allowing a Ping attack on its own Kernel. Giving the same position two own actions makes E5-G7 a forced win: repeat the Kernel attack after removing the guard. The action count changes the correct target completely.
3. In actual game 8443, Ice B3-D5 abandons protection against Trace F6-F5-C2. Moving enemy Trace to F7 or E6 makes that material capture bounded-safe. The correct distinction is the opponent's reachable shot geometry, not a general prohibition on capturing with a guard.
4. In actual game 9366 at action 52, capturing the enemy Daemon exposes the last own non-Kernel piece to a bare-Kernel terminal transition. Centralizing the own Kernel changes the adjudication and guarding geometry enough that E6-E4 becomes bounded-safe. With two own actions in the original geometry, a different first action, E6-C6, yields a current-turn win.

All five missed wins have a GEN02 value of approximately +0.99, but search still chooses a first action outside the exact winning set. This is a policy-selection failure in positions whose value is already recognized as winning; increasing the value target alone would not supply the missing first-action supervision.

The code here creates data, certificates and diagnostic results only. It does not train/export a model or release files into `data/ext`. Opus remains the sole model owner; release requires the parent's independent qualification workflow.
