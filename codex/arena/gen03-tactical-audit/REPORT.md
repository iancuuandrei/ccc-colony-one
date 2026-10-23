# GEN03 tactical audit against fixed GEN02

**No new certified move-selection regressions were found across 70 diagnostic states.** GEN03 preserves the frozen 48-state holdout's zero decision failures and improves three of 22 actual-site hard decisions. Important actual-site failures remain. This is not proof of complete tactical remediation or tournament strength.

GEN03: `release/gen03/model.onnx`, SHA256 `0cf937446ba89dc9b5b3652edbcb4025030bc785e924551782f4b2d298ebe96f`.

Fixed GEN02: `release/gen02/model.onnx`, SHA256 `579d3ec01da7328dc706003b8dcc5eba357bb74902252187e45633ca0595a9a0`.

## Frozen holdout

Both models were independently evaluated on all 48 unchanged cases using the supplied exact MCTS100, one ORT thread and seed850001, matching the frozen protocol. All 48 saved GEN02 action choices reproduced exactly. Input and model hashes matched before/after evaluation. Unknown certified values remained unknown and were excluded from value MAE.

| Metric | GEN02 | GEN03 |
|---|---:|---:|
| Missed available certified own-turn terminal wins | 0 | 0 |
| Avoidable certified next-turn losses | 0 | 0 |
| Value MAE over 32 exact-value states | 0.403069 | 0.382681 |

Two action selections changed, with no new certificate violation. There are pointwise valuation regressions despite lower aggregate MAE: `p-8a983f5658dad97a` remains an exact loss but value moves from −0.757810 to −0.524709; `p-defb6e6592ce3332` remains an exact win but value moves from −0.233866 to −0.453842. Both retain the same selected action. These are valuation diagnostics, not newly observed move-selection failures.

This holdout has 12 authored seeds, 11 motif groups and correlated rotations. Its constructed states are not proven reachable from the opening. It remains **QUARANTINED TEST ONLY**: no observations, certificates, or failures were exported to training. Results live only in this audit directory. No independent-game confidence interval is appropriate for its 48 variants.

## Actual GEN02 site packets, including new v3

All available v1/v2/v3 hard loss and win NPZs were included: 14 loss states and eight missed-win states. Each observation was matched byte-for-byte to its authoritative full-game replay state. Saved legal/safety masks were checked against the complete saved alternative lists. Recorded winning witnesses and selected unsafe-action witnesses were legally replayed. Both models used exact MCTS100 and seed0, matching the original site probes.

| Packet | States | GEN02 failures | GEN03 failures | Failure definition |
|---|---:|---:|---:|---|
| Loss v1 | 6 | 5 | 5 | Avoidable certified next-turn terminal loss |
| Loss v2 | 4 | 4 | 3 | Same |
| Loss v3 | 4 | 2 | 2 | Same |
| Wins v1 | 1 | 1 | 0 | Missed available own-turn terminal win |
| Wins v2 | 4 | 4 | 3 | Same |
| Wins v3 | 3 | 3 | 3 | Same |

Three loss snapshots have no certified safe alternative and are excluded from the avoidable count. Nearby states from one game are related diagnostic observations, not independent matches. The site packets may already be present in shared training/re-search data; these results do not establish unseen-data generalization.

The three improvements:

- **8944/action34:** GEN02 chooses Ice `E5->G7`, loses its last Ice to the saved Kernel witness and reaches a losing bare-kernel endgame. GEN03 chooses certified-safe `E5->D4`. Safety here means avoiding the enemy's immediate next-turn terminal win, not eventual victory.
- **8228/action35:** GEN03 selects `F6->E5`, then actual MCTS100 selects `E5->D4`, terminating with MAGENTA winning. GEN02's `G5->F5` discards the available current-turn win.
- **8623/action33:** GEN03 selects `E6->E5`, then actual MCTS100 selects `E5->G7`, terminating with CYAN winning. This is a different valid winning start from the originally saved `E6->E7` witness. GEN02's `E6->F5` misses the win.

Remaining avoidable loss cases for GEN03: v1 `7525/26`, `7658/55`, `8079/30`, `8185/42`, `8443/30`; v2 `9366/36`, `9366/52`, `8745/30`; v3 `11059/30`, `9928/28`.

Remaining missed wins for GEN03: v2 `8948/27`, `8948/47`, `8855/27`; v3 `10838/33`, `10838/37`, `10762/19`.

Two-action loss follow-through was also evaluated on `7658/55` and `10910/43`. Both models select the same first and second actions and allow the certified enemy terminal reply. The latter is already a no-safe-alternative state at this horizon, so it is not counted as a new avoidable failure.

## Evidence and scope

- `holdout-manifest.json`, `holdout-results.jsonl`, `holdout-summary.json`: frozen input hashes, every model decision, exact-value metrics and baseline reproduction.
- `site-manifest.json`, `site-results.jsonl`, `site-summary.json`: immutable packet/raw-game hashes, decisions, selected certificates and cohort totals.
- `improved-win-followthrough.json`: actual second MCTS actions completing the two newly found wins.
- `provenance.json`: model identities and all current engine/scene/helper/oracle hashes.
- `audit.py`, `followthrough.py`: reproducible read-only scripts; all writes are confined to this owned audit directory.

No model, training data, Astra file, frozen input, strength game or upload was created or changed. Runtime qualification and publication remain root's responsibility. On this bounded diagnostic evidence there is no newly observed tactical decision regression to block GEN03; the remaining actual-site failures should still be treated as unresolved by the shared model owner.
