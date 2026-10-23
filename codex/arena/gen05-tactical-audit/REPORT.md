# GEN05 tactical comparison against deployed GEN03

Candidate GEN05 SHA256 `af45a5f85894e694f7ddab47811c4fb45b9f9536345d78b0f2fa3ef06c99c91e`; deployed GEN03 SHA256 `0cf937446ba89dc9b5b3652edbcb4025030bc785e924551782f4b2d298ebe96f`.

**Three actual-site loss decisions improve; one previously found immediate win regresses.** This tradeoff is independent of the shared strength-gate result and must accompany any publication decision.

No model evaluation started until the native qualification owner confirmed PASS for the exact GEN05 hash. Both tactical comparisons use the supplied exact MCTS100 and one ORT inference thread. Only this audit directory receives outputs. No model, training artifact or quarantined input was changed.

## Frozen48

Both GEN03 and GEN05 were run on every unchanged holdout state with its original deterministic seed850001. All48 prior GEN03 selected actions reproduced exactly. Before/after input and model hashes matched.

| Metric | GEN03 | GEN05 |
|---|---:|---:|
| Missed available own-turn terminal wins | 0 | 0 |
| Avoidable certified next-turn losses | 0 | 0 |
| MAE across32 exact-value states | 0.382681 | 0.337635 |

The16 unknown values remain excluded. The48 variants come from12 authored seeds/11 motif groups and have unproven opening reachability; they are correlated diagnostics, not independent tournament matches. The cohort remains QUARANTINED TEST ONLY and was not exported to training.

## Actual-site hard decisions

All22 current v1/v2/v3 hard cases were replayed from complete actual protocols. NPZ observations matched their exact engine states byte-for-byte, saved legal/safety masks matched complete alternative lists, and selected losing witnesses were independently executed. Both models used the original probe seed0. These familiar diagnostic cases may already have entered shared training/re-search; no unseen-generalization claim follows.

| Packet | States | GEN03 failures | GEN05 failures | Metric |
|---|---:|---:|---:|---|
| Loss v1 |6|5|4|Avoidable enemy terminal win on next turn|
| Loss v2 |4|3|2|Same|
| Loss v3 |4|2|1|Same|
| Wins v1 |1|0|0|Missed available own-turn terminal win|
| Wins v2 |4|3|4|Same|
| Wins v3 |3|3|3|Same|

GEN05 improvements:

- **8079/action30:** `C3->C4` is certified horizon-safe; GEN03's `C3->C2` permits the saved terminal enemy reply.
- **9366/action52:** `E6->F6` is certified safe; GEN03's `E6->E4` allows the saved last-Daemon capture/bare-Kernel terminal witness.
- **9928/action28:** `D5->C6` is certified safe; GEN03's `D5->B3` bodyguard redirect allows the saved terminal Kernel continuation.

GEN05 regression:

- **8623/action33:** GEN03 selects `E6->E5`, retaining an own-turn terminal win (its second action `E5->G7` was independently followed through in the preceding GEN03 audit). GEN05 selects `E6->F5`, after which no own-turn terminal win remains. This is a missed immediate win; this audit does not claim it loses the eventual full game.

Overall avoidable-loss flags decrease10→7 while missed-win flags increase6→7. Three states with no certified safe option are not counted as avoidable. Nearby states from a single game remain correlated; counts are diagnostic state counts.

Full decisions, legal action indices, values, selected certificates and witness replays are in `site-results.jsonl`; cohort totals are in `site-summary.json`. The holdout equivalents are in `holdout-results.jsonl` and `holdout-summary.json`. Each manifest binds model/input/harness hashes. The audit creates no strength games and makes no automatic promotion decision.
