# GEN06 tactical audit against GEN05

**No new certified decision regression was found across the48 frozen holdout cases and22 existing site hard cases.** Site avoidable-loss flags improve7→6; missed available own-turn wins improve7→1. The GEN05 regression at8623/action33 is repaired through an actual terminal two-action continuation.

GEN06 SHA256: `123467a930e0598c464ae7694800ad565aeac71680b88dafc5f014bfe7cf50ac`.
GEN05 SHA256: `af45a5f85894e694f7ddab47811c4fb45b9f9536345d78b0f2fa3ef06c99c91e`.

All model evaluation waited for the native qualification owner's explicit PASS for this exact GEN06 hash. Opus owns model creation/export; this task wrote only independent diagnostic scripts and evidence.

## Existing internal gate replay

Before tactical inference, all400 histories in `logs/gen06-gate.json` were legally replayed to their recorded winners. Candidate and baseline checkpoint hashes match their release files. Verified result:297/400 (74.25%) against GEN05, Cyan167/200 (83.5%) and Magenta130/200 (65%), seeds906000–906199 both colors. Descriptive seed-paired bootstrap95% interval:70.25–78.25%. This is a reanalysis of existing selected-candidate evidence, not a fresh independent strength comparison or a selection-corrected confidence claim.

The original arena uses `seed*2+color`, not common RNG across swapped colors. Current engine/scene bundle hash is `2e6a926a41230225c7f7c58855b704d95c3c4fee1025211f860a34c766230ecd`. Gate JSON SHA256 is `7c2ffb88c67e516d9bf7d3eee7e98b4d9399a252724dc72461b1dd5f54dfa6b0`. Full per-history checks and current artifact hashes are in `../gen06-v-gen05/existing-gate-audit.json`. Legal replay does not regenerate the historical MCTS computation or establish a missing historical engine manifest.

## Frozen48

Both models independently ran the supplied MCTS100 with the frozen seed850001 and one ORT thread. All48 prior GEN05 selected actions reproduced exactly. Model and frozen-input hashes were unchanged before/after evaluation.

| Metric | GEN05 | GEN06 |
|---|---:|---:|
| Missed certified own-turn terminal wins |0|0|
| Avoidable certified next-turn terminal losses |0|0|
| Value MAE across32 exact-value states |0.337635|0.262984|

The16 unknown values remain excluded. The48 rotations/variants are correlated, constructed from12 authored seeds/11 motif groups and not proven opening-reachable. They remain quarantined TEST ONLY; no observation, certificate or failure was exported to training.

## Actual-site hard cases v1/v2/v3

All22 snapshots were matched byte-for-byte to their complete authoritative game replay states. Legal/safety masks matched the saved complete alternative lists. Selected unsafe actions' terminal witnesses were legally replayed. Both models used MCTS100 and the original seed0.

| Packet | States | GEN05 failures | GEN06 failures | Metric |
|---|---:|---:|---:|---|
|Loss v1|6|4|3|Avoidable enemy terminal win on next turn|
|Loss v2|4|2|2|Same|
|Loss v3|4|1|1|Same|
|Wins v1|1|0|0|Missed own-turn terminal win|
|Wins v2|4|4|1|Same|
|Wins v3|3|3|0|Same|

GEN06 newly avoids8185/action42 with `C5->B4`; GEN05's `C5->D4` admits the saved enemy terminal reply. Six previously missed winning starts improve:8948/27,8855/27,8623/33,10838/33,10838/37 and10762/19.

All seven GEN06-selected winning site starts, including the retained8228/35 win, were followed through using the model's actual second MCTS100 action. **All7/7 terminate with the current player winning**, as saved in `winning-followthrough.json`.

The explicitly requested8623/action33 sequence is **`E6->E5`, then `E5->G7`, terminal CYAN win**. GEN05 instead selected `E6->F5` and missed that available current-turn win.

Remaining avoidable-loss snapshots:7525/26,7658/55,8443/30,9366/36,8745/30 and11059/30. Remaining missed own-turn win:8948/47. Three loss snapshots without a certified safe alternative are excluded from the avoidable count; they do not authorize arbitrary policy labels. Related states from the same game are correlated. These familiar site packets may already be in shared training/re-search data, so the diagnostic improvement is not an unseen-generalization claim.

## Evidence and limits

`holdout-manifest.json`, `site-manifest.json`, the two `*-results.jsonl` files and two `*-summary.json` files preserve hashes, all decisions and the certificate comparisons. `audit.py` and `followthrough.py` are the independent read-only scripts. No raw input, Astra file, frozen membership, training data or model was edited.

This audit supports local tactical improvement with unresolved failures. Root separately authorized fresh fixed-GEN06 strength gates versus deployed GEN03 and fixed GEN02 after this diagnostic completed; those results belong to their own directories. Publication remains root's responsibility.
