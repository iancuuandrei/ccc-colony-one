# GEN05 curriculum qualification

Independent engine/source qualification PASS28/28. `qualify_site_gen05.py` recomputed each row's complete own-turn winning mask and, when no own-turn win exists, complete minimax classification through the next opponent turn using public full `env.clone()/step()`. It does not import the producer's tactical oracle.

The four actual states reproduce fresh complete raw-match replays of14594/14621/14878. Their observed remaining episode counts are4/3/4/3. All24 constructed states reproduce their declared piece/clock/action-budget modifications and rotations; opening reachability is unproved. Legal actions, typed pieces, complete policy denominator, policy/value masks and group aliases match. All28 have zero D4 overlap with the unchanged frozen TEST_ONLY holdout.

The action-budget counterexample is independently confirmed: actual14878/a36 with one action permits only Kernel F3-G2 to survive the next enemy turn; the same board with two mandatory actions permits F3-F2 or F3-G3 as safe first moves, while F3-G2 becomes unsafe. Survival is bounded and is not a full-game win claim.

The corrected chunk remains in `export_stage/`, pending trainer support and executed gradient qualification for independent `has_length`. It retains all16 exact values and14 policies, has four observed length targets, and masks the24 unknown constructed lengths. The source packet is unchanged. Do not train this staged file through a generic non-ext path: unrelated rows must not acquire next-policy auxiliary targets.

`report.json` contains per-row evidence, hashes, source groups and length metadata. `policy-mix-regression.json` separately records the current trainer regression: its pval_g branch bypasses authoritative override replacement when policy mixing is enabled. Both required trainer fixes were sent to Opus in `codex/coordination.md`; Codex did not edit his sources.
