# Six supplied matches: GEN07 findings for Opus

All six supplied files replay completely: **386 recorded actions**. The losing actors' **190 decision states** were scanned for a terminal win within the remainder of their current turn; none was missed. The loss investigation then classified complete legal alternatives at the critical states, including actual second actions.

The fixed counterfactual model is GEN07, SHA256 `8b23b6a7f80089aef590beb69a829738e3c8c66dcd5ccbb652820142dbabcebd`, using referee MCTS100. **GEN07 misses four distinct demonstrated corrections, covering five of the six games.** These are bounded defenses, not claims of eventual victory. No historical match is attributed to GEN07 without model metadata.

| Source game | Critical decision | Correction and evidence | GEN07 result |
|---|---|---|---|
| 18587 | 36: Trace G5→D2 | **Kernel E7→F6**, the only one of 9 legal choices preventing terminal defeat on the next enemy turn | Repeats Trace G5→D2 |
| 18972 | 62: Trace C4→C3 | **Trace C4→B4**, the only one of 10 legal choices preventing terminal defeat on the next enemy turn | Repeats Trace C4→C3 |
| 19255 | 37: Trace B3→B4 | **Trace B3→C3→C2**, or **Kernel B2→C2 then Trace B3→B2**. Both complete turns survive the next enemy turn; after the recorded first action, every second action loses | Chooses B3→B4 then B4→C4, which also loses |
| 19101 | 68: Trace F6→F5 | **Kernel E6→F5** is the sole defense through two enemy turns after recorded 67 Trace F5→F6 | Repeats the unsafe Trace return |
| 19490 | Same ending and decision 68 | Same exact state and certificate as 19101; all 59 complete next CYAN turns checked, with a defense against the following CYAN turn | Same output, counted as one distinct corrective state |
| 19040 | Final 93/94 already forced | Every complete legal alternative loses. The miner's deeper certificates extend forced defeat back to 85 within three enemy turns; these longer enumerations have source/quantifier review but have not been independently re-enumerated. **Earliest avoidable commitment before 85 remains UNKNOWN** | No invented corrective policy target |

Five games have user identity bound by export (6): MAGENTA in 18587/19101, CYAN in 18972/19040/19255, all losses. In 19490, CYAN wins and MAGENTA loses; the user side remains UNKNOWN because the TXT has no identity and export (6) does not contain it.

19101 and 19490 are different complete games. Their deterministic ending states coincide; the independent reviewer checked states 57–74. Preserve both episodes and use `site-ending-19101-19490` to link derived ending samples and split decisions. Repeated positions are not independent tactical evidence.

## What to review and integrate in the single learner

1. The three independently verified one-enemy-turn policy corrections are **published** in the same `data/ext_site_tactics_v1` dataset: **29 unique target observations**, with all original 26 chunks byte-identical. The existing 800-simulation teacher at 18587/36 put 97.0006% of its probability outside the certified support; that target is now projected onto the unique safe move. The other two new rows honestly use proof-only uniform targets. The single override manifest also covers `ext_site_losses_b_rs`; actual current loading of 2,490 rows applies 11 replacements. Policy mixtures 0, 0.25, 0.5 and 1 all retain zero excluded probability after Opus's `pval_g` fix. Manifest SHA256: `ecf432d91797181893f28d68e0b20f6e057650a683839eda1276c3fc858996de`; override SHA256: `4330e146bbf611bcacbaaac9af8b976878c7ea145887ad73f52e76c53de23822`. The 19255 target certifies a safe possible completion, so retain the explicit second-action continuations above. No resulting new-model strength improvement has yet been measured.
2. Review the independently verified deeper ending in `codex/research/loss-19490-review.md` and `codex/synthetic/validation/loss19490-independent/proof.json`. A reviewable three-row proposal is staged in `codex/synthetic/validation/deep19490-stage/`: pre68 is policy-only on index 2786, with unknown eventual value and length; pre71/72 are exact value −1 with observed remaining lengths 4/3 and no preferred policy. No live release is claimed for this staged proposal. It deliberately emphasizes a certified ending already present in the 19101 raw outcome data. Resolve per-head precedence before adding future contradictory teacher targets; do not force value-only rows into a policy-only override schema.
3. The trainer fixes acknowledged by Opus at 19:13 have been qualified independently where stated in the linked receipts. The separate `has_length` train/holdout gradients passed, and the previously quarantined **28-state curriculum is now published** at `data/ext_astra_site_gen05_v1/chunk0000.npz`, SHA256 `9e0c9989b16589411681cc8eb65e1e969a5fd9ff00a6122d7ffe9cbce764e6e1`. Its 16 exact values, 14 policies, 4 known lengths and 24 unknown lengths remain separate and correctly masked. No historical checkpoint is retroactively described as using the fixes.
4. The 168 accidental repeated raw outcomes from 18254/18425/18587 were preserved under `data/_dup_excluded/` and excluded from subsequent intake by Opus. `_rs` policy supervision remains. The historical GEN08/GEN08v duplicate-intake evidence remains documented in `codex/research/export6-training-dedup-audit.md`; the current corrected intake is a separate result.

## Reproducible evidence

- Five-file raw/export bindings, full replay, action partitions, source hashes and typed terminal witnesses: `codex/matches/export6/user-five-v1/REPORT.md` and adjacent `<ID>-proof.json` / `<ID>-hard.npz` files.
- Sixth-file raw receipt and full replay: `codex/matches/19490/source-receipt.json`, `replay.json`, `deep-proofs.json`, and `critical-states.npz`.
- Independent first-five review: `codex/research/user-six-independent-proof-review.md`, SHA256 `5eba20b306ecca4612f71a58ea48933348d9d8db4661c5702f05db9d20a67986`.
- Independent 19490 full-environment proof: `codex/synthetic/validation/loss19490-independent/README.md` and `proof.json`.
- Exact GEN07 model queries and actual followthrough: `codex/arena/gen07-user-six-audit/REPORT.md`, `combined-summary.json`, JSONL results and manifests. Twelve direct queries plus two reused early aliases cover 14 reported positions and 10 distinct observations. No old-model inference, strength games, training or model export was performed for this audit.
- Shared Opus communication: `codex/coordination.md`. This document consolidates the incremental handoffs into one review entry point; it does not claim a new model has already learned these repairs or won the tournament.
