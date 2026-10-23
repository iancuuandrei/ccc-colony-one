# Export6 training duplicate audit

FAIL for unintended duplicate raw-outcome weighting: the same three complete games occur in both `data/ext_site_v1` and `data/ext_site_losses_b`. They are identical in every normalized shared training field. Both completed GEN08 checkpoint manifests explicitly list all four source files, so this is actual intake evidence, beyond a possible future folder collision.

This audit was read-only except for this report. Match analysis held its manifest stable during the checks. No trainer/data/metadata edits, model construction, training, inference, export or tactical search occurred.

## Exact duplicate identities

`data/ext_site_losses_b/chunk0000.npz` SHA256:
`50c1ca9e10638c71577fd5968dfd47f1ac9b189ae56f93fc239d8487c75acc46`.
Its manifest SHA256:
`ce68f62b58daa5e775dcb20cd36ff8a2b55dfbcd8b861dbf8e4a97f0b71f54ea`.

The following ranges are zero-based, half-open slices in that 168-row chunk:

| Game / canonical group | Duplicate slice | Exact canonical raw copy | Rows |
|---|---|---|---:|
|18254 / site-game-18254|`[0:78]`|`data/ext_site_v1/chunk-site-18254.npz`|78|
|18425 / site-game-18425|`[78:130]`|`data/ext_site_v1/chunk-site-18425.npz`|52|
|18587 / site-game-18587|`[130:168]`|`data/ext_site_v1/chunk-site-18587.npz`|38|

Standalone NPZ file hashes, in the same order:

- 18254: `a69feaa1e37526ebf7d7994c94e6e0d2656e2667006bdd6ed1a9b5cd32fcc312`.
- 18425: `fb2295e34fdba54d4d2662bd11ba6c84693da6e3637804f1e42a44666cc6ad94`.
- 18587: `184cf6e310638f6f1a0b67e3a0a4b00519ea2219a085bbd8676718b8c28bf3df`.

Raw protocol hashes agree with both manifests:

- 18254: `ba024c2862b2ae48c55a6efcdf8fb16270c73fc3d1fe465cbddad62684f7aba3`.
- 18425: `76658f52f24ef4c42167f9e1619d53523c21d5cbbb07eacf6050f901706e496e`.
- 18587: `cd87e59c2381f8aae7b420417ad182b4d800ed52dc73776815ca562aeaec4652`.

The export6 JSON source SHA256 is `f6351e120c8f81950b460e67761cb0930e8e705db52b4c06e01b715ad515a731`. The stable `ext_site_v1/manifest.json` SHA256 was `8de0b5c7dd5c810e11bd117d96fa4642aa126d265c653408a5c11bfaa49f62aa` before and after the audit, with134 locally stored episodes /4,837 rows.

All nine normalized contract fields compare exactly, including action order: `obs`, `pidx`, `pval`, `z`, `q`, `has_pol`, `q_valid`, `has_value`, `left`. The standalone raw chunks omit `has_value`; the loader's documented default True equals the explicit True in `ext_site_losses_b`. Thus differing ZIP contents/schema defaults do not make these independent examples. Values and lengths agree; this is duplicate weighting, not conflicting labels.

Canonical semantic slice SHA256 values, in game order:

- 18254: `b528c899c19286566efca6b2ec0c6f823dd983672899dfb862847bd6fd9d557d`.
- 18425: `42afe153246431bc9fc1255cbc2b548536747bbfd66c623c634f7b75c9babb99`.
- 18587: `abf9fc23741b5db776e2b388cc14caac0267b51681d9d86124f9d512b45ebf3a`.

Semantic digest construction: normalize `obs/pval` to float16, `pidx` to int16, `z/q/left` to float32 and the three masks to bool; missing `has_value` is True. Iterate fields in lexicographic order, hashing `name|numpy_dtype|shape\n` then contiguous C-order bytes. Each standalone episode and its corresponding combined slice produce the same digest.

## Executed intake evidence

Current `learner.py` SHA256 `e6d448310977a51b66523fab9576145d67428125c12cafbd6322b49338a3954a` includes every `data/ext_*` directory with chunks and passes all of them to the trainer. Current `train.load` concatenates without episode-group or protocol deduplication.

The actual unmodified current numpy loader was executed on all external chunks:8,103 rows,10 authoritative policy replacements, and zero next-policy auxiliary targets for external rows. These8,103 rows include both raw copies of the168 episode positions. The external folder counts were:

| Folder | Rows |
|---|---:|
|ext_astra_policy_v1|144|
|ext_astra_site_gen02_v1|55|
|ext_astra_synthetic_v1|80|
|ext_site_deep10910_v1|2|
|ext_site_deep14958_v1|2|
|ext_site_gen05_losses|328|
|ext_site_gen05_losses_rs|328|
|ext_site_losses_b|168|
|ext_site_losses_b_rs|168|
|ext_site_tactics_v1|26|
|ext_site_v1|4,837|
|ext_site_v1_rs|1,965|

Both `ckpt/gen08.pt` and `ckpt/gen08v.pt` were read on CPU without constructing a model. Their `train_files` include `data/ext_site_losses_b/chunk0000.npz` and all three standalone files in the table above. Each records615,822 total training positions and `ext_frac=.06`. GEN08 records `policy_mix=0`; GEN08v records `.5`. Their command logs agree. Checkpoint file lists do not contain historical per-input content hashes; exact byte comparisons here bind the current preserved chunks.

The `_rs` folders are separate policy supervision and are not classified as accidental duplicate raw outcomes merely because observations overlap. Likewise, explicitly certified tactical/value reinforcement has separate declared semantics and is outside this raw-episode dedup correction.

## Current override and length status

`train.py` is still SHA256 `8e90cad91be762cbaad11c80d98466699784969a8380def045fb86303baf7210`, identical to the earlier executed regression probes. Those gradient/mix tests were not repeated blindly.

Source inspection confirms neither reported fix is present: `apply_overrides` replaces only `pidx/pval`, leaving the pre-replacement `pval_g`; and `has_length` is absent from KEYS while length auxiliary still uses `has_value`. The single26-entry policy manifest targets `ext_site_v1_rs` and `ext_site_gen05_losses_rs`, and actual primary-target routing replaces10 rows. It does not target raw outcome folders or newly created `ext_site_losses_b_rs` without corresponding certified rules.

The already demonstrated nonzero-policy-mix regression is therefore relevant to an actual configured GEN08v branch, not only a proposed future option. Its checkpoint records `.5`. Prior executed evidence remains in `codex/synthetic/validation/site-gen05-v1/policy-mix-regression.json` and `data/ext_site_tactics_v1/validation.json`; no new numerical claim about GEN08v's final behavior is inferred from the target defect alone. The28-row Astra curriculum remains unpublished pending separate length masking.

## Minimal safe Opus handoff

Use `ext_site_v1` as the canonical raw-outcome source for these three episodes and exclude only raw `data/ext_site_losses_b` from subsequent learner intake. Preserve its chunks/manifests and retain `data/ext_site_losses_b_rs` as policy supervision. This removes168 unintended repeated raw rows from the current external pool:8,103 becomes7,935. Existing checkpoints remain historical artifacts; do not relabel them as trained on deduplicated inputs.

For future imports, resolve complete-episode aliases by canonical `site-game-ID` plus exact protocol identity before constructing raw input lists. Do not globally deduplicate observation hashes: legitimate games can share openings, and policy/value targets can intentionally differ by source authority. Keep all aliases and counterfactual relatives together in any future group split. Match analysis already preserves this provenance; no source deletion or simultaneous metadata mutation is needed for the minimal learner-side exclusion.

Independently fix authoritative `pval_g` replacement before any nonzero policy mixture, and propagate a separate `has_length` mask before releasing constructed exact-value rows. These remain Opus-owned source changes.

## Superseding qualification — 2026-10-23 19:15 Europe/Bucharest

PASS for current dedup intake and independent length masking under new `train.py` SHA256 `b2d4bdf6a0b17ff7c225c3c67ed2cfc562bf643224d557d770b40b68e39f6bb3`. Earlier sections record the previous snapshot and historical GEN08/G08v intake; their duplication evidence is preserved. They no longer describe the current dedup or length-mask state.

The raw duplicate folder now resides at `data/_dup_excluded/ext_site_losses_b`. Its chunk still hashes to `50c1ca9e10638c71577fd5968dfd47f1ac9b189ae56f93fc239d8487c75acc46`; `data/ext_site_losses_b` is absent. Current learner glob excludes the preserved folder and retains `ext_site_losses_b_rs`. Executing the current real loader on all eligible external chunks yielded7,935 rows before the new28-state release, exactly168 fewer than the previous8,103. Current external rows have no fabricated next-policy adjacency.

New default semantics were executed against existing files: legacy real selfplay retains `has_length=True`; undeclared external lengths default False; explicitly declared external True is retained. Thus older external exact values remain useful while unjustified length auxiliaries are disabled.

The actual current nested loss functions passed CPU leaf-gradient checks in both train and holdout modes. All14 constructed exact-value/unknown-length rows have exactly zero length gradient and retained nonzero value gradient. Two actual rows with UNKNOWN value and observed length retain nonzero length gradient while value gradient is zero. Known length loss normalization matches an independent masked mean. Invalid Q and disabled policy/next-policy gradients remain zero. The actual holdout reporting branch was executed and passed `H['has_length']` to the loss; every production train/holdout loss call explicitly supplies its length mask.

The previously proved28-state curriculum is now atomically published at `data/ext_astra_site_gen05_v1/chunk0000.npz`, SHA256 `9e0c9989b16589411681cc8eb65e1e969a5fd9ff00a6122d7ffe9cbce764e6e1`. All16 exact values and14 policy targets are retained, with four observed lengths4/3/4/3 and24 unknown lengths explicitly masked. Its frozen staged chunk, provenance and all original proof/source files are unchanged. Actual published-path loader PASS; no next-policy auxiliary. External intake becomes7,963 rows immediately after this release, before any later independent policy extension.

Executed evidence and reproducible helper:

- `codex/synthetic/validation/qualify_site_gen05_loader.py`
- `codex/synthetic/validation/site-gen05-v1/loader-length-qualified.json`
- `codex/synthetic/validation/site-gen05-v1/published-loader.json`

Current source also contains the `pval_g` override assignment; independent execution of policy mixtures belongs to `onnx_validation` and was not duplicated here. A remaining nonblocking metric limitation is unchanged: mixed-holdout value aggregation weights batch means by all rows rather than value-valid counts. Ordinary current selfplay holdout rows have valid outcomes, so this does not invalidate the executed mask qualification or new release.
