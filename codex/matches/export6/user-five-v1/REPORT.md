# Five supplied losses: exact engine findings

All five supplied TXT files match export (6) protocols exactly by lines and replay completely with the supplied engine. Export (6) binds user identity; it contains no per-match model hash. Historical source models remain **UNKNOWN**. GEN07 counterfactual evaluation belongs to `codex/arena/gen07-user-six-audit`; no GEN06 decisions were rerun after the priority change.

| Game | User side | Total actions | User actions scanned | Supplied TXT SHA256 |
|---|---|---:|---:|---|
|18587|MAGENTA|38|18|`cd87e59c2381f8aae7b420417ad182b4d800ed52dc73776815ca562aeaec4652`|
|18972|CYAN|64|32|`5d1f3487cdee120770c30e8f70d13c69051b1fa37e024f771a261e495565dbba`|
|19040|CYAN|96|48|`b954ce8e23523bba781dc9fa19dedaa19499d98d419e470316d3c0aac43a429c`|
|19101|MAGENTA|74|36|`a545ac46fc49db2009957594f2ec55adea6d7e6e0b0ab2ccbaed1d9a2e282289`|
|19255|CYAN|40|20|`4f32f8f79eebc7f83c9fe02d1776740853696cc44391031084fbca89e7748d2a`|

Every one of the **154 user actions** was checked for a missed terminal win within the remaining current turn: **none found**. Last actions were checked for allowing a terminal win on the next enemy turn; preceding first actions were exhaustively checked when the last-action position was already forced.

## Certified corrective decisions

| Game/action | Actual commitment | Complete safe own continuation | Scope |
|---|---|---|---|
|18587/36|Trace G5→D2|Kernel E7→F6|Unique safe last action, 1/9 legal|
|18972/62|Trace C4→C3|Trace C4→B4|Unique safe last action, 1/10 legal|
|19255/37|Trace B3→B4|Trace B3→C3, then C3→C2; **or** Kernel B2→C2, then Trace B3→B2|Two safe first actions out of four|
|19101/68|Trace F6→F5|After the recorded first action 67 Trace F5→F6, play Kernel E6→F5|Reuse the identical 19490 two-enemy-turn partition|

“Safe” means some complete own continuation avoids terminal defeat through the stated bounded horizon, against every enemy reply. It is not an eventual-win certificate.

In 18587, the actual Trace shot leaves the Kernel exposed to Ice C4→C5→E7. In 18972, the Trace's move to C3 allows Ice E4→E5→C5 to capture the Kernel. In 19255, the bad first action leaves **zero** safe last actions at 38. The recorded Daemon E3→C3→C2 shot wins; an independent Ice F5→E4→C2 capture also wins. The two safe continuations preserve useful Kernel protection.

The GEN07 owner reports that pinned model SHA256 `8b23b6a7f80089aef590beb69a829738e3c8c66dcd5ccbb652820142dbabcebd` still chooses the bad decisions in 18587/36, 18972/62 and 19255/37, and repeats the unsafe Trace choice at the shared 19101/19490 action 68. Those are counterfactual findings, not identification of the models that played the original games.

## Forced tails and the earlier 19040 investigation

19101 actions 71/72 have zero safe choices (0/7 first actions; 0/5 last actions). They are already forced and must not receive an invented corrective policy target.

19040 actions 93/94 are also already forced (0/6 first actions; 0/10 last actions). A deeper composed proof finds every choice at 89/90 loses within two enemy turns. At action 86, Kernel C5→C4 and the **actual** Trace C3→C4 survive that shorter horizon; the other five choices do not. Extending the horizon proves every legal action 86 and every complete own continuation from action 85 loses within **three** enemy turns. Thus the actual 86 move cannot be called an avoidable error from these certificates. The earliest avoidable commitment before 85 remains **UNKNOWN**.

The main five packets were independently replayed and their masks verified by the read-only reviewer. The additional 19040 two-/three-turn outputs are miner evidence; the reviewer checked their quantifier structure but did not independently re-enumerate those longer certificates. See `codex/research/user-six-independent-proof-review.md`.

## Repeated ending, distinct games

19101 and 19490 have different raw hashes and different earlier trajectories. Full deterministic states, observation bytes and legal sets match at 63/65/67/68/71/72; the independent reviewer confirms equality throughout 57–74. The comparison includes board internals, piece movement flags, source scene, side, actions left, round, action number and terminal fields. Therefore 19490's exact deeper action 68 proof applies to 19101 without another search.

`19101-19490-state-equivalence.json` records the comparison. Preserve both source game IDs; link ending-derived supervision and split decisions with equivalence group `site-ending-19101-19490`. These are two source games and one repeated tactical ending, not two independent demonstrations of the same ending.

## Immutable packet paths

Each `<ID>-proof.json` in this directory contains raw/export hashes, identity, full replay status, all legal alternatives, typed winning witnesses and complete safe second-action continuations. Its `<ID>-hard.npz` contains `obs`, `legal_mask`, `safe_start_mask`, `mask_kind`, `match`, `ply`. Empty support denotes no admissible correction within that horizon.

| NPZ | SHA256 |
|---|---|
|18587-hard.npz|`a1faa64456bd620108337c8d77201ffb1457f760d94b32effb8af82df4399752`|
|18972-hard.npz|`119c118ecd246b3ea6b20e4c43cb9cf4985a555172acea4000ac027d886fc2fb`|
|19040-hard.npz|`ab54cba2c963b5fa509c607cf1be6932f3c321e7af437cdcd122a644aa5a6f7a`|
|19101-hard.npz|`82064934b50cea38a23a26b3758ed8258824745ffe2089cb04161492c848f263`|
|19255-hard.npz|`42226f3c87829e6a49003ce870c5517d97537605c1559eed7031a7e9c024bd22`|

Earlier 19040 proof files are `19040-deep-{86,89,90}.json` and `19040-deep3-{85,86}.json`. Shared ending proofs remain owned under `codex/matches/19490`; they were read without modifying them.

## Outcome corpus and preservation-safe duplicate remedy

Export (6), SHA256 `f6351e120c8f81950b460e67761cb0930e8e705db52b4c06e01b715ad515a731`, added 23 complete games / 1,060 rows: 9 user losses and 14 wins. The current corpus has 134 local episodes / 4,837 rows plus 7 external aliases / 328 rows: **141 unique complete games / 5,165 rows**. Validation passed; manifest SHA256 `8de0b5c7dd5c810e11bd117d96fa4642aa126d265c653408a5c11bfaa49f62aa`. Earlier 111 local chunks and external chunks remain unchanged. Seven metadata-only records globally remain excluded.

Independent audit proved that `ext_site_losses_b` repeats the 168 outcome rows of games 18254/18425/18587 already present in `ext_site_v1`, and both were in the recent learner intake. The minimal owner action is to exclude only raw `ext_site_losses_b` from future intake while retaining its bytes, receipts and `_rs` teacher supervision. No chunks were deleted and no concurrent dedup metadata was changed here. Details: `codex/research/export6-training-dedup-audit.md`.
