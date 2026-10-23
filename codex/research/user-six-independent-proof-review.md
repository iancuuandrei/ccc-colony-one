# Independent exact-proof review: user's six matches, GEN07 focus

Status: **PASS for the five supplied diagnostic packets and the stated bounded action sets.** The first four nonduplicated game mechanisms were independently enumerated through `env.clone/step`; 19101's repeated ending is bound to Astra's already independently verified 19490 proof, without repeating that deep enumeration. No source, data, model, trainer or exported candidate was changed. No model inference/search/arena was run in this review.

Current requested counterfactual baseline is GEN07 `release/gen07/model.onnx`, SHA256 `8b23b6a7f80089aef590beb69a829738e3c8c66dcd5ccbb652820142dbabcebd`, confirmed on disk. This does not identify the historical source model of the supplied text games. First-five user actor attribution comes from the supplied export-backed packets;19490 user/model identity remains UNKNOWN.

## Concrete findings and admissible targets

All paths below are in real board coordinates. Canonical policy indices use the actual moving side.

| Match / state | Full legal set | Independent classification | Valid interpretation |
|---|---:|---|---|
|18587 before 36, user MAGENTA loss|9|Exactly1 safe: Kernel E7→F6, index3306. Actual Trace G5→D2 unsafe.|Recoverable through the immediately following enemy turn; sole bounded-safe policy target. Eventual value UNKNOWN.|
|18972 before 62, user CYAN loss|10|Exactly1 safe: Trace C4→B4, index2209. Actual C4→C3 unsafe.|Same one-enemy-turn policy scope; no fabricated positive value.|
|19255 before 37, user CYAN loss|4|Exactly2 first actions have a safe completion: Trace B3→C3 (2666), then C3→C2; or Kernel B2→C2 (3186), then Trace B3→B2.|Recoverable before committing action37. These are existential own completions, not a guarantee that any second action is safe.|
|19255 before 38 after actual 37 Trace B3→B4|6|All 6 actions allow enemy terminal victory.|Already forced at 38; do not choose an arbitrary corrective policy there.|
|19040 before 93|6|All 49 actual own-turn completions allow enemy terminal victory.|Already forced before 93; zero corrective support.|
|19040 before 94|10|All 10 final actions allow enemy terminal victory.|Already forced; no corrective policy.|
|19101 before 71 /72|7 /5|Full observations/legal masks equal the corresponding independently proved 19490 states; all choices/completions lose next enemy turn.|Already forced ending. Reuse the bound proof; do not relabel a final move as an avoidable error.|

For 18587 the unsafe move allows Ice C4→C5→E7 to capture the Kernel. For 18972 the actual Trace move allows Ice E4→E5→C5. In 19255, the actual first move changes protection before any final action; the supplied Daemon E3→C3→C2 shot and alternate Ice F5→E4→C2 capture are legal terminal witnesses. Independent classification agrees with every packet mask.

The first safe19255 start has 6 possible own completions, and the second has 7; some completions after each are unsafe. Only the demonstrated appropriate second choices carry the safety argument. The actual first move has no safe completion, so changing only its subsequent choice cannot repair the observed turn. Current GEN07's actual first-plus-second choices are being measured by `candidate_arena`; this proof pass does not substitute a hypothetical best second action for the model's real continuation.

The minimum shared-data addendum is therefore **three nonempty policy states**:18587/36,18972/62,19255/37. Keep all legal pidx including zero-target rejected actions, and disable value/Q/length supervision for these bounded-safe policy-only rows. Empty forced states can separately support exact loss evidence when admitted under the appropriate schema, but not an arbitrary policy target. The existing value labels from observed complete episodes remain separate factual outcomes.

## Independent operational procedure and coverage

Strictly replayed every action of all five raw copied protocols using the supplied engine and parser. Every NPZ observation was compared byte/value-exactly with its freshly replayed pre-action state. Every full legal bool mask matched the engine; every safe mask was a subset of that complete legal set. The packet shapes are float32 `(N,16,8,8)` observations and bool `(N,4096)` masks, with explicit horizon `mask_kind` and ply identifiers.

Independent in-memory recursion completed the same team's remaining actions until true team change or terminal outcome. It then enumerated opponent complete turns until the true team change or terminal outcome. For an unsafe own completion, one explicit opponent terminal witness suffices; for a safe completion, every opponent complete continuation was checked. All own completions after every legal first action were visited, including after a safe completion had already been found. No generic budget cutoff or silent UNKNOWN classification was used. An unexpected no-legal-action state would have raised an assertion instead of being called safe.

For 18587,18972,19040 and 19255, this independently checked112 own completions,17,162 opponent complete-turn endpoints and 18,122 `env.step` operations in10.21 seconds. Per-state own completion counts were9,10,49,10,28,6 respectively. All reconstructed one-enemy-turn masks matched the supplied arrays exactly.19101's one-/two-enemy-turn enumeration was deliberately reused from the independently verified exact-equivalent19490 evidence rather than counted again.

Quantifiers are essential. A bounded-safe first action means **there exists an own second completion such that every opponent complete turn avoids terminal defeat**. A forced first action means **every legal own completion admits an opponent terminal winning continuation**. The two own actions are both controlled by the same player; perspective does not alternate after the first action. Safe through this horizon is not an eventual win or draw.

The five source packets live at `codex/matches/export6/user-five-v1/<ID>-proof.json`, `<ID>-hard.npz`, and `match-<ID>.txt`. The independent review was read-only and did not amend their source attribution or outputs.

| Match | Reviewed hard-NPZ SHA256 |
|---|---|
|18587|`a1faa64456bd620108337c8d77201ffb1457f760d94b32effb8af82df4399752`|
|18972|`119c118ecd246b3ea6b20e4c43cb9cf4985a555172acea4000ac027d886fc2fb`|
|19040|`ab54cba2c963b5fa509c607cf1be6932f3c321e7af437cdcd122a644aa5a6f7a`|
|19101|`82064934b50cea38a23a26b3758ed8258824745ffe2089cb04161492c848f263`|
|19255|`42226f3c87829e6a49003ce870c5517d97537605c1559eed7031a7e9c024bd22`|

Raw protocol hashes independently read at review:18587 `cd87e59c2381f8aae7b420417ad182b4d800ed52dc73776815ca562aeaec4652`;18972 `5d1f3487cdee120770c30e8f70d13c69051b1fa37e024f771a261e495565dbba`;19040 `b954ce8e23523bba781dc9fa19dedaa19499d98d419e470316d3c0aac43a429c`; 19101 `a545ac46fc49db2009957594f2ec55adea6d7e6e0b0ab2ccbaed1d9a2e282289`;19255 `4f32f8f79eebc7f83c9fe02d1776740853696cc44391031084fbca89e7748d2a`.

## 19101 and 19490: repeated ending, distinct source games

The raw protocols are **not byte-equal**, and their complete 74-action trajectories differ. Independent comparison found the last differing pre-action state at 56; every state57–74 agrees in full board occupancy, piece types/teams/has_moved flags, locked matrix, current team, actions_left, round, action number, terminated/winner, source scene and engine-order legal indices. Thus the critical63/65/67/68/71/72 states are exactly the same deterministic game states, not merely visually similar or encoded under an aliased clock.

The existing `19101-19490-state-equivalence.json`, SHA256 `2b262f7cebef1305b210e9439314fc77a7e2ba0aae359222856948f43a6999bf`, agrees with this independent comparison. The19101 NPZ legal/safe masks at 71/72 also exactly match19490's corresponding packet rows.

Astra's completed independent verifier is `codex/synthetic/validation/loss19490-independent/proof.json`, SHA256 `5513842e9e2c53066ff5130e5d9e03872db43bbbfde4728821a1ceb47fa2e114`, status PASS. It independently checked the full quantifier partitions with 242 endpoint states,1,431 complete defender turns and 1,314 terminal-oracle leaf checks, without importing the original mining minimax. It binds original19490 `deep-proofs.json` SHA `7809f9b9cb306d2499f53ea9fba14679ef2d8daa7dfe9fffdc1f5fcdc3943132` and critical NPZ SHA `333dbc28bfa7a06967c5267a0dc97e039ef9b0fb9c9a7e2d807d971b5a47495d`.

Consequently its verified pre 68 result transfers to19101 by full state equality: sole two-enemy-turn-safe action Kernel E6→F5, index2786. The actual Trace F6→F5 is safe through only one enemy turn and loses through the second. Before67 there are four two-enemy-turn-safe first starts, each with an appropriate own completion; the recorded Trace F5→F6 is one, but its recorded second action is not the required defense. At71/72 all choices are already forced losses. This is proof reuse, not a second independent discovery or a newly generated training row.

Preserve both raw episodes and their IDs/outcomes. Link the repeated ending as `site-ending-19101-19490` for state-derived proof counts, observation-level routing and train/holdout grouping. Do not delete or deduplicate the whole episodes merely because their endings converge, and do not count their identical hard ending as independent tactical evidence.

## Additional 19040 extension and remaining limits

The match-analysis owner subsequently supplied `19040-deep-{90,89,86}.json` and `19040-deep3-{86,85}.json`. Its claim is already-forced at 89/90 through two enemy turns, while at 86 the two-enemy-turn horizon still has survivors, including the recorded Trace action; the three-enemy-turn extension then marks all actions85/86 losing. A brief read of `deep3_19040.py` found the intended universal-own/existential-enemy turn composition and explicit UNKNOWN propagation. Those deeper output certificates were **not independently re-enumerated or released by this review**; earlier than 85 remains UNKNOWN here. Do not manufacture a policy correction from the currently forced final states or conflate those deeper horizon claims with this independently executed one-enemy-turn pass.

No blocker was found for the three bounded-safe policy states. The material release conditions are complete legal support, preserved canonical indexing, existential second-action semantics for 19255, disabled unknown value/Q/length targets, one shared override manifest, and no duplicate or conflicting replay labels. Actual trainer ingestion and current-model performance belong to their separate assigned reviewers/owners.
