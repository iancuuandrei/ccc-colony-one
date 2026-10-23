# GEN05 tactical proofs and export (5) data handoff

## Attribution correction

The seven individual TXT files contain a terminal winner but no user identity or model hash. Fresh JSON export (5), SHA256 `770ca3628647f4bc4e36064e71ee19e44d44c3baa54e6ff77804e1e67d8f7b3b`, explicitly binds user side:

| Games | User result and side |
|---|---|
|15310,15116|CYAN wins|
|15058|MAGENTA wins|
|14594,14621,14878|MAGENTA losses|
|14958|CYAN loss|

Thus the losing-actor proofs in 15310, 15116 and 15058 describe opponent mistakes. Their training outcomes remain valid. `site-gen05-attribution-v2.json` supersedes the earlier UI/UNKNOWN receipt. Actual source model identity remains UNKNOWN. Requested GEN05 SHA256 `af45a5f85894e694f7ddab47811c4fb45b9f9536345d78b0f2fa3ef06c99c91e` was used only as a counterfactual baseline at reached critical positions.

## Import and cross-folder deduplication

Opus owns the original seven-episode import and 800-simulation re-search. Those folders were not modified or duplicated. Their 328 outcome rows numerically matched the fresh JSON replays. Seven cross-folder episode aliases were recorded instead of creating duplicate chunks.

`data/ext_site_v1` now has **111 local complete episodes / 3,777 rows**, adding 22 new atomic chunks / 830 rows. Together with the seven external aliases / 328 rows, the union is **118 unique complete games / 4,105 rows**. Five metadata-only/incomplete episodes remain excluded. Earlier 89 local NPZ chunks remain unchanged. Actual mover-relative z, blank policy, q=z/q_valid=False and complete-game groups are preserved.

Independent validation passed all local hashes/schema, reconstructed legal sets and outcome signs, plus every external alias hash/row range/legal set/outcome and cross-folder uniqueness. Manifest SHA256 `50ce9af40cd67e8e7c27addd300dbf3dab00625daff47f37ed42b02169624b39`. Receipt `ext-site-gen05-v1-validation.json`; alias map `site-external-aliases.json`; download receipt `site-export5-source-v1.json`.

## Highest-priority user-loss corrections

These nonempty action sets avoid terminal defeat on the next enemy turn. They are not proofs of eventual victory. GEN05's counterfactual MCTS100 choice misses every listed defense.

| Game/action | Bad commitment | Certified alternatives |
|---|---|---|
|14878/36|Kernel F3->G3 enters Daemon C4->F4->G3 shot|Only F3->G2|
|14621/63|Trace F7->F6 wastes the first action before its round16 burn; no second action then saves the position|King E6->D6/D7/D5, with safe completions to C6/C6/C5|
|16056/36|King E5->D4 captures the last Trace and immediately loses bare-Kernel adjudication|E5->E4/D5/D6/F4|
|14009/33|King B2->B3 commits to a second-action position with no safe move|B2->C2/C3, then D2|
|13352/34|Ice C7->E5 bodyguard hit leaves Ice at C7; enemy King E5->D6->C7 captures it and wins bare-Kernel adjudication|Seven Ice retreats, including C7->C6/C5/D6|

Final turns 14594/35-36, 14958/41-42, 13711/61-62 and 13377/31-32 are already forced within the checked horizon. Empty admissible sets must not receive arbitrary policy targets. The 13711 witness includes the round 16 firewall burning the Kernel. Earlier causal analysis of 14958 was transferred to `distill_student`: at action 38 the actual Ice C3->B3 enables the recorded guard-removal reply and a forced loss; King B2->B3 is the sole candidate evading that recorded response, still UNKNOWN until that agent completes the all-reply proof.

The two 15116 first-action defenses at actions 31/35 belong to the opponent in our win. They remain useful tactical supervision but must retain correct attribution and the same game group.

## Winning-side check

Twenty confirmed user-winning episodes (17 newly imported plus 3 aliases) were checked over 390 own actions and 195 completed turns. One decision misses an available current-turn win: 15116/action 33, Daemon C6->F6->G5 shoots the enemy Kernel. Actual play and counterfactual GEN05 instead choose C6->C3. No completed winner turn allowed an immediate enemy terminal win. This winning-user observation and the opposing 15116 mistake observations remain one source game group.

## Immutable proof handoff

- `site-gen05-loss-hard-v1.npz`: 15 losing-actor observations, SHA256 `85b783756dc3d0a721b280a846597e8546aca90bb4a393a6cda8f61663f24a9e`.
- `site-gen05-loss-hard-v2.npz`: 8 observations from five new JSON-confirmed user losses, SHA256 `a71051f87c1c54deff9eae0140c3e50c9c2043f3c7df4a7e1a17742785509d24`.
- `site-gen05-wins-hard-v1.npz`:one winning-user observation, SHA256 `d42e9a23653a77f96ee2e5ddd6a9e9fd99aa1a769aab8e9825248c39d985081f`.

Typed witnesses and full alternatives are in `site-gen05-loss-mining-v1.json`, `site-gen05-loss-mining-v2.json`, the corresponding `site-gen05-earlier-first-actions` files, and `site-gen05-wins-mining-v1.json`. Re-search priorities with actual actor/user attribution are indexed in `data/ext_site_v1/hard_decisions_gen05_v1.json`. Astra and the independent policy-repair owner received the packets. Opus alone owns import of the seven original games, 800-sim re-search, training and model export; no duplicate computation or model was created here.
