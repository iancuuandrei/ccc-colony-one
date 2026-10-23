# Independent loss11059 review

Source: `codex/matches/site-all-v1-derived/match-11059.txt`, SHA256 `1e38342c16386b546e03104d29c77f7d5687fa1e5535c06126247c3b1b8923ec`. Independently replayed with `site_replay.parse(path, None)`: all32 typed actions, captures, turn order and final MAGENTA victory agree with the supplied engine. No model, GPU or800-simulation search was used. Text does not itself identify the submitted model hash.

The decisive **bounded tactical** commitment is Cyan action30, in round8 with one own action left: **Kernel C3→D4 captures the enemy Ice**. Just before it, Cyan has Kernel C3 and Trace B3; Magenta has Kernel H6, Ping G5, Trace F4 and Ice D4. Trace B3 guards Kernel C3. Capturing on D4 moves the Kernel two files from that guard, leaving it unguarded.

The actual legal winning reply is **Magenta Ping G5→E5, then Ping E5→D4**, both MOVE actions in the same Magenta turn. The first move is orthogonal range2; the second is a diagonal capture at range1. The second capture terminates immediately, before the round8 firewall can burn Magenta's Kernel H6. This is a quiet setup followed by a capture, not a current direct attack at the pre-blunder position.

Match analysis independently enumerated all10 legal action30 alternatives and found exactly one bounded-safe choice: **Cyan Kernel C3→C2**. I independently applied this precise defense and exhaustively checked the resulting opponent own-turn win mask: no terminal Magenta win exists within its two actions. The Kernel stays adjacent to Trace B3, while changing the Ice/Ping/Trace attack geometry. Simply retaining a bodyguard is not a general sufficient defense; the exact square matters.

Action29, Kernel B2→C3, therefore still admits the verified defensive completion C3→C2. The played action30 turns that recoverable next-turn horizon into a certified loss. This does **not** prove that the earlier whole game had a winning strategy or establish the globally earliest game-theoretic losing position; no full-game solver was run. An independent own-turn terminal-win scan at all16 Cyan decision states found **zero** missed Cyan wins within the remaining one/two actions, so an earlier overlooked immediate win is not supported by this replay.

Minimal shared training correction:

- At action30, keep every legal action in `pidx`, put policy mass on Kernel C3→C2 (canonical index2738), and zero target mass on the played Kernel C3→D4 (index2723). Use `has_value=False`: next-turn survival does not certify the eventual game result.
- The post-blunder Magenta action31 position supports an exact current-turn `+1` value and the winning quiet Ping setup policy. This teaches the evaluator the concrete two-action threat that root search must see after the bad capture.
- Keep these states and any counterfactual descendants in the same `site-game-11059` group. Prefer this narrow correction to a blanket material-capture penalty or a blanket rule that every guarded Kernel is safe.

## Approximate-teacher conflict

A correct tiny overlay does not cancel a contradictory duplicate800-simulation policy row. Independently matching all15 actual positions in the newer Astra site curriculum against `data/ext_site_v1_rs/chunk0000.npz` found exactly the existing six covered conflicts and no additional covered actual positions:

| Game/action | Original re-search row | Mass outside exact admissible set |
|---|---:|---:|
|7525/26|1680|93.25%|
|7658/55|1737|62.50%|
|8079/30|1808|95.37%|
|8185/42|1872|56.00%|
|8443/30|1962|96.00%|
|8228/35|1909|85.76% outside current-turn winning set|

The overlay owner has supplied `data/ext_site_tactics_v1/policy-overrides.json`, keyed by exact float16 observation SHA, with replacement targets, source-row/file hashes and proof bindings. Its in-memory replacement audit passes, but actual trainer application was **NOT RUN** at handoff. Opus must replace or deactivate the conflicting approximate policy target when loading the identical observation; merely adding six or55 extra exact rows leaves opposing gradients. Preserve genuine outcome/value supervision separately. Root and the overlay owner were notified; no teacher data or trainer source was changed by this review.

## New site-curriculum qualification

Separate high-priority assignment completed: all55 frozen states under `codex/synthetic/site_gen02_curriculum` were independently qualified via source reconstruction, full `env.step` own-turn/bounded-safety minimax, exact legal sets, typed witness replay, D4 mapping and held-out-orbit exclusion. Released `data/ext_astra_site_gen02_v1/chunk0000.npz`, SHA256 `2572aa1a2f75f913588a4e91b1b114b501a4b17798c7c762f820abe07d457c73`:15actual+40constructed states,52policy targets,9exact+1 and3exact−1 values,43UNKNOWN values masked with `has_value=False`, all `q_valid=False`. Three universally losing states have value supervision and no invented preferred policy. Source game groups, constructed reachability limits and duplicate-research warning are retained in provenance. This corpus predates11059 and does not already contain its new correction.
