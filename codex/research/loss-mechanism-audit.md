# GEN02 loss mechanisms under the immutable referee

2026-10-23. Five bounded single-position replays used released GEN02 SHA256 `579d3ec01da7328dc706003b8dcc5eba357bb74902252187e45633ca0595a9a0`, the supplied Level4 scene, exact `MCTSConfig(simulations=100)`, batch8, and `numpy.default_rng(0)`. Every position is after opening warmup. No candidate, search grid, arena, export, or training was run. Instrumentation wrapped `_set_priors` only to record node initial values; its original implementation and every search result were unchanged. Existing replay parser supplied `states[ply-1]` from immutable complete protocols under `codex/matches/site-all-v1-derived/`.

Source snapshots inspected: current `net.py` SHA256 `2d8c672ce1aab0421fde0b945565f997630180d9d39e59516395c29334061b1c`; supplied `mcts.py` SHA256 `1374b55952f08fcbe411307fd96ec62832d2712362b2b649a2307948ffbf355e`. Current Python source is explanatory; the released model hash is the actual replayed artifact. No claim that later source edits are part of GEN02.

## Measured mechanism: the search often sees the loss and still selects it

Here P is the legal normalized root prior, N completed root visits, and Q average value in the original mover's perspective. The server chooses **maximum N**, not maximum Q. Each root completed exactly100 simulations.

| Position/action | P | N | Q | Evidence |
|---|---:|---:|---:|---|
|8079/a30, selected Trace C3→C2|.472281|38|−.984135|Opponent leaf value+.990000; opponent Daemon C5→C2 gets36 visits, Q+.999722|
|8079/a30, unique bounded-safe C3→C4|.049264|5|−.930166|Opponent leaf+.747037; its C5→C4 stationary shot gets4 visits, Q+.975948|
|9366/a36, selected Daemon D5→G5|.487268|39|−.967520|Opponent leaf+.990000; killer F3→F6 gets36 visits, Q+.999722|
|9366/a36, unique bounded-safe D5→F5|.007366|1|−.923958|Opponent node receives only its first evaluation, no descendant visits|
|9366/a52, selected Daemon E6→E4|.787696|35|−.903904|Opponent initial value+.877028; C3→D4 gets32 visits, Q+.999877|
|9366/a52, bounded-safe E6→F6|.141813|28|−.654053|Better root Q than selected action|
|9366/a52, bounded-safe E6→G6|.016216|29|−.484913|Better root Q than selected action|
|11059/a30, selected Kernel C3→D4|.959059|90|−.996087|Opponent leaf+.990000; winning Ping G5→E5 gets88 visits, Q+.999886|

Thus “the exact override failed to recognize the opponent win” is false for8079,9366/a36 and11059 at their immediate opponent leaf. Those values already equal+.99, and search follows the winning continuation repeatedly. The unsafe moves nevertheless retain the most visits. In8079 the prior is9.59times the safe alternative; in9366/a36 it is66.15times larger. The safe alternatives also receive pessimistic approximate values. Both effects matter.

PUCT uses `Q + 1.5*P*sqrt(total+1)/(1+N)`, with virtual losses while batching. A high prior can allocate many visits to an action whose loss has already become apparent. Visits are not reset after discovering a losing line, and Q is an average over explored descendants rather than a solved minimax bound. Final move selection then uses those accumulated visits. For example9366/a52's Q-best listed retreat receives29 visits, while the worse high-prior E4 action retains35. That is a concrete finite-search policy-allocation problem, not a perspective or legal-mask error.

These measured facts support the already assigned proof-guided training-policy repair. They do not support another generic temperature grid or the claim that merely giving R2 winning replies a larger prior solves these root mistakes: the winning replies already have high priors (.925833,.585454,.933294 in the three kernel-kill cases).

## Measured distinct mechanism: a redirected edge never gains depth

At9928/a28, selected Ice D5→B3 attacks a guarded Kernel; the Ice actually stays at D5 and a guard disappears. Root P=.974491971, N=95, Q=+.400885046. There is **no cached child**. The resulting actual position is Cyan Kernel B3 versus Magenta Kernel G7 and Ice D5, Cyan to move with2 actions, round8. The released evaluator gives Cyan value−.400885046. That is the exact negative of the root edge Q.

The supplied `_descend` detects `actual_target_pos != aimed_to` and passes `cache=False` through `_child`; every descent therefore creates an unexpanded throwaway child. Source semantics plus the measured uncached95-visit edge establish that this branch repeatedly consumes the same post-redirect leaf evaluation instead of searching Cyan's B3→C4→D5 terminal win. Raising this same referee to800 or more simulations cannot remove that depth barrier. A policy prior inside the discarded child is insufficient; the scalar leaf value must recognize the own-turn win.

This is different from8079. Trace C3→C2 is an ordinary move and its child is cached. The enemy Daemon then move-captures the guard at range3 and shoots Kernel B3. Safe Trace C3→C4 instead makes the guard adjacent to the Daemon, so capturing it is a stationary shot. Same guard count, different follow-up reach. Engine legality and action application already encode this difference; the old value/prior estimates fail to value the defensive geometry accurately.

9366/a52 has a related uncovered terminal family even without a root redirect. After E6→E4, Cyan has only Kernel C3 and the enemy has Kernel G7 plus its last Daemon E4. Cyan can approach then capture the Daemon; the game ends through bare-Kernel centrality, not enemy-Kernel destruction. The cached search does discover one such winning line (C3→D4 then capture). The initial heuristic leaf is not an exact certificate, and high original prior still wins the root visit contest.

## What the current exact features do and do not guarantee

R1 certifies repeated attacks on an enemy Kernel when adjacent guard count is less than own remaining actions. It supplies value+.99 and +30 logits for legal actions aimed at that Kernel. R2 certifies a single actor relocating and then attacking an unguarded Kernel, allowing an appropriate guard capture and a vacated origin. Current source gives R2 the value override, but the policy bonus remains gated by R1 only.

An exact value at an evaluated node is not persistent solved status. At root, `_expand_root` returns a value that `run` ignores for edge backup. At an ordinary internal node, the value is backed up when that node is first expanded; later simulations descend further and back up new leaf values. The referee does not lock its minimax value at+.99 or impose a lower bound just because the model emitted that number. This is a source-proven limitation. It can matter for R2 policy starvation in other states, but starvation of the *opponent's winning R2 reply* is not the measured explanation for the three cases above.

R2 is sufficient, not a complete two-action solver. It can miss line-opening by another actor, a first actor capturing a blocker followed by a different attacker, stationary first shots enabling another piece, and winning capture-to-bare transitions. No false-positive R1/R2 or canonical legal-mask bug was demonstrated by these replays. Both winning witnesses were legal and their indices appeared in the actual node policies. Same-player perspective is correct in the supplied search: sign is unchanged after the first own action and flips only when the team changes. Blindly negating after every step would be a bug, but is not what happened here.

For new10910 the independent match-analysis worker reports all final choices at44 and all preceding starts at43 already unsafe. A policy repair must not manufacture a safe action there. Earlier trades/captures need analysis; the final forced position cannot itself identify the earlier strategic cause.

## One minimal model-side proposal, Opus sole owner

Extend the exact own-turn win feature with a **narrow three-piece capture-to-bare certificate**. Current side must have exactly its Kernel; enemy must have exactly its Kernel and one non-Kernel at t. With1 action left require adjacent legal capture of t. With2 actions require Chebyshev distance exactly2 and an explicit unlocked empty intermediate m permitting `s→m→t`. The final capture must make bare adjudication favor the acting side: `distance(t, enemyKernel)>2` and inward-ring(t) larger than the enemy Kernel's, or equal rings with actual side Magenta. Capture occurs on the last own action; bare adjudication preempts firewall.

This supplies+.99 at the discarded9928 leaf, where any deeper prior is ineffective, and recognizes the9366/a52 terminal family. Optionally bonus only certified first actions. It is not a general forced-loss flag and does not promise to fix the high-prior root mistakes above. The shared proof-policy data repair remains their targeted intervention.

Root subsequently authorized the pure helper and proof, without ONNX/net/training edits. They live in `codex/endgame/three_piece_certificate.py` and `three-piece-report.json/md`. Completed evidence:68,800 admitted positive geometries and111,920 independently engine-executed paths for one representative enemy type, all live lock stages/both sides/action counts;1,412 other-type states/2,124 paths;483 other-clock states/693 paths; both actual9928/9366 leaves. All assertions passed. Enemy-type and clock semantic reductions are proved in the companion report; this is not exhaustive five-type game solving. None means UNKNOWN, not loss. The helper/records are for Opus to port into the single shared model only after proof review, followed by exported parity and resource checks.
