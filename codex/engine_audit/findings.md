# Level 4 engine audit

Run from repository root:

```powershell
.venv/Scripts/python.exe workspace/colony-one-level4/codex/engine_audit/audit_engine.py
```

`results.json` binds the run to SHA256 of the six analyzed source files. This is bounded synthetic operational evidence, not grader acceptance, a self-play result, or full game-state reachability proof.

## 1. Existing attack + fewer guards than remaining actions is a forced win

**PROVED.** Let a legal action attack the enemy Kernel and let `g` be the exact number of its adjacent friendly non-Kernel guards. Repeating the same action `g+1` times destroys that Kernel. If `actions_left > g`, all actions occur before the opponent can act, and winning capture preempts turn-end rules/firewall.

Reason: on each redirect the attacker and target Kernel remain at their original coordinates, and exactly one guard disappears. Removing a different piece cannot block the previously legal attack path; jump geometry is also unchanged. A redirect cannot remove the attacker (guard must belong to enemy Kernel team). Every subsequent repeated action is legal. The first attack after the last guard is gone captures the Kernel immediately.

The actual engine permits at most two actions, so the relevant cases are zero guards with either remaining-action count, and one guard with two remaining actions. PASS: 430 fixtures, all six attacker types, both actual sides, five guard types, all compatible adjacent guard placements. Checked through `env.step`, including same-team continuation and final reward +1.

Sources: `board.py:260` guard selection; `board.py:290-355` execution; `env.py:172-186` action decrement, immediate Kernel victory, same-team continuation.

Useful ONNX rule: `has_current_legal_kernel_attack AND enemy_guard_count < 2*plane13` certifies value `+1`. Give each such actual attacking action a strong policy prior. A count clipped at three is sufficient for this comparison, since actions_left <= 2. This is sufficient, not necessary: other two-action sequences can win.

False positives arise if a geometric attack map includes blocked paths/Trace range one/illegal landing squares, if guard count excludes a piece type, or if the rule is applied to an opponent attack while the opponent is not the side to move. A current enemy threat is not itself a certified loss: the moving side can relocate its Kernel, capture the attacker, add a guard, or win first.

## 2. Search never looks beyond a redirected attack

**PROVED source defect.** `mcts.py:436` and `:442` pass `cache=False` after a bodyguard redirect. `_child` at `:448-469` then always creates a fresh unexpanded node. That node gets evaluated but is discarded. The next visit creates another fresh leaf. Search cannot traverse a second action after that edge, even with arbitrarily many simulations. If redirect occurs on the last action, the same problem prevents searching the opponent's reply, unless turn-end itself terminates.

Operational fixture: force one guarded Kernel attack as policy favorite, constant evaluator value 0.25, 32 sequential simulations. Root attack gets 32 visits, no cached child, and Q remains exactly 0.25; evaluator receives root then the identical post-guard-removal leaf 32 times. The actual repeated attack wins and has game-theoretic value +1. Board restoration also PASS. The script uses batch size 1 to make observation counts unambiguous; the cache defect is independent of batching and tournament batch 8 has the same barrier.

Therefore the post-redirect evaluator must recognize an unguarded legal attack with one action left. A prior alone at that discarded child cannot help: its actions are never explored. The root evaluator's own value is not backed up as an edge result (`_expand_root` result is ignored by `run`); the useful hard value must fire in the leaf as well.

Do not patch the supplied referee to make a candidate look better. Retain the exact search for comparison.

## 3. Two-action perspective

`env.py:172-186` preserves current team after first action, and `mcts.py:431-442` preserves sign during the same turn. `_turn_end` flips sign only on actual team change; `_backup` at `:534` multiplies node/value signs. Value is always from the encoded moving side, including unfinished same-player turns. Never negate a label after every action.

## 4. Observation is sufficient for the live clock

**PASS: 104 fixtures**, rounds 1 through 26, both sides, each remaining-action count. Despite `Board.from_observation` documentation at `board.py:116-119`, actions_left is exactly recoverable as `2*plane13`. Side is exactly plane15. Round is recoverable by combining locked-ring count and urgency:

- locked-square count 0/28/48/60 means respectively 0/1/2/3 shrinks already completed;
- `next_shrink=(8,16,24,26)[completed_shrinks]`;
- `round=round(next_shrink-8*(1-plane14))`.

Urgency alone aliases rounds, so use lock state too. Only applies to engine-consistent, live boards, not arbitrary malformed validator inputs or terminal observations with no remaining shrink. `has_moved` is absent but no rule uses it (`pieces.py:84-88`). Both piece coordinates and actions invert via the 180-degree Magenta rotation.

## 5. Firewall and bare-Kernel ordering

**PASS: bounded priority fixtures.** Immediate Kernel capture first; then, only at the end of the current turn, bare-Kernel adjudication; then, only after Magenta completes a shrink round, firewall. The round-8 perimeter is playable during all four actions of round 8. Cyan's turn ending in round 8 does not burn anything. At round end a single burned Kernel loses and both burned Kernels give Magenta the win. Bare-Kernel resolution preempts even a simultaneous burn that otherwise would favor Magenta (`env.py:188-229`, matching `mcts.py:_turn_end`).

Bare kernels: final Chebyshev distance <=2 means the side that just acted loses, regardless of rings. Otherwise closer to center wins, equal rings favor Magenta (`board.py:183-245`). The build's `ring=min(r,c,7-r,7-c)` grows toward center: larger build ring wins, even though board.py calls the opposite distance-from-center quantity 'ring'.

Useful exact bare-Kernel solver: only the current side moves before adjudication. If kernel distance <=actions_left, capture is possible this turn, value +1. Otherwise enumerate own-Kernel destinations reachable in exactly actions_left king steps inside the open rectangle. At actions_left=1 exclude the current square; with two actions out-and-back permits it. For each final position, distance <=2 loses; farther positions win iff own inward-ring index exceeds enemy's, or equal and own side is Magenta. Existence of any winning destination means +1; otherwise -1. This destination characterization follows the open concentric-rectangle geometry and is source-derived here; a separate exhaustive operational proof of this formula is **NOT RUN**.

False-positive risks: do not score current bare-Kernel position as already terminal; the current side still moves. Do not penalize every own Kernel on a doomed ring as certain loss: Magenta can win simultaneous burn, either side can capture first, and bare-Kernel resolution can override burn. At Magenta actions_left=1 on a shrink round, any nonterminal/non-bare resulting board with only own Kernel burning loses; both burning wins for Magenta.

## 6. Published heuristic attack map scope

**PASS** for capture geometry on enemy-occupied destinations: exported the actual internal graph outputs from `build_heuristic.py`, and compared against the authoritative legal generators on 128 synthetic static boards with consistent firewall locks. All 845 enemy-occupied-square comparisons match. This covers rays/locking/Trace minimum range/Daemon adjacency sufficiently to support the above attack flag, but is bounded evidence, not exhaustive proof.

`build_heuristic.py:71-91` rays and per-piece attack maps follow the capture geometry. They are not complete legal action maps: they include empty attack squares, friendly-occupied endpoints, and some locked endpoints. Multiplying by a legal enemy occupancy plane removes those issues on ordinary live boards. Do not use them alone as movement masks.

Destination-danger scores are approximations: a move changes blockers/removes a captured attacker; a shot keeps its source; a redirected move also keeps its source. The generic destination-danger policy at `build_heuristic.py:134-140` treats all three as if the actor arrives at the aimed square. The scalar material threat ignores guard redirection value, guard priority, future forced capture, and double actions. The heuristic never reads plane15, so it cannot exactly represent Magenta endgame/firewall asymmetry. `doomed_sq` at `:116` ramps for urgency>.75, not just the actual deadline, and its penalty does not account for side/remaining actions or simultaneous-burn victory.

## Action priorities

1. Hard +1 for a currently legal Kernel attack with fewer guards than own remaining actions, in all evaluator leaves.
2. Strong prior on the corresponding actual attacking actions; keep tournament search unchanged.
3. Respect side15 and clock when scoring imminent burns and bare-Kernel outcomes.
4. Use source/actual destination for shots and guard redirects when policy-scoring danger.
5. Treat broad current enemy attack maps and ring penalties as heuristic terms, not hard losses.

## Independent own-turn oracle and tactical-v2 corpus

`own_turn_oracle.py` exports `own_turn_wins(board, team, actions_left, round_number)` and `oracle_observation(observation)`. Result fields: `any_win`, `first_action_win_mask` (4096 bool), deterministic positive `witnesses`, and enumerated action counts. All legal first/second actions are considered; opponent replies are outside this oracle's horizon. A false result means no terminal victory within remaining own actions, not a forced loss. Winning outcomes include immediate Kernel capture, bare-Kernel adjudication, and firewall wins. All board mutations are restored through engine undo.

`build_oracle_corpus.py` generated `own_turn_corpus.npz` with `observations`, `first_action_win_masks`, `any_win`, `names`, and `origins`; JSON companion contains per-state witnesses. PASS: 121 states (55 diagnostic, 66 reachable from 192 seeded random sampling transitions), 40,870 first/second action branches, 274 positive witnesses replayed through `env.step`. Every diagnostic mask also matched independent full enumeration through cloned `env.step` calls. Observation reconstruction/clock recovery and board restoration matched in every corpus state.

`validate_tactical_v2.py` checked `codex/tactical/approach-diagnostic.onnx` against this independently generated corpus. PASS: all 28 certified approach actions are a subset of the exact winning-first-action masks; all 27 `value==1` states have an exact own-turn winning witness. Zero false positives in this bounded corpus; no completeness or strength claim. Exact model digest is recorded in `tactical_v2_oracle_validation.json`.

`oracle.py` adds `solve_this_turn(env, team=None)` and `check_forced_loss_by_next_turn(env, max_nodes=10000)`. The latter returns `VERIFIED`, `DISPROVED`, or `UNKNOWN`, plus per-first-action statuses: it universally searches current-turn choices and existentially searches an enemy terminal win before the next own turn. It includes actual firewall changes when handing over. Exhausted budgets and no-action positions return UNKNOWN rather than inventing rule outcomes. `validate_loss_oracle.py` independently compared full cloned-env trees for five bounded positions, including a true two-action enemy threat, an escape, a locked 2x2 position, and firewall resolution; all PASS. All zero-budget checks returned UNKNOWN and all boards were restored (`loss_oracle_validation.json`).
