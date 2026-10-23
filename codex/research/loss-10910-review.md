# Match 10910: the recoverable decision is before action 39

Independent engine replay: **PASS**, all 44 recorded actions legal, final winner CYAN. This review used no model inference, MCTS, 800-simulation re-search, training, or export. The site text does not bind the played moves to a model SHA; no model identity is inferred. GEN02 remains the shared comparison baseline, without any new strength comparison here.

**Demonstrated transition:** MAGENTA's recorded action 39, **Kernel F6→G6**, leaves a state in which every completion of its turn permits a CYAN strategy forcing terminal defeat by the end of CYAN's second following turn. At action 39, **Kernel F6→F7 followed by Trace G7→D4 SHOOT** avoids that same finite-horizon forced loss. This is a proved survival improvement, **not a proved eventual win**. The first full-game losing decision anywhere earlier in the match remains UNKNOWN; action 39 is the first recoverable transition identified by this bounded investigation.

## Exact snapshot and action identities

Before action 39: MAGENTA to move, two actions left, round 10, outer firewall ring locked. All coordinates below are absolute site coordinates.

| Side | Typed piece | Square | Zero-based `(row,col)` |
|---|---|---|---|
| MAGENTA | Kernel | F6 | `(2,5)` |
| MAGENTA | Trace | G7 | `(1,6)` |
| CYAN | Kernel | D6 | `(2,3)` |
| CYAN | Daemon | C5 | `(3,2)` |
| CYAN | Trace | D4 | `(4,3)` |

Canonical action indices use MAGENTA's 180-degree rotation:

- Recorded first action: Kernel F6→G6 = **2729**.
- Certified longer-horizon escape first action: Kernel F6→F7 = **2738**.
- Required second action after that escape: Trace G7→D4 SHOOT = **3164**. This captures CYAN's Trace on D4; MAGENTA's Trace remains on G7.

Lossless canonical sparse observation:

```json
{
  "action_number": 39,
  "side": "MAGENTA",
  "actions_left": 2,
  "round_number": 10,
  "piece_squares": [[], [], [], [49], [], [42], [], [37], [], [28], [], [44]],
  "locked_squares": [0,1,2,3,4,5,6,7,8,15,16,23,24,31,32,39,40,47,48,55,56,57,58,59,60,61,62,63],
  "global_planes": [1.0, 0.25, 1.0]
}
```

## Short-horizon checks explain why the final move is misleading

The standard oracle asks whether every remaining own-turn continuation permits terminal defeat during the immediately following enemy turn. All listed calls completed without exhausted budgets or UNKNOWN branches.

| MAGENTA action | Recorded choice | First choices surviving this horizon | Exact nodes |
|---|---|---:|---:|
| 31 | Kernel G8→G7 | 22/22 | 3,064 |
| 32 | Ice F6→D6 | 15/20 | 2,287 |
| 35 | Kernel G7→F6 | 4/4 | 1,450 |
| 36 | Trace G6→G7 | 5/9 | 1,784 |
| 39 | Kernel F6→G6 | 8/9 | 3,531 |
| 40 | Trace G7→D4 SHOOT | 3/6 | 1,774 |
| 43 | Kernel G6→F5 captures Daemon | 0/5 | 282 |
| 44 | Kernel F5→F4 | 0/8 | 8 |

The recorded choices at 31,32,35,36,39 and 40 all survive this short horizon; no current-turn winning first action was found in those snapshots. In particular, action 40 is **not** an avoidable immediate-next-turn loss, and 43/44 have no admissible safe action to manufacture as a policy label.

At action 44, Kernel F5 has eight legal destinations. All immediately adjudicate CYAN victory: F6/F4/E5/E6/E4 end within Chebyshev distance 2 of CYAN's Kernel D6; G5/G6/G4 are farther away but have centre-ring index 2 versus D6's index 1. Thus simply moving toward the centre cannot repair the position. The distance rule precedes centrality, so even E5/E4 lose.

## Composed exact proof after the recorded action 39

The longer check enumerates CYAN's entire next two-action turn. Each resulting live MAGENTA position is checked by the supplied exact `check_forced_loss_by_next_turn` oracle. A positive result therefore composes:

`exists CYAN first turn; for every MAGENTA response turn, exists CYAN terminal-winning next turn`.

Terminal wins during the first CYAN turn are handled directly by the engine. A negative result means the reverse quantifier statement through this horizon, not a full-game win. No learned value or search visit count enters these certificates.

After recorded 39 Kernel F6→G6, **all six legal second actions are losing under this longer horizon**:

| MAGENTA action 40 | Canonical index | One forcing CYAN first-turn witness | Endpoint proof |
|---|---:|---|---|
| Trace G7→F7 | 3186 | Daemon C5→F5, then Daemon F5→F7 captures Trace | MAGENTA next turn: 5/5 first actions lose; 401 nodes |
| Trace G7→D4 SHOOT | 3164 | Daemon C5→F5, then Daemon F5→G6 SHOOT sacrifices Trace G7 | MAGENTA next turn: 5/5 lose; 282 nodes |
| Kernel G6→G5 | 2657 | Daemon C5→F5, then Daemon F5→G5 SHOOT | Immediate CYAN Kernel capture |
| Kernel G6→F6 | 2666 | Kernel D6→C6, then Trace D4→F6 SHOOT sacrifices Trace G7 | MAGENTA next turn: 8/8 lose; 1,593 nodes |
| Kernel G6→F7 | 2674 | Daemon C5→F5, then Kernel D6→D5 | MAGENTA next turn: 6/6 lose; 3,022 nodes |
| Kernel G6→F5 | 2658 | Daemon C5→F5 captures Kernel | Immediate CYAN victory |

Total completed proof work for this six-action classification: **272,703 exact endpoint-oracle nodes, 96 distinct cached endpoint positions, zero UNKNOWN classifications**. This is stronger than the one-next-turn result. The actual action-40 snapshot is consequently a proved losing state under the composed certificate; a policy repair cannot rescue it by selecting one of its other five actions.

The recorded witness is especially clear: 41 CYAN Daemon C5→F5, 42 CYAN Daemon F5→G6 SHOOT sacrifices MAGENTA's Trace on G7. The Daemon remains on F5. At 43, all five MAGENTA first actions are already losing; capturing the Daemon merely exposes the bare-Kernel adjudication seen at 44.

## The verified alternative at action 39

Choose **39 Kernel F6→F7**. This counterfactual position has six legal second actions. Five still admit longer-horizon forced defeat. The sole escaping second action is **40 Trace G7→D4 SHOOT**, capturing CYAN's Trace while retaining its own bodyguard on G7.

For this completion, **all 239 legal CYAN first-turn completions were checked**, and every resulting live MAGENTA endpoint had an own continuation avoiding terminal defeat through CYAN's next turn. No budget exhaustion and no UNKNOWN endpoint occurred. The full six-second-action investigation used **805,317 exact endpoint-oracle nodes and 412 cached endpoint positions**. This proves a bounded survival improvement over recorded 39, without assigning an eventual `+1` value.

A concrete, separately replayed line illustrates why the one-square Kernel difference matters:

| Action | Side and piece | Move | Exact event |
|---:|---|---|---|
| 39 | MAGENTA Kernel | F6→F7 | Move |
| 40 | MAGENTA Trace | G7→D4 | Shoot captures CYAN Trace; attacker stays G7 |
| 41 | CYAN Daemon | C5→F5 | Move |
| 42 | CYAN Daemon | F5→F7 | Range-two attack redirects to MAGENTA Trace G7; Daemon stays F5 |
| 43 | MAGENTA Kernel | F7→E6 | Move |
| 44 | MAGENTA Kernel | E6→D6 | Captures CYAN Kernel: MAGENTA wins immediately |

This six-action counterfactual replay **PASS** establishes the displayed line only. The separate universal 239-completion check establishes bounded safety against alternative CYAN choices. Neither establishes an eventual minimax win from the action-39 snapshot.

One plausible-looking alternative was also excluded: **39 Kernel F6→G5** has eight legal second actions, all losing under the longer horizon, verified in 916,656 endpoint-oracle nodes. It should not be suggested as a repair just because it survives the short check.

## Single-pipeline handoff and scope

Send the action-39 snapshot, actual first action 2729, preferred bounded-survival first action 2738, and its required continuation 3164 to the shared Opus/Astra data path. Keep the source group `site:10910` across outcomes, re-search, and counterfactual relatives. The new preference needs an explicit **two-following-CYAN-turn** certificate scope; the existing one-opponent-turn repair schema must not silently reinterpret it. Other first actions at 39 were not completely classified under this longer horizon.

- Original pre-39 state's eventual value: **UNKNOWN**; use policy-only supervision with `has_value=False, q_valid=False` if the longer-scope certificate is independently accepted.
- Actual post-39/pre-40 state: **proved terminal loss** by the composed strategy; no safe policy alternative exists under that scope.
- Original 43/44 states: proved losses with zero safe alternatives; do not invent a preferred winning move.
- No model candidate, upload, promotion, or new training branch was created.

## Reproduction and bindings

From repository root:

```powershell
.venv/Scripts/python.exe workspace/colony-one-level4/codex/research/loss_10910_review.py
.venv/Scripts/python.exe workspace/colony-one-level4/codex/research/loss_10910_review.py --actions 40 --extended40
.venv/Scripts/python.exe workspace/colony-one-level4/codex/research/loss_10910_review.py --actions 39 --extended39-first 2738
.venv/Scripts/python.exe workspace/colony-one-level4/codex/research/loss_10910_review.py --actions 43,44 --counterfactual
```

The helper prints replay results, typed state/action records, all relevant classifications, endpoint certificates, and resource counts to stdout. Initial exploratory 300,000-node exhaustion was treated as UNKNOWN and not used as a proof; the final completed witnesses above use explicit larger finite budgets and no unresolved branches.

- Raw `codex/matches/site-all-v1-derived/match-10910.txt`: `98695de1b45eb2bee22b64d6ba004fb97864684a3eeb8e6219d48aa8dd4bc6bc`.
- Helper `loss_10910_review.py`: `5ef501abdc02fb1a096dc473f17d3ba8e4f540d8a183513020bbe2e32f76cb03`.
- `codex/engine_audit/oracle.py`: `4a9c4297aa107f4f5df3774aef51af84259299417ebacea8fe2bd7e605fa5dfc`.
- `codex/engine_audit/own_turn_oracle.py`: `993fd3c5a3aea4164991413ca110fbc85b392c612bf9ad8bad9903f3ae686ff9`.
- `env/colony_one/env.py`: `0bef52825302c38d8c84e6ed02b0502ef1295ab50c1543d0bdda8936fc0044f5`.
- `env/colony_one/board.py`: `a1b23455c22047db17f835c2fad2004754ecf9e5479b17afb1a9bce3281c8f5d`.
