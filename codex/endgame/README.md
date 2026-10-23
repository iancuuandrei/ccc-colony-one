# Exact two-bare-Kernel override

This optional wrapper preserves every original output on ordinary positions with any non-Kernel piece. It overrides only states with exactly one own Kernel, one enemy Kernel, no other pieces, no overlapping Kernels, neither Kernel on a locked square, and valid actions-left/actual-side scalars. The unchanged base models remain in their original paths. Nothing in this directory is promoted or uploaded.

## Candidate files

- `model-tactical-endgame.onnx`: tactical base SHA256 `8fef2d4aef5efe63e857432409f055974c4ed1a625d80871f5e5c2b177487b5b`; wrapper SHA256 `fcc4858fd61e77c6d7765acb87c57b4835aedcded10e8873009eb60d8e491969`; 372,391 bytes.
- `model-h1-endgame.onnx`: h1 base SHA256 `0940ddada8c2b761126ffbc5532d3103c3b8cb4eb20d89f7923f70d03f2e8b65`; wrapper SHA256 `23a7b2d15ee6cb59f39c7c7b6e18437b964f6421a0adbb38ba229aa34078d28d`; 349,512 bytes.
- Each wrapper adds 88 conventional nodes and 33,572 constant bytes. Its `.proof.onnx` sibling exposes the isolated solver only for verification; it is not an upload candidate.

## Exact mathematics and timing

Let `s` be own Kernel position, `e` the enemy position, and `A` the Chebyshev-distance-one adjacency matrix. Let `R(t)=min(row,col,7-row,7-col)` be inward ring index (larger means closer to centre). A final own-turn destination `t` wins iff it captures `e` immediately, or `distance(t,e)>2` and either `R(t)>R(e)` or equal rings and own actual side is Magenta. Landing on a locked square is excluded.

With one action left, this directly scores legal first moves. With two actions left, a first move to `t` wins iff it captures immediately or some legal adjacent second destination wins under the one-action rule. This is MAX then MAX, since the same player moves twice. The value is +1 iff any legal first move wins, otherwise -1. Winning actions receive logits +20 and losing actions -20; when all actions lose they tie, correctly reflecting their equal game-theoretic value. The referee masks other from/to pairs, so a destination-wise policy broadcast is sufficient in the bare-Kernel domain.

The environment terminates this domain by the end of the current turn. No opponent reply or future search is required. Kernel capture preempts every turn-end rule. Bare-Kernel adjudication precedes firewall, including round 26 and simultaneous impending burns. Therefore urgency is intentionally absent from this solver: present lock geometry controls legal movement, but the next shrink never executes on a noncapture bare-Kernel turn. Actual side plane15 is essential for equal-ring ties.

The override is a closed-form rule calculation, not a fitted value heuristic. It does not apply to mixed-piece positions, one missing Kernel, a current Kernel on a locked square, or invalid remaining-action/side scalar. It assumes engine-consistent binary spatial observations and concentric legal firewall geometry for its exhaustive domain claim. The official numeric-extreme corpus also executes finitely, but malformed tensors do not carry game semantics.

## Reproduction and evidence

From repository root with the existing `.venv/Scripts/python.exe`:

```powershell
.venv/Scripts/python.exe workspace/colony-one-level4/codex/endgame/build_endgame.py --base workspace/colony-one-level4/codex/tactical/model-tactical.onnx --output workspace/colony-one-level4/codex/endgame/model-tactical-endgame.onnx
.venv/Scripts/python.exe workspace/colony-one-level4/codex/endgame/prove_endgame.py --proof workspace/colony-one-level4/codex/endgame/model-tactical-endgame.proof.onnx --candidate workspace/colony-one-level4/codex/endgame/model-tactical-endgame.onnx --base workspace/colony-one-level4/codex/tactical/model-tactical.onnx --out workspace/colony-one-level4/codex/endgame/proof-results.json
.venv/Scripts/python.exe workspace/colony-one-level4/codex/endgame/verify_composed.py --model workspace/colony-one-level4/codex/endgame/model-tactical-endgame.onnx --out workspace/colony-one-level4/codex/endgame/composed-results.json
.venv/Scripts/python.exe workspace/colony-one-level4/codex/validation/qualify.py workspace/colony-one-level4/codex/endgame/model-tactical-endgame.onnx --mode graph --output workspace/colony-one-level4/codex/endgame/graph-validation.json
```

`prove_endgame.py` constructs every ordered pair of distinct open squares for both actual sides and both remaining-action counts. It enumerates all first actions through `env.step`; for a surviving first action with one own action remaining it enumerates every second action through `env.step`. It compares all legal action outcomes against the exported solver. Its domain includes every legal static bare-Kernel configuration; it does not claim every such configuration is reachable from the supplied starting scene. Covering this superset covers the reachable subset. Full enumeration repeats at rounds1,8,9,16,17,24,25,26 to exercise turn-end/shrink ordering. Source hashes bind evidence to the engine used.

`proof-results.json`: PASS, 44,352 state/round combinations and 283,648 legal first-action comparisons, with all available second continuations evaluated through the engine; zero policy/value mismatches and no unexpectedly disabled states. On 256 nonbare random-play positions, composed policy and value are bitwise equal to the base. Runtime 182 seconds on this host.

`composed-results.json`: PASS, all 22,176 distinct legal static configurations across four lock stages, both actual sides, both remaining-action counts, composed policy/value exactly equal isolated solver results. `graph-validation.json`: PASS authoritative structural/constant-index checks plus all 24 official sparse/numeric-extreme corpus cases at batch 1/8/16. The corresponding h1 composed and graph checks are recorded in `composed-h1-results.json` and `graph-h1-validation.json`. Full isolated resource and tournament-strength checks are separate work; these passes are not grader acceptance.
