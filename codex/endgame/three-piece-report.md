# Exact three-piece capture-to-bare certificate

Status: **PASS for the specified sufficient certificate**, with exhaustive admitted positive geometry checks and explicit reductions for enemy piece type and round. This is not a complete three-piece game solver, not exhaustive testing of every arbitrary tensor, and not a proof that every tested static board can arise from the supplied opening. No ONNX, model, network, training or export was changed.

Pure helper: `three_piece_certificate.py::certificate(env)`. It returns all advertised real-coordinate winning paths and canonical first-action indices, or `None`, meaning UNKNOWN rather than loss. The independent `verify_paths` executes each path through `env.clone()` and `env.step()`, asserts exact final victory, exactly two remaining Kernels, and an unchanged firewall lock matrix.

## Domain and mathematical argument

The current side has exactly one piece, its Kernel at s. The opponent has exactly two pieces, its Kernel at e and one non-Kernel at t. All three occupy distinct unlocked squares. The actual side to move and actions-left are taken from the environment, not inferred by alternating action parity.

Let d be Chebyshev distance and `R(p)=min(row,col,7-row,7-col)` the inward ring index, increasing toward the centre. Require `d(t,e)>2` and either `R(t)>R(e)`, or equal rings with actual mover Magenta.

- With one action left, require `d(s,t)=1`. Capture the enemy non-Kernel at t on the final own action.
- With two actions left, require `d(s,t)=2` and explicitly enumerate an unlocked **empty** intermediate m satisfying `d(s,m)=d(m,t)=1`. Move s→m, then capture t on the final own action. Only those exact first actions may receive a certificate-derived policy bonus.

Each action is a legal own-Kernel move. There is no self-check rule, so an attacked intermediate does not invalidate it. No opponent action occurs between these two actions. The first action in the two-action case is empty-square movement, so it neither captures a Kernel nor removes the last non-Kernel; the turn continues with the same player. The final action captures a non-Kernel, so Kernel bodyguard redirection does not apply.

After the final capture exactly the two Kernels remain, the acting Kernel is at t, and the opponent Kernel remains at e. The distance and inward-ring/actual-side tests are precisely the engine's bare-Kernel winner rule in this branch. This adjudication runs before firewall, including rounds8/16/24/26. Therefore every returned path ends in victory for the current side by the end of this same turn. The helper does not treat merely safe movement as a win and supplies no negative value certificate.

### Why enemy type does not affect this family

The opponent does not move during the witness. Only the moving Kernel generates actions; Kernel contact capture treats every enemy non-Kernel identically. The enemy non-Kernel is captured directly, not selected through the Kernel guard-priority mechanism. Enemy attack geometry does not constrain legal King moves because there is no check filter. Thus replacing Ping with Daemon, Exploit, Trace or Ice changes neither path legality, capture semantics nor bare-Kernel winner. The proof checks all geometry with Ping and separately executes all other types on spread samples; it does not falsely describe this as exhaustive five-type enumeration.

### Why current round does not affect this family

Given the same present locked squares, movement/capture rules do not consult round number. Final bare-Kernel adjudication preempts the only later round-dependent operation, firewall. Advancing the round after Magenta acts does not change the stored acting team used for bare adjudication. Hence the certificate extends to every legal round with that lock stage. Every admitted geometry was checked at its stage's shrink deadline; other legal clocks were additionally crosschecked. There are no admitted geometries on the2×2 board because final Kernel separation cannot exceed2.

## Executed evidence

Run from repository root:

```powershell
.venv/Scripts/python.exe workspace/colony-one-level4/codex/endgame/three_piece_certificate.py
```

Recorded in `three-piece-report.json`, source-bound to helper SHA256 `c97f7dc7ff2823d9a731a363fb1d8ac7907c379e5ba9ae523e165d8841d4f9f2` and exact environment source hashes:

- **68,800** admitted positive geometry/side/actions-left states, all advertised **111,920** paths executed independently through the engine. Exhaustive over every such geometry on all four live lock stages, both Cyan/Magenta and one/two remaining actions, with Ping as representative enemy non-Kernel.
- Other four enemy types: **1,412** spread states and **2,124** independently executed paths.
- Other legal round clocks: **483** spread states and **693** independently executed paths. Zero-eligibility strata are explicitly recorded.
- Two real actual-protocol leaves: three verified paths total. All assertions passed; no failure was silently skipped.
- Runtime145.15 seconds on this host. Validation digest `b4454365c2b20792e11542b29572658784f44386f2ee107b9697878e7c88c182`.

| Actual leaf | Current side / pieces | Certified continuation |
|---|---|---|
|9928 after action28 Ice D5→B3 guard redirect|Cyan Kernel B3; Magenta Kernel G7 + Ice D5; two actions|B3→C4→D5; first canonical index2658|
|9366 after action52 Daemon E6→E4|Cyan Kernel C3; Magenta Kernel G7 + Daemon E4; two actions|C3→D4→E4 or C3→D3→E4; first canonical indices2723/2731|

Actual protocol hashes and exact coordinates are included in JSON. In9928 the fixed referee's redirected edge never caches its child, so recognizing this certificate in the **scalar evaluator value** addresses information that deeper search cannot obtain on that edge. A child policy bonus alone cannot cross the cache barrier. In9366 search can discover the terminal family, but recognizing its leaf is still exact support rather than approximate value.

## Opus-only integration contract

This deliverable is the pure helper/spec/proof only. Opus owns any addition to the existing shared model and exporter. A possible port ORs this sufficient certificate into the existing own-turn win flag, preserving its+.99 convention, and optionally bonuses only the advertised winning first-action set. The current root player must not be labeled a loss just because the *opponent*, if it moved now, has a certificate. Apply the rule to the actual team-to-move observation.

Any port still needs its own torch/ONNX parity against this helper, batch1/8/16 and canonical-side checks, independent positive-certificate engine verification of the exported result, published resource validation and shared GEN02 competitive gate. None of those model checks was run by this task. This rule deliberately excludes adjacent-target/two-action quiet detours and all positions with additional material; `None` is UNKNOWN for every excluded case.
