# Fixed GEN06 confirmation against deployed GEN03

**GEN06 wins335/400 (83.75%) against deployed GEN03. Its separate fixed-GEN02 anchor is186/200 (93%). All600 new histories independently replayed legally, and the preceding400-game internal gate was also verified.**

Exact candidate: `release/gen06/model.onnx`, SHA256 `123467a930e0598c464ae7694800ad565aeac71680b88dafc5f014bfe7cf50ac`. It remained fixed throughout both gates. No variant, later generation, model creation, automatic promotion or upload was performed by this task.

| Fresh gate | Seed range /pairs | GEN06 wins | Cyan | Magenta | Seed-paired bootstrap95% |
|---|---|---|---|---|---|
|GEN06 vs confirmed deployed GEN03|252000–252199 /200|335/400 (83.75%)|177/200 (88.5%)|158/200 (79%)|80.5–87.0%|
|GEN06 vs fixed GEN02|253000–253099 /100|186/200 (93%)|93/100 (93%)|93/100 (93%)|89.0–96.5%|

Direct pair outcomes:136 GEN06 sweeps,63 splits,1 GEN03 sweep. Conservative paired Hoeffding95% interval:74.15–93.35%. Both color rates independently exceed50%; their point estimates differ and should remain visible rather than being described as equal.

Anchor pair outcomes:87 GEN06 sweeps,12 splits,1 GEN02 sweep. Conservative paired interval:79.42–100%. These local opponent results do not guarantee unknown tournament performance or establish universal strength.

GEN03 hash: `0cf937446ba89dc9b5b3652edbcb4025030bc785e924551782f4b2d298ebe96f`.
GEN02 hash: `579d3ec01da7328dc706003b8dcc5eba357bb74902252187e45633ca0595a9a0`.

## Protocol and verification

Both fresh gates used the unmodified supplied MCTS with100 fixed simulations, all other referee defaults, one ORT inference thread, and at most8 search workers. They ran sequentially after native qualification PASS was explicitly confirmed. Each seed was played in both colors, starting the same NumPy RNG state across the swap. Different branches can consume different randomness; identical openings are not assumed. The complete seed pair, rather than each individual game, is the uncertainty unit.

The model, engine, scene, runtime and harness were pinned in each `manifest.json`; hashes were rechecked at completion and during the independent legal-history replay. Engine/scene bundle SHA256: `2e6a926a41230225c7f7c58855b704d95c3c4fee1025211f860a34c766230ecd`. Scene SHA256: `5489fce67399e60fa5d291d249cb4e524dd90705da5221cb9e2f1a690338270f`.

Every recorded action was resolved against the engine's legal moves, with team/round/actions-left checked before execution. Final winner, round, game length, candidate-win flag, unique seed/color identity, complete pair membership and aggregate win counts all matched. All600 new histories passed. `engine-history-audit.json` in each directory binds the input hashes and retains per-game PASS checks; `replay.py` creates no new games.

Direct `games.jsonl` SHA256: `db859623bde51db4767ed72f6f83774a21c55cc1833893403ea1c7c4f7b92805`.
Anchor `games.jsonl` SHA256: `ba310d69dcfb69c0b22cf8420f9fb594589da0e45ba65b494bb2f2f3bfcf65f6`.

Each run retains its full histories, final summary, protocol manifest, and exact `harness_snapshot.py`. No raw challenge data was edited.

## Independent tactical check against GEN05

See `../gen06-tactical-audit/REPORT.md` and its source-bound manifests/results. Both models were independently evaluated on frozen48 and the22 actual-site hard snapshots from v1/v2/v3, after qualification PASS.

- Frozen48:0 missed certified own-turn wins and0 avoidable next-turn losses for both models. Exact-value MAE improves0.337635→0.262984 across32 exact-value states. All48 earlier GEN05 action choices reproduced exactly. The16 unknown values remain masked; correlated authored variants remain quarantined TEST ONLY.
- Actual-site cases: avoidable-loss flags improve7→6; missed-own-turn-win flags improve7→1; no new certified decision regression. Familiar site cases may already be in shared training/re-search and are not evidence of unseen generalization.
- **8623/action33 is repaired:** GEN06 actually executes `E6->E5`, then `E5->G7`, ending in a CYAN win. All7 selected winning site starts were followed through with the model's actual second MCTS100 action;7/7 terminate victorious.
- Remaining unresolved cases: six avoidable-loss snapshots and one missed current-turn win (`8948/47`), explicitly listed in the tactical report. This is improvement, not complete tactical remediation.

No frozen holdout observation, certificate, failure or membership was changed or exported to training.

## Earlier internal gate and qualification

The existing Opus `logs/gen06-gate.json` was separately audited before tactical inference:400/400 histories PASS,297/400 wins against GEN05 (167/200 Cyan,130/200 Magenta), seeds906000–906199. Candidate/baseline checkpoints match their release hashes. This earlier gate uses the original arena's different RNG streams across colors and is not conflated with the fresh common-RNG confirmation above. Its current artifact and history provenance is in `../gen06-v-gen05/existing-gate-audit.json`; legal replay does not regenerate historical MCTS or supply a missing historical engine hash manifest.

The native qualification owner confirmed PASS for exact candidate hash123467… before tactical/strength evaluation: unchanged Linux validator ACCEPTED, hard check and Level3 limits PASS, raw batches1/8/16 across24 corpus inputs PASS, and a complete58-action100-simulation selfmatch PASS. The actual graph has696 nodes and1,479,383 bytes. Qualification artifacts remain under `codex/validation/gen06-*`; parallel arena timings are not used as resource qualification. No remote grader acceptance is claimed here.

All owned GEN06 strength/tactical/replay processes have ended and all8 arena slots are free. Root was explicitly notified to restore actors as appropriate. Root and Opus own the final publication decision; no additional inference, generation chasing or gate was started.
