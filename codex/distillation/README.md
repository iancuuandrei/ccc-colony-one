# Astra distillation queue

`queue.py build` creates a compact exact-state packet from replayed matches and gen00. Packet0001 has20diverse positions including avoidable last-action losses, guard redirects, firewall timing and ambiguous search states. Piece names and readable coordinates use the real absolute board; action indices always use the current mover's canonical encoding. Piece placement, locks and scalar planes reconstruct the full float32 observation losslessly. Sources include content hashes and complete-game grouping keys.

The GPT-6 Astra teacher writes `teacher_labels/astra_packet0001.json` according to `teacher_answer.schema.json`: ranked legal actions with positive nonincreasing weights, mover-perspective value, policy confidence, optional separate value confidence, and short factual evidence. Strategic values are predictions; they are never described as actual self-play outcomes. No private reasoning is requested or stored.

`validation.py` checks exact state/source bindings and every ranked action against the supplied engine. Exact own-turn-win claims require a legal winning top action verified by `engine_audit/own_turn_oracle.py`; optional witnesses are replayed. Exact forced-loss-by-next-turn claims require the separate bounded oracle certificate, otherwise they are rejected. Exhausting a verification cap means unknown. Absence of a current-turn win does not imply a loss. Prediction evidence cannot claim unsupported guarantees.

Consume validated labels:

```powershell
.venv/Scripts/python.exe workspace/colony-one-level4/codex/distillation/queue.py consume --packet workspace/colony-one-level4/codex/distillation/packets/packet0001.json --labels workspace/colony-one-level4/codex/distillation/teacher_labels/astra_packet0001.json --out workspace/colony-one-level4/codex/distillation/validated/packet0001
```

This writes a validation receipt, `teacher_targets.npz` and `student_ready.json`. The NPZ contains observations, sparse policy indices/normalized weights, teacher values, policy/value confidence, position IDs, game groups and exact-claim flags. Student workers consume this distinct teacher corpus alongside, not disguised as, empirical outcomes. Repeat with a new packet ID and separate output directory for subsequent iterations; do not overwrite validated incumbent labels or models.

`queue.py ingest` is the incremental batch-loop entrypoint: scan packets for matching `teacher_labels/astra_<packet_id>.json`, validate each available pair and publish its student-ready manifest. Repeated identical inputs are allowed; changed packet/label contents require a fresh output directory. This does not invent teacher labels, initiate model calls or silently train/promote a student.

Use complete source-game groups for holdout: history source path+seed, or gen00 source path+game_index. Twenty labelled positions are only a bounded distillation pilot; independent paired matches and exact model validation remain necessary for any upload candidate.

Packet0002 adds12states from real platform-loss games4650/4674. Its upstream NPZ expanded70→134rows during labeling. Before provenance revision1, all12 observations, match/ply keys and recorded actions were checked identical against the expanded source, which was frozen under `sources/` by SHA256. Original packet revision0 remains in `packets/revisions/`. Teacher state hashes and labels did not change. Future site-packet builds freeze their source immediately. For grouping copies/revisions of site games, use logical `site:<match_id>` rather than the snapshot filename, so repeated matches cannot cross training/holdout.
