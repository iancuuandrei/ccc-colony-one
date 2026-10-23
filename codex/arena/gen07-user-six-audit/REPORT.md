# Current GEN07 six-game counterfactual audit

Fixed current checkpoint `ckpt/gen07-endgame.onnx`, SHA256 `8b23b6a7f80089aef590beb69a829738e3c8c66dcd5ccbb652820142dbabcebd`, matches `release/gen07/model.onnx`. Referee MCTS100 defaults, RNG seed 0 per decision, one CPU process / one ORT thread. Historical source-model identities remain UNKNOWN. No old-model inference, strength games, training or model export.

| Case | Actor / user | GEN07 first | Actual GEN07 second | Supplied certificate / actual result |
|---|---|---|---|---|
| 18587/36 | MAGENTA / MAGENTA | G5->D2 | n/a | missed safe set; terminal enemy reply replayed |
| 18972/62 | CYAN / CYAN | C4->C3 | n/a | missed safe set; terminal enemy reply replayed |
| 19040/93 | CYAN / CYAN | C4->C3 | C3->E5 | already forced: empty safe support; terminal enemy reply replayed |
| 19040/94 | CYAN / CYAN | D5->E5 | n/a | already forced: empty safe support; terminal enemy reply replayed |
| 19101/67 | MAGENTA / MAGENTA | F5->F6 | F6->F5 | two-enemy-turn safe start; actual completion bounded-loss VERIFIED |
| 19101/68 | MAGENTA / MAGENTA | F6->F5 | n/a | two-enemy-turn unsafe start; actual completion bounded-loss VERIFIED |
| 19101/71 | MAGENTA / MAGENTA | F5->F6 | E6->D6 | already forced: empty safe support; terminal enemy reply replayed |
| 19101/72 | MAGENTA / MAGENTA | E6->D6 | n/a | already forced: empty safe support; terminal enemy reply replayed |
| 19255/37 | CYAN / CYAN | B3->B4 | B4->C4 | missed safe set; terminal enemy reply replayed |
| 19255/38 | CYAN / CYAN | B4->C4 | n/a | already forced: empty safe support; terminal enemy reply replayed |
| 19490/67 | MAGENTA / UNKNOWN | F5->F6 | F6->F5 | two-enemy-turn safe start; actual completion bounded-loss VERIFIED |
| 19490/68 | MAGENTA / UNKNOWN | F6->F5 | n/a | two-enemy-turn unsafe start; actual completion bounded-loss VERIFIED |
| 19490/71 | MAGENTA / UNKNOWN | F5->F6 | E6->D6 | two-enemy-turn unsafe start; terminal enemy reply replayed |
| 19490/72 | MAGENTA / UNKNOWN | E6->D6 | n/a | two-enemy-turn unsafe start; terminal enemy reply replayed |

Corrections still needed:

- **18587/36:** unique safe Kernel E7->F6; GEN07 keeps the unsafe Trace G5->D2.
- **18972/62:** unique safe C4->B4; GEN07 chooses C4->C3.
- **19255/37:** B3->C3 then C3->C2, or Kernel B2->C2 then B3->B2, avoids the checked terminal loss. GEN07 chooses B3->B4 then B4->C4.
- **19490/68 and exact alias 19101/68:** unique two-enemy-turn-safe Kernel E6->F5 (canonical index 2786); GEN07 chooses Trace F6->F5 (2722). At pre67 it chooses F5->F6, which has a safe possible completion, but its actual second action is the same unsafe Trace return. A safe-first-action membership alone would conceal this failure.

Independent deterministic-state equivalence PASS covers 19101/19490 actions63,65,67,68,71,72. The earlier67/68 outputs are reused with zero new inference; final71/72 also exactly duplicate. These are two source games with one repeated ending, not separate tactical evidence. The final71/72 positions are already forced under the supplied horizon. The final 19040/93-94 positions likewise have no safe action target.

The separate 19040 miner followup reports every choice89/90 loses within two enemy turns and every own completion from85 plus every choice86 loses within three enemy turns. The actual86 Trace move survives the shorter two-turn horizon. No further nonempty corrective support was supplied, so no additional GEN07 query was needed. Those longer proofs were not independently re-enumerated by this inference audit; the earliest avoidable commitment before85 remains UNKNOWN. Sources are pinned in additional-proof-notes.json.

All exact packet observations and legal masks matched complete engine replays. Actual GEN07 followthrough was executed on every same-actor second action. Any available immediate enemy-terminal reply was replayed to its terminal outcome. No supplied winning-first-action certificate was selected or used to claim an actual model win.

Provenance is the captured `loop-state-start.json`, per-run loop-state snapshots, actual artifact hashes and per-run manifests. No unverified ablation label is asserted: the captured state records main arguments `--ext-frac 0.06`, variant arguments including `--policy-mix 0.5`, and chosen `gen07`. Those configuration records do not reconstruct the exact arguments loaded by a historical training process.

Every mask is used only at its documented horizon. Eventual values for the 19490 critical packet are UNKNOWN and masked. Forced-state choices are not avoidable tactical errors merely because a terminal enemy reply exists.
