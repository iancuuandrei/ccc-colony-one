# Match 19490: recoverable at action 68, forced by action 71

The raw replay is valid: **CYAN wins after 74 actions; MAGENTA is the losing actor.** The text identifies neither the user side nor the historical model. Neither is attributed to GEN07.

The decisive earlier error demonstrated here is **68 MAGENTA Trace F6→F5**. After the recorded first action **67 Trace F5→F6**, the sole alternative surviving the checked two-enemy-turn horizon is **68 Kernel E6→F5** (canonical index 2786). It survives all **59 complete next CYAN turns**, with a MAGENTA continuation preventing defeat through the following CYAN turn in every case. This is bounded recoverability; eventual victory remains **UNKNOWN**.

The recorded 68 instead allows the recorded **69 CYAN Kernel E3→E4; 70 Exploit C4→E3**. This reaches a state in which **every one of MAGENTA's 7 legal first actions, including every possible second action, loses by CYAN's next turn**. At action 72, all 5 remaining choices also lose. Blaming only the final Kernel E6→D6 move would miss that the loss was already forced.

| Before action | Remaining own actions | Legal starts | Terminal wins this turn | Safe through next enemy turn | Safe through two enemy turns |
|---|---:|---:|---:|---:|---:|
| 67 | 2 | 7 | 0 | 7 | 4 |
| 68 | 1 | 5 | 0 | 2 | 1 |
| 71 | 2 | 7 | 0 | 0 | 0 |
| 72 | 1 | 5 | 0 | 0 | 0 |

All legal alternatives in these sets are classified; no budget-exhausted UNKNOWN action remains. "Safe" is horizon-specific and does not assert an eventual win.

## Exact action sets and witnesses

Before 67: MAGENTA Kernel E6, Trace F5; CYAN Kernel E3, Exploit C4. The inner 4×4 board is open, round 17, two actions remain. The complete two-enemy-turn-safe first-action set and one corresponding safe own-turn completion are:

| First action | Canonical index | Demonstrated complete own turn |
|---|---:|---|
| Kernel E6→E5 | 2787 | Kernel E6→E5→F6 |
| Kernel E6→D5 | 2788 | Kernel E6→D5; Trace F5→E5 |
| Trace F5→F6 | 2218 | Trace F5→F6; Kernel E6→F5 |
| Trace F5→E5 | 2211 | Trace F5→E5; Kernel E6→D5 |

The other legal starts, Kernel E6→D6 (2796), Kernel E6→F6 (2794), and Trace F5→F4 (2202), are forced losses through two enemy turns, after **all** their own second-action alternatives.

Before 68: MAGENTA Kernel E6, Trace F6; CYAN Kernel E3, Exploit C4; one action remains. The full legal set is Kernel E6→E5 (2787), E6→D6 (2796), E6→D5 (2788), E6→F5 (2786), and Trace F6→F5 (2722). Only **2786** survives two enemy turns. The actual 2722 survives the first enemy turn but loses through the second; the other three already allow a first-turn terminal win.

Before 71: MAGENTA Kernel E6, Trace F5; CYAN Kernel E4, Exploit E3, round 18. Full legal set: Kernel E6→E5/D6/F6/D5 (2787/2796/2794/2788), Trace F5→F6/F4/E5 (2218/2202/2211). All lose within the next CYAN turn, under all MAGENTA completions. Complete typed witnesses for each own completion are in the JSON.

Before 72: MAGENTA Kernel E6, Trace F6; CYAN Kernel E4, Exploit E3. All 5 legal actions lose:

| MAGENTA action | CYAN terminal witness |
|---|---|
| Kernel E6→E5 | Kernel E4→E5 twice; first attack removes the adjacent Trace guard |
| Kernel E6→D6, recorded | Kernel E4→E5→D6 |
| Kernel E6→D5 | Kernel E4→E5→D5 |
| Kernel E6→F5 | Kernel E4→F5 twice; first attack removes the adjacent Trace guard |
| Trace F6→F5 | Kernel E4→F5→E6 |

The last witness also explains why retaining the Kernel on E6 and merely returning the Trace is insufficient after CYAN has already advanced its Kernel to E4.

## Evidence and limits

- Immutable source copy: `codex/matches/19490/match-19490.txt`, SHA256 `d52cc466fa8e9ed53b91fe561022ff6056973d6f89f0a7035bb9e44e063e97cb`; byte-equal to `D:/Downloads/match-19490.txt`.
- Strict full replay: `replay.json`; every action, team, piece, capture, and terminal outcome checked by the authoritative engine. `source-receipt.json` pins engine, oracle, scene, and parser hashes.
- `diagnostics.json` checks **all 36 MAGENTA decision states** for exact own-turn wins and forced loss through the next enemy turn. There was no missed terminal own-turn win at any of them. All recorded MAGENTA choices before 71 survive that shorter horizon. The deeper complete analysis is specifically actions 67/68; no claim is made that all earlier choices survive a longer horizon.
- `deep-proofs.json` composes complete own/enemy turns and exact one-next-turn-loss leaves. The two positions required 535 enumerated enemy completions, 74 distinct exact loss-leaf checks, and 29,621 leaf action nodes. A positive terminal own-turn win for the defender disproves a loss branch directly. The safe alternative at 68 exhausts all 59 enemy completions.
- `critical-states.npz` contains exact float32 observations for 67/68/71/72, all legal canonical indices, recorded actions, complete winning masks, and **separate one- and two-enemy-turn masks**. No horizon is silently substituted for another. Every row shares `source_group=site:19490`.
- Packet NPZ SHA256: `333dbc28bfa7a06967c5267a0dc97e039ef9b0fb9c9a7e2d807d971b5a47495d`. Companion JSON SHA256: `381f5b4132de5a0bfaf39d581e8b42a784baf1ce3c3c3444be2cc99a58ed4cc3`. Proof JSON SHA256: `7809f9b9cb306d2499f53ea9fba14679ef2d8daa7dfe9fffdc1f5fcdc3943132`.
- This is a diagnostic packet, **not a training release**: `has_pol`, `has_value`, and `q_valid` are all false; numeric values are finite masked placeholders. Forced-loss certificates at 71/72 do establish loss, but no arbitrary policy preference is invented there. The surviving 68 alternative receives no positive eventual value label.
- Current GEN07 counterfactual evaluation was handed to `candidate_arena` using the immutable exact states and horizon-separated masks. Independent deep-proof verification was requested from Astra. Results are owned by those agents; this file makes no GEN07 strength claim.

Reproduce from the level directory: run `analyze.py`, then `deep_probe.py 68 67`, then `finalize.py`, all under `codex/matches/19490/`. No model evaluation, training, export, 800-simulation search, or raw outcome corpus import was performed here.
