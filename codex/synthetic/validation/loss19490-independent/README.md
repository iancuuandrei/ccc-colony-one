# Independent 19490 certificate verification

**PASS.** All74 recorded actions, piece identities, captures, bodyguard redirections and final terminal outcome replayed through the supplied engine. CYAN wins; MAGENTA is the losing actor. The user side and historical model remain UNKNOWN.

| Before action | Legal starts | Safe through one enemy turn | Safe through two enemy turns |
|---|---:|---:|---:|
|67|7|7|4|
|68|5|2|1|
|71|7|0|0|
|72|5|0|0|

The complete two-enemy-turn-safe first actions at67 are Kernel E6-E5 (2787), Kernel E6-D5 (2788), Trace F5-F6 (2218), and Trace F5-E5 (2211). Each has a demonstrated safe own-turn completion. Every completion after each of the other three first actions admits a forced enemy win within that horizon.

After the actual67 Trace F5-F6, **68 Kernel E6-F5 (2786)** is uniquely bounded-safe. All59 complete CYAN replies were checked, and each permits a MAGENTA response avoiding terminal defeat during the following CYAN turn. Its eventual full-game value is UNKNOWN.

The actual68 Trace F6-F5 (2722) survives one enemy turn but loses through the second. The recorded69/70 sequence reproduces pre71 exactly; all seven first actions and every completion there lose within the next CYAN turn. All five pre72 actions also lose. No arbitrary best move is assigned to those forced-loss states.

`verify.py` uses an independent full `env.clone/step` turn composition, reusing the previously independently validated full-turn verifier and the supplied own-turn terminal oracle. It does not import the original19490 mining/minimax code. Positive leaf terminal witnesses were replayed through `env.step`. Work completed:242 endpoint states,1,431 complete defender turns,1,314 terminal-oracle leaf checks.

`proof.json` contains the complete checked action sets, quantifier witnesses, source hashes, round/action-budget bindings and resource counters. All four packet masks match the independent results, all frozen inputs are unchanged, and there are zero D4 orbit intersections with the quarantined holdout. No model inference, training, release or upload was performed. No earliest full-game mistake or eventual winning repair is claimed.
