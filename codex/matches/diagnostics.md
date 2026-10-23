# Bounded h1 failure probe

Four h1-vs-h1 games, seeds 12001-12004, authoritative Level 4 engine and MCTSConfig(simulations=100), default opening noise/temperature, one process, current model SHA256 recorded in each game. No tournament-strength inference is supported by four games.

- All four ended by kernel capture, at actions 30, 28, 18 and 16. Both colors won twice. No firewall or bare-kernel terminal occurred.
- Every loser finished its last turn with an unguarded kernel not under an immediate legal attack. The opponent moved an Ice or Daemon once, then captured the kernel as its second action.
- Exact exhaustive alternative-action checks at the loser's final action found 0/11, 4/27, 24/27 and 19/27 actions respectively that avoided an opponent win within its next turn. This is a one-turn safety qualification, not proof of eventual survival.
- Seed 12004: Cyan Kernel (4,3)->(3,3) captured Exploit. The resulting Magenta-to-move h1 value was -0.6818, yet Daemon (0,5)->(3,5)->(3,3) wins immediately. Chosen action prior was +4.42; safe Daemon (7,2)->(4,2) prior +0.70.
- Seed 12003: Magenta Kernel (1,6)->(2,5) captured Ping. The resulting Cyan-to-move value was -0.0422, yet Ice (6,1)->(4,3)->(2,5) wins. Chosen action prior +7.10; safe quiet actions were around +0.30.
- Immediate kernel-capture positions were valued only +0.4547 and +0.0928 in two games. An exact win override corrects this certainty even where MCTS found the capture.

Likely improvements, in priority order: add supervision or in-graph recognition for two-action approach/capture wins; use those exact labels to demote unsafe kernel captures and withdrawals from bodyguards; retain exact current-attack/guard-removal terminal overrides. h1's 1-action destination danger map cannot see threats requiring one quiet approach, and large material-capture priors compete against small guard terms. Its attack geometry follows the authoritative implementation, including all-eight-direction Daemon adjacent shots.

Code-level omissions without empirical endgame evidence here: h1 ignores observation plane 15, so cannot distinguish Magenta tiebreaks or side-specific firewall timing; the firewall kernel penalty only sees the current lowest ring and urgency, not exact remaining own actions; material threatened by firewall is not penalized. These need endgame positions or downloaded longer matches before attributing tournament losses to them.

Artifacts: selfplay-h1.json is the original capture; selfplay-h1.corrected.json corrects an initial terminal-reason diagnostic typo; selfplay-h1.summary.json contains corrected summaries. last-turn-avoidability.json records exact alternative actions and winning witness continuations. probe_matches.py replays capture and arena histories through the authoritative engine and verifies recorded winner; last_turn_probe.py performs bounded exact tactical enumeration.
