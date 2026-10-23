# Teacher labels and mathematical scope

Teacher packet 1 contains 20 engine-restored positions from local games/training data. Packet 2 contains 12 engine-restored positions from observed website games. The numerical judgments and within-tier preferences were authored by the assigned teacher. Helper scripts supply bounded exact evidence rather than pretending a hand-written evaluator is a language model.

## Exact label meaning

`value` is always from the current mover's perspective. The same team moves twice, so the value sign does not change after its first action.

An own-turn win is an existential certificate: at least one legal sequence of at most `actions_left` actions terminates with that mover winning. Winning first actions receive equal high policy weight, even when a deterministic order is required by the schema. The independent validator replays the supplied winning witness.

A next-turn loss is a universal certificate over all current-turn choices, followed by an existential terminal-winning opponent reply. For last-action positions the helper checks every legal first action. For two-action positions it checks all second actions until a safe continuation is found; a failed search exhausts all continuations. A safe continuation means only that this particular bounded tactical loss is absent. It is not a game-theoretic win or a guarantee of eventual survival.

Packet 1 has two certified winning positions and one certified losing position. Packet 2 has five certified winning positions and one certified losing position. Other values are predictions with separate, generally much lower value confidence. Equal-material opening estimates of zero mean uncertainty; the game has no draws. The all-action losing positions have uniform policy weights because the proof supplies no preference among their equal loss values.

## Useful feature relationships

1. A currently legal attack on the enemy Kernel with `g < actions_left` adjacent enemy guards certifies a win by repeating that attack. Each guard redirect leaves the attacker at its source and removes one guard. Kernel capture preempts all turn-end adjudication. This certificate must also fire in post-redirect evaluator leaves because the supplied MCTS discards redirected children rather than searching below them.
2. Quiet approach followed by capture depends on piece-specific movement and capture geometry. Ping moves orthogonally and captures diagonally. The actual website sequence Ping E6-E7-C5 illustrates a pattern a current attack map cannot see. Daemon adjacent shots are in all eight directions, despite orthogonal long-range movement. Trace has minimum shot range two.
3. Policy danger must refer to the actual post-action piece placement. A shot and a guard redirect leave the actor at source. A redirected Kernel attack sacrifices the first available guard in Ping, Exploit, Trace, Daemon, Ice order, so its capture value is the sacrificed piece's value, not a nominal Kernel value.
4. Guard loss interacts with remaining actions. At two actions left, an exposed first move can be repaired by a second move. With one action left, abandoning the final guard hands the opponent an entire two-action turn. In packet 2 the three-piece endgame makes this distinction especially clear: moving Ping away from its Kernel can permit an opposing Kernel's two-step capture.
5. Adjacent bodyguard count is a discrete defensive resource, not simply material. One guard defeats an approach-plus-capture sequence but does not defeat an already-available attack repeated twice. Two guards withstand two repeated attacks during a single opposing turn. These are sufficient local bounds, not full positional evaluations.
6. The actual actor and remaining-action count determine firewall timing. The ring is still open throughout round 8 until Magenta finishes; an immediate capture wins even if the attacking side's Kernel otherwise burns. With two actions left, stepping onto the perimeter and stepping off again need not lose. Do not encode every edge first move as terminal danger.
7. Source-vacancy and removed-target occupancy matter for two-action approach certificates. Removing a source can open a ray; capturing a guard can both clear the Kernel and change the attacker's location. A purely static reverse attack map is a sufficient approximation only when its omissions and false-positive conditions have been proved.

## Bounded observations relevant to fitting

- Packet 1 three avoidable final-action cases have respectively 4/27, 24/27 and 19/27 next-turn-safe actions. Every Kernel move loses in the third case; several non-Kernel guard/development moves survive the bounded probe.
- Packet 2 observed match 4650 final defensive action has exactly three safe actions: Kernel F4-F3/G5/G3. All Trace moves lose within the next opponent turn.
- Packet 2 observed match 4674 at Cyan Kernel C2 has three safe destinations B2/D2/B3 and two losing destinations C3/D3. A later Kernel D4 state has no safe move.
- Exact unsafe actions are recorded separately in every label. Within the safe tier, preferred development, material conversion and guard placement remain teacher predictions. Do not turn their confidence into proof metadata.

## Evaluation limits

The generated labels are actual supervised examples, suitable for fitting a small residual student. They are not exported teacher weights. Thirty-two examples do not establish general playing strength. Training and validation splits must group by source game, including adjacent website positions. A held-out policy fit on these examples is not independent tournament evidence. Preserve the incumbent until the student passes the exact ONNX contract and a separate, paired, exact-referee arena.
