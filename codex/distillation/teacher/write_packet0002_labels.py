"""Serialize teacher judgments on replayed actual website match positions."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;DIST=HERE.parent
p=json.loads((DIST/'packets'/'packet0002.json').read_text())
proof={x['position_id']:x for x in json.loads((HERE/'packet0002_probe.json').read_text())}
continuations={x['position_id']:x for x in json.loads((HERE/'packet0002_continuations.json').read_text())}
judgments=[
([2153,3121,2144,2152],-.65,.50,.35,'The lone Trace is the only possible bodyguard. Kernel G5-G6 followed by an appropriate Trace move can reconnect protection. Every first action has some next-turn-safe continuation, so no first move is certified best; avoid ending the turn with an exposed central Kernel.'),
([1697,1681,1682],-.70,.95,.40,'Exactly three actions avoid an opponent terminal win next turn: Kernel F4-G5, F4-G3 and F4-F3. All Trace moves and the other Kernel moves fail this probe. Prefer returning toward G5 and the remaining Trace, but relative ordering among the three survivors is predictive.'),
([3187,3171],1.,1.,1.,'Ice B2-D2 followed by D2-F4 captures the unguarded Kernel; B2-D4 also admits a second-action capture. These terminal wins preempt the round-8 firewall, despite Cyan Kernel A1 currently standing on the perimeter.'),
([3301],1.,1.,1.,'Ice D2-F4 captures the unguarded Kernel immediately. Almost every alternative permits an opponent terminal win next turn, including firewall loss while Cyan remains on the outer ring. Capture takes precedence over those later checks.'),
([3177,3186,3178],-.65,.30,.35,'Cyan has only a Kernel against Kernel plus Ping. Every available first move has a continuation avoiding a next-turn terminal loss. The Ping also guards the enemy Kernel, so direct approach is not equivalent to a bare-Kernel endgame. Value is an unfavorable prediction, not a proof.'),
([3241,3251,3249],-.65,.95,.40,'Kernel C2-B3, C2-D2 and C2-B2 avoid an enemy terminal win next turn. C2-C3 and C2-D3 do not. D3 specifically lets the enemy Kernel approach and capture in two actions. Relative preference among the three safe moves is predictive.'),
([2268,2267],1.,1.,1.,'Kernel E5-D4 followed by D4-D3 captures the unguarded Cyan Kernel. E5-E4 also has a winning capture continuation. Moving the Kernel to F5 abandons both certified winning approaches.'),
([2787,2211,2794,2793,2218,2217],.65,.90,.40,'Keep the Ping adjacent to Kernel F5 or move the Kernel with protection. Ping E6-E5 preserves that guard. Six actions leave a terminal reply, including Ping E6-E7/E4/D6/C6 and Kernel F5-F4/E4. The guarded advantage is predictive; no terminal win exists this action.'),
([2794,2786,2803,2802,2796,2804],-.65,.90,.35,'Kernel D3-D4 and D3-E4 have no second-action continuation avoiding the next opponent-turn win. The other six first moves retain at least one bounded safe continuation. Prefer C3 or C4 while preserving escape space; eventual outcome remains uncertain.'),
([], -1.,.15,1.,'All eight legal moves from Cyan Kernel D4 admit a terminal Magenta sequence on the next turn. The bounded exhaustive certificate covers each action. Equal policy weights avoid inventing a ranking among positions with the same proven loss value.'),
([2803,2796],1.,1.,1.,'Ping E6-E7 then E7-C5 captures the unguarded Cyan Kernel. Ping E6-D6 also admits a capture continuation. The Ping moves orthogonally to set up a diagonal capture; current diagonal attack maps alone miss this pattern.'),
([3301],1.,1.,1.,'Ping E7-C5 is a legal diagonal distance-two capture of the unguarded Cyan Kernel. It ends the game immediately. The model value is +1 from the current Magenta perspective, with no sign flip merely because this is the second action.')
]
suffix=(HERE/'write_packet0001_labels.py').read_text().split('labels=[]',1)[1]
exec('labels=[]'+suffix.replace('packet0001','packet0002'))
