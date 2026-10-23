"""Serialize Astra-authored judgments, preserving exact tactical evidence tiers."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;DIST=HERE.parent
p=json.loads((DIST/'packets'/'packet0001.json').read_text())
proof={x['position_id']:x for x in json.loads((HERE/'packet0001_probe.json').read_text())}
continuations={x['position_id']:x for x in json.loads((HERE/'packet0001_continuations.json').read_text())}
# Preferences below are teacher judgments; exact safety tiers come from engine runs.
# Fields: preferred actions, value estimate, policy confidence, value confidence, factual rationale.
judgments=[
([2786,2777,2793,2144],-.55,.90,.35,'Only four actions avoid an opponent terminal win next turn. Ice D3-C4, D3-B5 and D3-B3 add a bodyguard; Kernel B4-A4 is the other safe action. Prefer preserving the Ice near the Kernel over retreating to the perimeter. Eventual result remains uncertain.'),
([3186,3177,1633,1641,2931,2915],-.10,.78,.20,'Kernel G7-F6 captures a Ping but permits an Ice approach and Kernel capture on the next turn. Daemon G8-E8/D8 also abandons a necessary guard. The other 24 actions avoid this bounded loss; prefer maintaining protection while developing inward.'),
([3746,3754,3178,3171,2601],-.30,.88,.25,'Every Kernel move permits a terminal opponent sequence next turn; all 19 non-Kernel actions avoid it. Daemon C1-C4 and Ice B2-C3 approach or provide protection without spending the Kernel action on the tempting Exploit capture. No eventual-win claim.'),
([2261,2282,2273,2666],-.30,.68,.30,'Ice D4-F6 attacks the guarded Kernel, removes the enemy Ice at G7 and remains on D4. It gains a valuable piece without relocating onto F6. All legal first actions survived the next-turn tactical probe; material conversion preference is a prediction.'),
([2196,2218,2209,2666],.25,.76,.35,'Ice C4-E6 redirects onto the Daemon at F7, removes it and remains C4. This removes the enemy long-range attacker while retaining the Ice. Both current Kernel bodyguards and the exact source-preserving redirect matter; eventual result is not certified.'),
([1347,1351,3242,3234,3948],.05,.62,.25,'Ping F6-D8 captures a Daemon; F6-H8 instead sacrifices the defending Exploit and leaves the Ping on F6. Prefer the larger direct material gain, but that ordering is predictive. Kernel B2-C3 is excluded by an exact opponent-next-turn win witness.'),
([2774,2794,2796,3892,3251],.55,.78,.35,'Trace E6-B3 removes Cyan Trace A3 through Kernel guard redirection and stays E6, continuing to guard the Magenta Kernel. Moving Trace to E7 instead or several Kernel moves permits a next-turn terminal attack. Prefer the safe shot and retained protection.'),
([2794,2081,2786,2802,2803,2804],-.60,.88,.35,'Six first actions have a continuation avoiding an opponent win next turn; five have none. Kernel E6-F6 or Trace H5-G5 can arrange a guard. Kernel E6-E5 is a losing tempo under the bounded exhaustive probe. Material disadvantage still makes the position unfavorable.'),
([], -1., .15,1.,'All 11 legal actions were enumerated through the opponent turn and each admits an opponent terminal win. Equal policy weights reflect no certified distinction among these losing actions. The value is from Magenta, the current mover.'),
([2794,2786,2674,2802,2673],-.50,.66,.30,'Kernel B3-C4 has no continuation avoiding an opponent next-turn win. Prefer keeping or bringing Ice within guarding distance, rather than advancing an exposed Kernel. Other first actions have at least one safe continuation; this does not establish eventual survival.'),
([3697,3689,3633,3121,2915],-.10,.35,.15,'The Kernel starts with two adjacent bodyguards. Develop the Daemon or Kernel while preserving a guard; the nearby enemy Ping creates direct tactical pressure. No immediate terminal win was found, and all first actions have a bounded safe continuation. Ordering is strategic prediction.'),
([2203,2212,2196,2219],1.,1.,1.,'Four Ice moves admit an immediate second-action Kernel capture: C4-D5, C4-E4, C4-E6 and C4-D3. The opponent has no adjacent guard. Capturing the Kernel ends the game before firewall timing can matter.'),
([1757],1.,1.,1.,'Ice D5-F5 directly captures the unguarded Magenta Kernel. Immediate capture terminates before bare-Kernel or firewall checks. Moving the Cyan Kernel onto A4/A2 instead exposes a terminal opponent turn.'),
([2664,2673,3633,2731,2915],-.45,.45,.35,'Ice G6-H6 captures the advanced enemy Ping. A subsequent return toward G7 can guard the Kernel; the opponent retains more non-Kernel material. Prefer removing the nearby attacker over unsupported advancement. Value is an uncertain material-and-safety estimate.'),
([3632,3746,3754,3300,3177],0.,.20,.05,'Early position with equal material and distant Kernels. Kernel A1-A2 connects to Trace A3 and Ice B2; Daemon and Exploit development are alternatives. No terminal win exists within the current turn. Weak preference only; zero is uncertainty, not a predicted draw.'),
([3178,3241,2601,3177,3243],0.,.25,.10,'Material is balanced. Kernel G7-F6 improves inward placement while staying adjacent to Ice F7; other guard-preserving development is plausible. No action permits a terminal opponent win next turn. Preference and neutral value are uncertain predictions.'),
([3632,3746,3754,3300,3177],0.,.20,.05,'Equal material and a largely separated opening. Kernel H8-H7 connects with Trace H6 and Ice G7, while Daemon development improves reach. All first actions have at least one bounded safe continuation. Opening value and ranking have low confidence.'),
([3746,3754,3300,3177,3113],0.,.20,.05,'Both Kernels have nearby protection and equal material. Prefer activating the Daemon or Exploit before another edgeward move. All actions avoid a terminal opponent win on the next turn; relative ordering remains an unverified strategic preference.'),
([3178,3186,1741,1756,1748,2666],-.25,.35,.20,'Round 8 perimeter remains usable until Magenta ends its turn. Prefer a central Kernel destination or an active Ice move. Cyan can still leave and return from the perimeter in two actions, so edge first moves are not automatically losses. Material disadvantage is only a predictive value signal.'),
([3632,3876,3884,3300,3177],0.,.20,.05,'Equal material in an early position. Kernel A1-A2 retains Ice B2 protection and connects with Trace A3; Daemon E1-E4 activates the displaced Daemon. All legal actions avoid a next-turn terminal loss. Preference is weak and value is uncertain, not a draw claim.')
]
labels=[]
for i,(pos,j) in enumerate(zip(p['positions'],judgments)):
    pref,value,conf,vconf,evidence=j;pid=pos['position_id'];q=proof[pid]
    legal=[a['action_index'] for a in pos['legal_actions']]
    pref=[a for a in pref if a in legal]
    wins=[a['action_index'] for a in q['actions'] if a['win']]
    bad={a['action_index'] for a in q['actions'] if a.get('opponent_win_next_turn') or ('terminal' in a and a['terminal']!=q['team'])}
    if pid in continuations:bad|={a['action_index'] for a in continuations[pid]['actions'] if not a['safe_continuation']}
    safe=[a for a in legal if a not in bad]
    order=pref+[a for a in legal if a not in pref]
    if wins:order=[a for a in order if a in wins]+[a for a in order if a not in wins]
    elif safe:order=[a for a in order if a in safe]+[a for a in order if a not in safe]
    claim='forced_win_this_turn' if wins else 'forced_loss_by_next_turn' if len(bad)==len(legal) else 'prediction'
    rankings=[]
    for rank,a in enumerate(order,1):
        if wins:weight=1. if a in wins else .003
        elif len(bad)==len(legal):weight=1.
        elif a in bad:weight=.005
        elif a in pref:weight=1.2-0.04*pref.index(a)
        else:weight=.35 if conf>=.6 else .70
        rankings.append(dict(action_index=a,rank=rank,weight=round(weight,4)))
    # An exact safety tier does not prove that our strategic tie-break is best.
    lab=dict(position_id=pid,state_sha256=pos['state_sha256'],policy_rankings=rankings,value=value,confidence=conf,value_confidence=vconf,claim=claim,evidence=evidence,
        policy_tier_semantics='Winning-this-turn tier first when available; otherwise avoid verified next-turn losses. Within safe tiers the ordering is a teacher prediction.',
        exact_unsafe_action_indices=sorted(bad))
    if wins:lab['witness']=next(a['witness']['actions'] for a in q['actions'] if a['action_index']==order[0])
    labels.append(lab)
out=dict(packet_id=p['packet_id'],teacher=dict(model='gpt-6-astra',role='assigned teacher',method='Direct teacher-authored rankings and calibrated values, with exact supplied-engine tactical verification; no language-model weights exported'),labels=labels,
    evidence_files=['teacher/packet0001_probe.json','teacher/packet0001_continuations.json'],qualification='Local teacher judgments and bounded engine proofs; playing-strength improvement is untested.')
dest=DIST/'teacher_labels'/'astra_packet0001.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(dict(path=str(dest),labels=len(labels),exact_win=sum(l['claim']=='forced_win_this_turn' for l in labels),exact_loss=sum(l['claim']=='forced_loss_by_next_turn' for l in labels))))
