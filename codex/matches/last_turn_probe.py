"""Exhaustively check last-turn two-action tactics in captured exact-engine games."""
import json
from pathlib import Path
from probe_matches import *


def wins_this_turn(env, team):
    if env.terminated: return [] if env.winner != team else [None]
    if env.current_team != team: return []
    wins=[]
    for a in env.legal_actions():
        child=env.clone(); child.step(a)
        move=[list(a.from_pos),list(a.to_pos)]
        if child.terminated:
            if child.winner==team: wins.append([move])
        elif child.current_team==team:
            for rest in wins_this_turn(child,team): wins.append([move]+rest)
    return wins


def main():
    src=Path(__file__).parent/'selfplay-h1.json'
    out=[]
    ev=OnnxEvaluator(ROOT/'model-h1.onnx')
    for game in json.loads(src.read_text())['games']:
        env=ColonyOneEnv(); winner=game['winner'];loser='CYAN' if winner=='MAGENTA' else 'MAGENTA'
        target=[r for r in game['history'] if r['before']['team']==loser][-1]['before']['action']
        for row in game['history']:
            if env.action==target: break
            e=row['event']; a=next(a for a in env.legal_actions() if list(a.from_pos)==e['fr'] and list(a.to_pos)==e['to'])
            env.step(a)
        legal=env.legal_actions();logits,_=ev(env.observation()[None]);details=[]
        for a in legal:
            child=env.clone();child.step(a)
            wins=wins_this_turn(child,winner)
            val=float(ev(child.observation()[None])[1][0]) if not child.terminated else (1 if child.winner==child.current_team else -1)
            details.append(dict(fr=list(a.from_pos),to=list(a.to_pos),kind=a.kind,opponent_wins_in_turn=bool(wins),
                witness=wins[0] if wins else [],child_team=child.current_team,child_value=val,
                prior_logit=float(logits[0,encoding.action_index(a,env.current_team)])))
        out.append(dict(seed=game['seed'],loser=loser,action=target,chosen=row['event'],alternatives=details,
            safe_count=sum(not d['opponent_wins_in_turn'] for d in details),total=len(details)))
        print(json.dumps({k:v for k,v in out[-1].items() if k!='alternatives'}),flush=True)
    (Path(__file__).parent/'last-turn-avoidability.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__':main()
