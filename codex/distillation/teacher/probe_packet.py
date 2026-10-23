"""Teacher-only exact engine evidence; all action indices use current mover."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent
DIST=HERE.parent
sys.path.insert(0,str(DIST))
sys.path.insert(0,str(DIST.parent/'engine_audit'))
from queue import restore
from own_turn_oracle import own_turn_wins
from colony_one.encoding import action_index

packet_id=sys.argv[1] if len(sys.argv)>1 else 'packet0001'
packet=json.loads((DIST/'packets'/(packet_id+'.json')).read_text())
out=[]
for p in packet['positions']:
    env=restore(p['state']);team=env.current_team
    ow=own_turn_wins(env.board,team,env.actions_left,env.round_number)
    rows=[]
    for a in env.legal_actions():
        ix=action_index(a,team);child=env.clone();child.step(a)
        d=dict(action_index=ix,win=bool(ow.first_action_win_mask[ix]))
        if d['win']:d['witness']=ow.witnesses[ix]
        if child.terminated:d['terminal']=child.winner
        elif child.current_team!=team:
            reply=own_turn_wins(child.board,child.current_team,child.actions_left,child.round_number)
            d['opponent_win_next_turn']=reply.any_win
            if reply.any_win:d['opponent_witness']=next(iter(reply.witnesses.values()))
        rows.append(d)
    out.append(dict(position_id=p['position_id'],category=p['category'],team=team,left=env.actions_left,round=env.round_number,own_win=ow.any_win,actions=rows))
    print(json.dumps(out[-1]),flush=True)
(HERE/(packet_id+'_probe.json')).write_text(json.dumps(out,indent=2))
