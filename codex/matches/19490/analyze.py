"""Exact bounded replay diagnostic; no model, MCTS, or training data import."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
for p in [ROOT/'env',ROOT/'codex/matches',ROOT/'codex/engine_audit',ROOT/'codex/research']:
    sys.path.insert(0,str(p))
from site_replay import parse
from loss_10910_review import typed,sparse
from colony_one import encoding
from own_turn_oracle import own_turn_wins
from oracle import check_forced_loss_by_next_turn

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def chosen(env,row):
    return next(a for a in env.legal_actions() if list(a.from_pos)==row['event']['fr'] and list(a.to_pos)==row['event']['to'])
def line(env,indices):
    env=env.clone();result=[]
    for idx in indices:
        a=encoding.decode_action(int(idx),env.current_team,env.board)
        result.append(typed(env,a));env.step(a)
    return result
def threat(env,owner):
    if env.terminated:return dict(terminal=True,winner=env.winner,enemy_win=env.winner!=owner,witness=[])
    assert env.current_team!=owner
    w=own_turn_wins(env.board,env.current_team,env.actions_left,env.round_number)
    if not w.any_win:return dict(terminal=False,enemy_win=False,witness=[])
    indices=next(iter(w.witnesses.values()))['actions']
    return dict(terminal=False,enemy_win=True,witness=line(env,indices))
def continuation_details(env,action):
    owner=env.current_team;t=typed(env,action);first=env.clone();first.step(action)
    if first.terminated or first.current_team!=owner:
        return [dict(own_line=[t],**threat(first,owner))]
    result=[]
    for second in first.legal_actions():
        other=first.clone();t2=typed(first,second);other.step(second)
        result.append(dict(own_line=[t,t2],**threat(other,owner)))
    return result

def main():
    raw=HERE/'match-19490.txt';original=Path('D:/Downloads/match-19490.txt')
    assert raw.read_bytes()==original.read_bytes(),'immutable source mismatch'
    game,states=parse(raw,None)
    assert game['actions']==74 and game['winner']=='CYAN'
    write('replay.json',game)
    loser='MAGENTA';diagnostics=[];critical=[];start=time.monotonic()
    for env,row in zip(states,game['history']):
        if env.current_team!=loser:continue
        before=env.observation().copy();a=chosen(env,row);idx=int(encoding.action_index(a,loser))
        own=own_turn_wins(env.board,loser,env.actions_left,env.round_number)
        safety=check_forced_loss_by_next_turn(env,max_nodes=250000)
        assert np.array_equal(before,env.observation()),'oracle mutated state'
        alternatives=[]
        for act in env.legal_actions():
            d=typed(env,act);i=d['index'];d.update(winning_this_turn=bool(own.first_action_win_mask[i]),bounded_loss=safety['per_first_action'][i])
            if i in own.witnesses:d['winning_witness']=line(env,own.witnesses[i]['actions'])
            alternatives.append(d)
        d=dict(action_number=env.action,state=sparse(env),recorded=typed(env,a),own_turn_win=bool(own.any_win),
               winning_indices=np.flatnonzero(own.first_action_win_mask).tolist(),safety=safety,alternatives=alternatives,
               recorded_missed_win=bool(own.any_win and not own.first_action_win_mask[idx]),
               recorded_bounded_loss=safety['per_first_action'][idx])
        safe=[x for x in alternatives if x['bounded_loss']=='DISPROVED']
        d['avoidable_bounded_loss']=d['recorded_bounded_loss']=='VERIFIED' and bool(safe)
        if d['recorded_missed_win'] or d['avoidable_bounded_loss'] or env.action>=67:
            for x,act in zip(alternatives,env.legal_actions()):x['complete_own_continuations']=continuation_details(env,act)
            critical.append((env,d))
        diagnostics.append(d)
        print(json.dumps(dict(action=env.action,legal=len(alternatives),wins=len(d['winning_indices']),
            safe=len(safe),unsafe=sum(x['bounded_loss']=='VERIFIED' for x in alternatives),
            unknown=sum(x['bounded_loss']=='UNKNOWN' for x in alternatives),played_loss=d['recorded_bounded_loss'],
            missed_win=d['recorded_missed_win'],seconds=round(time.monotonic()-start,2))),flush=True)
    write('diagnostics.json',dict(winner=game['winner'],loser=loser,identity=game['model_identity'],positions=diagnostics))
    rows=[d for env,d in critical];n=len(rows);obs=np.stack([env.observation() for env,d in critical])
    legal=np.zeros((n,4096),bool);win=legal.copy();safe=legal.copy();unsafe=legal.copy();unknown=legal.copy()
    pidx=np.full((n,128),-1,np.int16)
    for j,d in enumerate(rows):
        ids=[a['index'] for a in d['alternatives']];assert len(ids)<=128;pidx[j,:len(ids)]=ids;legal[j,ids]=True
        win[j,d['winning_indices']]=True
        for a in d['alternatives']:
            {'DISPROVED':safe,'VERIFIED':unsafe,'UNKNOWN':unknown}[a['bounded_loss']][j,a['index']]=True
    assert np.array_equal(legal,safe|unsafe|unknown)
    np.savez_compressed(HERE/'critical-states.npz',obs=obs,action_number=np.array([r['action_number'] for r in rows]),
        recorded_action=np.array([r['recorded']['index'] for r in rows]),legal_mask=legal,pidx=pidx,
        winning_mask=win,safe_mask=safe,unsafe_mask=unsafe,unknown_mask=unknown,
        source_group=np.full(n,'site:19490'),position_id=np.array(['site19490:a'+str(r['action_number']) for r in rows]),
        value=np.zeros(n,np.float32),has_value=np.zeros(n,bool),has_pol=np.zeros(n,bool),q_valid=np.zeros(n,bool))
    write('critical-states.json',dict(schema='bounded-critical-positions-v1',npz_sha256=sha(HERE/'critical-states.npz'),
        source_sha256=game['sha256'],source_group='site:19490',historical_model='UNKNOWN',user_side='UNKNOWN',
        winning_scope='terminal win within remaining current own actions; all legal first actions classified exactly',
        safety_scope='existence of own turn completion that prevents enemy terminal win through next enemy turn only',
        unsafe_scope='all own completions admit some enemy terminal win by next own turn',
        value_scope='UNKNOWN: no eventual value claim or training release; numeric zero is masked placeholder',rows=rows))
    files=[ROOT/'env/colony_one/env.py',ROOT/'env/colony_one/board.py',ROOT/'env/colony_one/rules.py',
        ROOT/'env/colony_one/encoding.py',ROOT/'env/colony_one/scene.json',ROOT/'codex/engine_audit/own_turn_oracle.py',
        ROOT/'codex/engine_audit/oracle.py',ROOT/'codex/matches/site_replay.py',Path(__file__)]
    write('source-receipt.json',dict(source=str(original),raw_sha256=sha(raw),copy_equal=True,replay='PASS',actions=74,winner='CYAN',
        historical_model='UNKNOWN',user_side='UNKNOWN',file_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},
        scope='diagnostics only; no model loading, MCTS, training, outcome import, or release',seconds=time.monotonic()-start))

if __name__=='__main__':main()
