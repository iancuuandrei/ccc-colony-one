"""Independent full-env turn composition of the two-enemy-turn 10910 proof.

Does not import either research proof builder or its minimax helper. Uses the
separately qualified own-turn terminal oracle at leaves, and env.clone/step
for both sides' complete turns. No model or training runtime is loaded.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,re,hashlib,importlib.util,time
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'env'));sys.path.insert(0,str(ROOT/'codex'/'engine_audit'))
from colony_one.env import ColonyOneEnv
from colony_one import encoding as E
from own_turn_oracle import own_turn_wins
SPEC=importlib.util.spec_from_file_location('deep10910_queue',ROOT/'codex'/'distillation'/'queue.py');q=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(q)
SOURCE=ROOT/'codex'/'matches'/'site-all-v1-derived'/'match-10910.txt'
PACKET=ROOT/'codex'/'research'/'loss10910-training'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def actions(e):return sorted(map(int,e.legal_action_indices()))
def step(e,a):
    f=e.clone();f.step(a);return f
def turns(e):
    """Every legal completion of the actual current side's turn, exactly once."""
    own=e.current_team;firsts=actions(e);assert firsts,'No-action state requires UNKNOWN'
    for a in firsts:
        f=step(e,a)
        if f.terminated or f.current_team!=own:yield [a],f;continue
        assert f.actions_left==1
        seconds=actions(f);assert seconds,'No-action state requires UNKNOWN'
        for b in seconds:
            g=step(f,b);assert g.terminated or g.current_team!=own;yield [a,b],g
def typed(e,seq):
    f=e.clone();out=[]
    for idx in seq:
        a=next(a for a in f.legal_actions() if E.action_index(a,f.current_team)==idx);p=f.board.piece_at(*a.from_pos)
        record=dict(action=f.action,index=idx,side=f.current_team,piece=p.piece_type.value,from_rc=a.from_pos,to_rc=a.to_pos,kind=a.kind)
        _,_,_,info=f.step(idx);record.update(winner=f.winner,terminated=f.terminated);out.append(record)
    return out,f
def replay():
    text=SOURCE.read_text(encoding='utf-8-sig');pattern=re.compile(r'^\s*(\d+)\.\s+(CYAN|MAGENTA)\s+(\w+)\s+([A-H][1-8])\s+->\s+([A-H][1-8])\s+(MOVE|SHOOT)(.*)$')
    e=ColonyOneEnv();states={};records=[]
    for line in text.splitlines():
        m=pattern.match(line)
        if not m:
            assert not re.match(r'^\s*\d+\.',line),line
            continue
        num,side,abbr,fr,to,kind,extra=m.groups();assert e.action==int(num) and e.current_team==side
        rc=lambda s:(8-int(s[1]),ord(s[0])-65)
        a=next(a for a in e.legal_actions() if a.from_pos==rc(fr) and a.to_pos==rc(to));assert e.board.piece_at(*a.from_pos).piece_type.abbr==abbr
        rec=e.board.apply_action(a)
        try:
            redirected=rec.actual_target_pos is not None and rec.actual_target_pos!=a.to_pos and not rec.moved
            assert a.kind==kind.lower() or (kind=='SHOOT' and redirected)
        finally:e.board.undo_action(rec)
        states[e.action]=e.clone();line_t,end=typed(e,[int(E.action_index(a,side))]);records.extend(line_t);e=end
    assert len(records)==44 and e.terminated and e.winner=='CYAN';return states,records
memo={};leaf_checks=0;endpoint_turns=0
def can_survive_one_enemy_turn(e):
    """Exists a defender full turn after which enemy has no terminal win."""
    global leaf_checks,endpoint_turns
    key=hashlib.sha256(e.observation().tobytes()+str((e.current_team,e.actions_left,e.round_number)).encode()).hexdigest()
    if key in memo:return memo[key]
    assert len(memo)<10000,'Explicit endpoint cap; unresolved means no release'
    own=e.current_team;examined=[]
    for seq,f in turns(e):
        endpoint_turns+=1
        if f.terminated:
            safe=f.winner==own;detail=dict(own_actions=seq,terminal_winner=f.winner)
        else:
            assert f.current_team!=own and f.actions_left==2
            assert actions(f),'Enemy no-action state requires UNKNOWN'
            leaf_checks+=1;win=own_turn_wins(f.board,f.current_team,f.actions_left,f.round_number);safe=not win.any_win
            detail=dict(own_actions=seq,enemy_has_current_turn_win=win.any_win)
            if win.any_win:
                witness=win.witnesses[min(win.witnesses)]['actions'];line,end=typed(f,witness)
                assert end.terminated and end.winner==f.current_team;detail['enemy_witness']=witness
        examined.append(detail)
        if safe:
            result=dict(survives=True,own_safe_completion=seq,checked_own_completions=len(examined));memo[key]=result;return result
    result=dict(survives=False,all_own_completions=examined,checked_own_completions=len(examined));memo[key]=result;return result
def enemy_can_force_by_second_turn(e,own):
    """Exists enemy full turn, then forall defender replies exists enemy win."""
    assert e.terminated or e.current_team!=own
    if e.terminated:return dict(forced_loss=e.winner!=own,terminal_winner=e.winner,checked_enemy_completions=0)
    branches=[]
    for seq,f in turns(e):
        if f.terminated:
            loss=f.winner!=own;detail=dict(enemy_actions=seq,terminal_winner=f.winner)
        else:
            assert f.current_team==own and f.actions_left==2
            endpoint=can_survive_one_enemy_turn(f);loss=not endpoint['survives'];detail=dict(enemy_actions=seq,endpoint=endpoint,endpoint_state=q.encode_state(f))
        branches.append(detail)
        if loss:return dict(forced_loss=True,witness=detail,checked_enemy_completions=len(branches))
    return dict(forced_loss=False,all_enemy_completions=branches,checked_enemy_completions=len(branches))
def orbit(obs):
    return {np.ascontiguousarray(np.flip(np.rot90(obs,k,axes=(1,2)),axis=2) if flip else np.rot90(obs,k,axes=(1,2))).astype(np.float32).tobytes() for k in range(4) for flip in [False,True]}
def main():
    start=time.monotonic();assert sha(SOURCE)=='98695de1b45eb2bee22b64d6ba004fb97864684a3eeb8e6219d48aa8dd4bc6bc'
    packet=json.loads((PACKET/'packet.json').read_text());assert sha(PACKET/'training_packet.npz')==packet['files']['training_npz_sha256'];assert sha(PACKET/'certificates.json')==packet['files']['certificates_sha256']
    for name,h in packet['engine_sha256'].items():assert sha(ROOT/'env'/'colony_one'/name)==h
    for name,h in packet['oracle_sha256'].items():assert sha(ROOT/'codex'/'engine_audit'/name)==h
    states,replayed=replay();a=states[39];b=step(a,2729);c=step(a,2738);assert np.array_equal(b.observation(),states[40].observation())
    assert a.round_number==10 and a.actions_left==2 and a.current_team=='MAGENTA';assert len(actions(a))==9
    supplied=np.load(PACKET/'training_packet.npz');envs=[a,b,c]
    for i,e in enumerate(envs):
        assert np.array_equal(e.observation(),supplied['obs'][i]);assert set(actions(e))==set(map(int,supplied['pidx'][i][supplied['pidx'][i]>=0]))
    proof={}
    for name,e in [('B',b),('C',c)]:
        assert e.actions_left==1 and len(actions(e))==6;rows=[]
        for idx in actions(e):
            f=step(e,idx);r=enemy_can_force_by_second_turn(f,e.current_team);rows.append(dict(action_index=idx,**r));print(json.dumps(dict(row=name,action=idx,forced_loss=r['forced_loss'],enemy_completions=r['checked_enemy_completions'],leaf_checks=leaf_checks,endpoint_turns=endpoint_turns)),flush=True)
        proof[name]=rows
    assert all(r['forced_loss'] for r in proof['B']);assert [r['action_index'] for r in proof['C'] if not r['forced_loss']]==[3164]
    escape=next(r for r in proof['C'] if r['action_index']==3164);assert escape['checked_enemy_completions']==239
    assert supplied['has_pol'].tolist()==[False,False,True] and supplied['has_value'].tolist()==[False,True,False]
    assert supplied['classification_complete'].tolist()==[False,True,True] and supplied['unknown_action_mask'][0].sum()==7
    assert supplied['source_group'].tolist()==['site:10910']*3 and not supplied['q_valid'].any()
    assert supplied['z'].tolist()==[0.,-1.,0.] and supplied['pval'][0].sum()==0 and supplied['pval'][1].sum()==0
    assert supplied['pval'][2].sum()==1 and supplied['pidx'][2,np.argmax(supplied['pval'][2])]==3164
    assert supplied['pairwise_better_index'][0]==2738 and supplied['pairwise_worse_index'][0]==2729
    holdout=ROOT/'codex'/'synthetic'/'holdout_v1';frozen=json.loads((holdout/'FROZEN.json').read_text())
    for name,h in frozen['files'].items():assert sha(holdout/name)==h
    held=np.load(holdout/'holdout_states.npz')['obs'];holdout_orbits=set().union(*(orbit(o) for o in held))
    assert all(e.observation().astype(np.float32).tobytes() not in holdout_orbits for e in envs)
    # Concrete bodyguard/shot line verified with the full environment.
    line,end=typed(a,[2738,3164,1693,1869,3235,2331]) if False else ([],None)
    chosen=[((2,5),(1,5)),((1,6),(4,3)),((3,2),(3,5)),((3,5),(1,5)),((1,5),(2,4)),((2,4),(2,3))];e=a.clone();concrete=[]
    for fr,to in chosen:
        act=next(x for x in e.legal_actions() if x.from_pos==fr and x.to_pos==to);ix=int(E.action_index(act,e.current_team));l,e=typed(e,[ix]);concrete+=l
    assert e.terminated and e.winner=='MAGENTA'
    result=dict(status='PASS',scope='TERMINAL_LOSS_THROUGH_SECOND_FOLLOWING_OPPONENT_TURN',method='Independent env.clone/step full-turn quantifiers, own-turn terminal oracle leaves with all positive witness lines replayed. Research minimax helpers not imported.',source_sha256=sha(SOURCE),packet_sha256=sha(PACKET/'packet.json'),training_npz_sha256=sha(PACKET/'training_packet.npz'),certificates_sha256=sha(PACKET/'certificates.json'),script_sha256=sha(__file__),engine_sha256=packet['engine_sha256'],oracle_sha256=packet['oracle_sha256'],replayed_actions=replayed,concrete_counterfactual=concrete,A=dict(legal=9,classified_safe_subset=[2738],classified_unsafe_subset=[2729],unclassified=7,full_value='UNKNOWN',standard_policy_enabled=False),B=dict(all_six_forced_loss=True,value=-1,policy_enabled=False),C=dict(sole_bounded_safe=3164,all_six_classified=True,enemy_completions_checked=239,value='UNKNOWN',policy_enabled=True),proof=proof,resources=dict(endpoint_states=len(memo),endpoint_complete_turns=endpoint_turns,leaf_oracle_checks=leaf_checks,seconds=time.monotonic()-start),holdout=dict(frozen_sha256=sha(holdout/'FROZEN.json'),all_frozen_hashes_unchanged=True,d4_orbit_intersections=0),release_eligibility='B+C only; A metadata-only partial preference; source group site:10910')
    (HERE/'proof.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('proof','replayed_actions','concrete_counterfactual')}),flush=True)
if __name__=='__main__':main()
