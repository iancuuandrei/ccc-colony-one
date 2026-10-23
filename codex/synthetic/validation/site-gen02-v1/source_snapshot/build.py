"""Construct an exact-engine teacher curriculum; no training or model export."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib,importlib.util,shutil
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'env'));sys.path.insert(0,str(ROOT/'codex'/'engine_audit'))
from colony_one import constants as C,encoding as E
from colony_one.board import Board
from colony_one.env import ColonyOneEnv
from own_turn_oracle import own_turn_wins,recover_live_clock
from oracle import check_forced_loss_by_next_turn
spec=importlib.util.spec_from_file_location('site_teacher_queue',ROOT/'codex'/'distillation'/'queue.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def coord(p):return chr(65+p[1])+str(8-p[0])
def typed_replay(env,actions):
    e=env.clone();out=[]
    for ix in actions:
        assert not e.terminated
        legal={E.action_index(a,e.current_team):a for a in e.legal_actions()};a=legal[int(ix)];p=e.board.piece_at(*a.from_pos)
        row=dict(action_index=int(ix),team=e.current_team,piece=p.piece_type.value,move=coord(a.from_pos)+'-'+coord(a.to_pos),kind=a.kind)
        e.step(int(ix));row.update(terminated=e.terminated,winner=e.winner);out.append(row)
    return out,e
def rotate_env(env,k):
    e=env.clone();grid=[[None]*8 for _ in range(8)]
    for r in range(8):
        for c in range(8):
            rr,cc=r,c
            for _ in range(k):rr,cc=cc,7-rr
            grid[rr][cc]=env.board.grid[r][c]
    e.board.grid=grid;return e
def validate_board(env):
    types=set()
    for r,row in enumerate(env.board.grid):
        for c,p in enumerate(row):
            if p:
                assert not env.board.is_locked(r,c)
                key=(p.team,p.piece_type.value);assert key not in types;types.add(key)
    assert (C.CYAN,'KERNEL') in types and (C.MAGENTA,'KERNEL') in types
    assert env.board.locked==Board.locked_cells_for_round(env.round_number)
    assert len(env.legal_action_indices())>0
def certify(env,source,motif,seed_id,intent,teacher_note):
    validate_board(env);before=env.observation().tobytes();legal=list(map(int,env.legal_action_indices()))
    ow=own_turn_wins(env.board,env.current_team,env.actions_left,env.round_number);winning=np.flatnonzero(ow.first_action_win_mask).tolist();typed={}
    for ix,w in ow.witnesses.items():
        line,end=typed_replay(env,w['actions']);assert end.terminated and end.winner==env.current_team;typed[int(ix)]=line
    if winning:
        loss=dict(status='NOT_RUN_OWN_TURN_WIN_CERTIFIED',per_first_action={},nodes=0);safe=[];bad=[];unknown=[]
    else:
        loss=check_forced_loss_by_next_turn(env,max_nodes=300000)
        if loss.get('budget_exhausted'):loss=check_forced_loss_by_next_turn(env,max_nodes=1000000)
        safe=[int(a) for a,s in loss['per_first_action'].items() if s=='DISPROVED'];bad=[int(a) for a,s in loss['per_first_action'].items() if s=='VERIFIED'];unknown=[int(a) for a,s in loss['per_first_action'].items() if s=='UNKNOWN']
        assert set(safe+bad+unknown)==set(legal)
    assert env.observation().tobytes()==before
    value=1. if winning else -1. if loss['status']=='VERIFIED' else None;target=winning or safe;policy_valid=bool(target) and (bool(winning) or not unknown)
    row=q.package(env,source,motif);row.update(group_id='site-game-'+str(source['game_id']),seed_id=seed_id,motif_id=motif,teacher_intent=intent,teacher_explanation=teacher_note,label_source='exact_engine_certificate',teacher_label_role='LLM-authored scenario and factual explanation; engine-enumerated targets',certificate=dict(winning_first_actions=winning,winning_witnesses=ow.witnesses,typed_winning_witnesses=typed,loss_oracle=loss,safe_first_actions=safe,unsafe_first_actions=bad,unknown_first_actions=unknown,value=value,value_valid=value is not None,policy_valid=policy_valid,policy_scope='current-turn terminal win' if winning else 'survival through enemy next turn, with existential own completion' if safe else 'no policy preference certified'))
    row['policy_targets']=[dict(action_index=a,weight=(1./len(target) if a in target else 0.) if policy_valid else 0.) for a in legal]
    row['certificate_summary']=dict(legal=len(legal),winning=len(winning),bounded_safe=len(safe),bounded_unsafe=len(bad),unknown=len(unknown),value='UNKNOWN' if value is None else value)
    return row
def main():
    assert not (HERE/'MANIFEST.json').exists(),'Do not overwrite a frozen corpus.'
    proposal=json.loads((HERE/'proposals.json').read_text());snap=HERE/'sources';snap.mkdir(exist_ok=True);actual={};rows=[];source_info={}
    for name,expected in proposal['source_files'].items():
        original=ROOT/'codex'/'matches'/name;assert sha(original)==expected
        target=snap/name
        if not target.exists():shutil.copyfile(original,target)
        assert sha(target)==expected
        z=np.load(target);source_info[name]=dict(original_path=str(original.relative_to(ROOT)),snapshot_path=str(target.relative_to(HERE)),sha256=expected,rows=len(z['obs']))
        for i,obs in enumerate(z['obs']):
            game=str(z['match'][i]);ply=int(z['ply'][i]);key=game+':'+str(ply);team,left,rnd=recover_live_clock(obs)
            env=ColonyOneEnv();env.board=Board.from_observation(obs,team);env.current_team=team;env.actions_left=left;env.round_number=rnd;env.action=ply
            assert np.array_equal(env.observation(),obs)
            note=proposal['actual_teacher_notes'][key];source=dict(kind='ACTUAL_OBSERVED_REPLAY',game_id=game,ply=ply,path=str(target.relative_to(HERE)),sha256=expected,row_index=i,reachability='RECOVERED_FROM_ENGINE_REPLAY_BY_MATCH_ANALYSIS')
            row=certify(env,source,note['motif'],'actual-'+key,'Observed GEN02 tactical failure',note['explanation'])
            if 'legal_mask' in z:
                mask=np.zeros(4096,bool);mask[[a['action_index'] for a in row['legal_actions']]]=True;assert np.array_equal(mask,z['legal_mask'][i])
                if not row['certificate']['winning_first_actions']:
                    safe=np.zeros(4096,bool);safe[row['certificate']['safe_first_actions']]=True;assert np.array_equal(safe,z['safe_start_mask'][i]),key
            rows.append(row);actual[key]=(env,source,note)
            print(json.dumps(dict(actual=key,**row['certificate_summary'])),flush=True)
    for seed in proposal['counterfactuals']:
        original,src,note=actual[seed['source']];env=original.clone()
        if 'move_piece' in seed:
            a,b=seed['move_piece'];assert env.board.grid[a[0]][a[1]] is not None and env.board.grid[b[0]][b[1]] is None
            env.board.grid[b[0]][b[1]]=env.board.grid[a[0]][a[1]];env.board.grid[a[0]][a[1]]=None
        if 'actions_left' in seed:env.actions_left=seed['actions_left']
        env.action=(env.round_number-1)*4+(1 if env.current_team==C.CYAN else 3)+(2-env.actions_left)
        for k in proposal['variant_rotations_quarter_turns']:
            e=rotate_env(env,k);source=dict(kind='LLM_AUTHORED_COUNTERFACTUAL',game_id=src['game_id'],source_actual=seed['source'],source_sha256=src['sha256'],source_row=src['row_index'],proposal_id=seed['id'],rotation_quarter_turns=k,reachability=proposal['counterfactual_reachability'])
            row=certify(e,source,note['motif'],seed['id'],seed['intent'],'Constructed perturbation: '+seed['intent']+' Labels below are recomputed; the original tactical conclusion is not assumed to persist.')
            rows.append(row)
        print(json.dumps(dict(seed=seed['id'],states=len(rows),last=row['certificate_summary'])),flush=True)
    assert len({r['state_sha256'] for r in rows})==len(rows)
    (HERE/'states.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows),encoding='utf-8')
    (HERE/'packet.json').write_text(json.dumps(dict(packet_id=proposal['corpus_id'],positions=rows),indent=2),encoding='utf-8')
    n=len(rows);legal=np.zeros((n,4096),bool);win=np.zeros_like(legal);safe=np.zeros_like(legal);bad=np.zeros_like(legal);unknown=np.zeros_like(legal);policy=np.zeros((n,4096),np.float32)
    for i,r in enumerate(rows):
        c=r['certificate'];legal[i,[a['action_index'] for a in r['legal_actions']]]=1;win[i,c['winning_first_actions']]=1;safe[i,c['safe_first_actions']]=1;bad[i,c['unsafe_first_actions']]=1;unknown[i,c['unknown_first_actions']]=1
        for a in r['policy_targets']:policy[i,a['action_index']]=a['weight']
    np.savez_compressed(HERE/'certified_states.npz',obs=np.stack([q.observation(r['state']) for r in rows]),legal_mask=legal,winning_first_mask=win,safe_first_mask=safe,unsafe_first_mask=bad,unknown_first_mask=unknown,policy=policy,policy_valid=np.array([r['certificate']['policy_valid'] for r in rows]),value=np.array([np.nan if r['certificate']['value'] is None else r['certificate']['value'] for r in rows],np.float32),value_valid=np.array([r['certificate']['value_valid'] for r in rows]),group_id=np.array([r['group_id'] for r in rows]),position_id=np.array([r['position_id'] for r in rows]),provenance_kind=np.array([r['source']['kind'] for r in rows]),q_valid=np.zeros(n,bool))
    manifest=dict(corpus_id=proposal['corpus_id'],status='TEACHER_COMPLETE_INDEPENDENT_QUALIFICATION_REQUIRED',purpose=proposal['purpose'],states=n,actual_states=sum(r['source']['kind']=='ACTUAL_OBSERVED_REPLAY' for r in rows),constructed_states=sum(r['source']['kind']=='LLM_AUTHORED_COUNTERFACTUAL' for r in rows),game_groups=len(set(r['group_id'] for r in rows)),exact_win_values=sum(r['certificate']['value']==1 for r in rows),exact_loss_values=sum(r['certificate']['value']==-1 for r in rows),unknown_values=sum(not r['certificate']['value_valid'] for r in rows),policy_valid=int(sum(r['certificate']['policy_valid'] for r in rows)),sources=source_info,engine_sha256={name:sha(ROOT/'env'/'colony_one'/name) for name in ['env.py','board.py','rules.py','encoding.py','mcts.py']},oracle_sha256={name:sha(ROOT/'codex'/'engine_audit'/name) for name in ['own_turn_oracle.py','oracle.py']},files={name:sha(HERE/name) for name in ['proposals.json','build.py','states.jsonl','packet.json','certified_states.npz']},baseline=dict(path='release/gen02/model.onnx',sha256=proposal['baseline_sha256']),qualification='Exact current-turn wins and bounded next-turn safety only. No eventual-win inference from survival. Constructed histories unproven. No direct ext release. Frozen holdout_v1 is separate and untouched.')
    (HERE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
