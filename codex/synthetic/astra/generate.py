"""Certify teacher-authored constructed states and measure exact-referee difficulty.

No state is claimed reachable from the standard opening. Symmetry variants
share a family group. Oracle bounds prove only their stated tactical horizon.
"""
from pathlib import Path
import sys,json,hashlib,time,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'env'));sys.path.insert(0,str(ROOT/'codex'/'engine_audit'))
from colony_one import constants as C,encoding as E,mcts
from colony_one.board import Board
from colony_one.env import ColonyOneEnv
from colony_one.pieces import Piece,PieceType
from colony_one.onnx_runner import OnnxEvaluator
from own_turn_oracle import own_turn_wins
from oracle import check_forced_loss_by_next_turn
spec=importlib.util.spec_from_file_location('synthetic_queue',ROOT/'codex'/'distillation'/'queue.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rotate(r,c,k,flip):
    if flip:c=7-c
    for _ in range(k):r,c=c,7-r
    return r,c
def construct(seed,k,flip):
    env=ColonyOneEnv();b=Board();b.grid=[[None]*8 for _ in range(8)]
    b.locked=Board.locked_cells_for_round(seed['round'])
    seen=set();types=set()
    for team,kind,r,c in seed['pieces']:
        r,c=rotate(r,c,k,flip);actual=C.CYAN if team=='C' else C.MAGENTA
        assert (r,c) not in seen and not b.is_locked(r,c),(seed['id'],'overlap/locked')
        assert (actual,kind) not in types,(seed['id'],'duplicate piece type')
        seen.add((r,c));types.add((actual,kind));b.grid[r][c]=Piece(PieceType(kind),actual)
    assert sum(p[1]=='KERNEL' for p in seed['pieces'])==2
    env.board=b;env.current_team=C.CYAN if seed['side']=='C' else C.MAGENTA
    env.actions_left=seed['left'];env.round_number=seed['round'];env.action=(env.round_number-1)*4+(1 if env.current_team==C.CYAN else 3)+(2-env.actions_left)
    return env
def main():
    start=time.monotonic();proposal=(Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'seed_proposals.json').resolve();raw=json.loads(proposal.read_text())
    outdir=Path(sys.argv[2]) if len(sys.argv)>2 else HERE;outdir.mkdir(parents=True,exist_ok=True)
    models={'h1':ROOT/'model-h1.onnx','v2':ROOT/'codex'/'tactical'/'model-tactical-v2.onnx'}
    if len(sys.argv)>3 and sys.argv[3]=='gen01':models['gen01']=ROOT/'release'/'gen01'/'model.onnx'
    evals={name:OnnxEvaluator(path,num_threads=1) for name,path in models.items()}
    engine={name:sha(ROOT/'env'/'colony_one'/name) for name in ['env.py','board.py','rules.py','encoding.py','mcts.py']}
    output=[];unique=set();npzrows=[];stream=outdir/'certified_states.jsonl'
    with stream.open('w',encoding='utf-8') as f:
        for n,seed in enumerate(raw['seeds']):
            for flip in [False,True]:
                for k in range(4):
                    env=construct(seed,k,flip);obs=env.observation();state=q.encode_state(env);digest=q.state_hash(state)
                    if digest in unique:continue
                    unique.add(digest);legal=[int(x) for x in env.legal_action_indices()]
                    assert legal,'live no-action state unsupported'
                    ow=own_turn_wins(env.board,env.current_team,env.actions_left,env.round_number)
                    winning=np.flatnonzero(ow.first_action_win_mask).tolist()
                    loss=check_forced_loss_by_next_turn(env,max_nodes=30000)
                    unsafe=[int(a) for a,s in loss['per_first_action'].items() if s=='VERIFIED']
                    safe=[int(a) for a,s in loss['per_first_action'].items() if s=='DISPROVED']
                    unknown=[int(a) for a,s in loss['per_first_action'].items() if s=='UNKNOWN']
                    val=1. if winning else -1. if loss['status']=='VERIFIED' else None
                    claim='forced_win_this_turn' if winning else 'forced_loss_by_next_turn' if val==-1 else 'unknown_full_game_value'
                    target=winning if winning else safe
                    assert not(set(winning)&set(unsafe))
                    scored={}
                    for name,ev in evals.items():
                        logits,values=ev(obs[None]);rawidx=max(legal,key=lambda ix:float(logits[0,ix]))
                        before=env.observation().tobytes()
                        a=mcts.search_action(env.board,env.current_team,env.actions_left,env.round_number,ev,mcts.MCTSConfig(simulations=100),rng=np.random.default_rng(742000+n))
                        assert before==env.observation().tobytes(),'referee mutated board'
                        ix=int(E.action_index(a,env.current_team))
                        missed=bool(winning and ix not in winning);blunder=ix in unsafe and bool(safe)
                        scored[name]=dict(raw_policy_action=int(rawidx),value=float(values[0]),mcts100_action=ix,missed_certified_win=missed,selected_certified_unsafe=ix in unsafe,avoidable_tactical_blunder=blunder,
                            value_absolute_error=None if val is None else abs(float(values[0])-val))
                    difficult=any(s['missed_certified_win'] or s['avoidable_tactical_blunder'] for s in scored.values())
                    source=dict(path=str(proposal.relative_to(ROOT)),sha256=sha(proposal),seed_id=seed['id'],variant=dict(rotation=k,reflection=flip),kind='LLM_AUTHORED_CONSTRUCTED_SCENARIO',reachability='NOT_PROVEN_FROM_STANDARD_OPENING')
                    row=q.package(env,source,'synthetic_'+seed['family'])
                    row.update(group_id='astra-family-'+seed['family'],seed_id=seed['id'],proposal=seed['proposal'],certificate=dict(claim=claim,value=val,winning_first_actions=winning,witnesses=ow.witnesses,unsafe_first_actions=unsafe,safe_first_actions=safe,unknown_first_actions=unknown,loss_oracle=loss),baseline=scored,difficult_mcts100=difficult)
                    output.append(row);f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
                    npzrows.append((obs,legal,target,val,row['group_id'],row['position_id'],difficult,winning,unsafe,unknown))
            print(json.dumps(dict(seed=seed['id'],total=len(output),hard=sum(r['difficult_mcts100'] for r in output),seconds=round(time.monotonic()-start,1))),flush=True)
    assert engine=={name:sha(ROOT/'env'/'colony_one'/name) for name in engine},'engine drift'
    mask=np.zeros((len(output),4096),bool);targets=np.zeros((len(output),4096),np.float32);bad=np.zeros_like(mask);unknownmask=np.zeros_like(mask)
    for i,(_,legal,t,_,_,_,_,_,u,unknown) in enumerate(npzrows):
        mask[i,legal]=1;bad[i,u]=1;unknownmask[i,unknown]=1
        if t:targets[i,t]=1/len(t)
    np.savez_compressed(outdir/'certified_states.npz',obs=np.stack([r[0] for r in npzrows]),legal_mask=mask,policy=targets,value=np.array([np.nan if r[3] is None else r[3] for r in npzrows],np.float32),value_valid=np.array([r[3] is not None for r in npzrows]),policy_valid=np.array([bool(r[2]) for r in npzrows]),group_id=np.array([r[4] for r in npzrows]),position_id=np.array([r[5] for r in npzrows]),hard=np.array([r[6] for r in npzrows]),unsafe_mask=bad,unknown_mask=unknownmask)
    manifest=dict(schema_version=1,status='COMPLETE',provenance=raw['provenance'],teacher_seed_proposals=str(proposal),proposal_sha256=sha(proposal),engine_sha256=engine,baseline_sha256={n:sha(p) for n,p in models.items()},oracle_sha256={name:sha(ROOT/'codex'/'engine_audit'/name) for name in ['oracle.py','own_turn_oracle.py']},states=len(output),unique_families=len(set(r['group_id'] for r in output)),hard_states=sum(r['difficult_mcts100'] for r in output),certified_value_states=sum(r['certificate']['value'] is not None for r in output),unknown_value_states=sum(r['certificate']['value'] is None for r in output),seconds=time.monotonic()-start,qualification='Constructed legal states, not established reachable games. Certificates bound only the current/next-turn horizon. Difficulty is local exact-referee MCTS100 error, not playing-strength proof.',split_rule='All rotations, reflections and paired clocks from a family remain in the same group.',sampling='Prioritize hard states; retain counterexamples from paired action counts and firewall phases; never interpret UNKNOWN value as loss.')
    (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    packet=dict(packet_id='astra_synthetic0001',schema_version=1,teacher_model='gpt-6-astra',engine_sha256=engine,positions=output)
    (outdir/'packet.json').write_text(json.dumps(packet,indent=1))
    print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
