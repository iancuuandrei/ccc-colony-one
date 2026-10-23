"""Operational proof of conservative quiet-approach certificates."""
from pathlib import Path
import copy, hashlib, json, sys, time
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'env'))
from colony_one import constants,encoding,mcts
from colony_one.board import Board
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator
from colony_one.pieces import PieceType

MODEL=HERE/'model-tactical-v2.onnx'
evaluate=OnnxEvaluator(MODEL)
model=onnx.load(MODEL)
names=json.loads(MODEL.with_suffix('.nodes.json').read_text())
model.graph.output.extend([helper.make_tensor_value_info(names['approach_pairs'],TensorProto.FLOAT,['batch',64,64])])
debug=HERE/'approach-diagnostic.onnx'
onnx.save(model,debug)
opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
session=ort.InferenceSession(str(debug),sess_options=opts,providers=['CPUExecutionProvider'])

def get_predictions(obs):
    pol,val,pairs=session.run(None,{'observation':np.stack(obs)})
    return val,pairs.reshape(-1,4096)>0.5

def win_after_first(env,action):
    child=env.clone();team=child.current_team;child.step(action)
    if child.terminated:return child.winner==team
    if child.current_team!=team:return False
    for second in child.legal_actions():
        record=child.board.apply_action(second)
        won=record.captured_piece is not None and record.captured_piece.piece_type==PieceType.KERNEL
        child.board.undo_action(record)
        if won:return True
    return False

def verify_envs(envs,label):
    count=0;positives=0;states=0
    for offset in range(0,len(envs),16):
        batch=envs[offset:offset+16]
        val,masks=get_predictions([e.observation() for e in batch])
        for env,value,mask in zip(batch,val,masks):
            predicted=np.flatnonzero(mask)
            if len(predicted):
                states+=1;assert value==1.,(label,value)
            legal={encoding.action_index(a,env.current_team):a for a in env.legal_actions()}
            for idx in predicted:
                assert int(idx) in legal,(label,'illegal prediction',idx)
                assert win_after_first(env,legal[int(idx)]),(label,'false positive',idx)
                count+=1
            positives+=int(value==1.)
    return {'positions':len(envs),'approach_states':states,'approach_actions':count,'total_certified_win_states':positives,'false_positive_actions':0}

start=time.time()
# All six types, both actual sides, two-action quiet approach to unguarded
# Kernel. Artificial material avoids premature bare-Kernel terminals.
fixtures=[]
for team in [constants.CYAN,constants.MAGENTA]:
    for typ,pos in [(PieceType.PING,(5,3)),(PieceType.DAEMON,(0,5)),(PieceType.EXPLOIT,(7,5)),
                    (PieceType.TRACE,(1,0)),(PieceType.ICE,(6,1)),(PieceType.KERNEL,(5,3))]:
        obs=np.zeros((16,8,8),np.float32);obs[typ.encode_index(),*pos]=1
        if typ!=PieceType.KERNEL:obs[5,7,0]=1
        obs[11,3,3]=1;obs[6,0,7]=1
        env=ColonyOneEnv();env.board=Board.from_observation(obs,team=team)
        env.current_team=team;env.round_number=4;env.actions_left=2
        fixtures.append(env)
fixture_result=verify_envs(fixtures,'synthetic')
assert fixture_result['approach_states']==12

# Replay existing exact-engine diagnostic games; certify every new tactical
# approach at every position and record critical losing-turn boundaries.
replays=[];critical=[]
for game in json.loads((ROOT/'codex/matches/selfplay-h1.corrected.json').read_text())['games']:
    env=ColonyOneEnv()
    for row in game['history']:
        replays.append(env.clone())
        if env.current_team==game['winner'] and env.actions_left==2:
            critical.append((game['seed'],env.clone()))
        event=row['event']
        action=next(a for a in env.legal_actions() if list(a.from_pos)==event['fr'] and list(a.to_pos)==event['to'])
        env.step(action)
replay_result=verify_envs(replays,'replay')
last_winning_turn=[]
for seed in sorted({s for s,_ in critical}):
    env=[e for s,e in critical if s==seed][-1]
    v,p=get_predictions([env.observation()])
    assert v[0]==1. and p[0].any(),(seed,v)
    chosen=mcts.search_action(env.board,env.current_team,2,env.round_number,evaluate,mcts.MCTSConfig(simulations=100),rng=np.random.default_rng(19))
    assert win_after_first(env,chosen),(seed,chosen)
    last_winning_turn.append({'seed':seed,'team':env.current_team,'value':float(v[0]),'certified_actions':int(p.sum()),'mcts100_first_action':{'from':chosen.from_pos,'to':chosen.to_pos},'engine_wins':True})

# Reachable boards, including random paths to shrinking-firewall rings.
random_envs=[];rng=np.random.default_rng(20261023)
for game in range(128):
    env=ColonyOneEnv()
    while not env.terminated:
        random_envs.append(env.clone())
        legal=env.legal_actions();env.step(legal[int(rng.integers(len(legal)))])
random_result=verify_envs(random_envs,'random')
result={'status':'PASS','sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),'bytes':MODEL.stat().st_size,
        'synthetic':fixture_result,'replay':replay_result,'last_winning_turns':last_winning_turn,
        'random_reachable':random_result,'seconds':time.time()-start,
        'scope':'Sufficient certificate; false negatives allowed. No grader qualification.'}
(HERE/'proof-v2.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
