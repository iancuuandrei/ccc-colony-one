"""Temporary operational fixtures against authoritative engine, not grader proof."""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'env'))
from colony_one import constants, encoding, mcts
from colony_one.board import Board
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator
from colony_one.pieces import PieceType
from colony_one.rules import legal_actions_for_team

MODEL = HERE / 'model-tactical.onnx'
evaluate = OnnxEvaluator(MODEL)
proof = []

def fixture(pt, pos, guards, left, team):
    obs = np.zeros((16,8,8), np.float32)
    obs[pt.encode_index(), *pos] = 1
    if pt != PieceType.KERNEL: obs[5,7,7] = 1
    obs[11,3,3] = 1
    for typ, sq in [(PieceType.PING,(3,4)), (PieceType.EXPLOIT,(4,3))][:guards]:
        obs[6+typ.encode_index(), *sq] = 1
    board = Board.from_observation(obs, team=team)
    env = ColonyOneEnv()
    env.board, env.current_team, env.actions_left, env.round_number = board, team, left, 4
    return env

positions = [(PieceType.PING,(1,1)), (PieceType.DAEMON,(3,2)),
             (PieceType.DAEMON,(3,0)), (PieceType.EXPLOIT,(1,2)),
             (PieceType.TRACE,(0,0)), (PieceType.ICE,(1,3)), (PieceType.KERNEL,(2,3))]
start = time.time()
for team in [constants.CYAN, constants.MAGENTA]:
    for pt, pos in positions:
        for guards in [0,1,2]:
            for left in [1,2]:
                env = fixture(pt,pos,guards,left,team)
                target = encoding.real_pos((3,3), team)
                origin = encoding.real_pos(pos, team)
                action = next(a for a in env.legal_actions() if a.from_pos == origin and a.to_pos == target)
                pol, val = evaluate(env.observation()[None])
                expected = left > guards
                assert (val[0] == 1.) == expected, (team,pt,guards,left,val)
                if expected:
                    idx = encoding.action_index(action,team)
                    assert pol[0,idx] == max(pol[0,i] for i in env.legal_action_indices())
                    selected = mcts.search_action(env.board,team,left,4,evaluate,mcts.MCTSConfig(simulations=100),rng=np.random.default_rng(7))
                    assert selected.to_pos == target, (pt,guards,left,selected)
                    for step in range(guards+1):
                        assert action in env.legal_actions()
                        env.step(action)
                        if step < guards:
                            assert env.board.piece_at(*origin) is not None
                            assert env.board.piece_at(*target).piece_type == PieceType.KERNEL
                            assert not env.terminated
                    assert env.terminated and env.winner == team
                proof.append({'team':team, 'piece':pt.value,'from':list(pos),'guards':guards,'actions_left':left,
                              'forced_win':expected, 'value':float(val[0]), 'engine_sequence_checked':expected})

# Two independent stationary shot policies must equal captured piece value;
# movement penalties/rewards are exactly cancelled on these legal actions.
shots=[]
for typ, origin, target in [(PieceType.DAEMON,(3,3),(2,2)),(PieceType.TRACE,(0,0),(2,2))]:
    obs=np.zeros((16,8,8),np.float32)
    obs[typ.encode_index(),*origin]=1; obs[5,7,7]=1; obs[11,0,7]=1; obs[8,*target]=1
    board=Board.from_observation(obs)
    action=next(a for a in legal_actions_for_team(board,constants.CYAN) if a.from_pos==origin and a.to_pos==target)
    assert action.kind=='shoot'
    encoded=encoding.encode_board(board,constants.CYAN,1,4)
    pol,val=evaluate(encoded[None]); got=float(pol[0,encoding.action_index(action,constants.CYAN)])
    assert abs(got-3.52)<1e-5,got
    record=board.apply_action(action); assert not record.moved
    shots.append({'piece':typ.value,'policy':got,'expected':3.52,'engine_stationary':True})

for batch in [1,8,16]:
    obs=np.repeat(ColonyOneEnv().observation()[None],batch,axis=0)
    policy,value=evaluate(obs)
    assert policy.shape==(batch,4096) and value.shape==(batch,)
    assert np.isfinite(policy).all() and np.isfinite(value).all() and (np.abs(value)<=1).all()

# Cross-check exact win detector on independently generated reachable states.
# No reimplementation of attack geometry: legal_actions and engine guards
# provide the expected label, including blocking and firewall restrictions.
rng=np.random.default_rng(178)
random_obs=[]; random_expected=[]
for game in range(24):
    env=ColonyOneEnv()
    while not env.terminated:
        legal=env.legal_actions()
        enemy=constants.MAGENTA if env.current_team==constants.CYAN else constants.CYAN
        kernel_pos=next(( (r,c) for r in range(8) for c in range(8)
                         if (p:=env.board.piece_at(r,c)) is not None and p.team==enemy and p.piece_type==PieceType.KERNEL))
        guards=sum(1 for r in range(max(0,kernel_pos[0]-1),min(8,kernel_pos[0]+2))
                   for c in range(max(0,kernel_pos[1]-1),min(8,kernel_pos[1]+2))
                   if (p:=env.board.piece_at(r,c)) is not None and p.team==enemy and p.piece_type!=PieceType.KERNEL)
        expected=any(a.to_pos==kernel_pos for a in legal) and env.actions_left>guards
        random_obs.append(env.observation());random_expected.append(expected)
        env.step(legal[int(rng.integers(len(legal)))])
for offset in range(0,len(random_obs),16):
    _,val=evaluate(np.stack(random_obs[offset:offset+16]))
    assert np.array_equal(val==1.,random_expected[offset:offset+16])

result={'status':'PASS','sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(), 'bytes':MODEL.stat().st_size,
        'fixtures':proof,'stationary_shots':shots,'dynamic_batches':[1,8,16],
        'reachable_crosscheck':{'games':24,'positions':len(random_obs),'forced_win_labels':sum(random_expected)},
        'seconds':time.time()-start}
(HERE/'proof.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['fixtures','stationary_shots']}))
print('engine fixtures',len(proof),'forced wins',sum(x['forced_win'] for x in proof),'shots',len(shots))
