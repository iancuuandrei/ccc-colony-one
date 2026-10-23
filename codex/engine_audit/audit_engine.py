"""Bounded operational rule proof; no self-play/training or engine changes."""
from pathlib import Path
import sys, json, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'env'))
from colony_one import constants as C, encoding as E, mcts
from colony_one.board import Board
from colony_one.env import ColonyOneEnv
from colony_one.pieces import Piece, PieceType as PT
from colony_one.rules import legal_actions_for_team
from colony_one.action import Action

def board(items, round_number=1):
    b = Board()
    b.grid = [[None for _ in range(8)] for _ in range(8)]
    b.locked = Board.locked_cells_for_round(round_number)
    for team, kind, pos in items:
        assert b.grid[pos[0]][pos[1]] is None
        b.grid[pos[0]][pos[1]] = Piece(kind, team)
    return b

def env(b, team=C.CYAN, left=2, rnd=1):
    e = ColonyOneEnv()
    e.board, e.current_team, e.actions_left, e.round_number = b, team, left, rnd
    return e

def main():
    out = {}
    # Every piece type; both actual teams; every possible adjacent single
    # guard square/type compatible with a currently legal attack.
    attackers = [(PT.PING,(1,1)),(PT.DAEMON,(3,0)),(PT.EXPLOIT,(1,2)),
                 (PT.TRACE,(0,0)),(PT.ICE,(1,3)),(PT.KERNEL,(2,2))]
    cases = 0
    for team in [C.CYAN,C.MAGENTA]:
        enemy = C.MAGENTA if team == C.CYAN else C.CYAN
        for kind, src in attackers:
            for guard_kind in list(PT)[:-1]:
                for r in range(2,5):
                    for c in range(2,5):
                        gp=(r,c)
                        if gp in [src,(3,3)]: continue
                        items=[(team,kind,src),(enemy,PT.KERNEL,(3,3)),
                               (enemy,guard_kind,gp)]
                        if kind != PT.KERNEL: items.append((team,PT.KERNEL,(7,7)))
                        e=env(board(items),team)
                        hit=next((a for a in e.legal_actions() if a.from_pos==src and a.to_pos==(3,3)),None)
                        if hit is None: continue
                        _,reward,done,_=e.step(hit)
                        assert not done and reward==0 and e.current_team==team and e.actions_left==1
                        assert e.board.piece_at(*src).piece_type == kind
                        assert e.board.piece_at(*gp) is None
                        assert hit in e.legal_actions(), 'redirect invalidated original attack'
                        _,reward,done,_=e.step(hit)
                        assert done and reward==1 and e.winner==team
                        cases+=1
    out['one_guard_two_action_forced_win']={'status':'PASS','engine_fixtures':cases}

    # Exact blind spot: the root edge is visited repeatedly, but every child
    # after a redirect is a disposable evaluator leaf, never traversed.
    b=board([(C.CYAN,PT.PING,(1,1)),(C.CYAN,PT.KERNEL,(7,7)),
             (C.MAGENTA,PT.KERNEL,(3,3)),(C.MAGENTA,PT.PING,(4,4))])
    hit=next(a for a in legal_actions_for_team(b,C.CYAN) if a.from_pos==(1,1) and a.to_pos==(3,3))
    idx=E.action_index(hit,C.CYAN)
    seen=[]
    def evaluator(obs):
        seen.extend(obs.copy())
        logits=np.full((len(obs),4096),-100,np.float32)
        logits[:,idx]=100
        return logits,np.full(len(obs),0.25,np.float32)
    snapshot=E.encode_board(b,C.CYAN,2,1)
    config=mcts.MCTSConfig(simulations=32,batch_size=1,add_root_noise=False)
    root=mcts.run(b,C.CYAN,2,1,evaluator,config)
    assert root.N[idx]==32 and idx not in root.children
    assert abs(root.W[idx]/root.N[idx]-.25)<1e-9
    assert len(seen)==33 and all(np.array_equal(x,seen[1]) for x in seen[1:])
    assert np.array_equal(snapshot,E.encode_board(b,C.CYAN,2,1))
    out['mcts_redirect_blind_spot']={'status':'PROVEN','root_attack_visits':root.N[idx],
        'cached_child':False,'evaluator_calls':len(seen),'attack_Q':root.W[idx]/root.N[idx],
        'actual_repeat_attack_value':1.0,'board_restored':True}

    # Observation round inversion is exact for engine-consistent live states.
    checks=0
    for rnd in range(1,27):
        for team in [C.CYAN,C.MAGENTA]:
            for left in [1,2]:
                b=board([],rnd)
                obs=E.encode_board(b,team,left,rnd)
                locked_count=int(obs[12].sum())
                rings={0:0,28:1,48:2,60:3}[locked_count]
                recovered=round(C.SHRINK_ROUNDS[rings]-8*(1-float(obs[14,0,0])))
                assert recovered==rnd and round(2*obs[13,0,0])==left
                assert bool(obs[15,0,0])==(team==C.MAGENTA)
                checks+=1
    out['live_observation_clock_recoverability']={'status':'PASS','fixtures':checks}

    # Kernel-only distance rule is checked before rings, at turn end only.
    bare=board([(C.CYAN,PT.KERNEL,(3,3)),(C.MAGENTA,PT.KERNEL,(1,1))])
    assert bare.kernel_endgame_winner(C.CYAN)==C.MAGENTA
    assert bare.kernel_endgame_winner(C.MAGENTA)==C.CYAN
    far=board([(C.CYAN,PT.KERNEL,(1,1)),(C.MAGENTA,PT.KERNEL,(6,6))])
    assert far.kernel_endgame_winner(C.CYAN)==C.MAGENTA
    # Both Kernels burn; retain non-Kernel to avoid earlier bare resolution.
    burned=env(board([(C.CYAN,PT.KERNEL,(0,0)),(C.MAGENTA,PT.KERNEL,(0,7)),
                      (C.MAGENTA,PT.PING,(4,4))],8),C.MAGENTA,1,8)
    _,reward,done,_=burned.step(Action((4,4),(4,5),'move'))
    assert done and burned.winner==C.MAGENTA and reward==1
    # Same perimeter position is not burned when Cyan finishes its turn.
    delayed=env(board([(C.CYAN,PT.KERNEL,(0,0)),(C.MAGENTA,PT.KERNEL,(0,7)),
                       (C.CYAN,PT.PING,(4,4))],8),C.CYAN,1,8)
    delayed.step(Action((4,4),(4,5),'move'))
    assert not delayed.terminated and delayed.board.next_open_ring()==0
    # Bare-Kernel adjudication wins over a due firewall: Magenta loses at
    # distance 2 even though a subsequent simultaneous burn would favor it.
    priority=env(board([(C.CYAN,PT.KERNEL,(0,0)),(C.MAGENTA,PT.KERNEL,(0,3))],8),C.MAGENTA,1,8)
    priority.step(Action((0,3),(0,2),'move'))
    assert priority.terminated and priority.winner==C.CYAN
    assert priority.board.next_open_ring()==0
    out['endgame_and_firewall_priority']={'status':'PASS','fixtures':6}

    # Feed exact published build graph's internal attack maps through ORT;
    # compare only enemy-occupied destinations (legal capture geometry).
    import onnx, onnxruntime as ort
    source=(ROOT/'build_heuristic.py').read_text(encoding='utf-8')
    ns={'__name__':'audit_build','__file__':str(ROOT/'build_heuristic.py')}
    old=sys.argv
    try:
        sys.argv=[str(ROOT/'build_heuristic.py'),str(Path(__file__).with_name('audit-original.onnx'))]
        exec(compile(source,str(ROOT/'build_heuristic.py'),'exec'),ns)
    finally: sys.argv=old
    model=ns['model']
    model.graph.output.clear()
    for teamkey in ['own','en']:
        name=ns[teamkey]['attacks']
        model.graph.output.append(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,['batch',64]))
    attack_path=Path(__file__).with_name('attack_maps.onnx')
    onnx.save(model,attack_path)
    opts=ort.SessionOptions(); opts.intra_op_num_threads=1; opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(attack_path),opts,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(46304)
    obslist=[]; truth=[]
    for n in range(128):
        rnd=int(rng.integers(1,27)); b=board([],rnd)
        available=[(r,c) for r in range(8) for c in range(8) if not b.locked[r][c]]
        rng.shuffle(available)
        identities=[(tm,k) for tm in [C.CYAN,C.MAGENTA] for k in PT]
        rng.shuffle(identities)
        for pos,(tm,k) in zip(available,identities[:min(len(available),int(rng.integers(2,13)))]):
            b.grid[pos[0]][pos[1]]=Piece(k,tm)
        obslist.append(E.encode_board(b,C.CYAN,2,rnd))
        expected=[]
        for tm in [C.CYAN,C.MAGENTA]:
            captures=set(a.to_pos for a in legal_actions_for_team(b,tm)
                         if b.piece_at(*a.to_pos) is not None)
            expected.append(captures)
        truth.append((b,expected))
    maps=session.run(None,{'observation':np.stack(obslist)})
    comparisons=0
    for n,(b,expected) in enumerate(truth):
        for t,tm in enumerate([C.CYAN,C.MAGENTA]):
            for r in range(8):
                for c in range(8):
                    p=b.piece_at(r,c)
                    if p is None or p.team==tm: continue
                    assert bool(maps[t][n,r*8+c]>0)==((r,c) in expected[t])
                    comparisons+=1
    out['published_attack_maps_on_enemy_occupied_squares']={'status':'PASS',
        'bounded_static_boards':128,'square_comparisons':comparisons,
        'limitation':'Capture geometry only; these are not complete legal move masks or post-action maps.'}
    out['sources_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
        'env/colony_one/board.py','env/colony_one/env.py','env/colony_one/mcts.py',
        'env/colony_one/rules.py','env/colony_one/encoding.py','build_heuristic.py']}
    dest=Path(__file__).with_name('results.json')
    dest.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
