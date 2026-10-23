"""Exhaustive operational oracle comparisons; no changes to raw game files."""
import sys, json, hashlib, time, argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'env'))
from colony_one.env import ColonyOneEnv
from colony_one.board import Board
from colony_one.pieces import Piece, PieceType
from colony_one import constants,encoding


def env_for(own,enemy,team,actions,round_number):
    e=ColonyOneEnv()
    e.board.grid=[[None]*8 for _ in range(8)]
    e.board.locked=Board.locked_cells_for_round(round_number)
    e.board.grid[own//8][own%8]=Piece(PieceType.KERNEL,team)
    other=constants.MAGENTA if team==constants.CYAN else constants.CYAN
    e.board.grid[enemy//8][enemy%8]=Piece(PieceType.KERNEL,other)
    e.current_team=team;e.actions_left=actions;e.round_number=round_number
    e.action=1;e.terminated=False;e.winner=None
    return e


def oracle(e):
    team=e.current_team; scores={}
    for a in e.legal_actions():
        idx=encoding.action_index(a,team); child=e.clone();child.step(a)
        if child.terminated: win=child.winner==team
        else:
            wins=[]
            for b in child.legal_actions():
                last=child.clone();last.step(b)
                assert last.terminated,'bare second action did not adjudicate'
                wins.append(last.winner==team)
            win=any(wins)
        scores[idx]=win
    return scores


def main():
    p=argparse.ArgumentParser();p.add_argument('--proof',required=True);p.add_argument('--candidate');p.add_argument('--base');p.add_argument('--out',required=True)
    a=p.parse_args();start=time.time()
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    s=ort.InferenceSession(a.proof,sess_options=options,providers=['CPUExecutionProvider'])
    counts=dict(states=0,actions=0,winning_states=0,losing_states=0,policy_mismatches=0,value_mismatches=0,disabled=0)
    pending=[];mismatches=[]
    def flush():
        if not pending:return
        obs=np.stack([e.observation() for e,_ in pending]); good,val,enabled=s.run(None,{'observation':obs})
        for j,(e,sc) in enumerate(pending):
            want=1 if any(sc.values()) else -1
            counts['states']+=1;counts['actions']+=len(sc);counts['winning_states' if want==1 else 'losing_states']+=1
            if val[j]!=want:counts['value_mismatches']+=1
            if not enabled[j,0]:counts['disabled']+=1
            bad=[idx for idx,w in sc.items() if bool(good[j,idx%64]>.5)!=w]
            counts['policy_mismatches']+=len(bad)
            if (bad or val[j]!=want) and len(mismatches)<10:mismatches.append(dict(obs=obs[j].tolist(),scores=sc,bad=bad,value=float(val[j]),want=want))
        pending.clear()
    # All legal static configurations on each live concentric board, all
    # sides/actions. Repeat at shrink rounds to check endgame/firewall order.
    strata=[]
    for round_number in [1,8,9,16,17,24,25,26]:
        n0=counts['states'];locked=Board.locked_cells_for_round(round_number)
        squares=[s for s in range(64) if not locked[s//8][s%8]]
        for team in [constants.CYAN,constants.MAGENTA]:
            for actions in [1,2]:
                for own in squares:
                    for enemy in squares:
                        if own==enemy:continue
                        e=env_for(own,enemy,team,actions,round_number)
                        pending.append((e,oracle(e)))
                        if len(pending)>=128:flush()
        flush();strata.append(dict(round=round_number,states=counts['states']-n0));print(json.dumps(dict(round=round_number,counts=counts)),flush=True)
    unchanged=None
    if a.candidate and a.base:
        base=ort.InferenceSession(a.base,sess_options=options,providers=['CPUExecutionProvider'])
        cand=ort.InferenceSession(a.candidate,sess_options=options,providers=['CPUExecutionProvider'])
        e=ColonyOneEnv(seed=42);boards=[];rng=np.random.default_rng(42)
        for _ in range(256):
            if e.terminated:e=ColonyOneEnv(seed=42)
            if not e.board.only_kernels_remain():boards.append(e.observation())
            e.step(int(rng.choice(e.legal_action_indices())))
        obs=np.stack(boards);bp,bv=base.run(None,{'observation':obs});cp,cv=cand.run(None,{'observation':obs})
        unchanged=dict(boards=len(boards),policy_exact=np.array_equal(bp,cp),value_exact=np.array_equal(bv,cv))
    sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'env/colony_one/env.py',ROOT/'env/colony_one/board.py',ROOT/'env/colony_one/rules.py',Path(__file__)]}
    result=dict(counts=counts,strata=strata,mismatches=mismatches,nonbare_unchanged=unchanged,seconds=time.time()-start,sources=sources,proof_sha256=hashlib.sha256(Path(a.proof).read_bytes()).hexdigest())
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result))
    assert counts['policy_mismatches']==counts['value_mismatches']==counts['disabled']==0
    assert unchanged is None or (unchanged['policy_exact'] and unchanged['value_exact'])


if __name__=='__main__':main()
