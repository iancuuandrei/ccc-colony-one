"""Independent exact-referee MCTS100 difficulty replay on frozen constructed data."""
from pathlib import Path
import argparse,importlib.util,json,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('synthetic_gen01_queue',ROOT/'codex'/'distillation'/'queue.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
from colony_one import mcts,encoding
from colony_one.onnx_runner import OnnxEvaluator
def run(vdir):
    vdir=Path(vdir);snap=vdir/'source_snapshot';manifest=json.loads((snap/'manifest.json').read_text());seeds=json.loads((snap/'seed_proposals.json').read_text())['seeds'];seed_index={s['id']:i for i,s in enumerate(seeds)}
    rows=json.loads((snap/'packet.json').read_text())['positions'];model=ROOT/'release'/'gen01'/'model.onnx';digest=q.sha(model)
    if digest!=manifest['baseline_sha256']['gen01']:raise ValueError('gen01 source hash mismatch')
    ev=OnnxEvaluator(model,num_threads=1);out=[];start=time.monotonic()
    for i,row in enumerate(rows):
        env=q.restore(row['state']);before=q.encode_state(env)
        a=mcts.search_action(env.board,env.current_team,env.actions_left,env.round_number,ev,mcts.MCTSConfig(simulations=100),rng=np.random.default_rng(742000+seed_index[row['seed_id']]));idx=int(encoding.action_index(a,env.current_team))
        if q.encode_state(env)!=before:raise ValueError('MCTS mutated state')
        cert=row['certificate'];win=bool(cert['winning_first_actions']) and idx not in cert['winning_first_actions'];bad=bool(cert['safe_first_actions']) and idx in cert['unsafe_first_actions']
        expected=row['baseline']['gen01']['mcts100_action']
        out.append(dict(position_id=row['position_id'],action=idx,producer_action=expected,exact_action_match=idx==expected,missed_certified_win=win,avoidable_tactical_blunder=bad))
        if (i+1)%32==0:print(json.dumps(dict(positions=i+1,seconds=round(time.monotonic()-start,1))),flush=True)
    result=dict(model_sha256=digest,packet_sha256=q.sha(snap/'packet.json'),positions=len(out),action_mismatches=sum(not r['exact_action_match'] for r in out),missed_certified_wins=sum(r['missed_certified_win'] for r in out),avoidable_tactical_blunders=sum(r['avoidable_tactical_blunder'] for r in out),errors=sum(r['missed_certified_win'] or r['avoidable_tactical_blunder'] for r in out),seconds=time.monotonic()-start,records=out,qualification='Exact current-model local MCTS100 choices on constructed states; no independent-game or grader proof')
    (vdir/'gen01-independent-difficulty.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--validation-dir',required=True);a=p.parse_args();run(a.validation_dir)
