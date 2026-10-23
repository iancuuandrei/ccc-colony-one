"""Bounded current-GEN07 counterfactuals from independently supplied certificates."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'env'),str(ROOT/'codex/matches')]
from colony_one import mcts, encoding as E
from colony_one.onnx_runner import OnnxEvaluator
import site_replay as site

MODEL=ROOT/'ckpt/gen07-endgame.onnx'
MODEL_SHA='8b23b6a7f80089aef590beb69a829738e3c8c66dcd5ccbb652820142dbabcebd'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def path(p):
    p=Path(p)
    return p if p.is_absolute() else ROOT/p
def search(e,ev):
    before=e.observation().tobytes()
    a=mcts.search_action(e.board,e.current_team,e.actions_left,e.round_number,ev,
                         mcts.MCTSConfig(simulations=100),rng=np.random.default_rng(0))
    assert e.observation().tobytes()==before
    return a

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',required=True);ap.add_argument('--label',required=True)
    args=ap.parse_args();casepath=Path(args.cases);cases=json.loads(casepath.read_text(encoding='utf-8'))
    assert sha(MODEL)==MODEL_SHA
    inputs={};cache={};packets={};rows=[];start=time.monotonic()
    def register(p):inputs[str(p.resolve())]=sha(p)
    register(casepath)
    for p in sorted((ROOT/'env/colony_one').glob('*.py')):register(p)
    register(ROOT/'env/colony_one/scene.json');register(ROOT/'codex/matches/site_replay.py')
    register(ROOT/'codex/matches/probe_matches.py')
    ev=OnnxEvaluator(MODEL,num_threads=1)
    with (HERE/f'{args.label}-results.jsonl').open('w',encoding='utf-8') as stream:
        for case in cases:
            source=path(case['source_path']);register(source)
            assert sha(source)==case['source_sha256']
            for other in case.get('proof_files',[]):register(path(other))
            if str(source) not in cache:
                game,states=site.parse(source,None)
                cache[str(source)]=(game,{e.action:e for e in states})
            game,states=cache[str(source)];env=states[case['ply']];team=env.current_team
            assert team==case['actor']
            packet=path(case['packet_path']);register(packet)
            assert sha(packet)==case['packet_sha256']
            if str(packet) not in packets:
                with np.load(packet,allow_pickle=False) as d:packets[str(packet)]={k:d[k].copy() for k in d.files}
            data=packets[str(packet)];ix=case['packet_row']
            np.testing.assert_array_equal(env.observation(),data[case.get('obs_field','obs')][ix])
            legal=np.zeros(4096,bool)
            for a in env.legal_actions():legal[int(E.action_index(a,team))]=True
            masks={}
            for label,field in case.get('mask_fields',{}).items():
                masks[label]=np.asarray(data[field][ix],dtype=bool)
                assert masks[label].shape==(4096,)
                if label=='legal':assert np.array_equal(legal,masks[label])
                else:assert not (masks[label] & ~legal).any()
            action=search(env,ev);action_index=int(E.action_index(action,team));assert legal[action_index]
            child=env.clone();events=[site.apply(child,action)]
            current_win_remains=site.winning_line(child,team) is not None
            second=None;actual_indices=[action_index];second_matches_recorded_state=False
            if not child.terminated and child.current_team==team:
                second_matches_recorded_state=env.action+1 in states and np.array_equal(child.observation(),states[env.action+1].observation())
                follow=search(child,ev);second=site.move(follow)['notation']
                actual_indices.append(int(E.action_index(follow,team)));events.append(site.apply(child,follow))
            opponent='CYAN' if team=='MAGENTA' else 'MAGENTA'
            enemyline=site.winning_line(child,opponent);enemy_events=[]
            if enemyline is not None:
                reply=child.clone()
                for move in enemyline:
                    a=next(a for a in reply.legal_actions() if list(a.from_pos)==move['fr'] and list(a.to_pos)==move['to'])
                    enemy_events.append(site.apply(reply,a))
                assert reply.terminated and reply.winner==opponent
            deep_completion=None;deep_lookup_note=None
            if case.get('deep_proof_path'):
                dp=path(case['deep_proof_path']);register(dp);deep=json.loads(dp.read_text(encoding='utf-8'))
                position=next((r for r in deep['positions'] if r['action_number']==case['ply']),None)
                if position:
                    alt=next(r for r in position['alternatives'] if r['index']==action_index)
                    deep_completion=next((r for r in alt['own_completions'] if [s['index'] for s in r['own_line']]==actual_indices),None)
                    if deep_completion is None and second_matches_recorded_state:
                        next_position=next((r for r in deep['positions'] if r['action_number']==case['ply']+1),None)
                        if next_position:
                            next_alt=next(r for r in next_position['alternatives'] if r['index']==actual_indices[1])
                            deep_completion=next((r for r in next_alt['own_completions'] if [s['index'] for s in r['own_line']]==actual_indices[1:]),None)
                            deep_lookup_note='Actual first action exactly reproduces the next recorded observation; reused its complete second-action certificate.'
                    if deep_completion is None:deep_lookup_note='This actual completion is not enumerated in the saved short-circuit proof; do not infer whole-turn safety from existence of a safe first-action completion.'
            record={'case_id':case['case_id'],'game_id':case['game_id'],'ply':case['ply'],'actor':team,
                    'user_team':case.get('user_team','UNKNOWN'),'source_model':'UNKNOWN',
                    'group':f"site-game-{case['game_id']}",'historical_action':game['history'][case['ply']-1]['notation'],
                    'historical_winner':game['winner'],'observation_sha256':hashlib.sha256(env.observation().tobytes()).hexdigest(),
                    'model_sha256':MODEL_SHA,'gen07_action':site.move(action)['notation'],'gen07_action_index':action_index,
                    'certificate_membership':{label:bool(mask[action_index]) for label,mask in masks.items()},
                    'certificate_counts':{label:int(mask.sum()) for label,mask in masks.items()},
                    'after_first_current_turn_win_remains':current_win_remains,'gen07_actual_second_action':second,
                    'gen07_actual_action_indices':actual_indices,'saved_two_enemy_actual_completion_certificate':deep_completion,
                    'saved_two_enemy_completion_lookup_note':deep_lookup_note,
                    'gen07_turn_events':events,'gen07_actual_turn_terminal':child.terminated,
                    'gen07_actual_turn_winner':child.winner,
                    'gen07_actual_turn_win':child.terminated and child.winner==team,
                    'gen07_actual_turn_allows_enemy_terminal_win':enemyline is not None,
                    'enemy_terminal_witness':enemy_events,'certificate_note':case.get('certificate_note'),
                    'query_kind':'CURRENT_GEN07_COUNTERFACTUAL_MCTS100_NOT_HISTORICAL_MODEL_ATTRIBUTION'}
            rows.append(record);stream.write(json.dumps(record)+'\n');stream.flush()
            print(json.dumps({k:record[k] for k in ['case_id','gen07_action','certificate_membership',
                  'gen07_actual_second_action','gen07_actual_turn_win','gen07_actual_turn_allows_enemy_terminal_win']}),flush=True)
    assert sha(MODEL)==MODEL_SHA
    for p,h in inputs.items():assert sha(p)==h
    summary={'cases':len(rows),'game_ids':sorted({r['game_id'] for r in rows}),
             'selected_winning_first_actions':[r['case_id'] for r in rows if r['certificate_membership'].get('winning',False)],
             'actual_model_terminal_wins':[r['case_id'] for r in rows if r['gen07_actual_turn_win']],
             'selected_safe_first_actions':[r['case_id'] for r in rows if r['certificate_membership'].get('safe',False)],
             'actual_model_turn_allows_enemy_terminal_win':[r['case_id'] for r in rows if r['gen07_actual_turn_allows_enemy_terminal_win']],
             'end_model_and_input_hashes_verified':True,'seconds':time.monotonic()-start}
    (HERE/f'{args.label}-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    loop=(ROOT/'loop_state.json').read_bytes()
    (HERE/f'{args.label}-loop-state-end.json').write_bytes(loop)
    protocol={'model_sha256':MODEL_SHA,'inputs_sha256':inputs,'harness_sha256':sha(__file__),
              'processes':1,'ort_threads':1,'mcts_config':mcts.MCTSConfig(simulations=100).__dict__,
              'rng_seed_each_query':0,'loop_state_end_sha256':hashlib.sha256(loop).hexdigest(),
              'loop_state_end_best':json.loads(loop)['best'],'historical_source_model':'UNKNOWN',
              'claim_limit':'Counterfactual exact supplied cases only; no strength games; bounded enemy-turn safety is not eventual victory.'}
    (HERE/f'{args.label}-manifest.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
