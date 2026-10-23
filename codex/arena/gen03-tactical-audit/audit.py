"""Read-only GEN03/GEN02 tactical comparisons. Outputs stay outside quarantined data."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'env'), str(ROOT/'codex'/'engine_audit'), str(ROOT/'codex'/'matches')]
from colony_one import encoding as E, mcts
from colony_one.onnx_runner import OnnxEvaluator
from own_turn_oracle import own_turn_wins
import site_replay as site
spec = importlib.util.spec_from_file_location('queue_audit',ROOT/'codex'/'distillation'/'queue.py')
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)

MODELS = {
    'gen02': (ROOT/'release'/'gen02'/'model.onnx','579d3ec01da7328dc706003b8dcc5eba357bb74902252187e45633ca0595a9a0'),
    'gen03': (ROOT/'release'/'gen03'/'model.onnx','0cf937446ba89dc9b5b3652edbcb4025030bc785e924551782f4b2d298ebe96f'),
}
_ev = {}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evaluator(name):
    if name not in _ev: _ev[name] = OnnxEvaluator(MODELS[name][0],num_threads=1)
    return _ev[name]
def notation(a): return q.coord(a.from_pos)+'->'+q.coord(a.to_pos)


def evaluate(r):
    results = {}
    cert = r['certificate']
    for name in MODELS:
        env = q.restore(r['state']); before = env.observation().tobytes(); ev = evaluator(name)
        logits,value = ev(env.observation()[None])
        legal = list(map(int,env.legal_action_indices()))
        a = mcts.search_action(env.board,env.current_team,env.actions_left,env.round_number,
                              ev,mcts.MCTSConfig(simulations=100),rng=np.random.default_rng(r['rng_seed']))
        assert env.observation().tobytes()==before
        ix = int(E.action_index(a,env.current_team)); assert ix in legal
        wins = cert.get('winning_first_actions',[]); safe = cert.get('safe_first_actions',[])
        bad = cert.get('unsafe_first_actions',[])
        record = {'action_index':ix,'action':notation(a),'raw_top_action':max(legal,key=lambda i:float(logits[0,i])),
                  'value':float(value[0]),'missed_current_turn_win':bool(wins) and ix not in wins,
                  'avoidable_next_turn_loss':bool(safe) and ix in bad,
                  'selected_certified_safe':ix in safe,'selected_certified_unsafe':ix in bad,
                  'certified_value':cert.get('value')}
        if r['cohort']!='holdout_v1':
            original = env.current_team
            child=env.clone();child.step(a)
            record['selected_keeps_current_turn_win']=site.winning_line(child,original) is not None
            if r['kind']=='site_loss':
                alt=r['alternatives'][record['action']]
                record['saved_selected_certificate']=alt
                if 'safe' in alt and not alt['safe']:
                    witness=child.clone()
                    for step in alt['witness']:
                        action=next(x for x in witness.legal_actions() if list(x.from_pos)==step['fr'] and list(x.to_pos)==step['to'])
                        witness.step(action)
                    assert witness.terminated and witness.winner!=original
                    record['selected_unsafe_witness_replayed']=True
                if env.actions_left==2 and not child.terminated:
                    second=mcts.search_action(child.board,child.current_team,child.actions_left,child.round_number,
                                              ev,mcts.MCTSConfig(simulations=100),rng=np.random.default_rng(r['rng_seed']))
                    record['second_action']=notation(second);child.step(second)
                    opponent='MAGENTA' if original=='CYAN' else 'CYAN'
                    record['completed_turn_allows_enemy_terminal_win']=site.winning_line(child,opponent) is not None
        results[name]=record
    return {'case_id':r['case_id'],'cohort':r['cohort'],'kind':r['kind'],'group_id':r.get('group_id'),
            'position_id':r.get('position_id'),'rng_seed':r['rng_seed'],'models':results,
            'regression': any(results['gen03'][k] and not results['gen02'][k] for k in
                              ('missed_current_turn_win','avoidable_next_turn_loss')),
            'improvement': any(results['gen02'][k] and not results['gen03'][k] for k in
                               ('missed_current_turn_win','avoidable_next_turn_loss'))}


def holdout():
    folder=ROOT/'codex'/'synthetic'/'holdout_v1'; frozen=json.loads((folder/'FROZEN.json').read_text())
    assert frozen['training_allowed'] is False
    inputs=[folder/'FROZEN.json',folder/'EVALUATION_OVERRIDE.json',folder/'evaluation.json']
    for name,digest in frozen['files'].items(): assert sha(folder/name)==digest; inputs.append(folder/name)
    for name,digest in frozen['engine_sha256'].items(): assert sha(ROOT/'env'/'colony_one'/name)==digest
    for name,digest in frozen['oracle_sha256'].items(): assert sha(ROOT/'codex'/'engine_audit'/name)==digest
    rows=[]
    for line in (folder/'states.jsonl').read_text().splitlines():
        r=json.loads(line); assert r['training_allowed'] is False
        r.update(case_id=r['position_id'],cohort='holdout_v1',kind='constructed_holdout',rng_seed=850001)
        rows.append(r)
    return rows,inputs


def actual_site():
    folder=ROOT/'codex'/'matches'; rows=[]; inputs=[]; cache={}
    for npz in sorted(folder.glob('site-gen02-*-hard-v*.npz')):
        kind='site_loss' if '-loss-' in npz.name else 'site_win'; version=npz.stem.rsplit('-',1)[1]
        source=folder/f'site-gen02-{"loss" if kind=="site_loss" else "wins"}-mining-{version}.json'
        assert source.exists(); evidence=json.loads(source.read_text());inputs.extend([npz,source])
        with np.load(npz,allow_pickle=False) as data:
            for i in range(len(data['obs'])):
                gid=str(data['match'][i]);ply=int(data['ply'][i])
                if gid not in cache:
                    path=folder/'site-all-v1-derived'/f'match-{gid}.txt'
                    game,states=site.parse(path,None);cache[gid]=(game,{e.action:e for e in states});inputs.append(path)
                game,states=cache[gid];env=states[ply]
                assert np.array_equal(data['obs'][i],env.observation())
                own=own_turn_wins(env.board,env.current_team,env.actions_left,env.round_number)
                winning=np.flatnonzero(own.first_action_win_mask).tolist()
                cert={'winning_first_actions':winning,'safe_first_actions':[],'unsafe_first_actions':[],
                      'value':1.0 if winning else None}
                alternatives={}
                if kind=='site_loss':
                    proof=next((p for p in evidence['proofs'] if p['game_id']==gid and p['action']==ply),None)
                    if proof is not None: alts=proof['alternatives']
                    else:
                        proof=next(p for p in evidence['proofs'] if p['game_id']==gid and p['preceding_action']==ply)
                        alts=proof['first_alternatives']
                    alternatives={a['notation']:a for a in alts}
                    legal=np.zeros(4096,bool);safe=legal.copy()
                    for a in env.legal_actions():
                        ix=int(E.action_index(a,env.current_team));alt=alternatives[notation(a)];legal[ix]=True
                        safe[ix]=alt['safe'] if 'safe' in alt else alt['safe_completion'] is not None
                    assert np.array_equal(legal,data['legal_mask'][i])
                    assert np.array_equal(safe,data['safe_start_mask'][i])
                    cert['safe_first_actions']=np.flatnonzero(safe).tolist()
                    cert['unsafe_first_actions']=np.flatnonzero(legal & ~safe).tolist()
                else:
                    proof=next(p for p in evidence['records'] if p['game_id']==gid and p['action']==ply)
                    assert winning, 'Saved missed-win packet must contain an actual own-turn terminal win'
                    for flag in proof['flags']:
                        if flag['kind']=='missed_current_turn_win':
                            replay=env.clone()
                            for step in flag['witness']:
                                a=next(a for a in replay.legal_actions() if list(a.from_pos)==step['fr'] and list(a.to_pos)==step['to']);replay.step(a)
                            assert replay.terminated and replay.winner==env.current_team
                rows.append({'case_id':f'{kind}-{version}-{gid}-{ply}','cohort':f'{kind}_{version}',
                             'kind':kind,'group_id':f'site-game-{gid}','state':q.encode_state(env),
                             'certificate':cert,'alternatives':alternatives,'rng_seed':0})
    return rows,inputs


def summary(rows):
    out={'cases':len(rows),'regressions':[r['case_id'] for r in rows if r['regression']],
         'improvements':[r['case_id'] for r in rows if r['improvement']],'cohorts':{}}
    for cohort in sorted({r['cohort'] for r in rows}):
        rr=[r for r in rows if r['cohort']==cohort]
        out['cohorts'][cohort]={'cases':len(rr),'models':{}}
        for name in MODELS:
            out['cohorts'][cohort]['models'][name]={
                'missed_current_turn_wins':sum(r['models'][name]['missed_current_turn_win'] for r in rr),
                'avoidable_next_turn_losses':sum(r['models'][name]['avoidable_next_turn_loss'] for r in rr),
                'exact_value_mae':float(np.mean([abs(r['models'][name]['value']-r['models'][name]['certified_value'])
                                                for r in rr if r['models'][name]['certified_value'] is not None]))
                    if any(r['models'][name]['certified_value'] is not None for r in rr) else None}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['holdout','site'],required=True)
    p.add_argument('--workers',type=int,default=2);o=p.parse_args()
    for path,digest in MODELS.values(): assert sha(path)==digest
    cases,inputs=holdout() if o.phase=='holdout' else actual_site()
    inputs=list(dict.fromkeys(inputs)); hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    record=HERE/f'{o.phase}-results.jsonl'; manifest=HERE/f'{o.phase}-manifest.json'
    protocol={'models':{n:{'path':str(p),'sha256':h} for n,(p,h) in MODELS.items()},'inputs':hashes,
              'mcts_config':mcts.MCTSConfig(simulations=100).__dict__,'harness_sha256':sha(__file__),
              'quarantine':'READ ONLY; no training export; constructed holdout rotations correlated'}
    protocol=json.loads(json.dumps(protocol))
    if manifest.exists(): assert json.loads(manifest.read_text())==protocol
    else: manifest.write_text(json.dumps(protocol,indent=2))
    rows=[json.loads(x) for x in record.read_text().splitlines()] if record.exists() else []
    completed={r['case_id'] for r in rows}
    with record.open('a',encoding='utf-8') as out, ProcessPoolExecutor(o.workers) as ex:
        futures=[ex.submit(evaluate,r) for r in cases if r['case_id'] not in completed]
        for f in as_completed(futures):
            r=f.result();out.write(json.dumps(r)+'\n');out.flush();os.fsync(out.fileno());rows.append(r)
            report=summary(rows);(HERE/f'{o.phase}-summary.json').write_text(json.dumps(report,indent=2))
            if r['regression'] or r['improvement'] or len(rows)%8==0:
                print(json.dumps({'completed':len(rows),'case_id':r['case_id'],'regression':r['regression'],
                                  'improvement':r['improvement'],'models':r['models'] if r['regression'] or r['improvement'] else None}),flush=True)
    assert len(rows)==len(cases)
    for path,digest in MODELS.values():assert sha(path)==digest
    assert {str(p.relative_to(ROOT)):sha(p) for p in inputs}==hashes
    report=summary(rows);report['end_input_and_model_hashes_verified']=True
    if o.phase=='holdout':
        cached=json.loads((ROOT/'codex'/'synthetic'/'holdout_v1'/'evaluation.json').read_text())['results']
        byid={r['position_id']:r for r in cached if r['model']=='gen02'}
        assert all(r['models']['gen02']['action_index']==byid[r['position_id']]['action_index'] for r in rows)
        report['cached_gen02_all_48_decisions_reproduced']=True
    (HERE/f'{o.phase}-summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
