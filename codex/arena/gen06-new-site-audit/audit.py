"""Eight frozen website counterfactual queries; one process, one ORT thread."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MATCHES = ROOT / 'codex/matches'
sys.path[:0] = [str(ROOT/'env'), str(MATCHES)]
from colony_one import mcts, encoding as E
from colony_one.onnx_runner import OnnxEvaluator
import site_replay as site
from site_v2_first_action import safe_completion

MODEL = ROOT/'release/gen06/model.onnx'
MODEL_SHA = '123467a930e0598c464ae7694800ad565aeac71680b88dafc5f014bfe7cf50ac'
GEN05_SHA = 'af45a5f85894e694f7ddab47811c4fb45b9f9536345d78b0f2fa3ef06c99c91e'
TARGETS = [('14878',36,'v1'),('14621',63,'v1'),('16056',36,'v2'),
           ('14009',33,'v2'),('13352',34,'v2'),('15116',31,'v1'),
           ('15116',35,'v1'),('15116',33,'win')]
PINNED_NPZ = {'v1':'85b783756dc3d0a721b280a846597e8546aca90bb4a393a6cda8f61663f24a9e',
              'v2':'a71051f87c1c54deff9eae0140c3e50c9c2043f3c7df4a7e1a17742785509d24',
              'win':'d42e9a23653a77f96ee2e5ddd6a9e9fd99aa1a769aab8e9825248c39d985081f'}
inputs = {}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def register(p):
    inputs[str(p.relative_to(ROOT))] = sha(p)
    return p
def read(p): return json.loads(register(p).read_text(encoding='utf-8'))
def notation(a): return site.move(a)['notation']
def action_for(e, step):
    return next(a for a in e.legal_actions() if list(a.from_pos)==step['fr'] and list(a.to_pos)==step['to'])
def replay(e, line):
    child=e.clone(); events=[]
    for step in line:
        assert not child.terminated
        events.append(site.apply(child,action_for(child,step)))
    return child,events
def selected(e, ev):
    before=e.observation().tobytes()
    a=mcts.search_action(e.board,e.current_team,e.actions_left,e.round_number,ev,
                         mcts.MCTSConfig(simulations=100),rng=np.random.default_rng(0))
    assert e.observation().tobytes()==before
    assert int(E.action_index(a,e.current_team)) in e.legal_action_indices()
    return a

def main():
    assert sha(MODEL)==MODEL_SHA
    start=time.monotonic()
    index=read(ROOT/'data/ext_site_v1/hard_decisions_gen05_v1.json')
    assert index['baseline_sha256']==GEN05_SHA
    read(MATCHES/'site-gen05-attribution-v2.json')
    read(MATCHES/'site-export5-source-v1.json')
    register(MATCHES/'site-gen05-report-v1.md')
    evidence={v:read(MATCHES/f'site-gen05-loss-mining-{v}.json') for v in ['v1','v2']}
    first={v:read(MATCHES/f'site-gen05-earlier-first-actions-{v}.json') for v in ['v1','v2']}
    evidence['win']=read(MATCHES/'site-gen05-wins-mining-v1.json')
    for v in evidence: assert evidence[v]['baseline_sha256']==GEN05_SHA
    packets={}
    for v in evidence:
        path=MATCHES/('site-gen05-wins-hard-v1.npz' if v=='win' else f'site-gen05-loss-hard-{v}.npz')
        register(path); assert sha(path)==PINNED_NPZ[v]
        with np.load(path,allow_pickle=False) as d: packets[v]={k:d[k].copy() for k in d.files}
    for p in sorted((ROOT/'env/colony_one').glob('*.py')): register(p)
    register(ROOT/'env/colony_one/scene.json')
    for name in ['site_replay.py','site_v2_first_action.py','probe_matches.py']: register(MATCHES/name)
    ev=OnnxEvaluator(MODEL,num_threads=1)
    rows=[]; cache={}
    with (HERE/'results.jsonl').open('w',encoding='utf-8') as stream:
        for gid,ply,v in TARGETS:
            proofset=evidence[v]
            source=MATCHES/('site-gen05-raw' if v=='v1' else 'site-all-v1-derived')/f'match-{gid}.txt'
            register(source)
            if str(source) not in cache:
                game,states=site.parse(source,None)
                cache[str(source)]=(game,{s.action:s for s in states})
            game,states=cache[str(source)]; env=states[ply]; team=env.current_team
            opponent='CYAN' if team=='MAGENTA' else 'MAGENTA'
            attribution=next(r for r in index['records'] if str(r['game_id'])==gid and r['ply']==ply)
            assert attribution['analyzed_actor']==team and attribution['source_model']=='UNKNOWN'
            d=packets[v]; pos=next(i for i in range(len(d['ply'])) if str(d['match'][i])==gid and int(d['ply'][i])==ply)
            assert np.array_equal(d['obs'][pos],env.observation())
            record=next((r for r in proofset['records'] if r['game_id']==gid and r['action']==ply),None)
            actual=game['history'][ply-1]['notation']
            winning=[]; safe=[]; certificate_events=[]
            if v=='win':
                assert sha(source)==record['source_sha256']
                gen05=record['baseline_selected']
                for a in env.legal_actions():
                    child=env.clone();child.step(a)
                    if site.winning_line(child,team) is not None: winning.append(notation(a))
                assert winning
                for flag in record['flags']:
                    if flag['kind']=='MISSED_CURRENT_TURN_WIN':
                        terminal,events=replay(env,flag['typed_witness'])
                        assert terminal.terminated and terminal.winner==team
                        certificate_events.append(events)
                gen05_ok=gen05 in winning
                assert gen05_ok==record['baseline_keeps_win']
            else:
                coverage=next(r for r in proofset['coverage'] if r['game_id']==gid)
                assert sha(source)==coverage['source_sha256']
                if env.actions_left==2:
                    record=next(r for r in proofset['records'] if r['game_id']==gid and r.get('preceding_action')==ply)
                    alts=record['first_alternatives']
                    old=next(r for r in first[v] if r['game_id']==gid and r['action']==ply)
                    gen05=old['baseline_selected']; gen05_ok=old['baseline_first_safe']
                else:
                    alts=record['alternatives'];gen05=record['baseline_selected'];gen05_ok=record['baseline_safe']
                alts={a['notation']:a for a in alts}
                legal_mask=np.zeros(4096,bool); safe_mask=legal_mask.copy()
                for a in env.legal_actions():
                    ix=int(E.action_index(a,team));legal_mask[ix]=True
                    alt=alts[notation(a)];child=env.clone();child.step(a)
                    exact_safe=safe_completion(child,opponent) is not None
                    saved_safe=alt['safe'] if 'safe' in alt else alt['safe_completion'] is not None
                    assert exact_safe==saved_safe
                    safe_mask[ix]=exact_safe
                    if exact_safe:
                        safe.append(notation(a))
                        if alt.get('safe_completion') is not None:
                            end,events=replay(child,alt['safe_completion'])
                            assert safe_completion(end,opponent) is not None
                            certificate_events.append(events)
                    elif alt.get('witness') is not None:
                        end,events=replay(child,alt['witness'])
                        assert end.terminated and end.winner==opponent
                        certificate_events.append(events)
                assert np.array_equal(legal_mask,d['legal_mask'][pos])
                assert np.array_equal(safe_mask,d['safe_start_mask'][pos])
                assert gen05_ok==(gen05 in safe)
            a=selected(env,ev);choice=notation(a);child=env.clone();events=[site.apply(child,a)]
            gen06_ok=choice in (winning if v=='win' else safe)
            second=None
            if not child.terminated and child.current_team==team:
                a2=selected(child,ev);second=notation(a2);events.append(site.apply(child,a2))
            enemy_line=site.winning_line(child,opponent)
            if enemy_line is not None:
                end,reply_events=replay(child,enemy_line)
                assert end.terminated and end.winner==opponent
            else: reply_events=[]
            row={'case_id':f'{gid}/{ply}','game_id':gid,'ply':ply,'group':f'site-game-{gid}',
                 'historical_action':actual,'historical_winner':game['winner'],
                 'attribution':attribution,'source_model':'UNKNOWN',
                 'query_kind':'COUNTERFACTUAL_FIXED_GEN06_MCTS100',
                 'safe_first_actions':safe,'winning_first_actions':winning,
                 'gen05_saved_action':gen05,'gen05_saved_certificate_pass':gen05_ok,
                 'gen06_action':choice,'gen06_certificate_pass':gen06_ok,'gen06_actual_second_action':second,
                 'gen06_actual_turn_terminal':child.terminated,'gen06_actual_turn_winner':child.winner,
                 'gen06_actual_turn_win':child.terminated and child.winner==team,
                 'gen06_actual_turn_allows_enemy_terminal_win':enemy_line is not None,
                 'gen06_actual_events':events,'enemy_terminal_witness':reply_events,
                 'certificate_witness_replays':len(certificate_events),
                 'all_saved_action_certificates_and_npz_masks_independently_verified':True}
            if winning and gen06_ok: assert row['gen06_actual_turn_win'], 'Winning first action needs actual second-action terminal proof'
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
            print(json.dumps({k:row[k] for k in ['case_id','gen05_saved_action','gen06_action','gen06_certificate_pass',
                  'gen06_actual_second_action','gen06_actual_turn_win','gen06_actual_turn_allows_enemy_terminal_win']}),flush=True)
    assert sha(MODEL)==MODEL_SHA
    for p,h in inputs.items(): assert sha(ROOT/p)==h
    summary={'cases':len(rows),'independent_games':len(set(r['group'] for r in rows)),
             'gen05_saved_certificate_passes':sum(r['gen05_saved_certificate_pass'] for r in rows),
             'gen06_certificate_passes':sum(r['gen06_certificate_pass'] for r in rows),
             'remaining_own_loss_failures':[r['case_id'] for r in rows if r['attribution']['user_actor'] and not r['winning_first_actions'] and r['gen06_actual_turn_allows_enemy_terminal_win']],
             'remaining_opponent_defense_failures':[r['case_id'] for r in rows if not r['attribution']['user_actor'] and r['gen06_actual_turn_allows_enemy_terminal_win']],
             'remaining_missed_own_wins':[r['case_id'] for r in rows if r['winning_first_actions'] and not r['gen06_actual_turn_win']],
             'end_model_and_input_hashes_verified':True,'elapsed_seconds':time.monotonic()-start}
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    manifest={'model':str(MODEL.relative_to(ROOT)),'model_sha256':MODEL_SHA,'saved_gen05_sha256':GEN05_SHA,
              'inputs_sha256':inputs,'harness_sha256':sha(__file__),'mcts_config':mcts.MCTSConfig(simulations=100).__dict__,
              'rng_seed_per_query':0,'processes':1,'ort_threads':1,'source_model':'UNKNOWN',
              'scope':'Eight counterfactual state queries, no strength games; defense means no terminal defeat on next enemy turn, not eventual victory.'}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    lines=['# GEN06 new site counterfactual audit','',
           'Fixed GEN06 SHA256 `'+MODEL_SHA+'`; exact MCTS100, RNG seed 0 per query, one process / one ORT thread. GEN05 choices reused from immutable evidence. Historical source model UNKNOWN. All saved observations, legal/admissible masks and relevant witnesses independently verified against the engine. No strength games or training.','',
           '| Case | Actor / user actor | GEN05 saved | GEN06 first | Actual GEN06 second | Certificate / completion |',
           '|---|---|---|---|---|---|']
    for r in rows:
        outcome=('terminal win' if r['gen06_actual_turn_win'] else 'allows enemy terminal win' if r['gen06_actual_turn_allows_enemy_terminal_win'] else 'avoids checked terminal loss')
        lines.append(f"| {r['case_id']} | {r['attribution']['analyzed_actor']} / {r['attribution']['user_actor']} | {r['gen05_saved_action']} | {r['gen06_action']} | {r['gen06_actual_second_action'] or 'n/a'} | {'PASS' if r['gen06_certificate_pass'] else 'FAIL'}; {outcome} |")
    lines+=['','Only six independent games: the three 15116 positions share one game group. Defensive certificates are bounded to this turn plus the next enemy turn. These queries do not identify the model that played the website games or establish overall strength.','',json.dumps(summary,indent=2)]
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
