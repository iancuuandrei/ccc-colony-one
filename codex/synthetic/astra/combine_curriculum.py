"""Package immutable certified counterfactuals without inventing value labels."""
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('synthetic_queue',ROOT/'codex'/'distillation'/'queue.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[];sources=[]
for name in ['hard_refined','discovery']:
    src=HERE/name/'certified_states.jsonl';sources.append(dict(path=str(src),sha256=sha(src)))
    rows.extend(json.loads(x) for x in src.read_text().splitlines())
assert len({r['state_sha256'] for r in rows})==len(rows)
for r in rows:
    cert=r['certificate'];target=cert['winning_first_actions'] or cert['safe_first_actions']
    r['label_source']='engine_certificate';r['policy_valid']=bool(target) and (bool(cert['winning_first_actions']) or not cert['unknown_first_actions'])
    if 'discovery' in r['group_id']:r['group_id']='astra-family-other_actor_discovery'
    r['motif_id']=r['group_id'];r['variant_id']=r['source']['variant']
    order=target+[a['action_index'] for a in r['legal_actions'] if a['action_index'] not in target]
    r['policy_rankings']=[dict(action_index=a,rank=i+1,weight=1. if a in target else .005) for i,a in enumerate(order)]
    r['value_valid']=cert['value'] is not None
    r['value_confidence']=1. if r['value_valid'] else 0.
out=HERE/'curriculum';out.mkdir(exist_ok=True)
(out/'certified_states.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
npz={}
npz['obs']=np.stack([q.observation(r['state']) for r in rows]);npz['legal_mask']=np.zeros((len(rows),4096),bool);npz['policy']=np.zeros((len(rows),4096),np.float32)
npz['unsafe_mask']=np.zeros((len(rows),4096),bool);npz['unknown_mask']=np.zeros((len(rows),4096),bool)
for i,r in enumerate(rows):
    npz['legal_mask'][i,[a['action_index'] for a in r['legal_actions']]]=1
    w=r['policy_rankings'];s=sum(a['weight'] for a in w)
    for a in w:npz['policy'][i,a['action_index']]=a['weight']/s
    npz['unsafe_mask'][i,r['certificate']['unsafe_first_actions']]=1;npz['unknown_mask'][i,r['certificate']['unknown_first_actions']]=1
npz.update(value=np.array([np.nan if not r['value_valid'] else r['certificate']['value'] for r in rows],np.float32),value_valid=np.array([r['value_valid'] for r in rows]),value_confidence=np.array([r['value_confidence'] for r in rows],np.float32),policy_valid=np.array([r['policy_valid'] for r in rows]),group_id=np.array([r['group_id'] for r in rows]),position_id=np.array([r['position_id'] for r in rows]),hard=np.array([r['difficult_mcts100'] for r in rows]),gen01_hard=np.array([r['baseline']['gen01']['missed_certified_win'] or r['baseline']['gen01']['avoidable_tactical_blunder'] for r in rows]))
np.savez_compressed(out/'certified_states.npz',**npz)
hard=npz['hard'];np.savez_compressed(out/'hard_only.npz',**{k:v[hard] for k,v in npz.items()})
base=json.loads((HERE/'hard_refined'/'manifest.json').read_text())
manifest=dict(status='COMPLETE',provenance=base['provenance'],sources=sources,states=len(rows),llm_seed_proposals=len({r['seed_id'] for r in rows}),groups=len(set(r['group_id'] for r in rows)),hard_states=int(hard.sum()),exact_value_states=int(npz['value_valid'].sum()),unknown_full_game_values=int((~npz['value_valid']).sum()),policy_valid=int(npz['policy_valid'].sum()),per_baseline_errors={name:sum(r['baseline'][name]['missed_certified_win'] or r['baseline'][name]['avoidable_tactical_blunder'] for r in rows) for name in ['h1','v2','gen01']},engine_sha256=base['engine_sha256'],oracle_sha256=base['oracle_sha256'],baseline_sha256=base['baseline_sha256'],npz_sha256=sha(out/'certified_states.npz'),jsonl_sha256=sha(out/'certified_states.jsonl'),sampling=dict(rule='Uniform over motif family, then seed, then D4 variant. Hard examples may receive 2x weight inside a group. Keep all relatives of a motif in one split.',split='Leave-one-motif-family-out assessment; five motif groups, not 224 independent games.',negative_control='Earlier 176 easy control states remain separate under parent astra directory; do not count them as hard.',value='Mask all 144 unknown full-game values. Next-turn-safe policy actions are not labeled eventual wins.'),qualification='LLM-authored constructed counterfactuals; valid engine states, no reachability-from-opening proof. Local measured baseline errors and bounded engine certificates; no tournament-strength inference.')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
(out/'packet.json').write_text(json.dumps(dict(packet_id='astra_synthetic_curriculum0001',schema_version=1,teacher_model='gpt-6-astra',engine_sha256=base['engine_sha256'],positions=rows),indent=1))
print(json.dumps(manifest))
