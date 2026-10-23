"""Complete bounded safety masks without recomputing expensive baseline search."""
exec(__import__('pathlib').Path(__file__).with_name('generate.py').read_text().split('def main():')[0])
src=HERE/'hard';dest=HERE/'hard_refined';dest.mkdir(exist_ok=True)
rows=[json.loads(s) for s in (src/'certified_states.jsonl').read_text().splitlines()]
start=time.monotonic();changed=0
with (dest/'certified_states.jsonl').open('w',encoding='utf-8') as f:
    for row in rows:
        cert=row['certificate']
        if cert['unknown_first_actions'] and not cert['winning_first_actions']:
            env=q.restore(row['state']);result=check_forced_loss_by_next_turn(env,max_nodes=500000)
            cert['loss_oracle']=result
            for field,status in [('unsafe_first_actions','VERIFIED'),('safe_first_actions','DISPROVED'),('unknown_first_actions','UNKNOWN')]:
                cert[field]=[int(a) for a,s in result['per_first_action'].items() if s==status]
            if result['status']=='VERIFIED':cert['value']=-1.;cert['claim']='forced_loss_by_next_turn'
            changed+=1
            for score in row['baseline'].values():
                score['selected_certified_unsafe']=score['mcts100_action'] in cert['unsafe_first_actions']
                score['avoidable_tactical_blunder']=score['selected_certified_unsafe'] and bool(cert['safe_first_actions'])
                score['value_absolute_error']=None if cert['value'] is None else abs(score['value']-cert['value'])
            row['difficult_mcts100']=any(s['missed_certified_win'] or s['avoidable_tactical_blunder'] for s in row['baseline'].values())
            print(json.dumps(dict(refined=changed,seed=row['seed_id'],nodes=result['nodes'],unknown=len(cert['unknown_first_actions']),seconds=round(time.monotonic()-start,2))),flush=True)
        row['motif_id']=row['group_id'];row['variant_id']=row['source']['variant'];row['label_source']='engine_certificate'
        target=cert['winning_first_actions'] or cert['safe_first_actions']
        complete=bool(cert['winning_first_actions']) or not cert['unknown_first_actions']
        # Complete legal probability weights. Unknown safety states are excluded
        # from policy training; the masks remain available for pairwise research.
        row['policy_valid']=bool(target) and complete
        order=target+[a['action_index'] for a in row['legal_actions'] if a['action_index'] not in target]
        row['policy_rankings']=[dict(action_index=int(a),rank=i+1,weight=1. if a in target else .005) for i,a in enumerate(order)]
        row['policy_semantics']='Equal weight among certified winning-first actions, otherwise among actions with a continuation avoiding next-turn defeat; other legal actions are low weight. No eventual-win inference.'
        f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
mask=np.zeros((len(rows),4096),bool);targets=np.zeros((len(rows),4096),np.float32);unsafe=np.zeros_like(mask);unknown=np.zeros_like(mask)
for i,r in enumerate(rows):
    mask[i,[a['action_index'] for a in r['legal_actions']]]=1
    weights=r['policy_rankings'];den=sum(a['weight'] for a in weights)
    for a in weights:targets[i,a['action_index']]=a['weight']/den
    unsafe[i,r['certificate']['unsafe_first_actions']]=1;unknown[i,r['certificate']['unknown_first_actions']]=1
np.savez_compressed(dest/'certified_states.npz',obs=np.stack([q.observation(r['state']) for r in rows]),legal_mask=mask,policy=targets,value=np.array([np.nan if r['certificate']['value'] is None else r['certificate']['value'] for r in rows],np.float32),value_valid=np.array([r['certificate']['value'] is not None for r in rows]),policy_valid=np.array([r['policy_valid'] for r in rows]),group_id=np.array([r['group_id'] for r in rows]),position_id=np.array([r['position_id'] for r in rows]),hard=np.array([r['difficult_mcts100'] for r in rows]),unsafe_mask=unsafe,unknown_mask=unknown)
manifest=json.loads((src/'manifest.json').read_text());manifest.update(refinement=dict(source_sha256=sha(src/'certified_states.jsonl'),updated_states=changed,max_nodes=500000,seconds=time.monotonic()-start),hard_states=sum(r['difficult_mcts100'] for r in rows),per_baseline_errors={name:sum(r['baseline'][name]['missed_certified_win'] or r['baseline'][name]['avoidable_tactical_blunder'] for r in rows) for name in ['h1','v2','gen01']},complete_policy_states=sum(r['policy_valid'] for r in rows),certified_value_states=sum(r['certificate']['value'] is not None for r in rows))
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
(dest/'packet.json').write_text(json.dumps(dict(packet_id='astra_synthetic_hard0001',schema_version=1,teacher_model='gpt-6-astra',engine_sha256=manifest['engine_sha256'],positions=rows),indent=1))
print(json.dumps(manifest),flush=True)
