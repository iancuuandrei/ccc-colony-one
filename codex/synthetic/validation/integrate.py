"""Publish independently accepted engine labels in the separate distillation queue."""
from pathlib import Path
import argparse,importlib.util,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('synthetic_integrate_queue',ROOT/'codex'/'distillation'/'queue.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

def integrate(validation_dir,out=None):
    vdir=Path(validation_dir).resolve();report=json.loads((vdir/'report.json').read_text())
    if not report['accepted']:raise ValueError('Independent source validation did not accept this batch')
    packet=json.loads((vdir/'source_snapshot'/'packet.json').read_text());data=np.load(vdir/'independently_validated.npz')
    destination=Path(out).resolve() if out else vdir/'distillation';destination.mkdir(parents=True,exist_ok=True)
    positions=[];labels=[];qualified={r['position_id'] for r in report['checked'] if r['status']=='PASS'}
    for i,original in enumerate(packet['positions']):
        if original['position_id'] not in qualified:continue
        cert=original['certificate'];winning=cert['winning_first_actions'];safe=cert['safe_first_actions']
        claim=cert['claim'];exact=cert['value'] is not None
        if not exact and not safe:continue
        p=dict(original);p['source']=dict(p['source'],path=str((vdir/'source_snapshot'/'seed_proposals.json').relative_to(ROOT)),sha256=report['source_hashes']['seed_proposals.json'],game_group=p['group_id'])
        p['constructed']=True;p['reachability']='NOT_PROVEN_FROM_STANDARD_OPENING';p['split']=str(data['split'][i]);p['independent_validation_report']=str((vdir/'report.json').relative_to(ROOT));positions.append(p)
        preferred=winning or safe
        order=sorted(preferred)+sorted(a['action_index'] for a in p['legal_actions'] if a['action_index'] not in preferred)
        # A small legal-action floor is a training convention, not a win probability.
        if not preferred:order=sorted(a['action_index'] for a in p['legal_actions'])
        ranking=[dict(action_index=a,rank=j+1,weight=1. if a in preferred or not preferred else .005) for j,a in enumerate(order)]
        label=dict(position_id=p['position_id'],state_sha256=p['state_sha256'],policy_rankings=ranking,
            value=float(cert['value']) if exact else 0.,value_valid=exact,confidence=1.,value_confidence=1. if exact else 0.,
            claim=claim if exact else 'bounded_policy_only',label_source='ENGINE_TERMINAL_CERTIFICATE' if exact else 'ENGINE_BOUNDED_POLICY_CERTIFICATE',
            evidence='Independent authoritative env.step certificate: '+('terminal outcome before the next own turn.' if exact else 'preferred first actions admit a continuation avoiding terminal defeat through the opponent turn. Full game value UNKNOWN; value=0 is a masked storage placeholder.'),
            oracle_node_budget=500000)
        if winning:label['witness']=cert['witnesses'][str(order[0])]['actions']
        labels.append(label)
    packet=dict(packet,packet_id=packet['packet_id']+'_engine_v1',positions=positions,label_authority='ENGINE_CERTIFICATES_FROM_ASTRA_PROPOSED_STATES',constructed=True,reachability='NOT_PROVEN_FROM_STANDARD_OPENING',independent_report_sha256=q.sha(vdir/'report.json'))
    answer=dict(packet_id=packet['packet_id'],teacher={'model':'gpt-6-astra','role':'STATE_PROPOSAL_ONLY','label_authority':'ENGINE_NOT_LLM_STRATEGIC_PREDICTION'},labels=labels)
    for name,payload in [('packet.json',packet),('engine_labels.json',answer)]:
        path=destination/name
        if path.exists():raise ValueError('Preserve existing integration artifacts; use a new destination')
        path.write_text(json.dumps(payload,indent=1),encoding='utf-8')
    q.consume(destination/'packet.json',destination/'engine_labels.json',destination/'validated')
    # Carry the independently prescribed family split into a separate immutable sidecar.
    split={p['position_id']:p['split'] for p in positions};(destination/'grouped_split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    summary=dict(status='READY_FOR_STUDENT',packet=str(destination/'packet.json'),engine_labels=str(destination/'engine_labels.json'),targets=str(destination/'validated'/'teacher_targets.npz'),grouped_split=str(destination/'grouped_split.json'),rows=len(labels),exact_values=sum(l['value_valid'] for l in labels),masked_unknown_values=sum(not l['value_valid'] for l in labels),family_groups=len({p['group_id'] for p in positions}),reachability='NOT_PROVEN_FROM_STANDARD_OPENING',label_source='ENGINE_CERTIFICATES; Astra authored states, not these value/policy targets',qualification='Constructed curriculum; current/next-turn terminal certificates only; no playing-strength or reachability claim')
    (destination/'integration.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--validation-dir',required=True);p.add_argument('--out');a=p.parse_args();integrate(a.validation_dir,a.out)
