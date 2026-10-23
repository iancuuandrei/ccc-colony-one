"""Validate teacher labels against packet hashes, exact legal actions and bounded oracles."""
from __future__ import annotations
import argparse,importlib.util,json,math,re,sys
from pathlib import Path
import numpy as np
_qspec=importlib.util.spec_from_file_location('codex_distillation_queue',Path(__file__).with_name('queue.py'))
_qmod=importlib.util.module_from_spec(_qspec);sys.modules[_qspec.name]=_qmod;_qspec.loader.exec_module(_qmod)
HERE,ROOT,sha,restore,encode_state,state_hash=(_qmod.HERE,_qmod.ROOT,_qmod.sha,_qmod.restore,_qmod.encode_state,_qmod.state_hash)
from colony_one import encoding
sys.path.insert(0,str(ROOT/'codex'/'engine_audit'))
from own_turn_oracle import own_turn_wins

def number(v,lo,hi):return isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) and lo<=v<=hi
def validate_file(packet_path,labels_path):
    packet=json.loads(Path(packet_path).read_text(encoding='utf-8'));raw=json.loads(Path(labels_path).read_text(encoding='utf-8'))
    errors=[];validated=[];checks=[]
    if raw.get('packet_id')!=packet['packet_id']:errors.append('packet_id mismatch')
    if raw.get('teacher',{}).get('model')!='gpt-6-astra':errors.append('Teacher model metadata must be gpt-6-astra; this metadata alone is not proof of model execution')
    for f,h in packet['engine_sha256'].items():
        if sha(ROOT/'env'/'colony_one'/f)!=h:errors.append('engine source drift: '+f)
    positions={p['position_id']:p for p in packet['positions']};seen=set();source_checked={}
    labs=raw.get('labels',[])
    if not isinstance(labs,list) or not labs:errors.append('Nonempty labels array required');labs=[]
    for l in labs:
        pid=l.get('position_id');before=len(errors)
        try:
            if pid not in positions:raise ValueError('Unknown position_id')
            if pid in seen:raise ValueError('Duplicate position_id')
            seen.add(pid);p=positions[pid]
            src=p['source'];source_path=(ROOT/src['path']).resolve()
            if not source_path.is_relative_to(ROOT.resolve()):raise ValueError('Source path escapes task root')
            if source_path not in source_checked:source_checked[source_path]=sha(source_path)
            if source_checked[source_path]!=src['sha256']:raise ValueError('Source artifact hash drift')
            if l.get('state_sha256')!=p['state_sha256'] or state_hash(p['state'])!=p['state_sha256']:raise ValueError('State hash mismatch')
            env=restore(p['state'])
            if encode_state(env)!=p['state']:raise ValueError('State clock/observation roundtrip mismatch')
            legal=set(int(encoding.action_index(a,env.current_team)) for a in env.legal_actions())
            if legal!={a['action_index'] for a in p['legal_actions']}:raise ValueError('Packet legal actions differ from exact engine')
            pol=l.get('policy_rankings')
            if not isinstance(pol,list) or not pol:raise ValueError('Nonempty policy_rankings required')
            indices=[];ranks=[];weights=[]
            for row in pol:
                a=row.get('action_index');rank=row.get('rank');w=row.get('weight')
                if type(a)!=int or a not in legal:raise ValueError('Illegal ranked action '+str(a))
                if type(rank)!=int or rank<1:raise ValueError('Invalid policy rank')
                if not number(w,0,1e9) or w==0:raise ValueError('Policy weights must be positive finite numbers')
                indices.append(a);ranks.append(rank);weights.append(w)
            if len(indices)!=len(set(indices)) or sorted(ranks)!=list(range(1,len(ranks)+1)):raise ValueError('Unique actions and sequential ranks1..N required')
            ranked=sorted(pol,key=lambda x:x['rank'])
            if any(a['weight']<b['weight'] for a,b in zip(ranked,ranked[1:])):raise ValueError('Weights must not increase for worse ranks')
            if not number(l.get('value'),-1,1) or not number(l.get('confidence'),0,1):raise ValueError('Value/confidence range invalid')
            if not number(l.get('value_confidence',l['confidence']),0,1):raise ValueError('value_confidence range invalid')
            if not isinstance(l.get('evidence'),str) or not 1<=len(l['evidence'])<=500:raise ValueError('Evidence must contain1..500characters')
            claim=l.get('claim','prediction')
            if claim not in ('prediction','forced_win_this_turn','forced_loss_by_next_turn','bounded_policy_only'):raise ValueError('Unsupported exact assertion scope')
            proof={'position_id':pid,'claim':claim,'status':'PREDICTION_ONLY'}
            if claim=='forced_win_this_turn':
                result=own_turn_wins(env.board,env.current_team,env.actions_left,env.round_number)
                top=ranked[0]['action_index']
                if not result.any_win or not result.first_action_win_mask[top]:raise ValueError('Claimed top policy action does not certify an own-turn terminal win')
                if l['value']<.95:raise ValueError('Own-turn exact win conflicts with value below0.95')
                witness=l.get('witness',[])
                if witness:
                    startteam=env.current_team
                    for idx in witness:
                        if env.terminated or env.current_team!=startteam:raise ValueError('Witness exceeds current turn')
                        env.step(encoding.decode_action(idx,env.current_team,env.board))
                    if not env.terminated or env.winner!=startteam:raise ValueError('Witness does not end in claimed win')
                proof.update(status='VERIFIED',oracle_witness=result.witnesses[top])
            elif claim=='forced_loss_by_next_turn':
                path=ROOT/'codex'/'engine_audit'/'oracle.py'
                if not path.exists():raise ValueError('Forced-loss oracle unavailable; downgrade claim to prediction')
                spec=importlib.util.spec_from_file_location('distillation_loss_oracle',path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
                result=mod.check_forced_loss_by_next_turn(env,max_nodes=20000)
                if result.get('status')!='VERIFIED':raise ValueError('Forced-loss oracle did not verify: '+str(result.get('status')))
                if l['value']>-.95:raise ValueError('Exact loss conflicts with value above-0.95')
                proof.update(status='VERIFIED',oracle_certificate=result)
            elif claim=='bounded_policy_only':
                if l.get('label_source')!='ENGINE_BOUNDED_POLICY_CERTIFICATE' or l.get('value_valid') is not False or l.get('value_confidence')!=0:raise ValueError('Bounded-policy-only labels require engine source and explicitly masked unknown value')
                path=ROOT/'codex'/'engine_audit'/'oracle.py';spec=importlib.util.spec_from_file_location('distillation_safety_oracle',path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
                result=mod.check_forced_loss_by_next_turn(env,max_nodes=int(l.get('oracle_node_budget',100000)))
                top=ranked[0]['action_index']
                if result.get('per_first_action',{}).get(top)!='DISPROVED':raise ValueError('Top bounded-policy action lacks an exact escape through opponent-turn horizon')
                proof.update(status='VERIFIED_POLICY_ONLY',value='UNKNOWN',oracle_certificate=result)
            elif re.search(r'\b(?:guaranteed|forced[- ]win|forced[- ]loss|all moves lose|cannot avoid losing)\b',l['evidence'],re.I):
                raise ValueError('Prediction evidence contains an unsupported exact assertion; use bounded exact claim or qualified predictive language')
            clean=dict(l,claim=claim,policy_rankings=ranked,value_confidence=l.get('value_confidence',l['confidence']),value_valid=l.get('value_valid',True),label_source=l.get('label_source','ASTRA_STRATEGIC_PREDICTION' if claim=='prediction' else 'ASTRA_CLAIM_ENGINE_VERIFIED'))
            validated.append(clean);checks.append(proof)
        except Exception as exc:errors.append(str(pid)+': '+type(exc).__name__+': '+str(exc))
    return dict(accepted=not errors,packet_id=packet['packet_id'],packet_sha256=sha(packet_path),labels_sha256=sha(labels_path),validator_sha256=sha(__file__),oracle_sha256=sha(ROOT/'codex'/'engine_audit'/'own_turn_oracle.py'),synthetic_fixture=bool(raw.get('teacher',{}).get('synthetic_fixture',False)),errors=errors,checks=checks,labels=validated,unlabelled_positions=sorted(set(positions)-seen),qualification='Validated teacher predictions and bounded exact certificates; no playing-strength or grader-acceptance claim')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--labels',required=True);p.add_argument('--out');a=p.parse_args()
    r=validate_file(Path(a.packet),Path(a.labels));s=json.dumps(r,indent=2)
    if a.out:Path(a.out).write_text(s,encoding='utf-8')
    print(s);raise SystemExit(0 if r['accepted'] else 1)
