"""Independent constructed-state validation via authoritative env.clone/step.

This does not import the producer's tactical oracle: all asserted terminal
wins and bounded losses are recomputed using the public full environment.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
qs=importlib.util.spec_from_file_location('synthetic_validation_queue',ROOT/'codex'/'distillation'/'queue.py');q=importlib.util.module_from_spec(qs);qs.loader.exec_module(q)
from colony_one import constants as C,encoding as E

class BudgetExceeded(Exception):pass
def own_turn_win_mask(env):
    team=env.current_team;winning={};nodes=0
    for first in env.legal_actions():
        a=E.action_index(first,team);child=env.clone();child.step(first);nodes+=1
        if child.terminated:
            if child.winner==team:winning[a]=[a]
            continue
        if child.current_team!=team:continue
        for second in child.legal_actions():
            after=child.clone();after.step(second);nodes+=1
            if after.terminated and after.winner==team:
                winning.setdefault(a,[a,E.action_index(second,team)])
    return winning,nodes
def next_turn_loss(env,max_nodes=60000,only_first=None):
    team=env.current_team;nodes=0;memo={}
    def key(e):return e.observation().tobytes(),e.current_team,e.actions_left,e.round_number
    def lose(e):
        nonlocal nodes
        if e.terminated:return e.winner!=team
        if e.current_team==team and e.action>env.action and e.round_number>env.round_number+(0 if team==C.CYAN else 1):return False
        kk=key(e)
        if kk in memo:return memo[kk]
        actions=e.legal_actions()
        if not actions:raise BudgetExceeded('no legal action is unsupported')
        mychoice=e.current_team==team;outcomes=[]
        for a in actions:
            nodes+=1
            if nodes>max_nodes:raise BudgetExceeded('node cap')
            ch=e.clone();acting=e.current_team;ch.step(a)
            if ch.terminated:bad=ch.winner!=team
            elif acting!=team and ch.current_team==team:bad=False
            else:bad=lose(ch)
            outcomes.append(bad)
            if mychoice and not bad:memo[kk]=False;return False
            if not mychoice and bad:memo[kk]=True;return True
        result=all(outcomes) if mychoice else any(outcomes);memo[kk]=result;return result
    per={}
    try:
        for a in env.legal_actions():
            if only_first is not None and E.action_index(a,team) not in only_first:continue
            nodes+=1
            if nodes>max_nodes:raise BudgetExceeded('node cap')
            ch=env.clone();ch.step(a)
            bad=ch.winner!=team if ch.terminated else lose(ch)
            per[E.action_index(a,team)]='VERIFIED' if bad else 'DISPROVED'
        return {'status':'VERIFIED' if all(x=='VERIFIED' for x in per.values()) else 'DISPROVED','per_first_action':per,'nodes':nodes}
    except BudgetExceeded as exc:return {'status':'UNKNOWN','per_first_action':per,'nodes':nodes,'reason':str(exc)}

def transform(r,c,rotation,reflection):
    if reflection:c=7-c
    for _ in range(rotation):r,c=c,7-r
    return r,c
def expected_observation(seed,variant):
    o=np.zeros((16,8,8),np.float32);team=C.CYAN if seed['side']=='C' else C.MAGENTA
    types=['PING','DAEMON','EXPLOIT','TRACE','ICE','KERNEL']
    for side,kind,r,c in seed['pieces']:
        r,c=transform(r,c,variant['rotation'],variant['reflection'])
        if team==C.MAGENTA:r,c=7-r,7-c
        own=(side=='C')==(team==C.CYAN);o[types.index(kind)+(0 if own else 6),r,c]=1
    shrinks=sum(s<seed['round'] for s in (8,16,24,26))
    for r in range(8):
        for c in range(8):o[12,r,c]=min(r,c,7-r,7-c)<shrinks
    next_burn=next(s for s in (8,16,24,26) if s>=seed['round'])
    o[13]=seed['left']/2;o[14]=1-(next_burn-seed['round'])/8;o[15]=team==C.MAGENTA
    return o

def variant_mapping(basev,targetv,team):
    mapping={}
    for s in range(64):
        r,c=divmod(s,8);br,bc=transform(r,c,basev['rotation'],basev['reflection']);tr,tc=transform(r,c,targetv['rotation'],targetv['reflection'])
        if team==C.MAGENTA:br,bc,tr,tc=7-br,7-bc,7-tr,7-tc
        mapping[br*8+bc]=tr*8+tc
    return mapping

def validate(source_dir,out_dir,exact_only=False):
    source_dir=Path(source_dir).resolve();out=Path(out_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    live_manifest=json.loads((source_dir/'manifest.json').read_text())
    proposal_source=Path(live_manifest['teacher_seed_proposals']).resolve()
    if not proposal_source.is_relative_to(ROOT.resolve()):raise ValueError('Proposal source outside task root')
    sources={n:source_dir/n for n in ['packet.json','manifest.json','certified_states.jsonl','certified_states.npz']}
    sources.update({'seed_proposals.json':proposal_source,'generate.py':source_dir/'generate.py' if (source_dir/'generate.py').exists() else source_dir.parent/'generate.py'})
    source_hashes={n:q.sha(p) for n,p in sources.items()};snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for n,path in sources.items():
        dst=snapshot/n
        if dst.exists() and q.sha(dst)!=source_hashes[n]:raise ValueError('Snapshot already binds different source; choose new out dir')
        if not dst.exists():dst.write_bytes(path.read_bytes())
    manifest=json.loads((snapshot/'manifest.json').read_text());packet=json.loads((snapshot/'packet.json').read_text());proposal=json.loads((snapshot/'seed_proposals.json').read_text());seeds={s['id']:s for s in proposal['seeds']}
    rows=[json.loads(x) for x in (snapshot/'certified_states.jsonl').read_text().splitlines()];npz=np.load(snapshot/'certified_states.npz');errors=[];checked=[];ids=set();seed_groups={};start=time.monotonic();own_cache={}
    if manifest['proposal_sha256']!=source_hashes['seed_proposals.json']:errors.append('proposal hash mismatch')
    for name,digest in manifest['engine_sha256'].items():
        if q.sha(ROOT/'env'/'colony_one'/name)!=digest:errors.append('engine source drift:'+name)
    if packet['engine_sha256']!=manifest['engine_sha256']:errors.append('packet/manifest engine hashes differ')
    if len(rows)!=len(packet['positions']) or len(rows)!=len(npz['obs']) or len(rows)!=manifest['states']:errors.append('row count mismatch')
    for i,row in enumerate(rows):
        try:
            if row!=packet['positions'][i]:raise ValueError('JSONL/packet row mismatch')
            pid=row['position_id'];state=row['state'];seed=seeds[row['seed_id']];source=row['source'];o=q.observation(state)
            if pid in ids:raise ValueError('duplicate state ID')
            ids.add(pid)
            if q.state_hash(state)!=row['state_sha256']:raise ValueError('state hash mismatch')
            if source.get('reachability')!='NOT_PROVEN_FROM_STANDARD_OPENING' or source.get('kind')!='LLM_AUTHORED_CONSTRUCTED_SCENARIO':raise ValueError('constructed provenance missing')
            if source['sha256']!=source_hashes['seed_proposals.json']:raise ValueError('seed proposal source drift')
            if not np.array_equal(expected_observation(seed,source['variant']),o):raise ValueError('state does not match authored seed/D4 transform')
            if not np.array_equal(o,npz['obs'][i]):raise ValueError('NPZ/packet observation mismatch')
            if npz['position_id'][i]!=pid:raise ValueError('NPZ ID/order mismatch')
            if not np.isfinite(o).all() or not np.isin(o[:13],[0,1]).all():raise ValueError('invalid observation values')
            if any(len(s)>1 for s in state['piece_squares']):raise ValueError('more than one piece of same team/type')
            if len(state['piece_squares'][5])!=1 or len(state['piece_squares'][11])!=1:raise ValueError('exactly one kernel/team required')
            if np.any(o[:12].sum(0)>1) or np.any(o[:12].sum(0)*o[12]):raise ValueError('overlap or locked occupancy')
            if not 1<=state['round_number']<=26 or state['actions_left'] not in (1,2):raise ValueError('invalid live clock')
            env=q.restore(state)
            if q.encode_state(env)!=state:raise ValueError('clock/observation roundtrip')
            legal=set(env.legal_action_indices().tolist())
            if not legal or legal!={a['action_index'] for a in row['legal_actions']} or legal!=set(np.flatnonzero(npz['legal_mask'][i])):raise ValueError('legal action mismatch')
            group=row['group_id'];seed_groups.setdefault(row['seed_id'],set()).add(group)
            if group!='astra-family-'+seed['family'] or group!=npz['group_id'][i]:raise ValueError('motif group mismatch')
            cert=row['certificate']
            if exact_only and cert['value'] is None:
                if npz['value_valid'][i] or not np.isnan(npz['value'][i]):raise ValueError('unknown value incorrectly materialized')
                checked.append(dict(position_id=pid,seed_id=row['seed_id'],group_id=group,claim='unknown_full_game_value',status='STAGED_UNQUALIFIED_VALUE'))
                continue
            if row['seed_id'] not in own_cache:
                basewins,nodes=own_turn_win_mask(env);own_cache[row['seed_id']]=(row,basewins,nodes)
            base,basewins,nodes=own_cache[row['seed_id']];mp=variant_mapping(base['source']['variant'],row['source']['variant'],env.current_team)
            wins={mp[a//64]*64+mp[a%64]:w for a,w in basewins.items()};asserted=set(cert['winning_first_actions'])
            if asserted!=set(wins):raise ValueError('independent env.step winning mask mismatch')
            for first,wit in cert['witnesses'].items():
                if int(first) not in wins:raise ValueError('witness first action not certified')
                ch=env.clone();team=ch.current_team
                for act in wit['actions']:
                    if ch.terminated or ch.current_team!=team:raise ValueError('winning witness exceeds own turn')
                    ch.step(int(act))
                if not ch.terminated or ch.winner!=team:raise ValueError('winning witness does not terminate in win')
            if cert['claim']=='forced_win_this_turn':
                if not wins or cert['value']!=1:raise ValueError('own-turn win label mismatch')
            elif cert['claim']=='forced_loss_by_next_turn':
                independent=next_turn_loss(env)
                if independent['status']!='VERIFIED' or cert['value']!=-1:raise ValueError('independent loss verification failed:'+str(independent['status']))
            elif cert['claim']=='unknown_full_game_value':
                if cert['value'] is not None or npz['value_valid'][i] or not np.isnan(npz['value'][i]):raise ValueError('unknown value incorrectly materialized')
            else:raise ValueError('unsupported value horizon claim')
            # Independently check every safe/unsafe classification for one representative per seed;
            # all other D4 variants are checked for transformed mask covariance below.
            checked.append(dict(position_id=pid,seed_id=row['seed_id'],group_id=group,claim=cert['claim'],own_turn_nodes=nodes,independent_win_actions=sorted(wins),status='PASS'))
        except Exception as exc:errors.append({'row':i,'position_id':row.get('position_id'),'error':str(exc)})
    if any(len(g)!=1 for g in seed_groups.values()):errors.append('a seed spans multiple train/holdout groups')
    per_seed={};safety_checks=[]
    for row in rows:per_seed.setdefault(row['seed_id'],[]).append(row)
    for seed_id,variants in per_seed.items():
        if exact_only or all(r['certificate']['winning_first_actions'] for r in variants):continue
        base=variants[0];proof={'status':'PER_ACTION_VERIFICATION','per_first_action':{},'nodes':0}
        basev=base['source']['variant'];team=base['state']['side']
        for row in variants:
            targetv=row['source']['variant'];mapping={}
            for s in range(64):
                r,c=divmod(s,8);br,bc=transform(r,c,basev['rotation'],basev['reflection']);tr,tc=transform(r,c,targetv['rotation'],targetv['reflection'])
                if team==C.MAGENTA:br,bc,tr,tc=7-br,7-bc,7-tr,7-tc
                mapping[(br*8+bc)]=tr*8+tc
            inv={v:k for k,v in mapping.items()}
            for field,status in [('safe_first_actions','DISPROVED'),('unsafe_first_actions','VERIFIED')]:
                for a in row['certificate'][field]:
                    original=inv[a//64]*64+inv[a%64]
                    if original not in proof['per_first_action']:
                        action_proof=next_turn_loss(q.restore(base['state']),max_nodes=60000,only_first={original})
                        proof['nodes']+=action_proof['nodes'];proof['per_first_action'][original]=action_proof['per_first_action'].get(original,'UNKNOWN')
                    independent=proof['per_first_action'].get(original,'UNKNOWN')
                    if independent!=status:errors.append({'position_id':row['position_id'],'error':f'{field} action {a} independently {independent}, expected {status}'})
            # Complete own-turn masks must transform exactly, regardless of bounded safety caps.
            expected={mapping[a//64]*64+mapping[a%64] for a in base['certificate']['winning_first_actions']}
            if expected!=set(row['certificate']['winning_first_actions']):errors.append({'position_id':row['position_id'],'error':'D4 winning-mask covariance'})
        safety_checks.append({'seed_id':seed_id,'representative':base['position_id'],'independent_status':proof['status'],'nodes':proof['nodes'],'per_first_action':proof['per_first_action']})
        print(json.dumps({'verified_seed':seed_id,'nodes':proof['nodes'],'errors':len(errors),'seconds':round(time.monotonic()-start,1)}),flush=True)
    # Stable split assigns entire families; never split rotations or paired clocks.
    groups=sorted({r['group_id'] for r in rows});holdout=set(groups[::5]);split=np.array(['holdout' if r['group_id'] in holdout else 'train' for r in rows])
    if not errors:
        np.savez_compressed(out/'independently_validated.npz',**{k:npz[k] for k in npz.files},split=split,reachability=np.full(len(rows),'NOT_PROVEN_FROM_STANDARD_OPENING'),label_source=np.full(len(rows),'ENGINE_CERTIFICATE_FROM_LLM_PROPOSED_STATE'))
    result=dict(accepted=not errors,source_hashes=source_hashes,validator_sha256=q.sha(__file__),rows=len(rows),exact_only=exact_only,qualified_rows=sum(r['status']=='PASS' for r in checked),groups=len(groups),train_groups=sorted(set(groups)-holdout),holdout_groups=sorted(holdout),errors=errors,checked=checked,safety_checks=safety_checks,seconds=time.monotonic()-start,constructed=True,reachability='NOT_PROVEN_FROM_STANDARD_OPENING',label_authority='Engine-certified bounded outcomes; LLM authored proposals only; strategic LLM predictions absent',hard_states_original_manifest=manifest['hard_states'],proof_mode='Full env.clone/step winning mask per authored seed, exact D4 covariance plus actual witness replay/legal masks on every variant; unused next-turn safety diagnostics on already-winning states not qualified',qualification='Local invariants and independently recomputed env.step tactical certificates; no opening reachability, playing strength or server acceptance claim. STAGED_UNQUALIFIED_VALUE rows excluded from exact training export.')
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('checked','safety_checks')},indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-dir',default=str(HERE.parent/'astra'));p.add_argument('--out',default=str(HERE/'controls0001'));p.add_argument('--exact-only',action='store_true');a=p.parse_args();r=validate(a.source_dir,a.out,a.exact_only);raise SystemExit(0 if r['accepted'] else 1)
