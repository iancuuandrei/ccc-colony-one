"""Prepare a quarantined three-state training packet; never release ext_* data."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(RESEARCH))
from loss_10910_review import parse,typed,sparse,encoding,check_forced_loss_by_next_turn

SCOPE="TERMINAL_LOSS_THROUGH_SECOND_FOLLOWING_OPPONENT_TURN"
SOURCE=ROOT/'codex/matches/site-all-v1-derived/match-10910.txt'
EXPECTED_SOURCE="98695de1b45eb2bee22b64d6ba004fb97864684a3eeb8e6219d48aa8dd4bc6bc"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state_hash(env):return hashlib.sha256(env.observation().astype(np.float32).tobytes()+f"|{env.action}|{env.round_number}|{env.actions_left}|{env.current_team}".encode()).hexdigest()


def assess(child,own,budget,memo,endpoints):
    """Exact composition of enemy first turn and supplied second-turn oracle."""
    checked=[]
    if child.terminated:return dict(status='PROVED_LOSS' if child.winner!=own else 'PROVED_HORIZON_ESCAPE',branches=[])
    enemy=child.current_team
    if enemy==own:raise ValueError('Expected completed own turn')
    firsts=sorted(child.legal_actions(),key=lambda a:not(a.from_pos==(3,2) and a.to_pos==(3,5)))
    unknown=False
    for a in firsts:
        first=child.clone();line=[typed(first,a)];first.step(a)
        seconds=[None] if first.terminated else sorted(first.legal_actions(),key=lambda b:not(b.from_pos==(3,5) and b.to_pos==(2,6)))
        for b in seconds:
            final=first.clone();moves=list(line)
            if b is not None:moves.append(typed(final,b));final.step(b)
            if final.terminated:
                record=dict(line=moves,terminal_winner=final.winner)
                checked.append(record)
                if final.winner==enemy:return dict(status='PROVED_LOSS',branches=checked,witness=record)
                continue
            if final.current_team!=own:raise AssertionError('Opponent turn composition clock mismatch')
            digest=state_hash(final)
            if digest not in memo:
                if budget['nodes']>=budget['max']:return dict(status='UNKNOWN',branches=checked,reason='bounded exact-node limit')
                proof=check_forced_loss_by_next_turn(final,max_nodes=min(100000,budget['max']-budget['nodes']))
                budget['nodes']+=proof['nodes'];memo[digest]=proof
                endpoints[digest]=dict(state=sparse(final),obs_sha256=hashlib.sha256(final.observation().astype(np.float32).tobytes()).hexdigest(),oracle=proof)
            proof=memo[digest];record=dict(line=moves,endpoint_state_sha256=digest,oracle_status=proof['status'])
            checked.append(record)
            if proof['status']=='VERIFIED':return dict(status='PROVED_LOSS',branches=checked,witness=record)
            unknown |= proof['status']=='UNKNOWN'
    return dict(status='UNKNOWN' if unknown else 'PROVED_HORIZON_ESCAPE',branches=checked,
                checked_opponent_completions=len(checked))


def classify_last_action(env,budget,memo,endpoints):
    if env.actions_left!=1:raise AssertionError('Requires exactly one current action remaining')
    rows=[]
    for a in env.legal_actions():
        child=env.clone();child.step(a)
        result=assess(child,env.current_team,budget,memo,endpoints)
        rows.append(dict(action=typed(env,a),scope=SCOPE,**result))
        print(json.dumps(dict(state_action=env.action,candidate=typed(env,a),status=result['status'],nodes=budget['nodes'])),flush=True)
    return rows


def main():
    if sha(SOURCE)!=EXPECTED_SOURCE:raise AssertionError('Raw source artifact drift')
    game,states=parse(SOURCE,None);lookup={s.action:s for s in states}
    a=lookup[39].clone();b=lookup[40].clone();c=a.clone()
    alt=next(x for x in c.legal_actions() if encoding.action_index(x,c.current_team)==2738)
    c.step(alt)
    actual=a.clone();actual.step(next(x for x in actual.legal_actions() if encoding.action_index(x,actual.current_team)==2729))
    assert np.array_equal(actual.observation(),b.observation())
    budget={'nodes':0,'max':2000000};memo={};endpoints={}
    bproof=classify_last_action(b,budget,memo,endpoints)
    cproof=classify_last_action(c,budget,memo,endpoints)
    if any(r['status']!='PROVED_LOSS' for r in bproof):raise AssertionError('B all-loss proof failed')
    csafe=[r['action']['index'] for r in cproof if r['status']=='PROVED_HORIZON_ESCAPE']
    if csafe!=[3164] or any(r['status']=='UNKNOWN' for r in cproof):raise AssertionError('C complete classification failed')
    envs=[a,b,c];legal=[s.legal_actions() for s in envs];legal_idx=[[int(encoding.action_index(x,s.current_team)) for x in acts] for s,acts in zip(envs,legal)]
    legal_mask=np.zeros((3,4096),bool);demonstrated=np.zeros_like(legal_mask);unsafe=np.zeros_like(legal_mask)
    for i,idx in enumerate(legal_idx):legal_mask[i,idx]=True
    demonstrated[0,2738]=True;unsafe[0,2729]=True
    unsafe[1,legal_idx[1]]=True
    demonstrated[2,3164]=True;unsafe[2,[x for x in legal_idx[2] if x!=3164]]=True
    unknown=legal_mask & ~(demonstrated|unsafe)
    pidx=np.full((3,128),-1,np.int16);pval=np.zeros((3,128),np.float32)
    for i,indices in enumerate(legal_idx):pidx[i,:len(indices)]=indices
    pval[2,legal_idx[2].index(3164)]=1
    ids=np.array(['10910-A-pre39','10910-B-actual-pre40','10910-C-counterfactual-pre40'])
    npz=HERE/'training_packet.npz'
    np.savez_compressed(npz,obs=np.stack([s.observation() for s in envs]).astype(np.float32),pidx=pidx,pval=pval,
                        z=np.array([0,-1,0],np.float32),q=np.zeros(3,np.float32),
                        has_pol=np.array([False,False,True]),has_value=np.array([False,True,False]),q_valid=np.zeros(3,bool),
                        left=np.array([6,5,1],np.float32),legal_mask=legal_mask,
                        demonstrated_safe_mask=demonstrated,proved_unsafe_mask=unsafe,unknown_action_mask=unknown,
                        classification_complete=np.array([False,True,True]),
                        pairwise_valid=np.array([True,False,False]),pairwise_better_index=np.array([2738,-1,-1],np.int16),
                        pairwise_worse_index=np.array([2729,-1,-1],np.int16),
                        position_id=ids,source_group=np.array(['site:10910']*3),
                        scope=np.array([SCOPE]*3),policy_kind=np.array(['PROVED_SAFE_SUBSET','NO_ADMISSIBLE_ACTION','COMPLETE_BOUNDED_CLASSIFICATION']),
                        observed_outcome=np.array([-1,-1,0],np.float32),has_observed_outcome=np.array([True,True,False]))
    rows=[]
    for i,s in enumerate(envs):
        rows.append(dict(position_id=str(ids[i]),source_group='site:10910',state=sparse(s),state_sha256=state_hash(s),
                         obs_sha256=hashlib.sha256(s.observation().astype(np.float32).tobytes()).hexdigest(),
                         legal_actions=[typed(s,x) for x in legal[i]],scope=SCOPE,
                         policy_kind=('PROVED_SAFE_SUBSET','NO_ADMISSIBLE_ACTION','COMPLETE_BOUNDED_CLASSIFICATION')[i],
                         demonstrated_safe_actions=np.flatnonzero(demonstrated[i]).tolist(),
                         proved_unsafe_actions=np.flatnonzero(unsafe[i]).tolist(),unknown_actions=np.flatnonzero(unknown[i]).tolist(),
                         classification_complete=bool(i!=0),value=(-1 if i==1 else 'UNKNOWN'),
                         has_pol=bool(i==2),has_value=bool(i==1),q_valid=False,
                         length_target_kind=('MASKED_RECORDED_LENGTH','OBSERVED_RECORDED_CONTINUATION','MASKED_FINITE_PLACEHOLDER')[i],
                         provenance='actual_replay' if i<2 else 'one_step_counterfactual_from_actual_pre39'))
    proof=dict(scope=SCOPE,meaning='terminal defeat no later than the end of the second following CYAN turn; horizon escape is not eventual victory',
               A=dict(kind='PROVED_SAFE_SUBSET',safe_first=2738,required_second=3164,safe_certificate_ref='C/action3164',
                      unsafe_first=2729,unsafe_certificate_ref='B/all-actions',other_first_actions='UNKNOWN'),
               B=bproof,C=cproof,endpoints=endpoints,resource_budget=budget)
    proofpath=HERE/'certificates.json';proofpath.write_text(json.dumps(proof,indent=2),encoding='utf-8')
    packet=dict(schema_version=1,status='QUARANTINED_PENDING_INDEPENDENT_VERIFICATION',ext_release=False,
                source=dict(path=str(SOURCE),sha256=EXPECTED_SOURCE,source_group='site:10910',winner=game['winner'],actions=game['actions']),
                authoritative_model_context='GEN02 only; no model inference or identity assertion performed',
                rows=rows,files=dict(training_npz_sha256=sha(npz),certificates_sha256=sha(proofpath)),
                script_sha256=sha(__file__),engine_sha256={f:sha(ROOT/'env/colony_one'/f) for f in ['env.py','board.py','rules.py','encoding.py']},
                oracle_sha256={f:sha(ROOT/'codex/engine_audit'/f) for f in ['oracle.py','own_turn_oracle.py']},
                warnings=['A is a demonstrated safe subset, not a complete safe set. Generic full-legal KL is deliberately disabled for A.',
                          'A supports only pairwise preference2738 over proved losing2729; seven other legal first actions stay UNKNOWN.',
                          'Unknown value placeholders are finite zero and masked. Counterfactual C has no observed episode outcome.',
                          'B has no arbitrary policy target: all six actions are proved losing. B exact value is -1.',
                          'All three rows and their relatives must remain in source group site:10910 for train/holdout decisions.'])
    (HERE/'packet.json').write_text(json.dumps(packet,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=packet['status'],rows=3,files=packet['files'],nodes=budget['nodes'],endpoints=len(endpoints),
                         masks=[dict(id=str(ids[i]),legal=int(legal_mask[i].sum()),demonstrated=int(demonstrated[i].sum()),unsafe=int(unsafe[i].sum()),unknown=int(unknown[i].sum())) for i in range(3)])),flush=True)


if __name__=='__main__':main()
