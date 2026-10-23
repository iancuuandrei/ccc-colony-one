"""Execute current loader/losses and actor control flow; no models or live jobs."""
from pathlib import Path
import argparse, ast, hashlib, json, sys, types
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
import train
torch.set_num_threads(1)
HERE = Path(__file__).with_suffix('')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
obskey = lambda o: hashlib.sha256(np.ascontiguousarray(o, dtype=np.float16).tobytes()).hexdigest()
jsonbytes = lambda x: (json.dumps(x, indent=2, sort_keys=True) + '\n').encode()


def main():
    hashes = {p: sha(ROOT/p) for p in ('train.py', 'actor.py', 'learner.py')}
    ovdir = ROOT/'data/ext_site_tactics_v1'
    spec = json.loads((ovdir/'policy-overrides.json').read_text())
    assert len(spec['entries']) == 19
    assert len({e['observation_float16_sha256'] for e in spec['entries']}) == 19
    entries = {e['observation_float16_sha256']: e for e in spec['entries']}
    proofs = {}; bound_sources = []
    for e in entries.values():
        rep = e['replacement']; chunk = ovdir/rep['chunk']; proofpath = ovdir/e['proof_file']
        assert sha(chunk) == rep['chunk_sha256'] and sha(proofpath) == e['proof_file_sha256']
        proofdoc = proofs.setdefault(str(proofpath), json.loads(proofpath.read_text()))
        pr = next(s for s in proofdoc['states'] if s['observation_float16_sha256'] == e['observation_float16_sha256'])
        assert pr['proof_sha256'] == e['proof_record_sha256']
        assert hashlib.sha256(jsonbytes({k:v for k,v in pr.items() if k!='proof_sha256'})).hexdigest() == e['proof_record_sha256']
        with np.load(chunk) as z:
            row = rep['row']; assert obskey(z['obs'][row]) == e['observation_float16_sha256']
            assert np.array_equal(z['pidx'][row], rep['pidx']) and np.array_equal(z['pval'][row], rep['pval'])
            assert set(z['pidx'][row][z['pidx'][row]>=0]) == set(pr['legal_indices'])
            support = set(z['pidx'][row][z['pval'][row]>0]); allowed = set(pr['own_turn_winning_indices'] or pr['bounded_safe_indices'])
            assert support and support <= allowed and abs(float(z['pval'][row].sum())-1) < .001
        for s in e['teacher_source_rows']:
            source = ROOT/s['file']; assert sha(source) == s['file_sha256']
            with np.load(source) as z: assert obskey(z['obs'][s['row']]) == e['observation_float16_sha256']
            bound_sources.append((s['file'], s['row']))
    base = ROOT/'data/ext_site_v1_rs/chunk0000.npz'
    files = sorted(ovdir.glob('chunk*.npz')) + [base]
    D = train.load(files)
    assert len(D['obs']) == 1984 and int(D['_overrides_applied'][0]) == 6
    assert D['is_ext'].all() and not D['has_npol'].any()
    raw = dict(np.load(base)); changed = []
    for i,o in enumerate(raw['obs']):
        e = entries.get(obskey(o)); r = i+19
        if e:
            changed.append(i); rep=e['replacement']
            assert np.array_equal(D['pidx'][r], np.asarray(rep['pidx'],D['pidx'].dtype))
            assert np.array_equal(D['pval'][r], np.asarray(rep['pval'],D['pval'].dtype))
            assert D['has_pol'][r] and not D['has_value'][r] and not D['q_valid'][r]
        else:
            for k in train.KEYS: assert np.array_equal(D[k][r],raw[k][i]), (k,i)
    assert changed == [1680,1737,1808,1872,1909,1962]
    # Every 19 published observations must route under its designated source folder.
    source_obs = np.concatenate([np.load(f)['obs'] for f in files[:-1]])
    def fixture():
        n=len(source_obs)
        return dict(obs=source_obs.copy(),pidx=np.full((n,128),-1,np.int16),pval=np.zeros((n,128),np.float16),has_pol=np.zeros(n,bool),has_value=np.ones(n,bool),q_valid=np.ones(n,bool))
    X=fixture(); hits=train.apply_overrides(X,[(base,0,19),(ovdir/'x.npz',0,0)])
    assert hits==19
    for r,o in enumerate(source_obs):
        rep=entries[obskey(o)]['replacement'];assert np.array_equal(X['pidx'][r],rep['pidx'])
        assert np.array_equal(X['pval'][r],np.asarray(rep['pval'],np.float16))
    untouched=fixture(); assert train.apply_overrides(untouched,[(ROOT/'data/ext_site_v1/x.npz',0,19),(ovdir/'x.npz',0,0)])==0
    assert untouched['has_value'].all() and untouched['q_valid'].all()
    # Exact float16-key rule: float32 perturbations below rounding match; changed bytes do not.
    same=fixture();same['obs']=same['obs'].astype(np.float32);same['obs'][same['obs']==1]+=1e-5
    assert train.apply_overrides(same,[(base,0,19),(ovdir/'x.npz',0,0)])==19
    different=fixture();different['obs'][:,0,0,0]+=np.float16(.125)
    assert train.apply_overrides(different,[(base,0,19),(ovdir/'x.npz',0,0)])==0
    names=['ext_astra_site_gen02_v1','ext_astra_policy_v1','ext_site_deep10910_v1']
    newfiles=[ROOT/'data'/n/'chunk0000.npz' for n in names]
    N=train.load(newfiles); assert len(N['obs'])==201 and N['is_ext'].all() and not N['has_npol'].any()
    assert not N['q_valid'].any() and int(N['has_value'].sum())==13 and int(N['has_pol'].sum())==197
    assert np.all(N['left']>0) and all(np.isfinite(N[k]).all() for k in ('obs','pval','z','q','left'))
    input_counts=[]
    for name,f in zip(names,newfiles):
        z=dict(np.load(f));input_counts.append(dict(folder=name,sha256=sha(f),rows=len(z['obs']),value_rows=int(z['has_value'].sum()),policy_rows=int(z['has_pol'].sum())))
    rawfiles=sorted((ROOT/'data/ext_site_v1').glob('chunk*.npz'))
    mixed=train.load(files+newfiles+rawfiles)
    assert int(mixed['_overrides_applied'][0])==6 and not mixed['has_npol'].any()
    offset=len(D['obs'])+len(N['obs'])
    for f in rawfiles:
        z=dict(np.load(f)); n=len(z['z'])
        for k in train.KEYS:
            expected=z.get(k,np.zeros(n,bool) if k=='q_valid' else np.ones(n,bool))
            assert np.array_equal(mixed[k][offset:offset+n],expected),(str(f),k)
        offset+=n
    # Execute unmodified nested loss code with output tensors as leaves; no Net instantiated.
    tree=ast.parse((ROOT/'train.py').read_text()); mainfn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    funcs=[n for n in mainfn.body if isinstance(n,ast.FunctionDef) and n.name in ('legal_kl','losses')]
    n=len(N['obs']); logits=torch.zeros((n,4096),requires_grad=True); value=torch.full((n,),.37,requires_grad=True)
    aux=torch.full((n,2),.7,requires_grad=True); next_logits=torch.zeros_like(logits,requires_grad=True)
    fake=types.SimpleNamespace(forward_train=lambda o,with_next=False:(logits,value,aux,next_logits))
    cfg=types.SimpleNamespace(qmix=.2,opp_aux=.15,surprise=.5)
    ns=dict(torch=torch,F=F,net=fake,a=cfg);exec(compile(ast.Module(body=funcs,type_ignores=[]),'current_losses','exec'),ns)
    t=lambda k:torch.tensor(N[k]); hv=t('has_value');hp=t('has_pol')
    pl,vl,al=ns['losses'](None,t('pidx'),t('pval'),t('z'),t('q'),hp.float(),t('left'),t('q_valid').float(),hv.float(),npi=t('npidx'),npv=t('npval'),hn=t('has_npol').float())
    (pl+1.5*vl+.25*al).backward()
    assert value.grad[~hv].abs().sum()==0 and value.grad[hv].abs().sum()>0
    assert aux.grad[~hv,1].abs().sum()==0 and aux.grad[hv,1].abs().sum()>0 and aux.grad[:,0].abs().sum()==0
    assert logits.grad[~hp].abs().sum()==0 and logits.grad[hp & ~hv].abs().sum()>0
    assert next_logits.grad.abs().sum()==0
    grad=dict(status='PASS_EXECUTED_CPU_LEAF_GRADIENTS',value_unknown_zero=True,length_unknown_zero=True,q_invalid_zero=True,policy_only_active=True,no_policy_zero=True,external_next_policy_zero=True)
    # Current actor.main control flow, with sleep/subprocess replaced by terminating stubs.
    atree=ast.parse((ROOT/'actor.py').read_text()); actorfn=next(n for n in atree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    class EndProbe(Exception):pass
    class Parser:
        def add_argument(self,*a,**k):pass
        def parse_args(self):return types.SimpleNamespace(best_file='unused',prefix='audit',workers=8,batch=2,pool=[],pool_prob=.2)
    class FakeRoot:
        def __truediv__(self,x):return self
        def exists(self):return False
    actorcases=[]
    for workers in (0,-1,8):
        events=[]
        def sleep(seconds):events.append(('sleep',seconds));raise EndProbe
        def run(cmd,**kw):events.append(('subprocess_stub',cmd));raise EndProbe
        def best(path):events.append(('best_read',None));return 'ckpt/unused.onnx'
        an=dict(argparse=types.SimpleNamespace(ArgumentParser=Parser),ROOT=FakeRoot(),workers=lambda default:workers,time=types.SimpleNamespace(sleep=sleep,time=lambda:123),current_best=best,Path=Path,sys=sys,subprocess=types.SimpleNamespace(run=run),extra_args=lambda:[])
        exec(compile(ast.Module(body=[actorfn],type_ignores=[]),'current_actor_control','exec'),an)
        try:an['main']()
        except EndProbe:pass
        if workers<=0:assert events==[('sleep',20)]
        else:assert events[0][0]=='best_read' and events[1][0]=='subprocess_stub' and str(workers) in events[1][1]
        actorcases.append(dict(workers=workers,events=[x[0] for x in events]))
    log=(ROOT/'logs/gen07.txt').read_text(errors='replace')
    assert all('data/'+n in log for n in names)
    unchanged={p:sha(ROOT/p)==h for p,h in hashes.items()};assert all(unchanged.values())
    report=dict(status='PASS_CURRENT_SINGLE_MANIFEST_INTEGRATION',source_hashes=hashes,source_stable=unchanged,policy_overrides_sha256=sha(ovdir/'policy-overrides.json'),provenance_bindings='PASS_19_CHUNKS_PROOF_RECORDS_AND_6_TEACHER_ROWS',real_loader_rows=1984,actual_source_replacements=changed,future_hash_routing_fixture_hits=hits,float16_key_and_folder_scope='PASS',new_inputs=input_counts,mixed_rows=len(mixed['obs']),raw_outcomes_unchanged=True,external_next_policy_aux='PASS_NONE',gradient_masks=grad,actor_control_cases=actorcases,actor_processes_started=0,gen07_log_includes_all_new_inputs=True,models_instantiated=0,training_started=False,limits=['No replay of oracle trees in this integration audit; prior independent certificates are hash-bound.','No proof that an already-running gen07 process imported this exact train.py hash: its log does not record code/override hashes or replacement count.','Current production load routes one arbitrary override manifest if several are present; currently exactly one manifest is used.','Runtime apply_overrides trusts JSON and does not itself verify chunk/proof/source hashes; this audit verifies the current bindings.','Actor checks workers twice; a budget change between check and command can still pass 0 on a rare race.'])
    HERE.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
