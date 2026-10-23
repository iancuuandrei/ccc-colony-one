"""Qualify current length masks and atomically publish previously proved GEN05 rows."""
import ast,hashlib,json,shutil,sys,types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import train
torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=Path(__file__).resolve().parent/'site-gen05-v1';stage=out/'export_stage';target=ROOT/'data/ext_astra_site_gen05_v1';source=stage/'chunk0000.npz';sm=json.loads((stage/'provenance.json').read_text());engine=json.loads((out/'report.json').read_text())
    expected='b2d4bdf6a0b17ff7c225c3c67ed2cfc562bf643224d557d770b40b68e39f6bb3';assert sha(ROOT/'train.py')==expected
    assert engine['accepted'] and engine['checked']==28 and sha(source)==sm['chunk_sha256']=='9e0c9989b16589411681cc8eb65e1e969a5fd9ff00a6122d7ffe9cbce764e6e1'
    assert sha(out/'report.json')==sm['independent_report_sha256']
    original_stage={p.name:sha(p) for p in stage.iterdir() if p.is_file()};frozen=ROOT/'codex/synthetic/holdout_v1';freeze=json.loads((frozen/'FROZEN.json').read_text())
    for n,h in freeze['files'].items():assert sha(frozen/n)==h
    for n,h in engine['source_hashes'].items():assert sha(ROOT/'codex/synthetic/site_gen05_curriculum'/n)==h
    # Execute actual loader with target folder identity before exposing any new data.
    class Proxy:
        def __getattr__(self,n):return getattr(np,n)
        def load(self,f):return np.load(source)
    oldnp=train.np;train.np=Proxy()
    try:D=train.load([target/'chunk0000.npz'])
    finally:train.np=oldnp
    assert len(D['obs'])==28 and D['is_ext'].all() and not D['has_npol'].any()
    assert D['has_length'].tolist()==[True]*4+[False]*24 and D['left'].tolist()==[4,3,4,3]+[1]*24
    assert D['has_value'].sum()==16 and D['has_pol'].sum()==14 and not D['q_valid'].any()
    # Defaults are exercised on existing real files, without fabricated training data.
    ext=ROOT/'data/ext_astra_synthetic_v1/chunk0000.npz';legacy=train.load([ext]);assert legacy['has_value'].all() and not legacy['has_length'].any()
    selfplay=next(f for d in sorted((ROOT/'data').glob('sp_*')) for f in d.glob('chunk*.npz'))
    own=train.load([selfplay]);assert own['has_length'].all() and not own['is_ext'].any()
    explicit=train.load([ROOT/'data/ext_site_deep14958_v1/chunk0000.npz']);assert explicit['has_length'].all()
    tree=ast.parse((ROOT/'train.py').read_text());mainfn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');funcs=[n for n in mainfn.body if isinstance(n,ast.FunctionDef) and n.name in ('legal_kl','losses')]
    n=28;logits=torch.zeros((n,4096),requires_grad=True);v=torch.full((n,),.37,requires_grad=True);aux=torch.full((n,2),.7,requires_grad=True);next_logits=torch.zeros_like(logits,requires_grad=True)
    net=types.SimpleNamespace(forward_train=lambda o,with_next=False:(logits,v,aux,next_logits),eval=lambda:None)
    cfg=types.SimpleNamespace(qmix=.2,opp_aux=.15,surprise=.5);ns=dict(torch=torch,F=F,net=net,a=cfg);exec(compile(ast.Module(body=funcs,type_ignores=[]),'exact_new_losses','exec'),ns)
    T=lambda k:torch.tensor(D[k]);hv=T('has_value');hl=T('has_length');hp=T('has_pol');loss=ns['losses'];gradient=[]
    for training in (True,False):
        for t in (logits,v,aux,next_logits):t.grad=None
        args=(None,T('pidx'),T('pval'),T('z'),T('q'),hp.float(),T('left'),T('q_valid').float(),hv.float())
        pl,vl,al=loss(*args,train=training,npi=T('npidx'),npv=T('npval'),hn=T('has_npol').float(),hlv=hl.float());(pl+1.5*vl+.25*al).backward()
        assert v.grad[~hv].abs().sum()==0 and v.grad[hv].abs().sum()>0
        assert aux.grad[~hl,1].abs().sum()==0 and aux.grad[hl,1].abs().sum()>0 and aux.grad[:,0].abs().sum()==0
        assert v.grad[hv & ~hl].abs().sum()>0 and aux.grad[hv & ~hl,1].abs().sum()==0
        assert aux.grad[hl & ~hv,1].abs().sum()>0 and v.grad[hl & ~hv].abs().sum()==0
        assert logits.grad[~hp].abs().sum()==0 and logits.grad[hp].abs().sum()>0
        if training:assert next_logits.grad.abs().sum()==0
        expected_length=.05*((aux.detach()[hl,1]-torch.log(T('left')[hl]))**2).mean();assert torch.allclose(al.detach(),expected_length)
        gradient.append(dict(mode='train' if training else 'holdout',exact_value_unknown_length_rows=int((hv & ~hl).sum()),unknown_length_gradient=0,exact_value_gradient_retained=True,known_length_unknown_value_rows=int((hl & ~hv).sum()),known_length_gradient_retained=True,length_normalization='PASS'))
    # Execute exact current holdout reporting branch; intercept only to inspect passed masks.
    H={k:torch.tensor(v) for k,v in D.items()};calls=[]
    def tracked(*a,**kw):
        assert kw['train'] is False and torch.equal(kw['hlv'],H['has_length'].float());calls.append(True);return loss(*a,**kw)
    ns.update(H=H,rec={},losses=tracked)
    hnode=next(n for n in ast.walk(mainfn) if isinstance(n,ast.If) and ast.unparse(n.test)=='H is not None')
    exec(compile(ast.Module(body=[hnode],type_ignores=[]),'actual_holdout_reporting_branch','exec'),ns);assert calls
    # The production train call must explicitly pass its length mask, not rely on the legacy fallback.
    traincalls=[n for n in ast.walk(mainfn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='losses']
    assert all(any(k.arg=='hlv' for k in call.keywords) for call in traincalls)
    excluded=ROOT/'data/_dup_excluded/ext_site_losses_b';assert sha(excluded/'chunk0000.npz')=='50c1ca9e10638c71577fd5968dfd47f1ac9b189ae56f93fc239d8487c75acc46'
    assert not (ROOT/'data/ext_site_losses_b').exists() and (ROOT/'data/ext_site_losses_b_rs/chunk0000.npz').exists()
    files=[f for d in sorted((ROOT/'data').glob('ext_*')) for f in sorted(d.glob('chunk*.npz'))];intake=train.load(files)
    assert not any('_dup_excluded' in str(f) for f in files) and not intake['has_npol'].any()
    current_rows=len(intake['z']);assert current_rows>=7935
    assert all(sha(stage/n)==h for n,h in original_stage.items()) and sha(ROOT/'train.py')==expected
    audit=dict(status='PASS_CURRENT_LENGTH_DEFAULTS_TRAIN_HOLDOUT_GRADIENTS_AND_DEDUP',train_sha256=expected,staged_chunk_sha256=sha(source),independent_engine_report_sha256=sha(out/'report.json'),gradient=gradient,actual_holdout_branch_executed=True,production_train_and_holdout_pass_explicit_length_mask=True,defaults=dict(legacy_selfplay=True,undeclared_external=False,explicit_external_true_preserved=True),dedup=dict(excluded_chunk_bytes_preserved=True,excluded_raw_folder_not_in_intake=True,rs_policy_folder_retained=True,external_rows_before_release=current_rows),holdout_frozen_sha256=sha(frozen/'FROZEN.json'),source_stage_unchanged=True,no_models_or_training=True,pval_g_audit_owner='onnx_validation',nonblocking_limit='Mixed holdout value metric aggregation still weights chunk-batch means by all rows; current normal selfplay holdout has all value rows, and this does not change per-row loss masks.')
    (out/'loader-length-qualified.json').write_text(json.dumps(audit,indent=2))
    # Preserve frozen staged bytes and proof report; publish a fresh directory atomically.
    publish=out/'publish_stage';publish.mkdir(exist_ok=False);shutil.copyfile(source,publish/'chunk0000.npz')
    provenance=dict(sm,status='READY_INDEPENDENTLY_VERIFIED_TRAINING',loader_length_audit_sha256=sha(out/'loader-length-qualified.json'),loader_length_audit_path=str(out/'loader-length-qualified.json'),train_sha256=expected,staged_source_preserved=True,grouping='Keep site:<game>, site-game-<game> aliases and all rotations/counterfactuals together across splits.',length_semantics='Four recorded actual lengths4/3/4/3; all24 constructed lengths masked False independently of16 exact value labels.')
    (publish/'provenance.json').write_text(json.dumps(provenance,indent=2));assert not target.exists();publish.replace(target)
    actual=train.load([target/'chunk0000.npz']);assert np.array_equal(actual['has_length'],D['has_length']) and np.array_equal(actual['has_value'],D['has_value']) and np.array_equal(actual['left'],D['left']) and not actual['has_npol'].any()
    assert sha(target/'chunk0000.npz')==sha(source) and all(sha(stage/n)==h for n,h in original_stage.items())
    audit.update(published_path=str(target),published_chunk_sha256=sha(target/'chunk0000.npz'),actual_published_path_loader='PASS',external_rows_after_release=current_rows+28)
    (out/'published-loader.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
