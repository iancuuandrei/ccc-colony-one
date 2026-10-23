"""Release only independently proved B/C rows; A remains metadata-only."""
from pathlib import Path
import json,hashlib,ast
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];SOURCE=ROOT/'codex'/'research'/'loss10910-training';OUT=ROOT/'data'/'ext_site_deep10910_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof=json.loads((HERE/'proof.json').read_text());packet=json.loads((SOURCE/'packet.json').read_text())
assert proof['status']=='PASS' and proof['B']['all_six_forced_loss'] and proof['C']['sole_bounded_safe']==3164
assert sha(SOURCE/'packet.json')==proof['packet_sha256'] and sha(SOURCE/'training_packet.npz')==proof['training_npz_sha256']
assert sha(HERE/'verify.py')==proof['script_sha256']
assert proof['holdout']['d4_orbit_intersections']==0
holdout=ROOT/'codex'/'synthetic'/'holdout_v1';assert sha(holdout/'FROZEN.json')==proof['holdout']['frozen_sha256']
for name,h in json.loads((holdout/'FROZEN.json').read_text())['files'].items():assert sha(holdout/name)==h
train=ROOT/'train.py';train_hash=sha(train);tree=ast.parse(train.read_text(encoding='utf-8'))
load_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load')
keys_node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='KEYS' for t in n.targets))
ns={'np':np,'Path':Path};exec(compile(ast.Module(body=[keys_node,load_node],type_ignores=[]),str(train),'exec'),ns)
main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
loss_node=next(n for n in main_node.body if isinstance(n,ast.FunctionDef) and n.name=='losses')
assignments={n.targets[0].id:ast.unparse(n.value) for n in loss_node.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
assert '* hvv' in assignments['vl'] and '* hvv' in assignments['al'] and '* qvv' in assignments['al']
assert '* w' in assignments['pl'] and assignments['w']=='hh'
z=np.load(SOURCE/'training_packet.npz');take=np.array([1,2]);arrays={k:z[k][take] for k in ns['KEYS']}
arrays['obs']=arrays['obs'].astype(np.float16);arrays['pval']=arrays['pval'].astype(np.float16)
assert arrays['has_pol'].tolist()==[False,True] and arrays['has_value'].tolist()==[True,False] and arrays['z'].tolist()==[-1.,0.]
assert not arrays['q_valid'].any() and arrays['pidx'].shape==(2,128)
assert np.all(arrays['left']>0) and all(np.isfinite(a).all() for a in arrays.values())
for i in range(2):
    legal=z['legal_mask'][take[i]];idx=arrays['pidx'][i];assert set(idx[idx>=0].tolist())==set(np.flatnonzero(legal).tolist())
assert arrays['pval'][0].sum()==0 and arrays['pval'][1].sum()==1
assert arrays['pidx'][1,int(np.argmax(arrays['pval'][1]))]==3164
assert not OUT.exists(),'Preserve any existing release'
OUT.mkdir()
np.savez_compressed(OUT/'chunk0000.npz',**arrays)
loaded=ns['load']([OUT/'chunk0000.npz']);assert len(loaded['z'])==2 and loaded['is_ext'].all() and not loaded['has_npol'].any()
assert loaded['has_value'].tolist()==[True,False] and loaded['has_pol'].tolist()==[False,True]
assert sha(train)==train_hash,'Loader changed during audit'
audit=dict(status='PASS',method='Executed exact train.load AST with numpy/Path; inspected exact loss expressions for policy/value/Q/length masks. No torch import, model, GPU or training.',train_sha256=train_hash,loader_rows=2,external_rows=2,next_policy_disabled=True,loss_expressions={k:assignments[k] for k in ['pl','vl','al']},arrays={k:dict(dtype=str(a.dtype),shape=list(a.shape)) for k,a in arrays.items()},mask_contract=dict(has_pol=[False,True],has_value=[True,False],q_valid=[False,False],z=[-1,0],C_zero_is_masked_placeholder=True,A_exported=False))
(HERE/'loader_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
provenance=dict(status='READY_INDEPENDENTLY_VERIFIED_TRAINING',rows=2,source_group='site:10910',source_group_aliases=['site-game-10910','10910'],cross_corpus_grouping='All aliases denote the same real game; keep observed, counterfactual and replay rows in one split.',scope=proof['scope'],source_sha256=proof['source_sha256'],source_packet_sha256=proof['packet_sha256'],independent_proof_path=str(HERE/'proof.json'),independent_proof_sha256=sha(HERE/'proof.json'),loader_audit_path=str(HERE/'loader_audit.json'),loader_audit_sha256=sha(HERE/'loader_audit.json'),train_sha256=train_hash,chunk_sha256=sha(OUT/'chunk0000.npz'),holdout=proof['holdout'],records=[dict(row=0,position_id='10910-B-actual-pre40',provenance='ACTUAL_OBSERVED_REPLAY',z=-1,has_value=True,has_pol=False,q_valid=False,meaning='All six own actions lose by the second following enemy turn. No arbitrary policy preference.'),dict(row=1,position_id='10910-C-counterfactual-pre40',provenance='ONE_LEGAL_MOVE_FROM_ACTUAL_PRE39',z=0,has_value=False,has_pol=True,q_valid=False,preferred_action=3164,meaning='All six actions classified; only Trace G7-D4 avoids forced loss through the second following enemy turn. Full-game value UNKNOWN.')],excluded=dict(position_id='10910-A-pre39',reason='Only 2/9 first actions classified: 2738 demonstrated bounded-safe; 2729 proved unsafe; seven UNKNOWN. No standard CE target or exact full-game value.',pairwise_preference=dict(better=2738,worse=2729),metadata_path=str(SOURCE/'packet.json')))
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
(HERE/'release_receipt.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8');print(json.dumps(dict(status=provenance['status'],path=str(OUT),rows=2,chunk_sha256=provenance['chunk_sha256'],proof_sha256=provenance['independent_proof_sha256'],train_sha256=train_hash)))
