"""Prepare three independently certified rows; no live data publication."""
from pathlib import Path
import ast,hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'codex/matches/19490'))
from analyze import parse,sha,sparse

def dump(path,obj):path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def hobs(obs,dtype):return hashlib.sha256(np.ascontiguousarray(obs,dtype=dtype).tobytes()).hexdigest()

def main():
    source=ROOT/'codex/matches/19490';critical=source/'critical-states.npz'
    assert sha(critical)=='333dbc28bfa7a06967c5267a0dc97e039ef9b0fb9c9a7e2d807d971b5a47495d'
    proof=ROOT/'codex/synthetic/validation/loss19490-independent/proof.json'
    equivalence=ROOT/'codex/matches/export6/user-five-v1/19101-19490-state-equivalence.json'
    assert json.loads(proof.read_text())['status']=='PASS'
    eq=json.loads(equivalence.read_text());assert eq['proof_reuse']=='PASS'
    game,states=parse(source/'match-19490.txt',None);assert game['actions']==74
    original=np.load(critical);selected=[int(np.flatnonzero(original['action_number']==n)[0]) for n in (68,71,72)]
    obs32=original['obs'][selected].copy();obs16=obs32.astype(np.float16)
    assert np.array_equal(obs16.astype(np.float32),obs32),'lossy observation storage'
    legal=original['legal_mask'][selected];safe2=original['two_enemy_safe_mask'][selected]
    assert np.flatnonzero(safe2[0]).tolist()==[2786] and not safe2[1:].any()
    pidx=original['pidx'][selected].copy();pval=np.zeros_like(pidx,dtype=np.float16)
    pval[0,np.flatnonzero(pidx[0]==2786)[0]]=1
    fields=dict(obs=obs16,pidx=pidx,pval=pval,pval_g=pval.copy(),z=np.array([0,-1,-1],np.float32),
        q=np.zeros(3,np.float32),has_pol=np.array([1,0,0],bool),has_value=np.array([0,1,1],bool),
        q_valid=np.zeros(3,bool),left=np.array([1,4,3],np.float32),has_length=np.array([0,1,1],bool),
        legal_mask=legal,two_enemy_safe_mask=safe2,two_enemy_unsafe_mask=original['two_enemy_unsafe_mask'][selected],
        two_enemy_unknown_mask=original['two_enemy_unknown_mask'][selected],
        source_group=np.full(3,'site-ending-19101-19490'),source_aliases=np.array([['19101','19490']]*3),
        position_id=np.array(['site-ending-19101-19490:pre'+str(n) for n in (68,71,72)]),
        action_number=np.array([68,71,72],np.int16),recorded_action=original['recorded_action'][selected])
    rows=[]
    for j,n in enumerate((68,71,72)):
        assert np.array_equal(states[n-1].observation(),obs32[j])
        assert 74-n+1=={68:7,71:4,72:3}[n]
        row=dict(position_id=str(fields['position_id'][j]),source_group='site-ending-19101-19490',source_aliases=['19101','19490'],
            action_number=n,state=sparse(states[n-1]),observation_float32_sha256=hobs(obs32[j],np.float32),
            observation_float16_sha256=hobs(obs32[j],np.float16),legal_indices=fields['pidx'][j][fields['pidx'][j]>=0].tolist(),
            certified_safe_indices=np.flatnonzero(safe2[j]).tolist(),
            certificate_scope='own remainder, enemy turn, own turn, second enemy turn' if n==68 else 'all own completions lose within next enemy turn',
            has_pol=bool(fields['has_pol'][j]),has_value=bool(fields['has_value'][j]),q_valid=False,
            has_length=bool(fields['has_length'][j]),length_target=float(fields['left'][j]),
            length_semantics='UNKNOWN; 1 is masked placeholder' if n==68 else 'recorded remaining actions including current action, not an optimal-mate-distance claim',
            value_target=float(fields['z'][j]),value_semantics='UNKNOWN eventual value; 0 is masked placeholder' if n==68 else 'exact forced loss for current actor',
            policy_semantics='one-hot unique fully classified bounded-safe action 2786; identical pval_g' if n==68 else 'no policy supervision; all legal pidx retained with zero pval/pval_g')
        rows.append(row)
    # Inspect current ext data for exact-observation overlap and target contradictions.
    overlap=[]
    for path in sorted((ROOT/'data').glob('ext_*/*.npz')):
        z=np.load(path)
        if 'obs' not in z:continue
        obs=z['obs'].astype(np.float16)
        for j,target in enumerate(obs16):
            hits=np.flatnonzero(np.all(obs==target,axis=(1,2,3)))
            for i in hits:
                item=dict(file=path.relative_to(ROOT).as_posix(),file_sha256=sha(path),row=int(i),proposal_action=int(fields['action_number'][j]),
                    has_pol=bool(z['has_pol'][i]) if 'has_pol' in z else None,
                    has_value=bool(z['has_value'][i]) if 'has_value' in z else True,
                    q_valid=bool(z['q_valid'][i]) if 'q_valid' in z else False,
                    z=float(z['z'][i]) if 'z' in z else None,q=float(z['q'][i]) if 'q' in z else None)
                for target_name in ('pval','pval_g'):
                    if target_name in z and 'pidx' in z:
                        pi=z['pidx'][i];pv=z[target_name][i];valid=pi>=0
                        item[target_name+'_total_mass']=float(pv[valid].sum(dtype=np.float64))
                        if j==0:
                            item[target_name+'_safe2786_mass']=float(pv[pi==2786].sum(dtype=np.float64))
                            item[target_name+'_certified_unsafe_mass']=float(pv[valid&(pi!=2786)].sum(dtype=np.float64))
                overlap.append(item)
        z.close()
    raw19101=[x for x in overlap if x['file']=='data/ext_site_v1/chunk-site-19101.npz']
    assert len(raw19101)==3 and [r['row'] for r in raw19101]==[67,70,71]
    outdir=HERE/'ext_deep19490_proposal';outdir.mkdir(exist_ok=True)
    out=outdir/'chunk0000.npz';np.savez_compressed(out,**fields)
    # Execute the current pure loader definitions, avoiding torch/net imports.
    train=ROOT/'train.py';tree=ast.parse(train.read_text(encoding='utf-8'))
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('load','apply_overrides')]
    nodes += [node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='KEYS' for t in node.targets)]
    ns=dict(np=np,Path=Path,json=json);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(train),'exec'),ns)
    loaded=ns['load']([out]);assert loaded['is_ext'].all() and not loaded['has_npol'].any()
    for k in ns['KEYS']:assert np.array_equal(loaded[k],fields[k]),k
    for mix in (0.,0.5,1.):
        mixed=(1-mix)*loaded['pval']+mix*loaded['pval_g'];assert mixed[0,pidx[0]==2786].tolist()==[1.]
        assert mixed[1:].sum()==0
    bindings={str(p.relative_to(ROOT)):sha(p) for p in [critical,source/'critical-states.json',source/'deep-proofs.json',proof,equivalence,train,Path(__file__)]}
    packet=dict(status='STAGED_REVIEWABLE_NOT_RELEASED',live_ext_release=False,rows=rows,chunk=str(out.relative_to(ROOT)),chunk_sha256=sha(out),
        source_game_count=2,unique_certified_ending_count=1,supervised_rows=3,
        duplication='19101 complete raw-outcome coverage already exists in data/ext_site_v1/chunk-site-19101.npz (74 rows); these3 exact obs occur at0based67/70/71. No19490 raw outcome import. Intentional certified emphasis, not3 fresh independent samples.',
        split_rule='Link source groups site-game-19101,site-game-19490 and site-ending-19101-19490 for split and weighting; do not count this ending twice.',
        existing_overlap=overlap,bindings=bindings,loader_audit=dict(status='PASS',method='AST-extracted current train.load/apply_overrides/KEYS; exact masks and pval_g retained; is_ext=True; has_npol=False; policy mix0/.5/1 invariant',train_sha256=sha(train)),
        integration_blocker='Append-only is insufficient when same observations have contradictory approximate policy/Q rows. Opus+onnx_validation must coordinate source-qualified, observation-keyed precedence supporting policy-only and value-only modes; existing policy-only single manifest cannot silently represent both.')
    dump(HERE/'packet.json',packet)
    print(json.dumps(dict(status=packet['status'],chunk_sha256=packet['chunk_sha256'],overlap_rows=len(overlap),loader='PASS',packet_sha256=sha(HERE/'packet.json'))))

if __name__=='__main__':main()
