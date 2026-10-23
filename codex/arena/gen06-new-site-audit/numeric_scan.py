"""Raw ORT range scan of 118 exact replayed website games, without clipping."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import onnxruntime as ort
import onnx
import audit

HERE=audit.HERE;ROOT=audit.ROOT
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    started=time.monotonic();inputs={}
    def register(p):inputs[str(p.relative_to(ROOT))]=sha(p);return p
    def read(p):return json.loads(register(p).read_text(encoding='utf-8'))
    manifest=read(ROOT/'data/ext_site_v1/manifest.json')
    aliases=read(ROOT/'codex/matches/site-external-aliases.json')['aliases']
    assert len(manifest['episodes'])==111 and len(aliases)==7
    episodes=[dict(r,chunk_path=ROOT/'data/ext_site_v1'/r['chunk'],row_range=None) for r in manifest['episodes']]
    episodes += [dict(r,chunk_path=Path(r['external_chunk']),chunk_sha256=r['external_chunk_sha256'],
                      row_range=r['external_rows'],rows=r['external_rows'][1]-r['external_rows'][0]) for r in aliases]
    assert len({str(r['game_id']) for r in episodes})==118
    assert sum(r['rows'] for r in episodes)==4105
    assert sha(audit.MODEL)==audit.MODEL_SHA
    graph=onnx.load(str(audit.MODEL),load_external_data=False)
    allobs=[];metadata=[];chunks={}
    for ep in episodes:
        gid=str(ep['game_id']);p=ROOT/'codex/matches/site-all-v1-derived'/f'match-{gid}.txt';register(p)
        game,states=audit.site.parse(p,None);assert len(states)==ep['rows']
        assert game['winner']==ep['winner']
        cp=ep['chunk_path']
        if str(cp) not in chunks:
            register(cp);assert sha(cp)==ep['chunk_sha256']
            with np.load(cp,allow_pickle=False) as d:chunks[str(cp)]=d['obs'].copy()
        data=chunks[str(cp)];span=ep['row_range'];stored=data if span is None else data[span[0]:span[1]]
        obs=np.stack([s.observation() for s in states]).astype(np.float32)
        assert np.array_equal(obs.astype(stored.dtype),stored), 'Canonical replay must match stored game observations'
        for env,x in zip(states,obs):
            assert not env.terminated and env.legal_actions(), 'Observed input must be a legal nonterminal decision'
            metadata.append({'index':len(metadata),'game_id':gid,'ply':env.action,'actor':env.current_team,
                             'observation_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                             'exact_replay_source_sha256':sha(p),'historical_source_model':'UNKNOWN'})
        allobs.extend(obs)
    observations=np.ascontiguousarray(np.stack(allobs),dtype=np.float32)
    with (HERE/'numeric-observation-index.jsonl').open('w',encoding='utf-8') as out:
        for r in metadata:out.write(json.dumps(r)+'\n')
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(audit.MODEL),sess_options=options,providers=['CPUExecutionProvider'])
    input_name=session.get_inputs()[0].name;output_names=[x.name for x in session.get_outputs()]
    stats={};violations=[];bad_obs={}
    for batch in [1,8,16]:
        values=[];policy_nonfinite=0;calls=0;batchstart=time.monotonic()
        for offset in range(0,len(observations),batch):
            real=min(batch,len(observations)-offset)
            data=observations[offset:offset+real]
            if real<batch:data=np.concatenate([data,np.repeat(data[-1:],batch-real,axis=0)],axis=0)
            policy,value=session.run(output_names,{input_name:data})
            assert policy.shape==(batch,4096) and value.shape==(batch,)
            assert policy.dtype==np.float32 and value.dtype==np.float32
            value=value[:real];policy=policy[:real];values.extend(map(float,value));calls+=1
            policy_nonfinite+=int((~np.isfinite(policy)).sum())
            offending=np.flatnonzero((~np.isfinite(value)) | (value < -1.0) | (value > 1.0) | (~np.isfinite(policy)).any(axis=1))
            for j in offending:
                ix=offset+int(j);bad_obs[ix]=observations[ix]
                r=dict(metadata[ix],batch_size=batch,value=float(value[j]),
                       value_float32_bits=hex(int(value[j].view(np.uint32))),
                       nonfinite_policy_count=int((~np.isfinite(policy[j])).sum()))
                violations.append(r)
        finite=np.array(values)[np.isfinite(values)]
        stats[str(batch)]={'real_states':len(values),'calls':calls,'minimum':float(finite.min()),'maximum':float(finite.max()),
                           'nonfinite_values':int(sum(not np.isfinite(v) for v in values)),
                           'out_of_range_values':sum(np.isfinite(v) and (v < -1 or v > 1) for v in values),
                           'nonfinite_policy_entries':policy_nonfinite,'seconds':time.monotonic()-batchstart}
        print(json.dumps({'batch':batch,**stats[str(batch)]}),flush=True)
    assert sha(audit.MODEL)==audit.MODEL_SHA
    for p,h in inputs.items():assert sha(ROOT/p)==h
    if bad_obs:
        indices=sorted(bad_obs)
        np.savez_compressed(HERE/'numeric-failing-observations.npz',obs=np.stack([bad_obs[i] for i in indices]),index=np.array(indices))
    (HERE/'numeric-violations.json').write_text(json.dumps(violations,indent=2),encoding='utf-8')
    result={'status':'PASS_OBSERVED_CORPUS' if not violations else 'FAIL_RAW_OUTPUT_CONTRACT',
            'model_sha256':audit.MODEL_SHA,'episodes':118,'states':4105,'batch_sizes':[1,8,16],
            'all_replays_and_stored_float16_observations_match':True,
            'observation_collection_sha256':hashlib.sha256(observations.tobytes()).hexdigest(),
            'batches':stats,'violations':len(violations),'distinct_violating_states':len(bad_obs),
            'raw_session_run_without_evaluator_sanitizing_or_clipping':True,
            'ort_version':ort.__version__,'providers':session.get_providers(),'threads':1,'processes':1,
            'last_graph_node_type':graph.graph.node[-1].op_type,
            'inputs_sha256':inputs,'scan_harness_sha256':sha(__file__),
            'end_model_and_inputs_hash_verified':True,'elapsed_seconds':time.monotonic()-started,
            'limitation':'Observed legal states only, Windows ORT CPU. No proof over all legal/search states or Linux kernels.'}
    (HERE/'numeric-summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='inputs_sha256'}),flush=True)

if __name__=='__main__':main()
