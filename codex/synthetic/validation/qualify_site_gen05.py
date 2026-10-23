"""Independent full-environment GEN05 source/proof audit; stage before loader qualification."""
import json,sys,time
from pathlib import Path
import numpy as np
from validate_synthetic import ROOT,HERE,q,own_turn_win_mask,next_turn_loss
from audit_holdout import orbit_key
sys.path.insert(0,str(ROOT/'codex/matches'))
from site_replay import parse


def run():
    start=time.monotonic();src=ROOT/'codex/synthetic/site_gen05_curriculum';out=HERE/'site-gen05-v1';out.mkdir(exist_ok=True)
    manifest=json.loads((src/'MANIFEST.json').read_text());packet=json.loads((src/'packet.json').read_text());rows=[json.loads(s) for s in (src/'states.jsonl').read_text().splitlines()];props=json.loads((src/'proposals.json').read_text());data=dict(np.load(src/'training_packet.npz'));aux={r['position_id']:r for r in json.loads((src/'auxiliary_metadata.json').read_text())}
    assert len(rows)==len(packet['positions'])==len(data['obs'])==28
    hashes={n:q.sha(src/n) for n in [*manifest['files'],'MANIFEST.json','auxiliary_metadata.json','LENGTH_MASK_REQUIREMENT.md']}
    for n,h in manifest['files'].items():assert hashes[n]==h
    for n,h in manifest['engine_sha256'].items():assert q.sha(ROOT/'env/colony_one'/n)==h
    for n,h in manifest['oracle_sha256'].items():assert q.sha(ROOT/'codex/engine_audit'/n)==h
    snap=src/'sources'/props['source_npz'];assert q.sha(snap)==manifest['source_sha256']==props['source_sha256'];source=dict(np.load(snap))
    frozen=ROOT/'codex/synthetic/holdout_v1';freeze=json.loads((frozen/'FROZEN.json').read_text());assert freeze['training_allowed'] is False
    for n,h in freeze['files'].items():assert q.sha(frozen/n)==h
    held={orbit_key(o) for o in np.load(frozen/'holdout_states.npz')['obs']}
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for n,h in hashes.items():
        dest=snapshot/n
        if not dest.exists():dest.write_bytes((src/n).read_bytes())
        assert q.sha(dest)==h
    games={};actual={};proposal={r['id']:r for r in props['counterfactuals']};checks=[];errors=[];lengths=[];groups={};flips=[]
    for i,row in enumerate(rows):
        try:
            assert row==packet['positions'][i] and row['position_id']==data['position_id'][i] and row['state_sha256']==q.state_hash(row['state'])
            state=row['state'];env=q.restore(state);obs=env.observation();cert=row['certificate'];s=row['source'];gid=str(s['game_id']);groups.setdefault(gid,set()).add(row['group_id'])
            assert row['group_id']=='site:'+gid and row['group_alias']=='site-game-'+gid and data['source_group'][i]==row['group_id']
            assert q.encode_state(env)==state and np.array_equal(obs,data['obs'][i]) and orbit_key(obs) not in held
            assert not env.terminated and env.actions_left in (1,2) and 1<=env.round_number<=26
            assert np.isfinite(obs).all() and np.isin(obs[:13],[0,1]).all() and np.all(obs[:12].sum(0)<=1) and not np.any(obs[:12].sum(0)*obs[12])
            assert len(state['piece_squares'][5])==len(state['piece_squares'][11])==1 and all(len(x)<=1 for x in state['piece_squares'])
            if s['kind']=='ACTUAL_OBSERVED_REPLAY':
                if gid not in games:
                    path=ROOT/'codex/matches/site-gen05-raw'/('match-'+gid+'.txt');game,states=parse(path,None);games[gid]=(game,{e.action:e for e in states},q.sha(path))
                game,replay,_=games[gid];expected=replay[s['ply']];assert q.encode_state(expected)==state
                ri=s['source_row'];assert np.array_equal(source['obs'][ri],obs) and str(source['match'][ri])==gid and int(source['ply'][ri])==s['ply']
                actual[gid+':'+str(s['ply'])]=env.clone();length=game['actions']-s['ply']+1
                assert aux[row['position_id']]['has_length'] is True and aux[row['position_id']]['left']==length
                lengths.append((True,length))
            else:
                prop=proposal[s['proposal_id']];expected=actual[prop['source']].clone();assert s['reachability']=='LEGAL_STATE_CONSTRUCTION; NOT_PROVEN_FROM_STANDARD_OPENING; NOT_OBSERVED_MATCH_PLAY'
                if 'move_piece' in prop:
                    fr,to=prop['move_piece'];assert expected.board.grid[fr[0]][fr[1]] is not None and expected.board.grid[to[0]][to[1]] is None
                    expected.board.grid[to[0]][to[1]]=expected.board.grid[fr[0]][fr[1]];expected.board.grid[fr[0]][fr[1]]=None
                if 'actions_left' in prop:expected.actions_left=prop['actions_left']
                if 'round' in prop:expected.round_number=prop['round']
                expected.action=(expected.round_number-1)*4+(1 if expected.current_team=='CYAN' else 3)+(2-expected.actions_left)
                k=s['rotation_quarter_turns'];assert np.array_equal(np.rot90(expected.observation(),-k,axes=(1,2)),obs)
                assert aux[row['position_id']]['has_length'] is False;lengths.append((False,1))
            legal={int(q.encoding.action_index(a,env.current_team)):a for a in env.legal_actions()};assert legal
            assert set(legal)==set(np.flatnonzero(data['legal_mask'][i]))=={a['action_index'] for a in row['legal_actions']}
            pi=data['pidx'][i];n=len(legal);assert set(pi[pi>=0])==set(legal) and np.all(pi[n:]==-1) and np.all(data['pval'][i,n:]==0)
            for a in row['legal_actions']:
                ac=legal[a['action_index']];assert ac.kind==a['kind'] and list(ac.from_pos)==a['from_rc'] and list(ac.to_pos)==a['to_rc'] and env.board.piece_at(*ac.from_pos).piece_type.value==a['piece']
            wins,ownnodes=own_turn_win_mask(env);assert set(wins)==set(cert['winning_first_actions'])==set(np.flatnonzero(data['winning_first_mask'][i]))
            loss=None
            if wins:
                assert cert['value_valid'] and cert['value']==1;preferred=set(wins)
            else:
                loss=next_turn_loss(env,max_nodes=1000000);assert loss['status']!='UNKNOWN'
                statuses={int(a):v for a,v in cert['loss_oracle']['per_first_action'].items()};assert loss['per_first_action']==statuses
                preferred={a for a,v in statuses.items() if v=='DISPROVED'};bad={a for a,v in statuses.items() if v=='VERIFIED'}
                assert preferred==set(cert['safe_first_actions']) and bad==set(cert['unsafe_first_actions']) and not cert['unknown_first_actions']
                if preferred:assert not cert['value_valid'] and cert['value'] is None
                else:assert cert['value_valid'] and cert['value']==-1 and loss['status']=='VERIFIED'
            for field,cfield in [('safe_first_mask','safe_first_actions'),('unsafe_first_mask','unsafe_first_actions'),('unknown_first_mask','unknown_first_actions')]:assert set(np.flatnonzero(data[field][i]))==set(cert[cfield])
            assert bool(data['has_value'][i])==cert['value_valid'] and bool(data['has_pol'][i])==bool(preferred)==cert['policy_valid'] and not data['q_valid'][i]
            assert float(data['z'][i])==(cert['value'] if cert['value_valid'] else 0)
            weights={a['action_index']:a['weight'] for a in row['policy_targets']};assert set(weights)==set(legal)
            for j,a in enumerate(pi[:n]):assert abs(float(data['pval'][i,j])-weights[int(a)])<1e-7 and abs(weights[int(a)]-(1/len(preferred) if a in preferred else 0))<1e-7
            for key,w in cert['winning_witnesses'].items():
                ch=env.clone();team=ch.current_team
                for j,idx in enumerate(w['actions']):
                    ac=next(a for a in ch.legal_actions() if q.encoding.action_index(a,ch.current_team)==idx);typed=cert['typed_winning_witnesses'][str(key)][j]
                    assert typed['piece']==ch.board.piece_at(*ac.from_pos).piece_type.value and typed['kind']==ac.kind and typed['move']==q.coord(ac.from_pos)+'-'+q.coord(ac.to_pos)
                    ch.step(ac)
                assert ch.terminated and ch.winner==team
            checks.append(dict(row=i,position_id=row['position_id'],status='PASS',legal=n,own_nodes=ownnodes,safety_nodes=0 if loss is None else loss['nodes'],preferred=sorted(preferred),value=cert['value'],has_length=lengths[-1][0],left=lengths[-1][1]))
            if s.get('proposal_id')=='kernel_extra_action' and s['rotation_quarter_turns']==0:flips.append(checks[-1])
            print(json.dumps(dict(row=i,status='PASS',nodes=checks[-1]['safety_nodes'],seconds=round(time.monotonic()-start,1))),flush=True)
        except Exception as exc:errors.append(dict(row=i,position_id=row['position_id'],error=repr(exc)));print(json.dumps(errors[-1]),flush=True)
        (out/'progress.json').write_text(json.dumps(dict(checked=len(checks),errors=errors,seconds=time.monotonic()-start),indent=2))
    assert all(len(v)==1 for v in groups.values()) and len(groups)==3
    stable=all(q.sha(src/n)==h for n,h in hashes.items());assert stable
    report=dict(accepted=not errors,checked=len(checks),rows=len(rows),errors=errors,checks=checks,source_hashes=hashes,source_stable=stable,source_game_sha256={g:v[2] for g,v in games.items()},holdout_frozen_sha256=q.sha(frozen/'FROZEN.json'),D4_holdout_collisions=0,groups={g:sorted(v) for g,v in groups.items()},length_semantics='Four observed recorded episode lengths; all24 constructed lengths UNKNOWN',turn_order_flip=flips,seconds=time.monotonic()-start,method='Independent full env.clone/step own-turn winning mask and complete next-opponent-turn minimax for every row; producer oracle not imported. No model/search/training.',release_status='STAGED_PENDING_CURRENT_HAS_LENGTH_LOADER_AND_GRADIENT_QUALIFICATION')
    (out/'report.json').write_text(json.dumps(report,indent=2));assert report['accepted'],errors
    stage=out/'export_stage';stage.mkdir(exist_ok=True);export={k:data[k] for k in ('obs','pidx','pval','z','q','has_value','has_pol','q_valid')};export['obs']=export['obs'].astype(np.float16);export['pval']=export['pval'].astype(np.float16);export['has_length']=np.array([x[0] for x in lengths],bool);export['left']=np.array([x[1] for x in lengths],np.float32)
    np.savez_compressed(stage/'chunk0000.npz',**export)
    provenance=dict(status='STAGED_PENDING_LOADER_QUALIFICATION',rows=28,source_manifest_sha256=hashes['MANIFEST.json'],independent_report_sha256=q.sha(out/'report.json'),chunk_sha256=q.sha(stage/'chunk0000.npz'),policy_rows=int(export['has_pol'].sum()),value_rows=int(export['has_value'].sum()),length_rows=int(export['has_length'].sum()),records=[dict(row=i,position_id=r['position_id'],group_id=r['group_id'],group_alias=r['group_alias'],source=r['source'],has_value=bool(export['has_value'][i]),has_pol=bool(export['has_pol'][i]),has_length=bool(export['has_length'][i]),left=float(export['left'][i])) for i,r in enumerate(rows)],label_authority='Astra-authored scenarios; independently verified engine labels; constructed opening reachability unproved',quarantine='Frozen holdout TEST_ONLY unchanged, no D4 overlap')
    (stage/'provenance.json').write_text(json.dumps(provenance,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('checks','source_hashes')},indent=2))


if __name__=='__main__':run()
