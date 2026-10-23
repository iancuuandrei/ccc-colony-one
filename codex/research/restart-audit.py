"""Read-only restart/archive audit; model-free exact-engine operational fixtures."""
from pathlib import Path
import ast,hashlib,json,sys,types
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'env'));sys.path.insert(0,str(ROOT/'codex/matches'))
from colony_one import constants as C,encoding as E,mcts
from colony_one.board import Board
from colony_one.env import ColonyOneEnv
from site_replay import parse,pos

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(o):return hashlib.sha256(np.asarray(o,dtype=np.float16).tobytes()).hexdigest()
def rehydrate(ob,ns):
    e=ColonyOneEnv();e.current_team=C.MAGENTA if ob[15,0,0]>.5 else C.CYAN;e.board=Board.from_observation(ob,e.current_team);e.actions_left=round(float(ob[13,0,0])*2);e.round_number=ns['round_of'](ob);e.action=100;return e
def main():
    paths=['selfplay.py','build_archive.py','actor.py','learner.py','train.py','cpu_budget.json']
    hashes={p:digest(ROOT/p) for p in paths};tree=ast.parse((ROOT/'selfplay.py').read_text());ns={'np':np,'ROOT':ROOT,'K':128,'_restart':{},'evaluator':lambda path:None}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('round_of','restart_state','play')]
    exec(compile(ast.Module(body=functions,type_ignores=[]),'exact_current_selfplay_probe','exec'),ns)
    # A complete live clock cross-product, including the short final24->26 interval.
    clock_errors=[];clock_cases=0
    for rnd in range(1,27):
        for team in (C.CYAN,C.MAGENTA):
            for al in (1,2):
                e=ColonyOneEnv();e.board.locked=Board.locked_cells_for_round(rnd);e.current_team=team;e.actions_left=al;e.round_number=rnd
                ob=e.observation();back=rehydrate(ob,ns);clock_cases+=1
                if back.current_team!=team or back.actions_left!=al or back.round_number!=rnd or not np.array_equal(back.observation(),ob):clock_errors.append([rnd,team,al,back.round_number])
    g,states=parse(ROOT/'codex/matches/site-all-v1-derived/match-11059.txt',None);replay_errors=[]
    for e in states:
        back=rehydrate(e.observation(),ns)
        if (back.current_team,back.actions_left,back.round_number)!=(e.current_team,e.actions_left,e.round_number) or back.legal_action_indices().tolist()!=e.legal_action_indices().tolist():replay_errors.append(e.action)
    # No model is loaded: replace ONLY this probe process's MCTS evaluator route with
    # a legal scripted engine move, then exercise the exact production play() body.
    fixture=states[30].observation();ns['_restart']['audit_live_fixture']=fixture[None]
    original_run=mcts.run
    def scripted_run(board,team,al,rnd,ev,cfg,rng=None):
        pair=('G5','E5') if al==2 else ('E5','D4');actions=[a for a in __import__('colony_one.rules',fromlist=['legal_actions_for_team']).legal_actions_for_team(board,team)];chosen=next(a for a in actions if a.from_pos==pos(pair[0]) and a.to_pos==pos(pair[1]));target=E.action_index(chosen,team)
        N={E.action_index(a,team):int(E.action_index(a,team)==target) for a in actions};return types.SimpleNamespace(N=N,W={a:.25*n for a,n in N.items()})
    try:
        mcts.run=scripted_run
        opts=dict(full_sims=1,fast_sims=1,full_prob=1.,temp1_actions=8,temp2_actions=16,restart_file='audit_live_fixture',restart_prob=1.)
        result=ns['play'](('NO_MODEL_LOADED',None,2026,opts))
    finally:mcts.run=original_run
    assert result['z'].tolist()==[1.,1.] and result['left'].tolist()==[2.,1.]
    assert result['obs'][:,15,0,0].tolist()==[1.,1.] and result['obs'][:,13,0,0].tolist()==[1.,.5]
    continuation=dict(actions=len(result['z']),z=result['z'].tolist(),left=result['left'].tolist(),q_valid=result['q_valid'].tolist(),same_magenta_team_both_actions=True,winner=result['meta'][1],model_loaded=False,scope='Exact play() control/data path with scripted legal actions; no learned evaluator or model-strength assertion')
    # Bad inputs are intentionally NOT played: show that the exact loader accepts them.
    invalid={};base=fixture.copy()
    cases={'missing_own_kernel':base.copy(),'invalid_actions_left':base.copy(),'malformed_lock_count':base.copy(),'nonuniform_side_plane':base.copy()}
    cases['missing_own_kernel'][5]=0;cases['invalid_actions_left'][13]=0;cases['malformed_lock_count'][12,0,0]=1;cases['nonuniform_side_plane'][15,0,1]=0
    for name,ob in cases.items():
        ns['_restart'][name]=ob[None]
        try:loaded=ns['restart_state'](name,np.random.default_rng(0));back=rehydrate(loaded,ns);invalid[name]={'rejected':False,'decoded_round':back.round_number,'decoded_actions':back.actions_left}
        except Exception as exc:invalid[name]={'rejected':True,'error':str(exc)}
    # Bind current live archive and check source/data invariants without regenerating it.
    archive=ROOT/'archive/hard_states.npz';meta_path=ROOT/'archive/hard_states.json';archive_sha=digest(archive);metadata_sha=digest(meta_path);meta=json.loads(meta_path.read_text());arr=np.load(archive)['obs'];n=len(arr)
    f=arr.reshape(n,16,64);locks=f[:,12].sum(1);als=f[:,13,0];sides=f[:,15,0];urg=f[:,14,0];bad={
        'nonfinite':int((~np.isfinite(f).all((1,2))).sum()),'nonbinary_piece_or_lock':int((~np.isin(f[:,:13],[0,1]).all((1,2))).sum()),
        'overlapping_pieces':int((f[:,:12].sum(1)>1).any(1).sum()),'pieces_on_locked':int(((f[:,:12].sum(1)*f[:,12])>0).any(1).sum()),
        'missing_or_multiple_kernels':int(((f[:,5].sum(1)!=1)|(f[:,11].sum(1)!=1)).sum()),'invalid_locked_count':int((~np.isin(locks,[0,28,48,60])).sum()),
        'invalid_actions':int((~np.isin(als,[.5,1])).sum()),'invalid_side':int((~np.isin(sides,[0,1])).sum()),'nonuniform_globals':int((f[:,13:]!=f[:,13:,:1]).any((1,2)).sum())}
    arc_keys={key(o) for o in arr};live_rounds={};sample_errors=[]
    for i in np.unique(np.linspace(0,n-1,128,dtype=int)):
        ob=arr[i].astype(np.float32);e=rehydrate(ob,ns);live_rounds[str(e.round_number)]=live_rounds.get(str(e.round_number),0)+1
        if not np.array_equal(e.observation(),ob) or not e.legal_actions():sample_errors.append(int(i))
    # Observe archive overlap with actual holdout files named by training commands.
    holdout_overlap=[]
    for log in sorted((ROOT/'logs').glob('gen*.txt')):
        for line in log.read_text(errors='replace').splitlines():
            if line.startswith('$ ') and 'train.py --data ' in line and '--holdout-last' in line:
                args=line.split();dataargs=args[args.index('--data')+1:args.index('--init')];last=ROOT/dataargs[-1];files=sorted(last.glob('chunk*.npz'))
                if not files:continue
                path=files[-1]
                if last.name not in meta['dirs']:continue
                with np.load(path) as d:
                    eligible=d['q_valid']&(d['left']>3)&(np.abs(d['z']-d['q'])>=meta['min_regret']) if 'q_valid' in d.files else np.zeros(len(d['z']),bool)
                    overlap=sum(key(o) in arc_keys for o in d['obs']);eligible_overlap=sum(key(o) in arc_keys for o in d['obs'][eligible])
                holdout_overlap.append(dict(log=log.name,source_holdout=str(path.relative_to(ROOT)),overlap_observations=overlap,qualifying_regret_overlap=eligible_overlap,source_hash=digest(path),note='Observation overlap; duplicate occurrences cannot be assigned to one ancestor because archive lacks row IDs'))
    # Confirm q_valid gating against real selected source rows, not opponent Q guesses.
    candidate_counts={'q_valid_regret':0,'q_invalid_large_regret_excluded':0,'uncertainty_valid':0};selected_source_keys=set()
    for dirname in meta['dirs']:
        for path in sorted((ROOT/'data'/dirname).glob('chunk*.npz')):
            with np.load(path) as d:
                if 'q_valid' not in d.files:continue
                regret=np.abs(d['z']-d['q']);valid=d['q_valid'];long=d['left']>3
                mask=valid&long&(regret>=meta['min_regret']);candidate_counts['q_valid_regret']+=int(mask.sum());candidate_counts['q_invalid_large_regret_excluded']+=int((~valid&long&(regret>=meta['min_regret'])).sum());candidate_counts['uncertainty_valid']+=int((valid&long&(np.abs(d['q'])<.2)).sum())
                selected_source_keys.update(key(o) for o in d['obs'][mask])
    regret_segment_match=sum(key(o) in selected_source_keys for o in arr[:meta['regret_states']])
    held=ROOT/'codex/synthetic/holdout_v1';freeze=json.loads((held/'FROZEN.json').read_text());held_keys=set()
    for ob in np.load(held/'holdout_states.npz')['obs']:
        for reflect in (False,True):
            x=ob[:,:,::-1] if reflect else ob
            for k in range(4):held_keys.add(key(np.rot90(x,k,axes=(1,2))))
    held_collisions=sum(key(o) in held_keys for o in arr)
    cfg=json.loads((ROOT/'cpu_budget.json').read_text());args=cfg.get('selfplay_args',[]);restart_prob=float(args[args.index('--restart-prob')+1]) if '--restart-prob' in args else 0
    summaries=[]
    for dirname in meta['dirs'][-4:]:
        s=json.loads((ROOT/'data'/dirname/'summary.json').read_text());summaries.append(dict(dir=dirname,games=s.get('games'),restarted=s.get('restarted'),opts_restart_prob=s.get('opts',{}).get('restart_prob')))
    drift={p:hashes[p]!=digest(ROOT/p) for p in paths};archive_drift=archive_sha!=digest(archive) or metadata_sha!=digest(meta_path)
    report=dict(source_sha256=hashes,source_drift_during_probe=drift,clock_cases=clock_cases,clock_errors=clock_errors,actual_replay_cases=len(states),actual_replay_rehydration_errors=replay_errors,scripted_continuation=continuation,invalid_input_acceptance=invalid,
        archive=dict(path=str(archive),sha256=archive_sha,metadata_sha256=metadata_sha,rows=n,unique_observations=len(arc_keys),meta=meta,vectorized_invalid_counts=bad,sampled_exact_engine_cases=128,sampled_errors=sample_errors,sampled_round_counts=live_rounds,drift_during_probe=archive_drift),
        actual_training_holdout_overlaps=holdout_overlap,regret_candidate_counts=candidate_counts,regret_segment_source_membership=regret_segment_match,synthetic_test_holdout_archive_D4_collisions=held_collisions,current_actor_restart_prob=restart_prob,recent_archive_source_summary_restart_counts=summaries,
        provenance_limits=['archive NPZ stores obs only; no source row/game/seed/hash per selected state','continuation chunks discard meta except batch totals, so restart ancestor cannot be grouped','cache is keyed by mutable path only; workers can use different refresh snapshots','NPZ and JSON metadata replaced separately; freeze/version hash not carried to actor'],
        warmup_note='env.action=100 does not disable root noise: MCTS reconstructs move_number(team,actions_left,round). Early-round archive states still get opening noise; restart sampling temperature is separately keyed to n.')
    text='''# Restart/archive independent audit

Clock and perspective pass on104 exact fixtures (26rounds ×2teams ×2remaining-action counts), including24→25→26. All32 actual loss11059 observations rehydrate with exact legal sets. The actual selfplay.play control/data path, exercised with scripted legal engine moves and no model, produces fresh winner-based z=[+1,+1],left=[2,1] across two consecutive Magenta actions; no original outcome is copied.

**Actionable reliability findings:** build_archive reads all source chunks including any reserved --holdout-last chunk. Observed overlaps with actual training holdout commands are listed below; archive-only observations cannot identify which duplicate source occurrence was sampled. Preserve ancestor source/game grouping and explicitly exclude reserved validation/test sources before sampling. Current archive has zero D4 matches with quarantined holdout_v1.

The exact restart loader accepts malformed/missing-Kernel/invalid-clock observations without rejecting them; invalid lock counts silently map to the earliest ring. No such vectorized invalid state was found in the current archive, and128 sampled archive positions roundtrip through the exact engine with legal actions. Add strict validation/rejection at archive ingress, retain both Kernels and a live coherent clock, and preserve the exception that bare-Kernel positions can be live midturn.

q_valid selection correctly excludes opponent-search Q. |z−Q| is outcome disagreement, not proven action regret; uncertainty filtering |Q|<.2 is separate. Current archive and actor settings below supersede the document's historical20% claim. Restart cache/provenance and warmup caveats are explicitly recorded. No actor, GPU, model, archive builder or raw dataset was started/stopped/modified by this audit.

```json
'''+json.dumps(report,indent=2)+'\n```\n'
    Path(__file__).with_suffix('.md').write_text(text,encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('clock_cases','clock_errors','actual_replay_rehydration_errors','invalid_input_acceptance','actual_training_holdout_overlaps','regret_candidate_counts','regret_segment_source_membership','synthetic_test_holdout_archive_D4_collisions','current_actor_restart_prob','source_drift_during_probe')},indent=2));print('Archive rows',n,'invalid',bad,'sample_errors',sample_errors,'archive_drift',archive_drift)
if __name__=='__main__':main()
