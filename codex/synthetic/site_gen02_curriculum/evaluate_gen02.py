"""Evaluate the frozen teacher cohort on the sole authorized GEN02 baseline."""
exec(__import__('pathlib').Path(__file__).with_name('build.py').read_text().split('def main():')[0])
from colony_one import mcts
from colony_one.onnx_runner import OnnxEvaluator
manifest=json.loads((HERE/'MANIFEST.json').read_text())
for name,h in manifest['files'].items():assert sha(HERE/name)==h
model=ROOT/manifest['baseline']['path'];assert sha(model)==manifest['baseline']['sha256']
assert not (HERE/'evaluation_gen02.json').exists(),'Preserve completed evaluation.'
ev=OnnxEvaluator(model,num_threads=1);rows=[json.loads(x) for x in (HERE/'states.jsonl').read_text().splitlines()];results=[]
for r in rows:
    env=q.restore(r['state']);obs=env.observation();before=obs.tobytes();logits,value=ev(obs[None]);legal=[a['action_index'] for a in r['legal_actions']]
    action=mcts.search_action(env.board,env.current_team,env.actions_left,env.round_number,ev,mcts.MCTSConfig(simulations=100),rng=np.random.default_rng(850001))
    assert env.observation().tobytes()==before
    ix=int(E.action_index(action,env.current_team));assert ix in legal;c=r['certificate'];preferred=c['winning_first_actions'] or c['safe_first_actions']
    probabilities=np.exp(logits[0,legal]-np.max(logits[0,legal]));probabilities/=probabilities.sum()
    results.append(dict(position_id=r['position_id'],seed_id=r['seed_id'],group_id=r['group_id'],provenance_kind=r['source']['kind'],motif_id=r['motif_id'],model_sha256=manifest['baseline']['sha256'],action_index=ix,chosen_move=next(a['move'] for a in r['legal_actions'] if a['action_index']==ix),raw_top_action=max(legal,key=lambda i:float(logits[0,i])),preferred_prior_mass=float(sum(probabilities[j] for j,a in enumerate(legal) if a in preferred)),value=float(value[0]),certified_value=c['value'],missed_current_turn_win=bool(c['winning_first_actions']) and ix not in c['winning_first_actions'],avoidable_next_turn_loss=bool(c['safe_first_actions']) and ix in c['unsafe_first_actions'],all_choices_forced_loss=c['value']==-1))
    print(json.dumps(results[-1]),flush=True)
def summarize(rs):
    exact=[abs(r['value']-r['certified_value']) for r in rs if r['certified_value'] is not None]
    return dict(states=len(rs),missed_current_turn_wins=sum(r['missed_current_turn_win'] for r in rs),avoidable_next_turn_losses=sum(r['avoidable_next_turn_loss'] for r in rs),already_forced_loss=sum(r['all_choices_forced_loss'] for r in rs),exact_value_mae=float(np.mean(exact)) if exact else None)
summary=dict(corpus_id=manifest['corpus_id'],manifest_sha256=sha(HERE/'MANIFEST.json'),baseline=manifest['baseline'],mcts_simulations=100,rng_seed=850001,overall=summarize(results),actual=summarize([r for r in results if r['provenance_kind']=='ACTUAL_OBSERVED_REPLAY']),counterfactual=summarize([r for r in results if r['provenance_kind']=='LLM_AUTHORED_COUNTERFACTUAL']),qualification='Selected training curriculum, not held-out performance or full-game win rate. All rows retained regardless of model output. One CPU process, one ORT thread.')
for name,h in manifest['files'].items():assert sha(HERE/name)==h
(HERE/'evaluation_gen02.json').write_text(json.dumps(dict(summary=summary,results=results),indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
