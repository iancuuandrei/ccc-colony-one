"""Materialize exact, typed counterplay witnesses for evaluated model errors."""
exec(__import__('pathlib').Path(__file__).with_name('build.py').read_text().split('def main():')[0])
manifest=json.loads((HERE/'MANIFEST.json').read_text());evaluation=json.loads((HERE/'evaluation_gen02.json').read_text());rows={r['position_id']:r for r in [json.loads(x) for x in (HERE/'states.jsonl').read_text().splitlines()]}
for name,h in manifest['files'].items():assert sha(HERE/name)==h
def completed_turns(env,first):
    e=env.clone();e.step(first)
    if e.terminated or e.current_team!=env.current_team:return [([first],e)]
    out=[]
    assert e.actions_left==1
    for second in e.legal_action_indices():
        f=e.clone();f.step(int(second));out.append(([first,int(second)],f))
    return out
results=[]
for result in evaluation['results']:
    if not (result['missed_current_turn_win'] or result['avoidable_next_turn_loss']):continue
    r=rows[result['position_id']];env=q.restore(r['state']);record=dict(position_id=r['position_id'],seed_id=r['seed_id'],teacher_explanation=r['teacher_explanation'],chosen_action=result['action_index'])
    if result['missed_current_turn_win']:
        record.update(kind='MISSED_CURRENT_TURN_WIN',winning_witnesses=r['certificate']['typed_winning_witnesses'],claim='There exists a terminal win this turn; no eventual-loss claim about the selected move.')
    else:
        branches=[]
        for own_actions,e in completed_turns(env,result['action_index']):
            if e.terminated:
                assert e.winner not in (None,env.current_team);actions=own_actions
            else:
                assert e.current_team!=env.current_team
                ow=own_turn_wins(e.board,e.current_team,e.actions_left,e.round_number);assert ow.any_win
                key=min(ow.witnesses);actions=own_actions+ow.witnesses[key]['actions']
            typed,end=typed_replay(env,actions);assert end.terminated and end.winner not in (None,env.current_team)
            branches.append(dict(own_completion=own_actions,typed_terminal_witness=typed,winner=end.winner))
        assert branches
        safeproof=None
        for first in r['certificate']['safe_first_actions']:
            for own_actions,e in completed_turns(env,first):
                if e.terminated:
                    safe=e.winner in (None,env.current_team)
                else:
                    ow=own_turn_wins(e.board,e.current_team,e.actions_left,e.round_number);safe=not ow.any_win
                if safe:
                    typed,end=typed_replay(env,own_actions);safeproof=dict(own_actions=own_actions,typed_own_sequence=typed,terminal=end.terminated,winner=end.winner,opponent_terminal_win_within_next_turn=False,full_game_outcome='UNKNOWN' if not end.terminated else end.winner);break
            if safeproof:break
        assert safeproof
        record.update(kind='AVOIDABLE_LOSS_BEFORE_NEXT_OWN_TURN',all_own_completions_after_selected_first_action=branches,bounded_safe_alternative=safeproof,claim='Every completion of the selected first action has an enemy terminal response; an alternative avoids loss through that bounded horizon only.')
    results.append(record)
out=dict(corpus_id=manifest['corpus_id'],manifest_sha256=sha(HERE/'MANIFEST.json'),evaluation_sha256=sha(HERE/'evaluation_gen02.json'),verified_error_positions=len(results),actual_env_step_replay=True,records=results)
(HERE/'failure_certificates.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(dict(verified_error_positions=len(results),path=str(HERE/'failure_certificates.json'))))
