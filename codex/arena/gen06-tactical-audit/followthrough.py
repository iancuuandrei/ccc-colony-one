"""Read-only GEN06 actual terminal follow-through for selected certified winning starts."""
import json
import numpy as np
import audit

cases,_=audit.actual_site();byid={r['case_id']:r for r in cases}
saved=[json.loads(x) for x in (audit.HERE/'site-results.jsonl').read_text().splitlines()]
results=[]
for row in saved:
    if row['kind']!='site_win' or row['models']['gen06']['missed_current_turn_win']:continue
    env=audit.q.restore(byid[row['case_id']]['state']);team=env.current_team
    actions=[row['models']['gen06']['action']]
    env.step(row['models']['gen06']['action_index'])
    if not env.terminated and env.current_team==team:
        a=audit.mcts.search_action(env.board,env.current_team,env.actions_left,env.round_number,
                                  audit.evaluator('gen06'),audit.mcts.MCTSConfig(simulations=100),rng=np.random.default_rng(0))
        actions.append(audit.notation(a));env.step(a)
    won=env.terminated and env.winner==team
    results.append({'case_id':row['case_id'],'actions':actions,'terminal_own_turn_win':won,
                    'winner':env.winner,'status':'PASS' if won else 'FAIL_MCTS_FOLLOWTHROUGH'})
report={'cases':len(results),'terminal_own_turn_wins':sum(r['terminal_own_turn_win'] for r in results),'results':results,
        'models':{n:{'path':str(p),'sha256':audit.sha(p)} for n,(p,h) in audit.MODELS.items()}}
assert all(report['models'][n]['sha256']==h for n,(p,h) in audit.MODELS.items())
(audit.HERE/'winning-followthrough.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
