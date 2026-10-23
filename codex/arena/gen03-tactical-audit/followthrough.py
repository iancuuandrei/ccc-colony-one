"""Verify the two newly selected GEN03 winning starts complete under MCTS100."""
import hashlib
import json
from pathlib import Path
import numpy as np
import audit

rows,_=audit.actual_site()
states={r['case_id']:r for r in rows}
saved={r['case_id']:r for r in map(json.loads,(audit.HERE/'site-results.jsonl').read_text().splitlines())}
report=[]
for case in ['site_win-v1-8228-35','site_win-v2-8623-33']:
    r=states[case];env=audit.q.restore(r['state']);original=env.current_team
    first=saved[case]['models']['gen03']['action_index'];env.step(first)
    actions=[saved[case]['models']['gen03']['action']]
    if not env.terminated and env.current_team==original:
        a=audit.mcts.search_action(env.board,env.current_team,env.actions_left,env.round_number,
                                  audit.evaluator('gen03'),audit.mcts.MCTSConfig(simulations=100),
                                  rng=np.random.default_rng(0))
        actions.append(audit.notation(a));env.step(a)
    assert env.terminated and env.winner==original
    report.append({'case_id':case,'gen03_actions':actions,'terminal_winner':env.winner,'status':'PASS'})
(audit.HERE/'improved-win-followthrough.json').write_text(json.dumps(report,indent=2))
files=list((audit.ROOT/'env'/'colony_one').glob('*.py'))+[audit.ROOT/'env'/'colony_one'/'scene.json',
        audit.ROOT/'codex'/'engine_audit'/'own_turn_oracle.py',audit.ROOT/'codex'/'matches'/'site_replay.py',
        audit.ROOT/'codex'/'distillation'/'queue.py']
provenance={'models':{n:{'path':str(p),'sha256':audit.sha(p)} for n,(p,h) in audit.MODELS.items()},
            'files':{str(p.relative_to(audit.ROOT)):audit.sha(p) for p in sorted(files)},
            'scope':'Read-only diagnostic; holdout remains TEST_ONLY and cannot enter training.'}
assert all(provenance['models'][n]['sha256']==h for n,(p,h) in audit.MODELS.items())
(audit.HERE/'provenance.json').write_text(json.dumps(provenance,indent=2))
print(json.dumps(report))
