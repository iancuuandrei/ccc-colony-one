"""Independently replay only the two authorized GEN05 gates; no new games."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'env'))
from colony_one.env import ColonyOneEnv

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    reports=[]
    for name in ['gen05-v-gen03','gen05-v-gen02']:
        folder=ROOT/'codex'/'arena'/name
        manifest=json.loads((folder/'manifest.json').read_text());p=manifest['protocol']
        summary=json.loads((folder/'summary.json').read_text())
        assert summary['complete'] and summary['end_provenance_verified']
        assert sha(p['a'])==p['a_sha256'] and sha(p['b'])==p['b_sha256']
        assert sha(ROOT/'codex'/'arena'/'paired_arena.py')==p['harness_sha256']
        for path,digest in p['engine_files'].items():assert sha(ROOT/path)==digest
        rows=[json.loads(x) for x in (folder/'games.jsonl').read_text().splitlines()]
        expected={(s,c) for s in range(p['start'],p['start']+p['pairs']) for c in (True,False)}
        assert len(rows)==2*p['pairs'] and {(r['seed'],r['a_cyan']) for r in rows}==expected
        checks=[]
        for row in rows:
            env=ColonyOneEnv(seed=row['seed'])
            for h in row['history']:
                assert (env.current_team,env.round_number,env.actions_left)==(h['team'],h['round'],h['actions_left'])
                legal=[a for a in env.legal_actions() if list(a.from_pos)==h['from'] and list(a.to_pos)==h['to']]
                assert len(legal)==1;env.step(legal[0])
            assert env.terminated and env.winner==row['winner'] and env.round_number==row['round']
            assert len(row['history'])==row['actions']
            assert row['a_won']==(env.winner==('CYAN' if row['a_cyan'] else 'MAGENTA'))
            checks.append({'seed':row['seed'],'a_cyan':row['a_cyan'],'actions':row['actions'],'status':'PASS'})
        assert sum(r['a_won'] for r in rows)==summary['a_wins']
        report={'status':'PASS','histories':len(rows),'a_wins':summary['a_wins'],
                'a_sha256':p['a_sha256'],'b_sha256':p['b_sha256'],'engine_sha256':p['engine_sha256'],
                'manifest_sha256':sha(folder/'manifest.json'),'games_jsonl_sha256':sha(folder/'games.jsonl'),
                'script_sha256':sha(__file__),'checks':checks}
        (folder/'engine-history-audit.json').write_text(json.dumps(report,indent=2))
        reports.append({k:v for k,v in report.items() if k!='checks'})
    print(json.dumps(reports))

if __name__=='__main__':main()
