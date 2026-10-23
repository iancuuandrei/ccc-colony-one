"""Independent legal replay and identity audit of existing Opus GEN06 gate only."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'env'),str(ROOT/'codex'/'arena')]
from colony_one.env import ColonyOneEnv
from paired_arena import summarize

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    source=ROOT/'logs'/'gen06-gate.json'
    if not source.exists():
        report={'status':'NO_HISTORIES','source':str(source)}
        (HERE/'existing-gate-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return
    data=json.loads(source.read_text());rows=data.get('games')
    if not isinstance(rows,list) or not rows:
        report={'status':'NO_HISTORIES','source':str(source),'summary':data.get('summary')}
        (HERE/'existing-gate-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return
    assert len(rows)==400
    summary=data['summary'];candidate=ROOT/'release'/'gen06'/'model.onnx';baseline=ROOT/'release'/'gen05'/'model.onnx'
    assert sha(candidate)=='123467a930e0598c464ae7694800ad565aeac71680b88dafc5f014bfe7cf50ac'
    assert sha(baseline)=='af45a5f85894e694f7ddab47811c4fb45b9f9536345d78b0f2fa3ef06c99c91e'
    assert sha(Path(summary['a']))==sha(candidate) and sha(Path(summary['b']))==sha(baseline)
    files=sorted((ROOT/'env'/'colony_one').glob('*.py'))+[ROOT/'env'/'colony_one'/'scene.json']
    engine={str(p.relative_to(ROOT)):sha(p) for p in files}
    binding={str(p):sha(p) for p in [source,candidate,baseline,Path(summary['a']),Path(summary['b'])]}
    seen=set();checks=[]
    for row in rows:
        key=row['seed'],row['a_cyan'];assert key not in seen;seen.add(key)
        env=ColonyOneEnv()
        for team,fr,to in row['history']:
            assert env.current_team.startswith(team)
            aa=[a for a in env.legal_actions() if list(a.from_pos)==fr and list(a.to_pos)==to]
            assert len(aa)==1;env.step(aa[0])
        assert env.terminated and env.winner==row['winner'] and env.round_number==row['round']
        assert env.action==row['actions']==len(row['history'])+1
        assert row['a_won']==(env.winner==('CYAN' if row['a_cyan'] else 'MAGENTA'))
        checks.append({'seed':row['seed'],'a_cyan':row['a_cyan'],'actions':len(row['history']),'status':'PASS'})
    seeds=sorted({r['seed'] for r in rows});assert len(seeds)==200
    assert seen=={(s,c) for s in seeds for c in (True,False)}
    assert sum(r['a_won'] for r in rows)==summary['a_wins']
    assert {str(p):sha(p) for p in map(Path,binding)}==binding
    assert {str(p.relative_to(ROOT)):sha(p) for p in files}==engine
    report={'status':'PASS','histories':len(rows),'observed_summary':summary,'derived_paired_statistics':summarize(rows,400),
            'seed_min':min(seeds),'seed_max':max(seeds),'unique_seed_pairs':len(seeds),
            'source_model_hashes':binding,'engine_file_hashes':engine,
            'engine_sha256':hashlib.sha256(json.dumps(engine,sort_keys=True).encode()).hexdigest(),
            'audit_sha256':sha(__file__),'checks':checks,
            'scope':'Existing histories replayed legally; no new MCTS/strength games. Original arena uses seed*2+color, not shared color RNG. Current checkpoint/release hashes agree; original log lacks run-time engine manifest.'}
    (HERE/'existing-gate-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','engine_file_hashes')}))

if __name__=='__main__':main()
