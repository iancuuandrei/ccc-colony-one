"""Continuous self-play actor: always plays the current best model, in batches.

python actor.py --best-file loop_state.json --prefix sp --workers 24 --batch 400
python actor.py --best-file models/current.txt --prefix vps --workers 15 --batch 300   (VPS)
Each batch lands in data/<prefix>_<modeltag>_<unixtime>/ with summary.json written last.
Worker count is re-read from cpu_budget.json (key actor_workers) before every batch.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent

def current_best(path: Path) -> str:
    txt = path.read_text().strip()
    return json.loads(txt)['best'] if path.suffix == '.json' else txt

def extra_args():
    try:
        return [str(x) for x in json.loads((ROOT / 'cpu_budget.json').read_text()).get('selfplay_args', [])]
    except Exception:
        return []

def workers(default):
    try:
        return int(json.loads((ROOT / 'cpu_budget.json').read_text()).get('actor_workers', default))
    except Exception:
        return default

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--best-file', required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--batch', type=int, default=400)
    p.add_argument('--pool', nargs='*', default=None)
    p.add_argument('--pool-prob', type=float, default=0.2)
    a = p.parse_args()
    best_file = ROOT / a.best_file
    while True:
        if (ROOT / f'STOP_{a.prefix}').exists():
            print('stop file present, exiting', flush=True); return
        if workers(a.workers) <= 0:   # explicit pause (e.g. for an idle-machine qualification)
            time.sleep(20); continue
        model = current_best(best_file)
        tag = Path(model).stem
        stamp = int(time.time())
        out = ROOT / 'data' / f'{a.prefix}_{tag}_{stamp}'
        pool = a.pool
        if pool is None:
            st = json.loads((ROOT / 'loop_state.json').read_text())
            pool = [x for x in st['pool'] if Path(x).stem != tag][-4:]
        cmd = [sys.executable, 'selfplay.py', '--model', model, '--games', a.batch,
               '--workers', workers(a.workers), '--seed0', stamp * 1000 % 2_000_000_000, '--out', out,
               '--pool-prob', a.pool_prob]
        if pool:
            cmd += ['--pool', *pool]
        cmd += extra_args()
        t = time.time()
        r = subprocess.run(list(map(str, cmd)), cwd=ROOT, capture_output=True, text=True)
        print(json.dumps({'out': out.name, 'model': model, 'rc': r.returncode, 'sec': round(time.time() - t, 1),
                          'tail': r.stdout.strip().splitlines()[-1][:200] if r.stdout.strip() else r.stderr[-300:]}),
              flush=True)
        if r.returncode != 0:
            time.sleep(30)

if __name__ == '__main__':
    main()
