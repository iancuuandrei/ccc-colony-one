"""Expert-iteration loop: self-play -> train -> gate -> promote.

python loop.py --start 1 --gens 20
State lives in loop_state.json (best model, pool, history) so the loop can resume.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable
STATE = ROOT / 'loop_state.json'
WSL_PY = '/mnt/d/dev/CCC-OMNI/workspace/colony-one-level4/wpy.sh'
TACTICAL = 'codex/tactical/model-tactical.onnx'

def run(cmd, log):
    with open(log, 'a') as f:
        f.write(f'\n$ {" ".join(map(str, cmd))}\n'); f.flush()
        subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    lines = [l for l in Path(log).read_text().splitlines() if l.startswith('{')]
    return json.loads(lines[-1])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, default=1)
    p.add_argument('--gens', type=int, default=20)
    p.add_argument('--games', type=int, default=5000)
    p.add_argument('--window', type=int, default=6)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--gate-seeds', type=int, default=200)
    p.add_argument('--gate', type=float, default=0.55)
    a = p.parse_args()
    st = json.loads(STATE.read_text())
    for g in range(a.start, a.start + a.gens):
        tag = f'gen{g:02d}'; log = ROOT / 'logs' / f'{tag}.txt'; log.parent.mkdir(exist_ok=True)
        rec = {'gen': g, 'best_before': st['best'], 't0': time.time()}
        data = ROOT / 'data' / tag
        if not (data / 'summary.json').exists():
            pool = [x for x in st['pool'] if x != st['best']][-4:]
            rec['selfplay'] = run([PY, 'selfplay.py', '--model', st['best'], '--pool', *pool, '--pool-prob', 0.2,
                                   '--games', a.games, '--workers', 28, '--seed0', g * 1_000_000, '--out', data], log)
        gens = sorted(d for d in (ROOT / 'data').glob('gen*') if (d / 'summary.json').exists() and d.name <= tag)
        window = [f'data/{d.name}' for d in gens[-a.window:]]
        ck = ROOT / 'ckpt' / f'{tag}.pt'
        if not ck.exists():
            # Training runs on the GPU inside WSL (prepared reference env), relative paths only.
            rec['train'] = run(['wsl', '-e', 'bash', WSL_PY, 'train.py', '--data', *window, '--init', st['latest_ckpt'],
                                '--out', f'ckpt/{tag}.pt',
                                '--steps', a.steps, '--lr', 1.5e-3, '--holdout-last'], log)
        cand = str(ck.with_suffix('.onnx'))
        rec['check'] = run([PY, 'check_onnx.py', cand], log)
        gate = run([PY, 'arena.py', cand, st['best'], '--games', a.gate_seeds, '--workers', 30,
                    '--start', 900000 + g * 1000, '--out', ROOT / 'logs' / f'{tag}-gate.json'], log)
        anchor = run([PY, 'arena.py', cand, TACTICAL, '--games', 100, '--workers', 30,
                      '--start', 800000 + g * 1000, '--out', ROOT / 'logs' / f'{tag}-tactical.json'], log)
        rec.update(gate=gate['a_rate'], anchor_tactical=anchor['a_rate'], cand=cand)
        st['latest_ckpt'] = f'ckpt/{tag}.pt'
        if gate['a_rate'] >= a.gate:
            st['best'] = cand; st['pool'].append(cand); rec['promoted'] = True
        else:
            rec['promoted'] = False
        rec['seconds'] = time.time() - rec.pop('t0')
        st['history'].append(rec)
        STATE.write_text(json.dumps(st, indent=1))
        print(json.dumps(rec), flush=True)

if __name__ == '__main__':
    main()
