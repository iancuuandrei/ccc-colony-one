"""Asynchronous learner: train on the freshest self-play while actors keep playing.

python learner.py --window-pos 700000 --min-new 120000
Each cycle: wait for >= min-new fresh positions -> train (GPU, WSL) on the newest
window (by completion time) + data/ext_* -> wrap with the proved endgame solver ->
policy check -> gate vs best -> promote. Actors pick up a promoted best at their next batch.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable
STATE = ROOT / 'loop_state.json'
WSL_PY = '/mnt/d/dev/CCC-OMNI/workspace/colony-one-level4/wpy.sh'
TACTICAL = 'codex/tactical/model-tactical-v2.onnx'

def budget(key, default):
    try:
        return int(json.loads((ROOT / 'cpu_budget.json').read_text()).get(key, default))
    except Exception:
        return default

def run(cmd, log, tries=3):
    for _ in range(tries):
        with open(log, 'a') as f:
            f.write(f'\n$ {" ".join(map(str, cmd))}\n'); f.flush()
            r = subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode == 0:
            break
        time.sleep(10)          # transient worker death (BrokenProcessPool) -> retry
    else:
        raise RuntimeError(f'command failed {tries}x: {cmd[:3]}')
    lines = [l for l in Path(log).read_text(errors='replace').splitlines() if l.startswith('{')]
    return json.loads(lines[-1])

def cfg():
    try:
        return json.loads((ROOT / 'learner_cfg.json').read_text())
    except Exception:
        return {}

def train_async(tag, data, init, steps, extra, log):
    cmd = ['wsl', '-e', 'bash', WSL_PY, 'train.py', '--data', *data, '--init', init, '--out', f'ckpt/{tag}.pt',
           '--steps', steps, '--lr', 1.5e-3, '--holdout-last', *extra]
    f = open(log, 'a'); f.write('\n$ ' + ' '.join(map(str, cmd)) + '\n'); f.flush()
    return subprocess.Popen(list(map(str, cmd)), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT), f

def finished_dirs():
    out = []
    for d in (ROOT / 'data').iterdir():
        s = d / 'summary.json'
        if d.is_dir() and not d.name.startswith('ext_') and s.exists():
            out.append((s.stat().st_mtime, d, json.loads(s.read_text())['positions']))
    return sorted(out)

def main():
    while not (ROOT / 'STOP_learner').exists():
        try:
            cycle_main()
            return
        except Exception as e:   # transient CUDA/worker failures: log and retry the cycle
            print(json.dumps({'cycle_error': str(e)[:300], 'retry_in_s': 30}), flush=True)
            time.sleep(30)

def cycle_main():
    p = argparse.ArgumentParser()
    p.add_argument('--window-pos', type=int, default=700000)
    p.add_argument('--min-new', type=int, default=120000)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--gate-seeds', type=int, default=200)
    p.add_argument('--gate', type=float, default=0.55)
    p.add_argument('--start', type=int, default=3)
    a = p.parse_args()
    st = json.loads(STATE.read_text())
    g = max(a.start, max([h['gen'] for h in st['history']] + [0]) + 1)   # resume numbering after retries
    last_mtime = st.get('learner_last_mtime', 0.0)
    while not (ROOT / 'STOP_learner').exists():
        dirs = finished_dirs()
        fresh = sum(n for m, _, n in dirs if m > last_mtime)
        if fresh < a.min_new:
            time.sleep(30); continue
        window, tot = [], 0
        for m, d, n in reversed(dirs):
            window.append(d); tot += n
            if tot >= a.window_pos:
                break
        window = list(reversed(window))                     # newest last -> holdout
        ext = [d for d in sorted((ROOT / 'data').glob('ext_*')) if any(d.glob('chunk*.npz'))]
        data = [f'data/{d.name}' for d in ext + window]
        tag = f'gen{g:02d}'; log = ROOT / 'logs' / f'{tag}.txt'
        rec = {'gen': g, 'best_before': st['best'], 'positions': tot, 'fresh': fresh,
               'dirs': [d.name for d in window], 'ext': [d.name for d in ext], 't0': time.time()}
        newest = dirs[-1][0]
        c = cfg()
        main_extra = [str(x) for x in c.get('main_args', [])]
        var_extra = c.get('variant_args')
        jobs = {}
        if not (ROOT / 'ckpt' / f'{tag}.pt').exists():
            jobs[tag] = train_async(tag, data, st['latest_ckpt'], a.steps, main_extra, log)
        if var_extra is not None and not (ROOT / 'ckpt' / f'{tag}v.pt').exists():
            jobs[tag + 'v'] = train_async(tag + 'v', data, st['latest_ckpt'], a.steps, [str(x) for x in var_extra],
                                          ROOT / 'logs' / f'{tag}v.txt')
        for name, (proc, fh) in jobs.items():
            rc = proc.wait(); fh.close()
            if rc != 0:
                raise RuntimeError(f'training {name} failed rc={rc}')
        rec['main_args'] = main_extra; rec['variant_args'] = var_extra
        results = {}
        names = [tag] + ([tag + 'v'] if var_extra is not None else [])
        for name in names:
            base = str(ROOT / 'ckpt' / f'{name}.onnx'); cand = str(ROOT / 'ckpt' / f'{name}-endgame.onnx')
            try:
                run([PY, 'codex/endgame/build_endgame.py', '--base', base, '--output', cand], log)
                run([PY, 'check_onnx.py', cand], log, tries=1)
            except RuntimeError:
                rec.setdefault('rejected', []).append(name)   # failed server policy: never gate/promote it
                continue
            gate = run([PY, 'arena.py', cand, st['best'], '--games', a.gate_seeds, '--workers', budget('arena_workers', 20),
                        '--start', 900000 + g * 1000, '--out', ROOT / 'logs' / f'{name}-gate.json'], log)
            results[name] = (gate['a_rate'], cand)
        rec['gates'] = {k: v[0] for k, v in results.items()}
        if not results:
            raise RuntimeError('no candidate passed the policy check')
        best_name = max(results, key=lambda k: results[k][0])
        rate, cand = results[best_name]
        rec.update(gate=rate, cand=cand, chosen=best_name)
        st = json.loads(STATE.read_text())                  # re-read: others may have edited the pool
        st['latest_ckpt'] = f'ckpt/{best_name}.pt'
        rec['promoted'] = rate >= float(c.get('gate', a.gate))
        if rec['promoted']:
            st['best'] = cand; st['pool'].append(cand)
        last_mtime = newest
        st['learner_last_mtime'] = last_mtime
        rec['seconds'] = round(time.time() - rec.pop('t0'), 1)
        st['history'].append(rec)
        STATE.write_text(json.dumps(st, indent=1))
        print(json.dumps(rec), flush=True)
        g += 1

if __name__ == '__main__':
    main()
