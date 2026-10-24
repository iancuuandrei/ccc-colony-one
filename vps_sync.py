"""Keep the VPS actor fed with the current best model and pull its finished batches.

python vps_sync.py           (runs forever; STOP_vps_sync to exit)
"""
from __future__ import annotations
import json, subprocess, time
from pathlib import Path

ROOT = Path(__file__).parent
HOST = 'corpus-staging'
REMOTE = 'ccc-colony'

def ssh(cmd, **kw):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST, cmd],
                          capture_output=True, text=True, timeout=600, **kw)

def push_model(path: Path):
    with open(path, 'rb') as f:
        r = subprocess.run(['ssh', '-o', 'BatchMode=yes', HOST,
                            f'cat > {REMOTE}/models/{path.name}.tmp && mv {REMOTE}/models/{path.name}.tmp {REMOTE}/models/{path.name}'
                            f' && echo models/{path.name} > {REMOTE}/models/current.txt'],
                           stdin=f, capture_output=True, timeout=300)
    return r.returncode == 0

def pull(name: str):
    dst = ROOT / 'data' / name
    tmp = ROOT / 'data' / f'.incoming_{name}'
    tmp.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', HOST, f'tar -C {REMOTE}/data/{name} -cf - .'],
                         stdout=subprocess.PIPE)
    r = subprocess.run(['tar', '-C', str(tmp), '-xf', '-'], stdin=p.stdout)
    p.wait()
    if r.returncode == 0 and p.returncode == 0 and (tmp / 'summary.json').exists():
        summary = tmp / 'summary.json'
        data = summary.read_bytes(); summary.unlink()
        tmp.rename(dst)
        (dst / 'summary.json').write_bytes(data)       # summary last: learner sees only complete dirs
        ssh(f'mkdir -p {REMOTE}/pulled && mv {REMOTE}/data/{name} {REMOTE}/pulled/')
        return True
    return False

def push_archive(path: Path):
    with open(path, 'rb') as f:
        r = subprocess.run(['ssh', '-o', 'BatchMode=yes', HOST,
                            f'mkdir -p {REMOTE}/archive && cat > {REMOTE}/archive/hard_states.tmp.npz && '
                            f'mv {REMOTE}/archive/hard_states.tmp.npz {REMOTE}/archive/hard_states.npz'],
                           stdin=f, capture_output=True, timeout=300)
    return r.returncode == 0

def main():
    pushed = None
    archive_mtime = 0.0
    while not (ROOT / 'STOP_vps_sync').exists():
        try:
            best = json.loads((ROOT / 'loop_state.json').read_text())['best']
            if best != pushed and push_model(ROOT / best):
                pushed = best; print(json.dumps({'pushed': best}), flush=True)
            arc = ROOT / 'archive' / 'hard_states.npz'
            if arc.exists() and arc.stat().st_mtime != archive_mtime and push_archive(arc):
                archive_mtime = arc.stat().st_mtime; print(json.dumps({'pushed_archive': archive_mtime}), flush=True)
            r = ssh(f'cd {REMOTE}/data && for d in vps_*; do [ -f "$d/summary.json" ] && echo "$d"; done')
            for name in [x for x in r.stdout.split() if x.startswith('vps_')]:
                ok = pull(name)
                print(json.dumps({'pulled': name, 'ok': ok}), flush=True)
        except Exception as e:  # keep syncing through transient network errors
            print(json.dumps({'error': str(e)[:200]}), flush=True)
        time.sleep(60)

if __name__ == '__main__':
    main()
