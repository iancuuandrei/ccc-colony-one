"""Re-search value-only positions (e.g. real site games) to add policy targets.

python research_ext.py --src data/ext_site_v1 --out data/ext_site_v1_rs --model ckpt/gen02-endgame.onnx --sims 800
Writes policy-only rows (has_value=False, so outcomes are not double counted).
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'env'))
K = 128
_ev = {}

def round_of(o):
    locked = int(o[12].sum())
    done = {0: 0, 28: 1, 48: 2, 60: 3}.get(locked, 0)
    nxt = (8, 16, 24, 26)[min(done, 3)]
    return max(1, int(round(nxt - 8 * (1 - float(o[14, 0, 0])))))

def work(job):
    from colony_one import constants, mcts
    from colony_one.board import Board
    from colony_one.onnx_runner import OnnxEvaluator
    model, obs, sims = job
    if model not in _ev:
        _ev[model] = OnnxEvaluator(model)
    ev = _ev[model]
    cfg = mcts.MCTSConfig(simulations=sims, add_root_noise=False)
    out = []
    for o in obs:
        o = o.astype(np.float32)
        team = constants.MAGENTA if o[15, 0, 0] > 0.5 else constants.CYAN
        board = Board.from_observation(o, team)
        al = int(round(float(o[13, 0, 0]) * 2))
        root = mcts.run(board, team, al, round_of(o), ev, cfg, rng=np.random.default_rng(0))
        idx, cnt = mcts.visit_distribution(root)
        cnt = np.asarray(cnt, np.float64)
        q = float(sum(root.W.values()) / max(1, sum(root.N.values())))
        out.append((idx[:K], (cnt / max(cnt.sum(), 1))[:K], q))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True); p.add_argument('--out', required=True)
    p.add_argument('--model', required=True); p.add_argument('--sims', type=int, default=800)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    obs = np.concatenate([np.load(f)['obs'] for f in sorted(Path(a.src).glob('chunk*.npz'))])
    left = np.concatenate([np.load(f)['left'] for f in sorted(Path(a.src).glob('chunk*.npz'))])
    model = str(Path(a.model).resolve())
    parts = np.array_split(np.arange(len(obs)), max(1, len(obs) // 40))
    t = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = [r for chunk in ex.map(work, [(model, obs[ix], a.sims) for ix in parts]) for r in chunk]
    m = len(res)
    pidx = np.full((m, K), -1, np.int16); pval = np.zeros((m, K), np.float16)
    for i, (ix, pv, _) in enumerate(res):
        pidx[i, :len(ix)] = ix; pval[i, :len(ix)] = pv
    q = np.array([r[2] for r in res], np.float32)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'chunk0000.npz', obs=obs.astype(np.float16), pidx=pidx, pval=pval,
                        z=q.copy(), q=q, has_pol=np.ones(m, bool), q_valid=np.ones(m, bool),
                        has_value=np.zeros(m, bool), left=left.astype(np.float32))
    (out / 'manifest.json').write_text(json.dumps({'source': a.src, 'model': a.model, 'sims': a.sims, 'rows': m,
                                                   'note': 'policy-only re-search; has_value=False',
                                                   'seconds': round(time.time() - t, 1)}, indent=1))
    print(json.dumps({'rows': m, 'seconds': round(time.time() - t, 1)}))

if __name__ == '__main__':
    main()
