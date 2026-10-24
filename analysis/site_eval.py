"""Real-opponent evaluation: tournament history stats + value MSE vs true outcomes by game phase.

python analysis/site_eval.py   (writes analysis/site_history_eval.json)
Uses data/ext_site_hist_v1 (28 complete tournament games replayed exactly by the engine);
no model listed here was trained on these games.
"""
import collections, json
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
ort.set_default_logger_severity(3)
hist = json.load(open(ROOT / 'data/_raw_site_hist/tournament-matches.json'))
games = [m for m in hist if m.get('protocol') and 'WINS' in m['protocol']]
rows = []
for m in games:
    p = m['protocol']
    winner = 'CYAN' if 'CYAN WINS' in p else 'MAGENTA'
    plies = sum(1 for l in p.splitlines() if l.strip()[:1].isdigit())
    rows.append(dict(id=m['match_id'], side=m['you'], opponent=m['opponent'] or '(unnamed)', win=winner == m['you'], plies=plies))
stats = {'games': len(rows), 'wins': sum(r['win'] for r in rows),
         'by_side': {s: f"{sum(r['win'] for r in rows if r['side'] == s)}/{sum(r['side'] == s for r in rows)}" for s in ('CYAN', 'MAGENTA')},
         'mean_plies_win': float(np.mean([r['plies'] for r in rows if r['win']])),
         'mean_plies_loss': float(np.mean([r['plies'] for r in rows if not r['win']]))}
by_opp = collections.defaultdict(lambda: [0, 0])
for r in rows:
    by_opp[r['opponent']][0] += r['win']; by_opp[r['opponent']][1] += 1
stats['by_opponent'] = {k: f'{w}/{n}' for k, (w, n) in by_opp.items()}

d = np.load(ROOT / 'data/ext_site_hist_v1/chunk0000.npz')
obs, z = d['obs'].astype(np.float32), d['z']
lock = obs[:, 12].sum((1, 2))
phase = np.where(lock == 0, 'r1-8', np.where(lock <= 28, 'r9-16', 'r17+'))
o = ort.SessionOptions(); o.intra_op_num_threads = 4
mse = {}
for name, path in [('gen07', 'ckpt/gen07-endgame.onnx'), ('gen09', 'release/gen09/model.onnx'), ('gen10', 'ckpt/gen10-endgame.onnx'),
                   ('gen12', 'release/gen12/model.onnx'), ('E5', 'release/E5/model.onnx')]:
    if not (ROOT / path).exists():
        continue
    s = ort.InferenceSession(str(ROOT / path), o, providers=['CPUExecutionProvider'])
    v = np.concatenate([s.run(None, {'observation': obs[i:i + 256]})[1] for i in range(0, len(obs), 256)])
    e = (v - z) ** 2
    mse[name] = {'all': round(float(e.mean()), 3), **{p: round(float(e[phase == p].mean()), 3) for p in ('r1-8', 'r9-16', 'r17+')}}
out = {'tournament': stats, 'positions': int(len(z)), 'positions_by_phase': {p: int((phase == p).sum()) for p in ('r1-8', 'r9-16', 'r17+')},
       'value_mse_vs_true_outcome': mse}
(ROOT / 'analysis/site_history_eval.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
