"""Archive hard states for search-control restarts (Go-Exploit / RGSC-style).

python build_archive.py --dirs 12 --top 30000
Regret proxy: |z - rootQ| on the learner's own searched positions (the search was most
wrong about the outcome), excluding the last few actions; plus every real-site position.
Writes archive/hard_states.npz (obs only + provenance) atomically.
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dirs', type=int, default=12)
    p.add_argument('--top', type=int, default=30000)
    p.add_argument('--min-regret', type=float, default=0.8)
    a = p.parse_args()
    dirs = sorted([d for d in (ROOT / 'data').iterdir() if (d / 'summary.json').exists() and not d.name.startswith('ext_')],
                  key=lambda d: (d / 'summary.json').stat().st_mtime)[-a.dirs:]
    obs, reg, unc = [], [], []
    for d in dirs:
        for f in sorted(d.glob('chunk*.npz')):
            z = np.load(f)
            if 'q_valid' not in z.files:
                continue
            r = np.abs(z['z'] - z['q'])
            keep = z['q_valid'] & (z['left'] > 3) & (r >= a.min_regret)
            obs.append(z['obs'][keep]); reg.append(r[keep])
            # Uncertainty-guided branching (NeurIPS 2025): LCR = 2v(1-v) > 0.48 with v = (q+1)/2  <=>  |q| < 0.2
            ukeep = z['q_valid'] & (z['left'] > 3) & (np.abs(z['q']) < 0.2)
            unc.append(z['obs'][ukeep])
    obs = np.concatenate(obs); reg = np.concatenate(reg)
    order = np.argsort(-reg)[:a.top]
    hard = obs[order]
    site = []
    site_dirs = [d for d in sorted((ROOT / 'data').glob('ext_site*')) if not d.name.endswith('_rs')]
    for f in sorted(f for d in site_dirs for f in d.glob('chunk*.npz')):
        z = np.load(f)
        if 'left' in z.files and len(z['left']):
            site.append(z['obs'][z['left'] > 3])
    site = np.concatenate(site) if site else np.zeros((0, 16, 8, 8), np.float16)
    # Late-game states (rounds 9+): self-play rarely reaches them and site losses happen there.
    late = []
    for d in dirs:
        for f in sorted(d.glob('chunk*.npz')):
            z2 = np.load(f)
            lk = z2['obs'][:, 12].astype(np.float32).sum((1, 2))
            late.append(z2['obs'][(lk >= 1) & (z2['left'] > 3)])
    late = np.concatenate(late) if late else np.zeros((0, 16, 8, 8), np.float16)
    late = late[np.random.default_rng(1).permutation(len(late))[:a.top // 2]]
    unc = np.concatenate(unc) if unc else np.zeros((0, 16, 8, 8), np.float16)
    rng = np.random.default_rng(len(unc))
    unc = unc[rng.permutation(len(unc))[:a.top // 2]]
    allobs = np.concatenate([hard, unc.astype(hard.dtype), late.astype(hard.dtype), site.astype(hard.dtype)])
    out = ROOT / 'archive'; out.mkdir(exist_ok=True)
    tmp = out / 'hard_states.tmp.npz'
    np.savez_compressed(tmp, obs=allobs)
    tmp.replace(out / 'hard_states.npz')
    meta = {'dirs': [d.name for d in dirs], 'regret_states': int(len(hard)), 'site_states': int(len(site)), 'uncertain_states': int(len(unc)), 'late_states': int(len(late)),
            'min_regret': a.min_regret, 'mean_regret_kept': float(reg[order].mean()) if len(order) else 0.0}
    (out / 'hard_states.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))

if __name__ == '__main__':
    main()
