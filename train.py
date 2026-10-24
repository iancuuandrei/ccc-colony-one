"""Train the policy/value net on self-play chunks with 8-fold board symmetry.

python train.py --data data/gen00 data/gen01 --init ckpt/g00.pt --out ckpt/g01.pt --steps 6000
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from net import Net, export

def sq_maps():
    maps = []
    for t in range(8):
        m = np.zeros(64, np.int64)
        for s in range(64):
            r, c = divmod(s, 8)
            for _ in range(t % 4):
                r, c = c, 7 - r          # rotate 90
            if t >= 4:
                c = 7 - c                # mirror
            m[s] = r * 8 + c
        maps.append(m)
    return np.stack(maps)

def chunk_files(dirs, holdout_last):
    files = [f for d in dirs for f in sorted(Path(d).glob('chunk*.npz'))]
    if holdout_last:
        last = sorted(Path(dirs[-1]).glob('chunk*.npz'))[-1]
        return [f for f in files if f != last], [last]
    return files, []

KEYS = ('obs', 'pidx', 'pval', 'pval_g', 'z', 'q', 'has_pol', 'q_valid', 'has_value', 'has_length', 'left')

def apply_overrides(d, files_rows):
    """Codex authoritative policy overrides: exact float16 obs SHA256 -> certified pidx/pval,
    applied to rows from the folders named in each policy-overrides.json."""
    import hashlib
    ovs = {}
    for f, _, _ in files_rows:
        ov = Path(f).parent / 'policy-overrides.json'
        if ov.exists():
            spec = json.loads(ov.read_text())
            for e in spec['entries']:
                ovs[e['observation_float16_sha256']] = (spec['apply_to_source_folders'], e['replacement'])
    if not ovs:
        return 0
    n = 0
    for f, lo, hi in files_rows:
        folder = 'data/' + Path(f).parent.name
        for r in range(lo, hi):
            h = hashlib.sha256(np.ascontiguousarray(d['obs'][r].astype(np.float16)).tobytes()).hexdigest()
            hit = ovs.get(h)
            if hit and folder in hit[0]:
                rep = hit[1]
                d['pidx'][r] = np.asarray(rep['pidx'], d['pidx'].dtype); d['pval'][r] = np.asarray(rep['pval'], d['pval'].dtype)
                d['pval_g'][r] = np.asarray(rep.get('pval_g', rep['pval']), d['pval_g'].dtype)   # certified target for every mix
                d['has_pol'][r] = True; d['has_value'][r] = False; d['q_valid'][r] = False
                n += 1
    return n

def load(files):
    parts = []
    for f in files:
        z = np.load(f)
        d = {k: z[k] for k in z.files if k in KEYS}
        m = len(d['z'])
        if 'q_valid' not in d:  # legacy chunks: Q may come from a pool opponent's search
            d['q_valid'] = np.zeros(m, bool)
        if 'pval_g' not in d:     # older chunks: no Gumbel target -> identical to visits
            d['pval_g'] = d['pval'].copy()
        if 'has_value' not in d:  # policy-only rows (e.g. UNKNOWN-value curriculum) set this False
            d['has_value'] = np.ones(m, bool)
        d['is_ext'] = np.full(m, Path(f).parent.name.startswith('ext_'), bool)
        if 'has_length' not in d:   # real self-play has true lengths; external rows only when declared
            d['has_length'] = np.full(m, not d['is_ext'][0], bool)
        # Next-mover policy target (KataGo opponent-policy aux): row i -> row i+1 of the same game.
        # If the mover changes, map indices through the 180-degree canonical rotation (s -> 63 - s).
        npidx = np.full_like(d['pidx'], -1); npval = np.zeros_like(d['pval']); has_n = np.zeros(m, bool)
        if m > 1 and not d['is_ext'][0]:
            cont = d['left'][:-1] > 1
            side = d['obs'][:, 15, 0, 0] > 0.5
            nxt_pidx = d['pidx'][1:].astype(np.int32)
            flip = side[:-1] != side[1:]
            fr, to = nxt_pidx // 64, nxt_pidx % 64
            rot = (63 - fr) * 64 + (63 - to)
            mapped = np.where((nxt_pidx >= 0) & flip[:, None], rot, nxt_pidx)
            mapped = np.where(nxt_pidx >= 0, mapped, -1)
            sel = cont & d['has_pol'][1:]
            npidx[:-1][sel] = mapped[sel].astype(npidx.dtype)
            npval[:-1][sel] = d['pval'][1:][sel]
            has_n[:-1] = sel
        d['npidx'], d['npval'], d['has_npol'] = npidx, npval, has_n
        parts.append(d)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    rows, lo = [], 0
    for f, p in zip(files, parts):
        m = len(p['z']); rows.append((f, lo, lo + m)); lo += m
    # Only rows of folders that some override file targets are hashed (cheap).
    targets = set()
    for f in files:
        ov = Path(f).parent / 'policy-overrides.json'
        if ov.exists():
            targets.update(json.loads(ov.read_text())['apply_to_source_folders'])
    sel = [r for r in rows if 'data/' + Path(r[0]).parent.name in targets]
    ovf = {Path(f).parent / 'policy-overrides.json' for f in files if (Path(f).parent / 'policy-overrides.json').exists()}
    if sel and ovf:
        out['_overrides_applied'] = np.array([apply_overrides(out, sel + [(str(next(iter(ovf)).parent / 'x.npz'), 0, 0)])])
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', nargs='+', required=True)
    p.add_argument('--holdout-last', action='store_true', help='hold out the last chunk of the last data dir')
    p.add_argument('--init')
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--wd', type=float, default=3e-5)
    p.add_argument('--qmix', type=float, default=0.2)
    p.add_argument('--channels', type=int, default=48)
    p.add_argument('--blocks', type=int, default=5)
    p.add_argument('--emb', type=int, default=32)
    p.add_argument('--ext-frac', type=float, default=0.04, help='batch share drawn from data/ext_* rows')
    p.add_argument('--late-weight', type=float, default=1.0, help='sampling weight of rounds 9-16 (x2 for 17+); site analysis: late-game values overfit')
    p.add_argument('--policy-mix', type=float, default=0.0, help='share of the Gumbel completed-Q target in the policy target')
    p.add_argument('--opp-aux', type=float, default=0.0, help='weight of next-mover policy aux loss (KataGo: 0.15)')
    p.add_argument('--surprise', type=float, default=0.0, help='policy-surprise weighting share (KataGo-style, e.g. 0.5)')
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_files, hold_files = chunk_files(a.data, a.holdout_last)
    D = load(train_files)
    n = len(D['z'])
    T = lambda k, dt: torch.tensor(D[k]).to(dev, dt)
    if a.policy_mix > 0:
        D['pval'] = ((1 - a.policy_mix) * D['pval'].astype(np.float32) + a.policy_mix * D['pval_g'].astype(np.float32)).astype(np.float16)
    obs = T('obs', torch.float16); pidx = T('pidx', torch.int16); pval = T('pval', torch.float16)
    npidx = T('npidx', torch.int16); npval = T('npval', torch.float16); hasn = T('has_npol', torch.float32)
    z = T('z', torch.float32); q = T('q', torch.float32); qv = T('q_valid', torch.float32); has = T('has_pol', torch.bool); left = T('left', torch.float32)
    hv = T('has_value', torch.float32); hl_ = T('has_length', torch.float32)
    ext_idx = torch.nonzero(T('is_ext', torch.bool)).reshape(-1)
    own_idx = torch.nonzero(~T('is_ext', torch.bool)).reshape(-1)
    n_ext = int(round(a.batch * a.ext_frac)) if len(ext_idx) else 0
    # Phase-balanced sampling of self-play rows: locked squares 0 -> rounds 1-8, 28 -> 9-16, >=48 -> 17+.
    nlock = obs[:, 12][own_idx].float().sum((1, 2))
    own_w = torch.where(nlock < 1, 1.0, torch.where(nlock < 40, a.late_weight, 2 * a.late_weight)).to(dev)
    print(json.dumps({'phase_rows': [int((nlock < 1).sum()), int(((nlock >= 1) & (nlock < 40)).sum()), int((nlock >= 40).sum())],
                      'late_weight': a.late_weight}), flush=True)
    print(json.dumps({'rows': n, 'ext_rows': int(len(ext_idx)), 'ext_per_batch': n_ext}), flush=True)
    H = None
    if hold_files:
        Dh = load(hold_files)
        H = {k: torch.tensor(Dh[k]).to(dev) for k in Dh}
    maps = torch.tensor(sq_maps(), device=dev)
    if a.init:
        ck = torch.load(a.init, map_location='cpu')
        net = Net(**ck['cfg']); net.load_state_dict(ck['state'], strict=False)
    else:
        net = Net(a.channels, a.blocks, a.emb)
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)

    def map_idx(pi, m):
        pi = pi.long()
        valid = pi >= 0
        fr, to = pi.clamp(min=0) // 64, pi.clamp(min=0) % 64
        npi = torch.gather(m, 1, fr) * 64 + torch.gather(m, 1, to)
        return torch.where(valid, npi, torch.full_like(npi, -1))

    def augment(o, pi, t):
        b = o.shape[0]
        m = maps[t]                                    # (B,64)
        flat = o.reshape(b, 16, 64).float()
        out = torch.empty_like(flat)
        out.scatter_(2, m[:, None, :].expand(b, 16, 64), flat)
        return out.reshape(b, 16, 8, 8), map_idx(pi, m)

    def legal_kl(logits, pi, pv):
        valid = pi >= 0
        lg = torch.gather(logits, 1, pi.clamp(min=0)).masked_fill(~valid, -1e9)
        logp = F.log_softmax(lg, 1)
        return -(pv * logp).sum(1) + (pv * torch.log(pv.clamp(min=1e-8))).sum(1)

    def losses(o, pi, pv, zz, qq, hh, ll, qvv, hvv=None, train=True, npi=None, npv=None, hn=None, hlv=None):
        if hvv is None:
            hvv = torch.ones_like(zz)
        if hlv is None:
            hlv = hvv
        pi = pi.long(); pv = pv.float()
        use_next = train and a.opp_aux > 0 and npi is not None
        out = net.forward_train(o, with_next=use_next)
        logits, v, aux = out[0], out[1], out[2]
        kl = legal_kl(logits, pi, pv)
        w = hh
        if train and a.surprise > 0:
            # KataGo policy-surprise weighting: upweight rows where the search disagreed with the net.
            ks = (kl.detach() * hh)
            w = hh * ((1 - a.surprise) + a.surprise * ks / ks.sum().clamp(min=1e-6) * hh.sum().clamp(min=1))
        pl = (kl * w).sum() / hh.sum().clamp(min=1)                 # KL to search policy
        if use_next:
            nk = legal_kl(out[3], npi.long(), npv.float())
            pl = pl + a.opp_aux * (nk * hn).sum() / hn.sum().clamp(min=1)
        mix = a.qmix * qvv
        tgt = (1 - mix) * zz + mix * qq
        vl = (((v - tgt) ** 2) * hvv).sum() / hvv.sum().clamp(min=1)
        al = ((torch.tanh(aux[:, 0]) - qq) ** 2 * qvv).sum() / qvv.sum().clamp(min=1)             + 0.05 * (((aux[:, 1] - torch.log(ll.clamp(min=1))) ** 2) * hlv).sum() / hlv.sum().clamp(min=1)
        return pl, vl, al

    t0 = time.time(); log = []
    for step in range(a.steps):
        net.train()
        if a.late_weight != 1.0:
            i = own_idx[torch.multinomial(own_w, a.batch - n_ext, replacement=True)]
        else:
            i = own_idx[torch.randint(0, len(own_idx), (a.batch - n_ext,), device=dev)]
        if n_ext:
            i = torch.cat([i, ext_idx[torch.randint(0, len(ext_idx), (n_ext,), device=dev)]])
        t = torch.randint(0, 8, (a.batch,), device=dev)
        o, pi = augment(obs[i], pidx[i], t)
        npi = map_idx(npidx[i], maps[t]) if a.opp_aux > 0 else None
        pl, vl, al = losses(o, pi, pval[i], z[i], q[i], has[i].float(), left[i], qv[i], hv[i],
                            npi=npi, npv=npval[i] if npi is not None else None, hn=hasn[i], hlv=hl_[i])
        loss = pl + 1.5 * vl + 0.25 * al
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step()
        if (step + 1) % 500 == 0 or step + 1 == a.steps:
            rec = {'step': step + 1, 'policy_kl': float(pl), 'value_mse': float(vl), 'aux': float(al),
                   'sec': round(time.time() - t0, 1)}
            if H is not None:
                net.eval()
                with torch.no_grad():
                    hl = [0.0, 0.0]; m = len(H['z']); cnt = 0; pcnt = 0
                    for s in range(0, m, 4096):
                        sl = slice(s, s + 4096)
                        o = H['obs'][sl].float(); w = H['has_pol'][sl].float()
                        pl2, vl2, _ = losses(o, H['pidx'][sl].long(), H['pval'][sl].float(), H['z'][sl],
                                             H['q'][sl], w, H['left'][sl], H['q_valid'][sl].float(), H['has_value'][sl].float(), train=False,
                                             hlv=H['has_length'][sl].float())
                        k = len(H['z'][sl]); kp = float(w.sum())
                        hl[0] += float(pl2) * kp; hl[1] += float(vl2) * k; cnt += k; pcnt += kp
                rec.update(hold_policy_kl=hl[0] / max(pcnt, 1), hold_value_mse=hl[1] / cnt)
            log.append(rec); print(json.dumps(rec), flush=True)
        if (step + 1) % 1000 == 0:  # recoverable state (true resume, not just warm start)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'cfg': net.cfg, 'state': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                        'step': step + 1, 'rng': torch.cuda.get_rng_state() if dev == 'cuda' else torch.get_rng_state(),
                        'args': vars(a)}, Path(a.out).with_suffix('.resume.pt'))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in net.state_dict().items()}
    export(net, out.with_suffix('.onnx'))          # ONNX first: a .pt means a complete export
    torch.save({'cfg': net.cfg, 'state': state, 'log': log, 'args': vars(a), 'positions': n,
                'train_files': [str(f) for f in train_files], 'holdout_files': [str(f) for f in hold_files]}, out)
    print(json.dumps({'saved': str(out), 'onnx': str(out.with_suffix('.onnx')), 'positions': n}))

if __name__ == '__main__':
    main()
