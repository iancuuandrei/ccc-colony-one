"""Paired from-scratch training: attention net vs CNN control, identical data/batches/losses.

wsl -e bash .../wpy.sh arch_attn/train_arch.py --arch attn --out arch_attn/ckpt_attn.pt --data ... --holdout-last
Reuses train.py's loading (chunk_files, load, sq_maps) and copies its augmentation and losses
(policy KL to search policy over legal moves, value MSE to (1-qmix*qv)*z + qmix*qv*q, aux heads).
Batch sampling and symmetry choice come from a dedicated seeded generator, so both arms see the
same rows in the same order with the same augmentations.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import torch.nn.functional as F
from train import sq_maps, chunk_files, load      # noqa: E402
from net_attn import build, export, n_params       # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['attn', 'cnn'], required=True)
    p.add_argument('--data', nargs='+', required=True)
    p.add_argument('--holdout-last', action='store_true')
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--lr', type=float, default=1.5e-3)
    p.add_argument('--wd', type=float, default=3e-5)
    p.add_argument('--qmix', type=float, default=0.2)
    p.add_argument('--ext-frac', type=float, default=0.04)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--eval-every', type=int, default=500)
    # CNN control
    p.add_argument('--channels', type=int, default=48)
    p.add_argument('--blocks', type=int, default=5)
    p.add_argument('--emb', type=int, default=32)
    # attention
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--ffn', type=int, default=2)
    p.add_argument('--conv-layers', default='')
    p.add_argument('--no-pos', action='store_true')
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    g = torch.Generator(device=dev); g.manual_seed(a.seed + 1)

    train_files, hold_files = chunk_files(a.data, a.holdout_last)
    D = load(train_files)
    n = len(D['z'])
    T = lambda k, dt: torch.tensor(D[k]).to(dev, dt)
    obs = T('obs', torch.float16); pidx = T('pidx', torch.int16); pval = T('pval', torch.float16)
    z = T('z', torch.float32); q = T('q', torch.float32); qv = T('q_valid', torch.float32)
    has = T('has_pol', torch.bool); left = T('left', torch.float32); hv = T('has_value', torch.float32)
    ext_idx = torch.nonzero(T('is_ext', torch.bool)).reshape(-1)
    own_idx = torch.nonzero(~T('is_ext', torch.bool)).reshape(-1)
    n_ext = int(round(a.batch * a.ext_frac)) if len(ext_idx) else 0
    del D
    Dh = load(hold_files)
    H = {k: torch.tensor(Dh[k]).to(dev) for k in Dh}
    maps = torch.tensor(sq_maps(), device=dev)

    if a.arch == 'attn':
        cfg = dict(d=a.d, layers=a.layers, heads=a.heads, ffn=a.ffn, emb=a.emb, pos=not a.no_pos,
                   conv_layers=[int(c) for c in a.conv_layers.split(',') if c])
    else:
        cfg = dict(channels=a.channels, blocks=a.blocks, emb=a.emb)
    net = build(a.arch, cfg).to(dev)
    print(json.dumps({'arch': a.arch, 'cfg': cfg, 'params': n_params(net), 'rows': n, 'ext_rows': int(len(ext_idx)),
                      'ext_per_batch': n_ext, 'holdout_rows': int(len(H['z'])),
                      'holdout_files': [str(f) for f in hold_files]}), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)

    def map_idx(pi, m):
        pi = pi.long(); valid = pi >= 0
        fr, to = pi.clamp(min=0) // 64, pi.clamp(min=0) % 64
        npi = torch.gather(m, 1, fr) * 64 + torch.gather(m, 1, to)
        return torch.where(valid, npi, torch.full_like(npi, -1))

    def augment(o, pi, t):
        b = o.shape[0]; m = maps[t]
        flat = o.reshape(b, 16, 64).float(); out = torch.empty_like(flat)
        out.scatter_(2, m[:, None, :].expand(b, 16, 64), flat)
        return out.reshape(b, 16, 8, 8), map_idx(pi, m)

    def legal_kl(logits, pi, pv):
        valid = pi >= 0
        lg = torch.gather(logits, 1, pi.clamp(min=0)).masked_fill(~valid, -1e9)
        logp = F.log_softmax(lg, 1)
        return -(pv * logp).sum(1) + (pv * torch.log(pv.clamp(min=1e-8))).sum(1)

    def losses(o, pi, pv, zz, qq, hh, ll, qvv, hvv):
        pi = pi.long(); pv = pv.float()
        logits, v, aux = net.forward_train(o)
        kl = legal_kl(logits, pi, pv)
        pl = (kl * hh).sum() / hh.sum().clamp(min=1)
        mix = a.qmix * qvv
        tgt = (1 - mix) * zz + mix * qq
        vl = (((v - tgt) ** 2) * hvv).sum() / hvv.sum().clamp(min=1)
        al = ((torch.tanh(aux[:, 0]) - qq) ** 2 * qvv).sum() / qvv.sum().clamp(min=1) \
            + 0.05 * (((aux[:, 1] - torch.log(ll)) ** 2) * hvv).sum() / hvv.sum().clamp(min=1)
        return pl, vl, al

    def holdout():
        net.eval()
        with torch.no_grad():
            hl = [0.0, 0.0]; m = len(H['z']); cnt = 0; pcnt = 0
            for s in range(0, m, 4096):
                sl = slice(s, s + 4096)
                o = H['obs'][sl].float(); w = H['has_pol'][sl].float()
                pl2, vl2, _ = losses(o, H['pidx'][sl].long(), H['pval'][sl].float(), H['z'][sl], H['q'][sl], w,
                                     H['left'][sl], H['q_valid'][sl].float(), H['has_value'][sl].float())
                k = len(H['z'][sl]); kp = float(w.sum())
                hl[0] += float(pl2) * kp; hl[1] += float(vl2) * k; cnt += k; pcnt += kp
        return hl[0] / max(pcnt, 1), hl[1] / cnt

    t0 = time.time(); log = []
    for step in range(a.steps):
        net.train()
        i = own_idx[torch.randint(0, len(own_idx), (a.batch - n_ext,), device=dev, generator=g)]
        if n_ext:
            i = torch.cat([i, ext_idx[torch.randint(0, len(ext_idx), (n_ext,), device=dev, generator=g)]])
        t = torch.randint(0, 8, (a.batch,), device=dev, generator=g)
        o, pi = augment(obs[i], pidx[i], t)
        pl, vl, al = losses(o, pi, pval[i], z[i], q[i], has[i].float(), left[i], qv[i], hv[i])
        loss = pl + 1.5 * vl + 0.25 * al
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            rec = {'step': step + 1, 'policy_kl': float(pl), 'value_mse': float(vl), 'aux': float(al),
                   'sec': round(time.time() - t0, 1)}
            hk, hm = holdout()
            rec.update(hold_policy_kl=hk, hold_value_mse=hm)
            log.append(rec); print(json.dumps(rec), flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in net.state_dict().items()}
    export(net, out.with_suffix('.onnx'))
    torch.save({'arch': a.arch, 'cfg': cfg, 'state': state, 'log': log, 'args': vars(a), 'positions': n,
                'params': n_params(net), 'train_files': [str(f) for f in train_files],
                'holdout_files': [str(f) for f in hold_files]}, out)
    print(json.dumps({'saved': str(out), 'onnx': str(out.with_suffix('.onnx')), 'final': log[-1],
                      'train_sec': round(time.time() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()
