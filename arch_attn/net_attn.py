"""Chessformer-lite policy/value net for Colony: One (drop-in alternative to net.Net).

Same Features (33 exact planes), same bilinear from x to policy head, value head, aux heads,
win override and export post-processing as ../net.py; only the trunk changes:

  3x3 conv stem (33 -> d) -> 64 square tokens + learned absolute position
  -> L pre-LN transformer layers, each multi-head self-attention with a LEARNED GEOMETRIC
     RELATIVE POSITION BIAS (per head, table over (dr, dc) in [-7, 7]^2, 225 entries),
     SiLU FFN, optional residual 3x3 conv after chosen layers
  -> final LayerNorm -> back to (B, d, 8, 8) so net.Net.heads/outputs/forward_train apply unchanged.

The 64x64 bias is table[:, RIDX] with RIDX a constant index buffer; at export the Gather has two
constant inputs and is folded into a (H, 64, 64) initializer (and would be allow-listed anyway,
since its index input is constant). Every reshape keeps the batch axis as -1: no Shape/Expand/
ConstantOfShape/Range is emitted.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from net import Features, N_FEAT, Net, export  # noqa: F401  (export re-exported for callers)


def rel_index() -> np.ndarray:
    """(64*64,) index into a 15x15 table: (dr + 7) * 15 + (dc + 7) for query i, key j."""
    idx = np.zeros((64, 64), np.int64)
    for i in range(64):
        ri, ci = divmod(i, 8)
        for j in range(64):
            rj, cj = divmod(j, 8)
            idx[i, j] = (rj - ri + 7) * 15 + (cj - ci + 7)
    return idx.reshape(-1)


class RelMHA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.d, self.h, self.dh = d, heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.table = nn.Parameter(torch.zeros(heads, 225))
        self.register_buffer('ridx', torch.tensor(rel_index()), persistent=False)

    def bias(self):
        return self.table[:, self.ridx].reshape(self.h, 64, 64)

    def forward_ref(self, x):                                 # textbook formulation (for tests)
        h, dh = self.h, self.dh
        y = self.qkv(x).reshape(-1, 64, 3 * h, dh).transpose(1, 2)   # (B, 3H, 64, dh)
        q, k, v = y[:, :h], y[:, h:2 * h], y[:, 2 * h:]
        s = torch.matmul(q * (dh ** -0.5), k.transpose(2, 3)) + self.bias()
        o = torch.matmul(torch.softmax(s, -1), v)            # (B, H, 64, dh)
        return self.out(o.transpose(1, 2).reshape(-1, 64, self.d))

    def forward(self, x):                                     # x: (B, 64, d)
        # Same math without activation transposes: per-head weights (weight-only reshapes/permutes
        # are constant-folded at export), heads produced by a broadcast MatMul, output projection
        # applied per head and summed over the head axis.
        h, dh, d = self.h, self.dh, self.d
        w = self.qkv.weight.reshape(3, h, dh, d).permute(1, 3, 0, 2).reshape(h, d, 3 * dh)
        b = self.qkv.bias.reshape(3, h, dh).permute(1, 0, 2).reshape(h, 1, 3 * dh)
        y = torch.matmul(x.reshape(-1, 1, 64, d), w) + b      # (B, H, 64, 3dh)
        q, k, v = y[..., :dh], y[..., dh:2 * dh], y[..., 2 * dh:]
        s = torch.matmul(q * (dh ** -0.5), k.transpose(2, 3)) + self.bias()
        o = torch.matmul(torch.softmax(s, -1), v)            # (B, H, 64, dh)
        wo = self.out.weight.t().reshape(h, dh, d)
        return torch.matmul(o, wo).sum(1) + self.out.bias


class Layer(nn.Module):
    def __init__(self, d, heads, ffn, conv):
        super().__init__()
        self.d = d
        self.ln1 = nn.LayerNorm(d)
        self.attn = RelMHA(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ffn * d)
        self.ff2 = nn.Linear(ffn * d, d)
        self.conv = nn.Conv2d(d, d, 3, padding=1, bias=False) if conv else None
        self.cbn = nn.BatchNorm2d(d) if conv else None

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff2(F.silu(self.ff1(self.ln2(x))))
        if self.conv is not None:
            g = x.transpose(1, 2).reshape(-1, self.d, 8, 8)
            g = F.relu(self.cbn(self.conv(g)))
            x = x + g.reshape(-1, self.d, 64).transpose(1, 2)
        return x


class AttnNet(Net):
    """Inherits heads(), outputs(), forward(), forward_train() from net.Net."""

    def __init__(self, d=64, layers=4, heads=4, ffn=2, conv_layers=(), emb=32, pos=True):
        nn.Module.__init__(self)
        conv_layers = tuple(int(i) for i in conv_layers)
        self.cfg = dict(d=d, layers=layers, heads=heads, ffn=ffn, conv_layers=list(conv_layers), emb=emb, pos=pos)
        self.d = d
        self.feat = Features()
        self.stem = nn.Sequential(nn.Conv2d(N_FEAT, d, 3, padding=1, bias=False), nn.BatchNorm2d(d), nn.ReLU())
        self.pos = nn.Parameter(torch.zeros(1, 64, d)) if pos else None
        self.layers = nn.ModuleList([Layer(d, heads, ffn, i in conv_layers) for i in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        # Heads: identical modules/names to net.Net with channels = d.
        self.emb = emb
        self.p_conv = nn.Conv2d(d, 2 * emb + 2, 1)
        self.v_conv = nn.Conv2d(d, 8, 1)
        self.v_fc1 = nn.Linear(8 * 64 + d, 128)
        self.v_fc2 = nn.Linear(128, 1)
        self.aux = nn.Linear(128, 2)
        self.p2_conv = nn.Conv2d(d, 2 * emb + 2, 1)
        self.register_buffer('ptemp', torch.ones(1), persistent=False)
        self.register_buffer('vscale', torch.ones(1), persistent=False)
        self.register_buffer('open_gain', torch.ones(1), persistent=False)

    def trunk(self, x):
        feats, win, ek = self.feat(x)
        t = self.stem(feats).reshape(-1, self.d, 64).transpose(1, 2)   # (B, 64, d)
        if self.pos is not None:
            t = t + self.pos
        for layer in self.layers:
            t = layer(t)
        t = self.ln_f(t)
        return t.transpose(1, 2).reshape(-1, self.d, 8, 8), win, ek


def build(arch, cfg):
    return AttnNet(**cfg) if arch == 'attn' else Net(**cfg)


def load_ckpt(path):
    ck = torch.load(path, map_location='cpu')
    net = build(ck.get('arch', 'cnn'), ck['cfg'])
    net.load_state_dict(ck['state'], strict=False)
    return net, ck


def n_params(net):
    return sum(p.numel() for n, p in net.named_parameters() if not n.startswith('feat.'))


if __name__ == '__main__':
    # python arch_attn/net_attn.py out.onnx [d layers heads ffn conv_layers(comma)]  -> random-init export
    import json
    out = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / 'smoke_attn.onnx')
    a = sys.argv[2:]
    cfg = dict(d=int(a[0]), layers=int(a[1]), heads=int(a[2]), ffn=int(a[3]),
               conv_layers=[int(c) for c in a[4].split(',') if c] if len(a) > 4 else []) if a else {}
    torch.manual_seed(0)
    net = AttnNet(**cfg)
    with torch.no_grad():                                   # non-trivial bias so folding is exercised
        for l in net.layers:
            l.attn.table.normal_(0, 0.5)
    with torch.no_grad():
        for l in net.layers:
            nn.init.normal_(l.attn.qkv.bias, 0, 0.1); nn.init.normal_(l.attn.out.bias, 0, 0.1)
        t = torch.randn(3, 64, net.d)
        ref_diff = max(float((l.attn(t) - l.attn.forward_ref(t)).abs().max()) for l in net.layers)
    print(json.dumps({'fast_vs_ref_attention_max_abs_diff': ref_diff}))
    x = (torch.rand(4, 16, 8, 8) > 0.9).float()
    net.eval()
    with torch.no_grad():
        lg, v = net(x)
    export(net, out)
    import onnxruntime as ort
    s = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    lo, vo = s.run(None, {'observation': x.numpy()})
    print(json.dumps({'onnx': out, 'cfg': net.cfg, 'params': n_params(net),
                      'max_abs_diff_policy': float(np.abs(lo - lg.numpy()).max()),
                      'max_abs_diff_value': float(np.abs(vo - v.numpy()).max())}))
