"""Function-preserving Net2Net expansion of a trained Net: more channels and blocks, same outputs.

python widen.py ckpt/gen07.pt ckpt/big_init.pt --channels 80 --blocks 6
New channels read from old ones (small random) but write nothing back (zero weights into old
units and heads); new blocks start as identity (zero BN gamma/beta on the residual branch).
"""
import argparse
import torch
from torch import nn
from net import Net

def main():
    p = argparse.ArgumentParser()
    p.add_argument('src'); p.add_argument('dst')
    p.add_argument('--channels', type=int, default=80); p.add_argument('--blocks', type=int, default=6)
    a = p.parse_args()
    ck = torch.load(a.src, map_location='cpu')
    old = Net(**ck['cfg']); old.load_state_dict(ck['state'], strict=False); old.eval()
    C, B, E = ck['cfg']['channels'], ck['cfg']['blocks'], ck['cfg']['emb']
    C2, B2 = a.channels, a.blocks
    new = Net(C2, B2, E)
    torch.manual_seed(0)

    def small_(t, s=0.02):
        with torch.no_grad():
            t.normal_(0, s)

    with torch.no_grad():
        # stem: old output channels copied, new ones small random
        small_(new.stem[0].weight); new.stem[0].weight[:C] = old.stem[0].weight
        bn_n, bn_o = new.stem[1], old.stem[1]
        for k in ('weight', 'bias', 'running_mean', 'running_var'):
            getattr(bn_n, k)[:C] = getattr(bn_o, k)
        for i in range(B2):
            nb = new.blocks[i]
            if i < B:
                ob = old.blocks[i]
                for cn, co in ((nb.c1, ob.c1), (nb.c2, ob.c2)):
                    small_(cn.weight)
                    cn.weight[:C, :C] = co.weight
                    cn.weight[:C, C:] = 0          # new channels never feed old units
                for bn, bo in ((nb.b1, ob.b1), (nb.b2, ob.b2)):
                    for k in ('weight', 'bias', 'running_mean', 'running_var'):
                        getattr(bn, k)[:C] = getattr(bo, k)
                nb.b2.weight[C:] = 0.1             # new residual channels start small
                nb.b2.bias[C:] = 0.0
                if (nb.se is None) != (ob.se is None):
                    raise SystemExit('SE layout mismatch')
                if nb.se is not None:
                    hs_o, hs_n = C // 4, C2 // 4
                    small_(nb.se.fc1.weight); nb.se.fc1.bias.zero_()
                    nb.se.fc1.weight[:hs_o, :C] = ob.se.fc1.weight; nb.se.fc1.weight[:hs_o, C:] = 0
                    nb.se.fc1.bias[:hs_o] = ob.se.fc1.bias
                    W, b = nb.se.fc2.weight, nb.se.fc2.bias
                    W.zero_(); b.zero_()
                    # gate rows: old i -> i ; bias rows: old C+i -> C2+i ; new rows stay 0 (gate 0.5, bias 0)
                    W[:C, :hs_o] = ob.se.fc2.weight[:C]; b[:C] = ob.se.fc2.bias[:C]
                    W[C2:C2 + C, :hs_o] = ob.se.fc2.weight[C:]; b[C2:C2 + C] = ob.se.fc2.bias[C:]
            else:  # identity block
                small_(nb.c1.weight); small_(nb.c2.weight)
                nb.b2.weight.zero_(); nb.b2.bias.zero_()
                if nb.se is not None:
                    nb.se.fc2.weight.zero_(); nb.se.fc2.bias.zero_()
        # heads read only old channels
        for hn, ho in ((new.p_conv, old.p_conv), (new.v_conv, old.v_conv), (new.p2_conv, old.p2_conv)):
            hn.weight.zero_(); hn.weight[:, :C] = ho.weight; hn.bias.copy_(ho.bias)
        new.v_fc1.weight.zero_()
        new.v_fc1.weight[:, :8 * 64] = old.v_fc1.weight[:, :8 * 64]
        new.v_fc1.weight[:, 8 * 64:8 * 64 + C] = old.v_fc1.weight[:, 8 * 64:]
        new.v_fc1.bias.copy_(old.v_fc1.bias)
        for k in ('v_fc2', 'aux'):
            getattr(new, k).load_state_dict(getattr(old, k).state_dict())
    new.eval()
    # verify function preservation on real observations
    import numpy as np, glob
    f = sorted(glob.glob('data/sp_*/chunk*.npz'))[-1]
    x = torch.tensor(np.load(f)['obs'][:512].astype(np.float32))
    with torch.no_grad():
        lo, vo = old(x); ln, vn = new(x)
    err = {'policy_max_abs': float((lo - ln).abs().max()), 'value_max_abs': float((vo - vn).abs().max())}
    print(err)
    params = sum(p.numel() for p in new.parameters())
    torch.save({'cfg': new.cfg, 'state': new.state_dict(), 'widened_from': a.src, 'verify': err}, a.dst)
    print({'saved': a.dst, 'cfg': new.cfg, 'params': params})

if __name__ == '__main__':
    main()
