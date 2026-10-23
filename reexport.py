"""Re-export a checkpoint with calibration: python reexport.py ckpt.pt out.onnx --ptemp 1.2 --vscale 0.7"""
import argparse, torch
from net import Net, export
p = argparse.ArgumentParser(); p.add_argument('ckpt'); p.add_argument('out')
p.add_argument('--ptemp', type=float, default=1.0); p.add_argument('--vscale', type=float, default=1.0)
a = p.parse_args()
ck = torch.load(a.ckpt, map_location='cpu'); net = Net(**ck['cfg']); net.load_state_dict(ck['state'], strict=False)
export(net, a.out, a.ptemp, a.vscale); print({'out': a.out, 'ptemp': a.ptemp, 'vscale': a.vscale})
