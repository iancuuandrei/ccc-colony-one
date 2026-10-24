"""Export calibration variants of a checkpoint: python calib_export.py ckpt/gen01.pt calib/gen01"""
import sys, torch
from pathlib import Path
from net import Net, export
ck = torch.load(sys.argv[1], map_location='cpu'); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
grid = [(1.0, 1.0, 3.0), (1.0, 1.0, 8.0), (0.7, 1.0, 1.0), (1.4, 1.0, 1.0), (1.0, 0.6, 1.0)]
for t, s, og in grid:
    net = Net(**ck['cfg']); net.load_state_dict(ck['state'], strict=False)
    export(net, out / f'T{t}_s{s}_o{og}.onnx', t, s, og); print('exported', t, s, og)
