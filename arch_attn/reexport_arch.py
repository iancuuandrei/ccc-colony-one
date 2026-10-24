"""Re-export arch_attn checkpoints through the current net.export (picks up net.py output changes).

wsl -e bash .../wpy.sh arch_attn/reexport_arch.py arch_attn/ckpt_attnc.pt [more.pt ...]
Writes <ckpt>.onnx next to each checkpoint and checks ORT vs torch on random sparse boards.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from net_attn import load_ckpt, export

for c in sys.argv[1:]:
    net, ck = load_ckpt(c)
    out = Path(c).with_suffix('.onnx')
    export(net, out)
    import onnxruntime as ort
    x = (torch.rand(16, 16, 8, 8) > 0.9).float()
    net.eval()
    with torch.no_grad():
        lg, v = net(x)
    s = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    lo, vo = s.run(None, {'observation': x.numpy()})
    print(json.dumps({'onnx': str(out), 'arch': ck.get('arch'), 'cfg': ck['cfg'],
                      'max_abs_diff_policy': float(np.abs(lo - lg.numpy()).max()),
                      'max_abs_diff_value': float(np.abs(vo - v.numpy()).max())}))
