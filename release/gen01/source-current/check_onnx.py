"""Fail fast on an exported model: server acceptance policy, op histogram, latency.

python check_onnx.py model.onnx
"""
import json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'env'))
from colony_one.submission_policy import inspect_and_dry_run_model, ALLOWED_OPERATORS

path = Path(sys.argv[1])
m = onnx.load(str(path))
ops = Counter(n.op_type for n in m.graph.node)
bad = sorted(set(ops) - set(ALLOWED_OPERATORS))
dry = inspect_and_dry_run_model(path)  # raises ModelRejected on any policy violation
o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
s = ort.InferenceSession(str(path), o, providers=['CPUExecutionProvider'])
x = (np.random.rand(8, 16, 8, 8) > 0.9).astype(np.float32)
for _ in range(5): s.run(None, {'observation': x})
t = time.perf_counter()
for _ in range(50): s.run(None, {'observation': x})
lat = (time.perf_counter() - t) / 50
print(json.dumps({'model': str(path), 'bytes': path.stat().st_size, 'nodes': len(m.graph.node),
                  'disallowed_ops': bad, 'dry_run_s': round(dry, 4), 'batch8_ms': round(lat * 1000, 3),
                  'ops': dict(ops)}))
sys.exit(1 if bad else 0)
