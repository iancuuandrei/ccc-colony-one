"""Interleaved batch-8 single-thread ORT latency for several ONNX models (robust to a busy CPU).

python arch_attn/bench.py ckpt/gen04.onnx arch_attn/ckpt_attn.onnx [--rounds 30 --iters 40]
Reports per model the median and min (over rounds) of the per-call mean, plus node/initializer counts.
"""
import argparse, json, time
from collections import Counter
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

p = argparse.ArgumentParser()
p.add_argument('models', nargs='+')
p.add_argument('--rounds', type=int, default=30)
p.add_argument('--iters', type=int, default=40)
a = p.parse_args()
rng = np.random.default_rng(0)
x = (rng.random((8, 16, 8, 8)) > 0.9).astype(np.float32)
sess = {}
for m in a.models:
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    sess[m] = ort.InferenceSession(m, o, providers=['CPUExecutionProvider'])
    for _ in range(20):
        sess[m].run(None, {'observation': x})
times = {m: [] for m in a.models}; cpu = {m: [] for m in a.models}
for r in range(a.rounds):
    for m in (a.models if r % 2 == 0 else a.models[::-1]):
        t = time.perf_counter(); c = time.thread_time()
        for _ in range(a.iters):
            sess[m].run(None, {'observation': x})
        times[m].append((time.perf_counter() - t) / a.iters * 1000)
        cpu[m].append((time.thread_time() - c) / a.iters * 1000)
for m in a.models:
    g = onnx.load(m).graph
    print(json.dumps({'model': m, 'bytes': Path(m).stat().st_size, 'nodes': len(g.node), 'inits': len(g.initializer),
                      'batch8_ms_median': round(float(np.median(times[m])), 3),
                      'batch8_ms_min': round(float(np.min(times[m])), 3),
                      'batch8_cpu_ms_median': round(float(np.median(cpu[m])), 3),
                      'ops': dict(Counter(n.op_type for n in g.node))}))
