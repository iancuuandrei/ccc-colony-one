"""Append Clip(-1, 1) to the value output (float rounding in composed wrappers can give 1.0000001).

python final_clip.py in.onnx out.onnx
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
prod = [n for n in g.node if 'value' in n.output]
assert len(prod) == 1, 'expected one producer of value'
for i, o in enumerate(prod[0].output):
    if o == 'value':
        prod[0].output[i] = 'value_unclipped'
for n in g.node:  # any consumer of value (none expected) follows the rename
    for i, x in enumerate(n.input):
        if x == 'value':
            n.input[i] = 'value_unclipped'
g.initializer.extend([numpy_helper.from_array(np.array(-1.0, np.float32), 'value_clip_lo'),
                      numpy_helper.from_array(np.array(1.0, np.float32), 'value_clip_hi')])
g.node.append(helper.make_node('Clip', ['value_unclipped', 'value_clip_lo', 'value_clip_hi'], ['value'],
                               name='value_final_clip'))
onnx.checker.check_model(m)
onnx.save(m, dst, save_as_external_data=False)
import json; print(json.dumps({'in': src, 'out': dst, 'nodes': len(g.node)}))
