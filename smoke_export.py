import torch
from net import Net, export
torch.manual_seed(0)
n = Net()
export(n, 'ckpt/smoke.onnx')
print('exported', torch.__version__)
