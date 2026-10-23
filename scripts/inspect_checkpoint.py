"""Export configuration from a trusted project checkpoint without serializing tensors."""
import hashlib
import json
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
path = root / 'checkpoints/gen01.pt'
ck = torch.load(path, map_location='cpu', weights_only=True)
result = {
    'checkpoint': 'checkpoints/gen01.pt',
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'keys': sorted(ck),
    'cfg': ck.get('cfg'),
    'args': ck.get('args'),
    'positions': ck.get('positions'),
    'train_files': ck.get('train_files'),
    'holdout_files': ck.get('holdout_files'),
    'training_log': ck.get('log'),
    'state_parameter_count': sum(v.numel() for v in ck['state'].values()),
    'source_qualification': 'Current source snapshots are separately hash-bound. Exact generation-build source was not stored in this checkpoint and is UNKNOWN.',
    'resume_qualification': 'Contains model state/config and run arguments. --init is a warm start; optimizer/scheduler/RNG/step continuation is not established.',
}
(root / 'provenance/gen01-checkpoint.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: result[k] for k in ('sha256', 'cfg', 'positions', 'keys')}))
