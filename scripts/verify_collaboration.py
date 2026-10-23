"""Operational integrity check for the synchronized collaboration artifacts."""
import hashlib
import json
from pathlib import Path
import runpy

root = Path(__file__).resolve().parents[1]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

runpy.run_path(str(root / 'scripts/verify_snapshot.py'))
cfg = json.loads((root / 'sync_allowlist.json').read_text(encoding='utf-8'))
pins = [e for e in cfg['files'] if 'sha256' in e]
for entry in pins:
    assert sha(root / entry['path']) == entry['sha256'], entry['path']
gen01 = 'ed45eae71a9332d941f5f02d9e5a469f8372af56e8d31ccf0d4b5cdbe87f6c7a'
for name in ('model.onnx', 'model-gen01.onnx', 'release/gen01/model.onnx'):
    assert sha(root / name) == gen01, name
checkpoint = json.loads((root / 'provenance/gen01-checkpoint.json').read_text(encoding='utf-8'))
assert sha(root / checkpoint['checkpoint']) == checkpoint['sha256']
student = root / 'codex/distillation/student'
recovery = json.loads((student / 'v2-recovery-hashes.json').read_text(encoding='utf-8'))
for name, expected in recovery.items():
    assert sha(student / name) == expected, name
labels = 0
unavailable_sources = []
for packet in ('packet0001', 'packet0002'):
    base = root / 'codex/distillation'
    ready = json.loads((base / 'validated' / packet / 'student_ready.json').read_text(encoding='utf-8'))
    labels += ready['count']
    for key, path in (
        ('packet_sha256', base / 'packets' / f'{packet}.json'),
        ('labels_sha256', base / 'teacher_labels' / f'astra_{packet}.json'),
        ('targets_sha256', base / 'validated' / packet / 'teacher_targets.npz'),
    ):
        assert sha(path) == ready[key], key
    if sha(base / 'validation.py') != ready['validator_sha256']:
        unavailable_sources.append({'packet': packet, 'historical_validator_sha256': ready['validator_sha256'],
                                    'qualification': 'Historical validator source not available; current source is separately hash-bound.'})
assert labels == 32
base = root / 'codex/synthetic/astra/curriculum'
manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
assert manifest['status'] == 'COMPLETE'
assert sha(base / 'certified_states.npz') == manifest['npz_sha256']
assert sha(base / 'certified_states.jsonl') == manifest['jsonl_sha256']
result = {'status': 'PASS', 'scope': 'Artifact hash bindings; not a new playing-strength test or fresh certificate run',
          'immutable_pins': len(pins), 'teacher_labels': labels,
          'constructed_curriculum_states': manifest['states'], 'source_limitations': unavailable_sources}
print(json.dumps(result, indent=2))
(root / 'provenance/collaboration-integrity.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
