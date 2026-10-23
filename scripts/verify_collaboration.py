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
for name in ('model-gen01.onnx', 'release/gen01/model.onnx'):
    assert sha(root / name) == gen01, name
gen03 = '0cf937446ba89dc9b5b3652edbcb4025030bc785e924551782f4b2d298ebe96f'
for name in ('model-gen03.onnx', 'release/gen03/model.onnx'):
    assert sha(root / name) == gen03, name
current = json.loads((root / 'provenance/current-release.json').read_text(encoding='utf-8'))
assert sha(root / 'model.onnx') == current['sha256']
gen06 = '123467a930e0598c464ae7694800ad565aeac71680b88dafc5f014bfe7cf50ac'
for name in ('model-gen06.onnx', 'release/gen06/model.onnx'):
    assert sha(root / name) == gen06
assert sha(root / 'release/gen02/model.onnx') == '579d3ec01da7328dc706003b8dcc5eba357bb74902252187e45633ca0595a9a0'
checkpoint = json.loads((root / 'provenance/gen01-checkpoint.json').read_text(encoding='utf-8'))
assert sha(root / checkpoint['checkpoint']) == checkpoint['sha256']
for generation in ('gen02', 'gen03', 'gen05', 'gen06'):
    item = json.loads((root / f'provenance/{generation}-checkpoint.json').read_text(encoding='utf-8'))
    assert sha(root / item['checkpoint']) == item['sha256']
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
import numpy as np
datasets = []
for name, expected_rows in (('ext_astra_policy_v1', 144), ('ext_astra_site_gen02_v1', 55), ('ext_site_tactics_v1', 26)):
    base = root / 'data' / name
    rows = value_rows = policy_rows = 0
    for path in base.glob('chunk*.npz'):
        with np.load(path, allow_pickle=False) as z:
            n = len(z['obs']); rows += n
            assert z['obs'].shape == (n, 16, 8, 8)
            assert not z['q_valid'].any()
            value_rows += int(z['has_value'].sum()); policy_rows += int(z['has_pol'].sum())
    assert rows == expected_rows, name
    if name != 'ext_site_tactics_v1':
        prov = json.loads((base / 'provenance.json').read_text(encoding='utf-8'))
        assert sha(base / 'chunk0000.npz') == prov['chunk_sha256']
    else:
        check = json.loads((base / 'validation.json').read_text(encoding='utf-8'))
        assert sha(base / 'manifest.json') == check['manifest_sha256']
        summary = json.loads((base / 'summary.json').read_text(encoding='utf-8'))
        for field in ('policy_override', 'proof', 'report'):
            filename = summary[field + '_file']
            key = 'proof_file_sha256' if field == 'proof' else field + '_sha256'
            assert sha(base / filename) == summary[key]
        for item in check.get('checks', []):
            path = base / f"chunk-site-tactics-{item['game']}-ply{item['ply']:03d}.npz"
            if 'chunk_sha256' in item:
                assert sha(path) == item['chunk_sha256']
            else:
                assert path.exists() and item['result'] == 'PASS'
        for item in check.get('preserved_old_six_chunks_byte_exact', []):
            assert sha(base / item['chunk']) == item['sha256']
    datasets.append({'name': name, 'rows': rows, 'policy_rows': policy_rows, 'exact_value_rows': value_rows})
result['reviewed_external_datasets'] = datasets
archive = json.loads((root / 'provenance/ext-site-tactics-six.json').read_text(encoding='utf-8'))
for item in archive['files']:
    assert sha(root / item['path']) == item['sha256']
result['historical_six_row_snapshot'] = 'PASS: exact archived hashes preserved'
archive19 = json.loads((root / 'provenance/ext-site-tactics-nineteen.json').read_text(encoding='utf-8'))
for item in archive19['files']:
    assert sha(root / item['path']) == item['sha256']
    if item['path'].endswith('.npz'):
        assert sha(root / 'data/ext_site_tactics_v1' / Path(item['path']).name) == item['sha256']
result['historical_nineteen_row_snapshot'] = 'PASS: exact archived hashes preserved'
deep = root / 'data/ext_site_deep10910_v1'
prov = json.loads((deep / 'provenance.json').read_text(encoding='utf-8'))
assert sha(deep / 'chunk0000.npz') == prov['chunk_sha256']
deep_proof = root / 'codex/synthetic/validation/deep10910'
assert sha(deep_proof / 'proof.json') == prov['independent_proof_sha256']
assert sha(deep_proof / 'loader_audit.json') == prov['loader_audit_sha256']
assert sha(root / 'codex/research/loss10910-training/packet.json') == prov['source_packet_sha256']
with np.load(deep / 'chunk0000.npz', allow_pickle=False) as z:
    assert len(z['obs']) == 2
    assert z['has_value'].tolist() == [True, False]
    assert z['has_pol'].tolist() == [False, True]
    assert not z['q_valid'].any()
    assert z['z'].tolist() == [-1.0, 0.0]
result['deep10910_two_row_snapshot'] = 'PASS: chunk, proof, packet, loader audit and value/policy masks'
override_audit = json.loads((root / 'codex/research/overrides-integration-audit.json').read_text(encoding='utf-8'))
audit_sources = {}
for name, expected in override_audit['source_hashes'].items():
    candidate = root / 'release/gen06/source-current' / name
    audit_sources[name] = {'expected': expected, 'captured': sha(candidate), 'matches_audit_source': sha(candidate) == expected}
result['override_audit_captured_source_binding'] = audit_sources
gate_checks = []
for directory, expected_games, expected_wins in (
    ('gen05-v-gen03', 400, 286), ('gen05-v-gen02', 200, 172),
    ('gen06-v-gen03', 400, 335), ('gen06-v-gen02', 200, 186),
):
    base = root / 'codex/arena' / directory
    audit = json.loads((base / 'engine-history-audit.json').read_text(encoding='utf-8'))
    summary = json.loads((base / 'summary.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'PASS'
    assert summary['complete'] and summary['games'] == expected_games and summary['a_wins'] == expected_wins
    assert sha(base / 'manifest.json') == audit['manifest_sha256']
    assert sha(base / 'games.jsonl') == audit['games_jsonl_sha256']
    assert len((base / 'games.jsonl').read_text(encoding='utf-8').splitlines()) == expected_games
    gate_checks.append({'directory': directory, 'histories': expected_games, 'candidate_wins': expected_wins, 'receipt_bindings': 'PASS'})
result['final_gate_receipts'] = gate_checks
site = json.loads((root / 'data/ext_site_v1/summary.json').read_text(encoding='utf-8'))
assert site['episode_count'] == 111 and site['rows'] == 3777
assert site['unique_complete_episodes_across_folders'] == 118
assert site['unique_rows_across_folders'] == 4105
assert sha(root / 'data/ext_site_v1/manifest.json') == '50ce9af40cd67e8e7c27addd300dbf3dab00625daff47f37ed42b02169624b39'
result['export5_receipts'] = 'PASS: 118 unique episodes / 4105 rows; bounded receipts only'
deep14958 = root / 'data/ext_site_deep14958_v1'
prov14958 = json.loads((deep14958 / 'provenance.json').read_text(encoding='utf-8'))
assert sha(deep14958 / 'chunk0000.npz') == prov14958['chunk_sha256'] == '6fd955351da4676e58d0a7e48c0b32e7c6038a4c9c731d6b1dcd2672f43c93a3'
assert sha(root / 'codex/synthetic/site_gen05_curriculum/validation14958/proof.json') == prov14958['independent_proof_sha256']
assert sha(root / 'codex/synthetic/validation/deep14958-final/report.json') == prov14958['final_audit_sha256']
with np.load(deep14958 / 'chunk0000.npz', allow_pickle=False) as z:
    assert len(z['obs']) == 2 and z['has_value'].all() and z['has_length'].all()
    assert not z['has_pol'].any() and not z['q_valid'].any()
    assert z['z'].tolist() == [-1.0, -1.0] and z['left'].tolist() == [8.0, 7.0]
result['deep14958_two_value_rows'] = 'PASS: exact chunk/proof/final-audit bindings, explicit value-only and observed-length masks'
new_audit = root / 'codex/arena/gen06-new-site-audit'
numeric = json.loads((new_audit / 'numeric-summary.json').read_text(encoding='utf-8'))
assert numeric['status'] == 'PASS_OBSERVED_CORPUS' and numeric['model_sha256'] == gen06
assert numeric['states'] == 4105 and numeric['batch_sizes'] == [1, 8, 16]
for batch in numeric['batches'].values():
    assert batch['nonfinite_values'] == batch['out_of_range_values'] == batch['nonfinite_policy_entries'] == 0
assert json.loads((new_audit / 'numeric-violations.json').read_text(encoding='utf-8')) == []
result['preserved_numeric_audit'] = 'PASS: source report covers GEN06 raw ORT outputs on 4105 observed states at batch 1/8/16; not a new execution'
primary = json.loads((root / 'data/ext_site_tactics_v1/validation.json').read_text(encoding='utf-8'))
assert primary['data_validation'] == 'PASS' and primary['published_rows'] == 26
assert primary['actual_loader_primary_policy_and_value_Q_masks'] == 'PASS'
assert primary['gumbel_alternate_target_guard'].startswith('FAIL_FOR_NONZERO_POLICY_MIX')
known_failures = root / 'codex/synthetic/validation/site-gen05-v1'
length_bug = json.loads((known_failures / 'length-mask-before-fix.json').read_text(encoding='utf-8'))
mix_bug = json.loads((known_failures / 'policy-mix-regression.json').read_text(encoding='utf-8'))
assert length_bug['status'].startswith('FAIL_') and mix_bug['status'].startswith('FAIL_')
result['trainer_qualification'] = {'primary_default_policy_overrides': 'PASS in preserved source-specific report',
    'length_mask': length_bug['status'], 'nonzero_policy_mix': mix_bug['status'],
    'scope': 'Known failures pending Opus fixes; this sync changes no trainer or deployed model.'}
print(json.dumps(result, indent=2))
(root / 'provenance/collaboration-integrity.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
