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
for name, expected_rows in (('ext_astra_policy_v1', 144), ('ext_astra_site_gen02_v1', 55), ('ext_site_tactics_v1', 29)):
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
archive26 = json.loads((root / 'provenance/ext-site-tactics-twentysix.json').read_text(encoding='utf-8'))
for item in archive26['files']:
    assert sha(root / item['path']) == item['sha256']
    if item['path'].endswith('.npz'):
        assert sha(root / 'data/ext_site_tactics_v1' / Path(item['path']).name) == item['sha256']
result['historical_twentysix_row_snapshot'] = 'PASS: archived hashes and current chunk bytes preserved'
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
assert primary['data_validation'] == 'PASS' and primary['published_rows'] == 29
assert primary['current_policy_rows_value_Q_length_masks'] == 'PASS'
assert primary['actual_pval_g_mix_guard'].startswith('PASS:')
assert primary['actual_loader_rows'] == 2490 and len(primary['actual_loader_rs_replacements']) == 11
known_failures = root / 'codex/synthetic/validation/site-gen05-v1'
length_bug = json.loads((known_failures / 'length-mask-before-fix.json').read_text(encoding='utf-8'))
mix_bug = json.loads((known_failures / 'policy-mix-regression.json').read_text(encoding='utf-8'))
assert length_bug['status'].startswith('FAIL_') and mix_bug['status'].startswith('FAIL_')
result['historical_trainer_regressions'] = {'length_mask': length_bug['status'], 'nonzero_policy_mix': mix_bug['status']}
length_fixed = json.loads((known_failures / 'loader-length-qualified.json').read_text(encoding='utf-8'))
published28 = json.loads((known_failures / 'published-loader.json').read_text(encoding='utf-8'))
fixed_train = root / 'source_snapshots/trainer-fixed-b2d4/train.py'
assert sha(fixed_train) == primary['trainer_sha256'] == length_fixed['train_sha256'] == 'b2d4bdf6a0b17ff7c225c3c67ed2cfc562bf643224d557d770b40b68e39f6bb3'
assert length_fixed['status'] == 'PASS_CURRENT_LENGTH_DEFAULTS_TRAIN_HOLDOUT_GRADIENTS_AND_DEDUP'
for gradient in length_fixed['gradient']:
    assert gradient['unknown_length_gradient'] == 0 and gradient['exact_value_gradient_retained']
    assert gradient['known_length_gradient_retained'] and gradient['length_normalization'] == 'PASS'
release28 = root / 'data/ext_astra_site_gen05_v1'
prov28 = json.loads((release28 / 'provenance.json').read_text(encoding='utf-8'))
assert sha(release28 / 'chunk0000.npz') == prov28['chunk_sha256'] == published28['published_chunk_sha256']
assert sha(known_failures / 'report.json') == prov28['independent_report_sha256']
assert sha(known_failures / 'loader-length-qualified.json') == prov28['loader_length_audit_sha256']
assert sha(root / 'codex/synthetic/site_gen05_curriculum/MANIFEST.json') == prov28['source_manifest_sha256']
assert published28['actual_published_path_loader'] == 'PASS'
with np.load(release28 / 'chunk0000.npz', allow_pickle=False) as z:
    assert len(z['obs']) == 28 and int(z['has_value'].sum()) == 16
    assert int(z['has_pol'].sum()) == 14 and int(z['has_length'].sum()) == 4
    assert not z['q_valid'].any()
result['trainer_qualification'] = {'qualified_source': primary['trainer_sha256'], 'length_mask': 'PASS_TRAIN_AND_HOLDOUT_GRADIENTS',
    'policy_mix': 'PASS_0_0.25_0.5_1', 'published28_loader': 'PASS',
    'scope': 'Preserved source-specific fixed qualification; no historical model-build attribution, training or gradient execution by this sync.'}
assert sha(root / 'codex/research/user-six-gen07-handoff.md') == '2bb467969847907f27c04a447899ff34ced9e24f9bcf65a696ff69c418442241'
assert sha(root / 'codex/research/user-six-independent-proof-review.md') == primary['independent_review_sha256'] == '5eba20b306ecca4612f71a58ea48933348d9d8db4661c5702f05db9d20a67986'
six = root / 'codex/arena/gen07-user-six-audit'
for name, expected in json.loads((six / 'artifact-hashes.json').read_text(encoding='utf-8')).items():
    assert sha(six / name) == expected, name
summary_six = json.loads((six / 'combined-summary.json').read_text(encoding='utf-8'))
assert summary_six['model_sha256'] == '8b23b6a7f80089aef590beb69a829738e3c8c66dcd5ccbb652820142dbabcebd'
assert summary_six['reported_case_count_including_reused_aliases'] == 14 and summary_six['distinct_queried_observations'] == 10
assert summary_six['distinct_missed_nonempty_support_observations'] == 4
raw19490 = root / 'codex/matches/19490'
raw_receipt = json.loads((raw19490 / 'source-receipt.json').read_text(encoding='utf-8'))
assert sha(raw19490 / 'match-19490.txt') == raw_receipt['raw_sha256']
assert raw_receipt['replay'] == 'PASS' and raw_receipt['user_side'] == raw_receipt['historical_model'] == 'UNKNOWN'
staged19490 = root / 'codex/synthetic/validation/deep19490-stage/ext_deep19490_proposal/chunk0000.npz'
assert sha(staged19490) == '42e9d162b2e4105d1f8e5bec262ce62e7813d1d0a57cd506666191ff6c349bc0'
assert not (root / 'data/ext_deep19490_proposal').exists()
with np.load(staged19490, allow_pickle=False) as z:
    assert len(z['obs']) == 3 and z['has_pol'].tolist() == [True, False, False]
    assert z['has_value'].tolist() == [False, True, True] and z['has_length'].tolist() == [False, True, True]
    assert not z['q_valid'].any()
children = json.loads((root / 'codex/research/19255-safe-children-independent-receipt.json').read_text(encoding='utf-8'))
assert children['status'] == 'PASS_STAGE_REVIEW_NOT_RELEASED'
assert sha(root / 'data/ext_site_tactics_v1/staged-export6-19255-safe-children.json') == children['stage_sha256']
assert sha(root / 'codex/matches/export6/user-five-v1/match-19255.txt') == children['source_protocol_sha256']
result['six_game_gen07_review'] = 'PASS: frozen handoff, artifact-hash table, reviewed raw19490 source and diagnostic/staged mask bindings; no GEN07 model or deployment added'
print(json.dumps(result, indent=2))
(root / 'provenance/collaboration-integrity.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
