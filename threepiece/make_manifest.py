"""Aggregate build + verification evidence into gen06-endgame-3p.manifest.json (fails closed)."""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda n: json.loads((HERE / f'verify-{n}.json').read_text())
build = json.loads((HERE / 'gen06-endgame-3p.build.json').read_text())
model, proof, base = HERE / 'gen06-endgame-3p.onnx', HERE / 'gen06-endgame-3p.proof.onnx', ROOT / 'ckpt' / 'gen06-endgame.onnx'
assert sha(model) == build['sha256'] and sha(proof) == build['proof_sha256'] and sha(base) == build['base_sha256']
V = {n: load(n) for n in ('sweep', 'positives', 'audit', 'selfplay', 'batches', 'passthrough', 'check')}
problems = []
for n in ('sweep', 'positives', 'audit', 'selfplay'):
    c = V[n]['counts']
    for k, v in c.items():
        if k.startswith('mismatch_') and v: problems.append(f'{n}:{k}={v}')
    if c.get('engine_failures', 0): problems.append(f'{n}:engine_failures')
    if c.get('nonfamily', 0) != c.get('nonfamily_bitwise_identical', 0): problems.append(f'{n}:nonfamily_not_identical')
    if c.get('engine_checked_actions', 0) != c.get('engine_wins', 0): problems.append(f'{n}:engine')
pt = V['passthrough']['counts']
if pt.get('policy_bitwise_mismatch', 0) or pt.get('value_bitwise_mismatch', 0) or not pt.get('states'): problems.append('passthrough')
for b, r in V['batches']['results'].items():
    if not all(v for k, v in r.items() if isinstance(v, bool)): problems.append('batches:' + b)
if V['check']['wrapper']['exit'] != 0 or V['check']['wrapper']['report']['disallowed_ops']: problems.append('check_onnx')
w = V['audit']['witnesses']
if any(x['helper'] != x['wrapper'] for x in w): problems.append('witnesses')
counts = lambda n: {k: v for k, v in V[n]['counts'].items()}
manifest = dict(
    status='PASS' if not problems else 'FAIL', problems=problems,
    model=dict(path=str(model.relative_to(ROOT)), sha256=build['sha256'], bytes=build['bytes'],
               nodes=build['total_nodes'], base_nodes=build['base_nodes'], added_nodes=build['added_nodes'],
               initializers=build['total_initializers'], added_initializers=build['added_initializers'],
               added_weight_bytes=build['added_weight_bytes'], opset=build['opset'], node_cap=1024, initializer_cap=512),
    base=dict(path=str(base.relative_to(ROOT)), sha256=build['base_sha256']),
    proof_sidecar=dict(path=str(proof.relative_to(ROOT)), sha256=build['proof_sha256'], note='verification only, not an upload candidate'),
    rule=dict(family='mover: only one Kernel; enemy: one Kernel + exactly one non-Kernel; all unlocked, distinct; '
                     'Cheb(t,e)>2 and ring(t)>ring(e) or equal ring with actual mover Magenta (plane 15); '
                     'al=1: Cheb(s,t)=1, first action s->t; al=2: Cheb(s,t)=2, first actions s->m for every empty unlocked common neighbour m',
              in_family='value=+0.99; policy=base+30 on certified first actions only; policy = base - Bn, Bn in {-30, +0.0} (bitwise pass-through elsewhere)',
              outside='policy and value bitwise identical to base (UNKNOWN)'),
    sources={str(p.relative_to(ROOT)): sha(p) for p in [HERE / 'build_threepiece.py', HERE / 'verify_threepiece.py', HERE / 'make_manifest.py',
             ROOT / 'codex/endgame/three_piece_certificate.py', ROOT / 'codex/arena/three-piece-independent-audit/audit.py',
             ROOT / 'env/colony_one/env.py', ROOT / 'env/colony_one/board.py', ROOT / 'env/colony_one/rules.py',
             ROOT / 'env/colony_one/encoding.py', ROOT / 'env/colony_one/submission_policy.py', ROOT / 'check_onnx.py']},
    verification=dict(
        exhaustive_sweep=dict(scope=V['sweep'].get('scope', 'all ordered distinct (s,e,t) over 64 squares x 4 lock stages x 2 teams x al 1/2; type+round seeded per state'),
                              counts=counts('sweep'), seconds=V['sweep']['seconds']),
        helper_enumeration_all_types=dict(specs=V['positives']['specs'], counts=counts('positives'), seconds=V['positives']['seconds']),
        independent_audit_fixtures=dict(audit_status=V['audit']['audit_status'], audit_fixture_states=V['audit']['audit_fixture_states'],
                                        counts=counts('audit'), witnesses=w,
                                        excluded_terminated=V['audit']['terminated_fixtures_outside_observation_domain']),
        selfplay=dict(files=V['selfplay']['files'], counts=counts('selfplay'), seconds=V['selfplay']['seconds']),
        batch_invariance=V['batches']['results'],
        signed_zero_passthrough=dict(counts=dict(pt, policy_bitwise_mismatch=pt.get('policy_bitwise_mismatch', 0),
                                                 value_bitwise_mismatch=pt.get('value_bitwise_mismatch', 0)),
                                     specials=V['passthrough']['specials'],
                                     scope='increment-only graph, base outputs fed with -0.0/+0.0/denormal/+-1e30/+-3.4e38/random, batch 1/8/16/512, ORT default and disabled optimisation'),
        check_onnx=dict(wrapper=V['check']['wrapper'], base=V['check']['base']),
        latency=V['check']['latency']))
out = HERE / 'gen06-endgame-3p.manifest.json'
out.write_text(json.dumps(manifest, indent=2)); print(json.dumps(dict(status=manifest['status'], problems=problems, sha256=build['sha256'])))
sys.exit(0 if not problems else 1)
