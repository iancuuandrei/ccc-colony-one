"""Verify gen06-endgame-3p.onnx against Codex's three-piece helper and the engine.

Modes (default: all):
  sweep      every ordered triple of distinct squares (own Kernel s, enemy Kernel e,
             enemy non-Kernel t; locked squares included) x 4 lock stages x both
             actual teams x actions_left 1/2; enemy type and round drawn per state
             from a seeded RNG over all 5 non-Kernel types and the stage's rounds.
  positives  the helper's own enumeration (every admitted geometry at each stage's
             deadline round) for ALL five enemy non-Kernel types.
  audit      re-runs the independent audit's fixture script (output redirected to
             threepiece/audit_rerun) while recording every certificate() call.
  selfplay   every observation of data/sp_*/chunk*.npz: wrapper flag, and bitwise
             base==wrapper on all non-family positions; flagged ones reconstructed
             into an engine state and checked like the fixtures.
  batches    batch 1/8/16 invariance of the wrapper on mixed family/non-family obs.
  check      check_onnx.py on the wrapper and interleaved batch-8 latency.
Every state: helper flag == wrapper flag, helper winning_first_indices == bonused
indices, wrapper policy == where(helper set, base+30f, base) and wrapper value ==
where(helper flag, 0.99f, base) bitwise. Every bonused action is played through
env.clone()/step and must win for the mover by the end of the current turn.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[_k] = '1'
import argparse, glob, hashlib, importlib.util, json, subprocess, sys, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'env'))
from colony_one import constants as C, encoding as E
from colony_one.board import Board
from colony_one.env import ColonyOneEnv
from colony_one.pieces import Piece, PieceType as PT

PY = sys.executable
HELPER = ROOT / 'codex' / 'endgame' / 'three_piece_certificate.py'
HELPER_SHA = 'c97f7dc7ff2823d9a731a363fb1d8ac7907c379e5ba9ae523e165d8841d4f9f2'
AUDIT = ROOT / 'codex' / 'arena' / 'three-piece-independent-audit' / 'audit.py'
BASE = ROOT / 'ckpt' / 'gen06-endgame.onnx'
WRAP = HERE / 'gen06-endgame-3p.onnx'
PROOF = HERE / 'gen06-endgame-3p.proof.onnx'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HELPER) == HELPER_SHA, 'helper changed'
_spec = importlib.util.spec_from_file_location('tp_helper_under_port', HELPER)
helper = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(helper)

TYPES = [PT.PING, PT.DAEMON, PT.EXPLOIT, PT.TRACE, PT.ICE]
STAGE_ROUNDS = {0: list(range(1, 9)), 1: list(range(9, 17)), 2: list(range(17, 25)), 3: [25, 26]}
LOCKS = {r: Board.locked_cells_for_round(r) for r in range(1, 27)}
B30, V99 = np.float32(30.0), np.float32(0.99)
BATCH = 512
G = {}


def opp(team): return C.MAGENTA if team == C.CYAN else C.CYAN


def make(env, s, e, t, team, al, rnd, typ):
    env.board.grid = [[None] * 8 for _ in range(8)]
    for q, piece in ((s, Piece(PT.KERNEL, team)), (e, Piece(PT.KERNEL, opp(team))), (t, Piece(typ, opp(team)))):
        env.board.grid[q // 8][q % 8] = piece
    env.board.locked = [row[:] for row in LOCKS[rnd]]
    env.current_team = team; env.actions_left = al; env.round_number = rnd
    env.action = (rnd - 1) * 4 + (1 if team == C.CYAN else 3) + (2 - al)
    env.terminated = False; env.winner = None
    return env


def sessions():
    import onnxruntime as ort
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1; o.log_severity_level = 3
    mk = lambda p: ort.InferenceSession(str(p), sess_options=o, providers=['CPUExecutionProvider'])
    return mk(PROOF), mk(BASE), mk(WRAP)


def init_worker():
    G['proof'], G['base'], G['wrap'] = sessions()
    G['env'] = ColonyOneEnv(); G['env2'] = ColonyOneEnv()


def proof_run(feed):
    return G['proof'].run(None, feed)[:2]  # (flag [B] bool, certified mask [B,4096] bool)


def engine_win(env, idx):
    """Play canonical index idx for the actual mover; True iff the mover wins this turn."""
    team = env.current_team
    a = E.decode_action(int(idx), team, env.board)  # raises if illegal
    child = env.clone(); child.step(a)
    if child.terminated: return child.winner == team
    if child.current_team != team or child.actions_left != 1: return False
    for b in child.legal_actions():
        g = child.clone(); g.step(b)
        if g.terminated and g.winner == team: return True
    return False


def new_stats():
    return Counter()


def evaluate(obs, certs, envs, st, tags=None, bad=None, run_engine=True):
    """obs [n,16,8,8] float32; certs: helper result or None; envs: engine clone for
    states that either side flags (else None). Updates Counter st in place."""
    flag, mask = proof_run({'observation': obs})
    bp, bv = G['base'].run(None, {'observation': obs})
    wp, wv = G['wrap'].run(None, {'observation': obs})
    diff = wp.view(np.uint32) != bp.view(np.uint32)
    for j, cert in enumerate(certs):
        st['states'] += 1
        hf = cert is not None
        hidx = sorted(cert['winning_first_indices']) if hf else []
        pidx = np.flatnonzero(mask[j]).tolist()
        didx = np.flatnonzero(diff[j]).tolist()
        tag = tags[j] if tags else None
        if hf: st['helper_positive'] += 1; st['helper_first_actions'] += len(hidx)
        if flag[j]: st['wrapper_flagged'] += 1; st['wrapper_bonused_actions'] += len(didx)
        exp_p = bp[j].copy()
        if hf: exp_p[hidx] = bp[j][hidx] + B30
        exp_v = V99 if hf else bv[j]
        errs = []
        if bool(flag[j]) != hf: errs.append('flag')
        if pidx != hidx: errs.append('proof_index')
        if didx != hidx: errs.append('bonused_index')
        if not np.array_equal(wp[j].view(np.uint32), exp_p.view(np.uint32)): errs.append('policy_bitwise')
        if np.float32(wv[j]).view(np.uint32) != np.float32(exp_v).view(np.uint32): errs.append('value_bitwise')
        if not hf:
            st['nonfamily'] += 1
            if not errs: st['nonfamily_bitwise_identical'] += 1
        if tag is not None: st['tag:' + tag] += 1
        for err in errs:
            st['mismatch_' + err] += 1
            if bad is not None and len(bad) < 20: bad.append(dict(err=err, tag=tag, helper=hidx, wrapper=didx, flag=bool(flag[j])))
        if run_engine and (flag[j] or hf):
            env = envs[j]
            if isinstance(env, tuple): env = make(G['env2'], *env)
            if env is None:
                st['engine_failures'] += 1; continue
            for idx in didx:
                st['engine_checked_actions'] += 1
                ok = engine_win(env, idx)
                st['engine_wins' if ok else 'engine_failures'] += 1
                if not ok and bad is not None and len(bad) < 20: bad.append(dict(err='engine', tag=tag, idx=idx))
            if didx: st['engine_checked_states'] += 1
    return st


def sweep_task(args):
    stage, s = args
    env = G['env']; st = new_stats(); bad = []; rng = np.random.default_rng([7, stage, s]); rounds = STAGE_ROUNDS[stage]
    obs, certs, envs, tags = [], [], [], []
    def flush():
        if obs: evaluate(np.stack(obs), certs, envs, st, tags, bad)
        obs.clear(); certs.clear(); envs.clear(); tags.clear()
    for e in range(64):
        if e == s: continue
        for t in range(64):
            if t in (s, e): continue
            for team in (C.CYAN, C.MAGENTA):
                for al in (1, 2):
                    typ = TYPES[int(rng.integers(5))]; rnd = rounds[int(rng.integers(len(rounds)))]
                    make(env, s, e, t, team, al, rnd, typ)
                    cert = helper.certificate(env)
                    o = env.observation()
                    envs.append((s, e, t, team, al, rnd, typ))  # rebuilt only if either side flags
                    obs.append(o); certs.append(cert)
                    tags.append(f'stage{stage}/{team}/al{al}/{typ.value}' + ('/pos' if cert else ''))
                    if len(obs) >= BATCH: flush()
    flush()
    return st, bad


def positives_specs():
    specs = []
    dist = lambda a, b: max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    ring = lambda p: min(p[0], p[1], 7 - p[0], 7 - p[1])
    for rg, rnd in [(0, 8), (1, 16), (2, 24), (3, 26)]:
        sq = [(r, c) for r in range(rg, 8 - rg) for c in range(rg, 8 - rg)]
        for team in (C.CYAN, C.MAGENTA):
            for al in (1, 2):
                for s in sq:
                    for t in sq:
                        if dist(s, t) != al: continue
                        for e in sq:
                            if e in (s, t) or dist(t, e) <= 2: continue
                            if not (ring(t) > ring(e) or (ring(t) == ring(e) and team == C.MAGENTA)): continue
                            for ti in range(5):
                                specs.append((s[0] * 8 + s[1], e[0] * 8 + e[1], t[0] * 8 + t[1], team, al, rnd, ti))
    return specs


def positives_task(specs):
    env = G['env']; st = new_stats(); bad = []; obs, certs, envs, tags = [], [], [], []
    for k, (s, e, t, team, al, rnd, ti) in enumerate(specs):
        make(env, s, e, t, team, al, rnd, TYPES[ti])
        cert = helper.certificate(env)
        obs.append(env.observation()); certs.append(cert); envs.append((s, e, t, team, al, rnd, TYPES[ti]))
        tags.append(f'{TYPES[ti].value}/{team}/al{al}')
        if len(obs) >= BATCH or k == len(specs) - 1:
            evaluate(np.stack(obs), certs, envs, st, tags, bad); obs.clear(); certs.clear(); envs.clear(); tags.clear()
    return st, bad


def env_from_obs(o, env=None):
    """Rebuild an engine state from an engine observation; re-encode must be bitwise equal."""
    team = C.MAGENTA if o[15, 0, 0] > .5 else C.CYAN
    env = env if env is not None else ColonyOneEnv()
    env.board.grid = [[None] * 8 for _ in range(8)]
    env.board.locked = [[False] * 8 for _ in range(8)]
    for p in range(12):
        for cr, cc in zip(*np.nonzero(o[p] > .5)):
            r, c = E.canonical_pos((int(cr), int(cc)), team)
            env.board.grid[r][c] = Piece(list(PT)[p % 6], team if p < 6 else opp(team))
    for cr, cc in zip(*np.nonzero(o[12] > .5)):
        r, c = E.canonical_pos((int(cr), int(cc)), team); env.board.locked[r][c] = True
    stage = sum(env.board.locked[k][k] for k in range(4))
    u = float(o[14, 0, 0])
    rnd = C.SHRINK_ROUNDS[stage] - int(round((1 - u) * 8)) if stage < 4 else 27
    env.current_team = team; env.actions_left = int(round(float(o[13, 0, 0]) * 2)); env.round_number = rnd
    env.terminated = False; env.winner = None
    return env, bool(np.array_equal(env.observation(), o))


def selfplay_task(path):
    """Every observation: reconstruct the engine state, run the helper, compare all outputs."""
    X = np.load(path)['obs'].astype(np.float32); st = new_stats(); bad = []; env = G['env']
    for i in range(0, len(X), BATCH):
        x = X[i:i + BATCH]; certs, envs, tags = [], [], []
        for j in range(len(x)):
            _, same = env_from_obs(x[j], env); st['reencode_equal'] += int(same)
            cert = helper.certificate(env) if same else None
            certs.append(cert); envs.append(env.clone() if cert is not None else None)
            tags.append('helper_positive' if cert is not None else ('nonfamily' if same else 'not_reencodable'))
        # a wrapper-flagged state without a clone is rebuilt for the engine check
        flag, _ = proof_run({'observation': x})
        for j in range(len(x)):
            if flag[j] and envs[j] is None: envs[j] = env_from_obs(x[j])[0]
        evaluate(x, certs, envs, st, tags, bad)
    return st, bad


def merge(results):
    st = Counter(); bad = []
    for s, b in results: st.update(s); bad.extend(b)
    return dict(sorted(st.items())), bad[:20]


def write(name, payload):
    (HERE / f'verify-{name}.json').write_text(json.dumps(payload, indent=2)); print(name, json.dumps({k: v for k, v in payload.items() if k != 'tags'})[:1500], flush=True)


def run_sweep(pool):
    t = time.time(); tasks = [(stage, s) for stage in range(4) for s in range(64)]
    st, bad = merge(pool.imap_unordered(sweep_task, tasks, chunksize=1))
    write('sweep', dict(mode='sweep', counts={k: v for k, v in st.items() if not k.startswith('tag:')},
                        tags={k[4:]: v for k, v in st.items() if k.startswith('tag:')}, mismatches=bad, seconds=time.time() - t))


def run_positives(pool):
    t = time.time(); specs = positives_specs(); n = 400
    st, bad = merge(pool.imap_unordered(positives_task, [specs[i:i + n] for i in range(0, len(specs), n)]))
    write('positives', dict(mode='positives', specs=len(specs), counts={k: v for k, v in st.items() if not k.startswith('tag:')},
                            tags={k[4:]: v for k, v in st.items() if k.startswith('tag:')}, mismatches=bad, seconds=time.time() - t))


def run_selfplay(pool):
    t = time.time(); files = sorted(glob.glob(str(ROOT / 'data' / 'sp_*' / 'chunk*.npz')))
    st, bad = merge(pool.imap_unordered(selfplay_task, files, chunksize=1))
    write('selfplay', dict(mode='selfplay', files=len(files), counts={k: v for k, v in st.items() if not k.startswith('tag:')},
                           tags={k[4:]: v for k, v in st.items() if k.startswith('tag:')}, mismatches=bad, seconds=time.time() - t))


def run_audit():
    t = time.time(); init_worker()
    spec = importlib.util.spec_from_file_location('tp_audit_rerun', AUDIT)
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    out = HERE / 'audit_rerun'; out.mkdir(exist_ok=True); audit.HERE = out  # never write into Codex's folder
    orig = audit.certmod.certificate; calls = []
    def recorder(env):
        cert = orig(env); calls.append((env.observation(), cert, env.clone(), env.terminated)); return cert
    audit.certmod.certificate = recorder
    audit.main()
    rep = json.loads((out / 'report.json').read_text())
    names = [r['fixture'] for r in rep['fixture_results']] + ['witness-' + w['game'] for w in rep['real_witnesses']]
    assert len(names) == len(calls), (len(names), len(calls))
    st = new_stats(); bad = []; excluded = []
    keep = [k for k, c in enumerate(calls) if not c[3]]
    excluded = [names[k] for k, c in enumerate(calls) if c[3]]
    for i in range(0, len(keep), BATCH):
        ks = keep[i:i + BATCH]
        evaluate(np.stack([calls[k][0] for k in ks]), [calls[k][1] for k in ks], [calls[k][2] for k in ks], st,
                 ['witness' if names[k].startswith('witness') else 'fixture' for k in ks], bad)
    # terminated fixtures: the observation does not encode termination; report what the wrapper does
    term = []
    for k, c in enumerate(calls):
        if c[3]:
            f, _ = proof_run({'observation': c[0][None]}); term.append(dict(fixture=names[k], wrapper_flag=bool(f[0])))
    witnesses = []
    for k, name in enumerate(names):
        if name.startswith('witness'):
            _, m = proof_run({'observation': calls[k][0][None]})
            witnesses.append(dict(name=name, helper=calls[k][1]['winning_first_indices'], wrapper=np.flatnonzero(m[0]).tolist()))
    write('audit', dict(mode='audit', audit_status=rep['status'], audit_fixture_states=rep['fixture_states'],
                        audit_positive=rep['positive_fixtures'], audit_negative=rep['negative_or_unknown_fixtures'],
                        audit_paths=rep['independent_fixture_paths'], recorded_calls=len(calls),
                        counts={k: v for k, v in st.items() if not k.startswith('tag:')},
                        terminated_fixtures_outside_observation_domain=term, witnesses=witnesses,
                        mismatches=bad, seconds=time.time() - t))


def run_batches():
    init_worker(); rng = np.random.default_rng(5)
    sp = np.load(sorted(glob.glob(str(ROOT / 'data' / 'sp_*' / 'chunk*.npz')))[-1])['obs'][:200].astype(np.float32)
    fam = []; env = G['env']
    for s, e, t, team, al, rnd, ti in [x for i, x in enumerate(positives_specs()) if i % 997 == 0][:120]:
        make(env, s, e, t, team, al, rnd, TYPES[ti]); fam.append(env.observation())
    X = np.concatenate([sp, np.stack(fam)]); X = X[rng.permutation(len(X))]
    res = {}
    flag_all, _ = proof_run({'observation': X})
    for b in (1, 8, 16):
        P, V, BP, BV = [], [], [], []
        for i in range(0, len(X), b):
            x = X[i:i + b]
            if len(x) < b: continue
            p, v = G['wrap'].run(None, {'observation': x}); bp, bv = G['base'].run(None, {'observation': x})
            P.append(p); V.append(v); BP.append(bp); BV.append(bv)
        P, V, BP, BV = map(np.concatenate, (P, V, BP, BV)); n = len(P); fl = flag_all[:n]; nf = ~fl
        derived = np.any(P.view(np.uint32) != BP.view(np.uint32), axis=1) | (V.view(np.uint32) != BV.view(np.uint32))
        res[f'batch{b}'] = dict(states=int(n), flagged=int(fl.sum()), derived_flag_equals_proof=bool(np.array_equal(derived, fl)),
                                nonfamily_policy_bitwise=bool(np.array_equal(P[nf].view(np.uint32), BP[nf].view(np.uint32))),
                                nonfamily_value_bitwise=bool(np.array_equal(V[nf].view(np.uint32), BV[nf].view(np.uint32))),
                                family_value_099=bool(np.all(V[fl] == V99)),
                                family_policy_base_or_base_plus_30=bool(np.all((P[fl] == BP[fl]) | (P[fl] == BP[fl] + B30))))
    write('batches', dict(mode='batches', results=res))


def increment_session():
    """Only the tp_ nodes of the wrapper, with the base outputs fed as graph inputs."""
    import onnx, onnxruntime as ort
    from onnx import helper as oh, TensorProto as OT
    m = onnx.load(str(WRAP)); g = m.graph; keep = [n for n in g.node if n.name.startswith('tp_')]
    used = {i for n in keep for i in n.input}
    inc = oh.make_model(oh.make_graph(keep, 'increment', [g.input[0], oh.make_tensor_value_info('tp_base_policy', OT.FLOAT, ['batch', 4096]),
                        oh.make_tensor_value_info('tp_base_value', OT.FLOAT, ['batch'])], list(g.output),
                        [x for x in g.initializer if x.name in used]), opset_imports=m.opset_import)
    inc.ir_version = m.ir_version
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1; o.log_severity_level = 3
    return ort.InferenceSession(inc.SerializeToString(), o, providers=['CPUExecutionProvider']), inc


def run_passthrough():
    """Signed zeros, denormals, huge values: wrapper must pass base logits through bitwise
    outside the certified set, and add exactly +30 (float32) on it, for both default and
    disabled ORT graph optimisation."""
    import onnxruntime as ort
    init_worker(); rng = np.random.default_rng(11); env = G['env']
    sess, inc = increment_session()
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1; o.log_severity_level = 3
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_noopt = ort.InferenceSession(inc.SerializeToString(), o, providers=['CPUExecutionProvider'])
    obs, certs = [], []
    for spec in [x for i, x in enumerate(positives_specs()) if i % 1301 == 0][:256]:
        make(env, *spec[:6], TYPES[spec[6]]); obs.append(env.observation()); certs.append(helper.certificate(env))
    sp = np.load(sorted(glob.glob(str(ROOT / 'data' / 'sp_*' / 'chunk*.npz')))[0])['obs'][:256].astype(np.float32)
    obs.extend(sp); certs.extend([None] * len(sp))
    X = np.stack(obs); n = len(X)
    specials = np.array([0.0, -0.0, 1e-45, -1e-45, 1e-38, -1e-38, 1e30, -1e30, 3.4e38, -3.4e38, 1.0, -1.0], np.float32)
    st = Counter()
    for trial in range(4):
        bp = rng.standard_normal((n, 4096)).astype(np.float32) * 10
        pick = rng.random((n, 4096)) < 0.5
        bp[pick] = specials[rng.integers(len(specials), size=int(pick.sum()))]
        bv = np.where(rng.random(n) < 0.3, np.float32(-0.0), rng.uniform(-1, 1, n).astype(np.float32)).astype(np.float32)
        for name, sx in (('default_opt', sess), ('no_opt', sess_noopt)):
            for b in (1, 8, 16, n):
                for i in range(0, n - n % b, b):
                    p, v = sx.run(None, {'observation': X[i:i + b], 'tp_base_policy': bp[i:i + b], 'tp_base_value': bv[i:i + b]})
                    for j in range(len(p)):
                        k = i + j; cert = certs[k]; ep = bp[k].copy()
                        if cert: ep[cert['winning_first_indices']] = bp[k][cert['winning_first_indices']] + B30
                        ev = V99 if cert else bv[k]
                        st['states'] += 1; st['family' if cert else 'nonfamily'] += 1
                        st['neg_zero_logits_in'] += int(np.sum((bp[k] == 0) & np.signbit(bp[k])))
                        if not np.array_equal(p[j].view(np.uint32), ep.view(np.uint32)): st['policy_bitwise_mismatch'] += 1
                        if np.float32(v[j]).view(np.uint32) != np.float32(ev).view(np.uint32): st['value_bitwise_mismatch'] += 1
    write('passthrough', dict(mode='passthrough', counts=dict(st), specials=[float(x) for x in specials]))


def run_check():
    import onnxruntime as ort
    out = {}
    for name, p in (('base', BASE), ('wrapper', WRAP)):
        r = subprocess.run([PY, str(ROOT / 'check_onnx.py'), str(p)], capture_output=True, text=True, cwd=str(ROOT))
        out[name] = dict(exit=r.returncode, report=json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else r.stderr[-500:])
    import onnx
    from onnx import helper as oh, TensorProto as OT
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1; o.log_severity_level = 3
    # Isolated increment: only the tp_ nodes, with the base outputs fed as inputs.
    m = onnx.load(str(WRAP)); g = m.graph; keep = [n for n in g.node if n.name.startswith('tp_')]
    used = {i for n in keep for i in n.input}
    inc = oh.make_model(oh.make_graph(keep, 'increment', [g.input[0], oh.make_tensor_value_info('tp_base_policy', OT.FLOAT, ['batch', 4096]),
                        oh.make_tensor_value_info('tp_base_value', OT.FLOAT, ['batch'])], list(g.output),
                        [x for x in g.initializer if x.name in used]), opset_imports=m.opset_import)
    inc.ir_version = m.ir_version
    S = {'base': ort.InferenceSession(str(BASE), o, providers=['CPUExecutionProvider']),
         'wrapper': ort.InferenceSession(str(WRAP), o, providers=['CPUExecutionProvider']),
         'increment_only': ort.InferenceSession(inc.SerializeToString(), o, providers=['CPUExecutionProvider'])}
    real = np.load(sorted(glob.glob(str(ROOT / 'data' / 'sp_*' / 'chunk*.npz')))[-1])['obs'][:16].astype(np.float32)
    lat = {}
    for b in (1, 8, 16):
        x = real[:b]
        feeds = {'base': {'observation': x}, 'wrapper': {'observation': x},
                 'increment_only': {'observation': x, 'tp_base_policy': np.random.default_rng(1).standard_normal((b, 4096)).astype(np.float32),
                                    'tp_base_value': np.zeros(b, np.float32)}}
        for k, s in S.items():
            for _ in range(50): s.run(None, feeds[k])
        acc = {k: [] for k in S}; runs = 200; rounds = 80
        for r in range(rounds):
            for k in (list(S) if r % 2 else list(S)[::-1]):
                t = time.perf_counter()
                for _ in range(runs): S[k].run(None, feeds[k])
                acc[k].append((time.perf_counter() - t) / runs * 1000)
        med = {k: float(np.median(v)) for k, v in acc.items()}; mn = {k: float(np.min(v)) for k, v in acc.items()}
        lat[f'batch{b}'] = dict(ms_median=med, ms_min_of_rounds=mn,
                                delta_ms_median=med['wrapper'] - med['base'], delta_pct_median=(med['wrapper'] / med['base'] - 1) * 100,
                                delta_ms_min=mn['wrapper'] - mn['base'], delta_pct_min=(mn['wrapper'] / mn['base'] - 1) * 100,
                                paired_delta_ms_median=float(np.median(np.array(acc['wrapper']) - np.array(acc['base']))),
                                paired_delta_pct_median=float(np.median(np.array(acc['wrapper']) / np.array(acc['base']) - 1) * 100),
                                rounds=rounds, runs_per_round=runs)
    out['latency'] = dict(method='wall perf_counter per call, single-thread ORT, interleaved alternating order, real self-play obs; host shared with self-play actors', **lat)
    write('check', dict(mode='check', **out))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('modes', nargs='*', default=['audit', 'batches', 'passthrough', 'check', 'positives', 'selfplay', 'sweep'])
    ap.add_argument('--workers', type=int, default=4); a = ap.parse_args()
    assert a.workers <= 4
    if 'audit' in a.modes: run_audit()
    if 'batches' in a.modes: run_batches()
    if 'passthrough' in a.modes: run_passthrough()
    if 'check' in a.modes: run_check()
    pooled = [m for m in ('positives', 'selfplay', 'sweep') if m in a.modes]
    if pooled:
        with Pool(a.workers, initializer=init_worker) as pool:
            for m in pooled: globals()['run_' + m](pool)
