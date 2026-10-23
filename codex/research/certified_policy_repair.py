"""Project existing GEN02 800-simulation policies onto exact tactical certificates.

No searches or value labels are created. Missing-teacher cases receive explicit
proof-only uniform targets, never attributed to an 800-simulation teacher.
Engine clone/step exhaustively
checks own-turn wins and existence of an own-turn completion that prevents an
opponent terminal win on its immediately following turn. This bounded safety
property does not prove eventual victory. No held-out synthetic input is read.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'env'), str(ROOT / 'codex/matches'), str(ROOT / 'codex/engine_audit')]
from colony_one import encoding
from own_turn_oracle import own_turn_wins
from site_replay import parse

BASE_SHA = '579d3ec01da7328dc706003b8dcc5eba357bb74902252187e45633ca0595a9a0'
K = 128

def sha(data):
    return hashlib.sha256(data).hexdigest()

def obs_key(obs):
    return sha(np.ascontiguousarray(obs, dtype=np.float16).tobytes())

def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()

def atomic_json(path, value, immutable=False):
    data = json_bytes(value)
    if immutable and path.exists():
        if path.read_bytes() != data:
            raise ValueError('Would mutate immutable evidence: ' + str(path))
        return
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(data)
    temp.replace(path)

def step_copy(env, action):
    child = env.clone()
    child.step(action)
    return child

def winning_path(env, team, counts):
    """Find one terminal win before this team's turn ends, via env.step only."""
    if env.terminated:
        return [] if env.winner == team else None
    if env.current_team != team:
        return None
    for action in env.legal_actions():
        counts['enemy_or_own_win_branches'] += 1
        idx = int(encoding.action_index(action, team))
        child = step_copy(env, action)
        rest = winning_path(child, team, counts)
        if rest is not None:
            return [idx] + rest
    return None

def safety_completion(env, owner, opponent, counts):
    """Existential own completion; unsafe means every continuation has a witness."""
    if env.terminated:
        safe = env.winner == owner
        return safe, {'terminal_winner': env.winner}
    if env.current_team == opponent:
        witness = winning_path(env, opponent, counts)
        return witness is None, {'opponent_winning_actions': witness,
                                  'exhaustive_no_win': witness is None}
    if env.current_team != owner:
        raise ValueError('Unknown turn owner')
    failures = []
    for action in env.legal_actions():
        counts['own_completion_branches'] += 1
        idx = int(encoding.action_index(action, owner))
        safe, certificate = safety_completion(step_copy(env, action), owner, opponent, counts)
        if safe:
            return True, {'safe_own_action': idx, 'continuation': certificate}
        failures.append({'own_action': idx, 'counterexample': certificate})
    if not failures:
        raise ValueError('No legal own completion; cannot certify safety or loss')
    return False, {'all_own_continuations_lose_within_bound': failures}

def prove(env):
    owner = env.current_team
    opponent = 'CYAN' if owner == 'MAGENTA' else 'MAGENTA'
    original = env.observation().copy()
    legal = env.legal_action_indices().astype(np.int32)
    counts = {'enemy_or_own_win_branches': 0, 'own_completion_branches': 0}
    win = np.zeros(4096, bool)
    safe = np.zeros(4096, bool)
    details = []
    for action in env.legal_actions():
        idx = int(encoding.action_index(action, owner))
        child = step_copy(env, action)
        witness = winning_path(child, owner, counts)
        win[idx] = witness is not None
        safe[idx], certificate = safety_completion(child, owner, opponent, counts)
        details.append({'first_action': idx, 'own_turn_win': win[idx].item(),
                        'winning_actions': [idx] + witness if witness is not None else None,
                        'bounded_safe': safe[idx].item(), 'safety_certificate': certificate})
    # Different implementation: reversible board operations rather than env clones.
    independent = own_turn_wins(env.board, owner, env.actions_left, env.round_number)
    if not np.array_equal(win, independent.first_action_win_mask):
        raise ValueError('Independent own-turn oracle disagreement')
    if not np.array_equal(original, env.observation()):
        raise ValueError('Proof mutated the source environment')
    if np.any(win & ~safe):
        raise ValueError('Winning action was classified unsafe')
    return legal, win, safe, {'counts': counts, 'actions': details,
                             'independent_own_turn_oracle': 'PASS'}

def teacher_index(folder):
    manifest_data = (folder / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_data)
    if manifest['sims'] != 800:
        raise ValueError('Expected existing 800-simulation teacher')
    model_path = ROOT / manifest['model']
    if sha(model_path.read_bytes()) != BASE_SHA:
        raise ValueError('Teacher-declared model path differs from authoritative GEN02')
    index, files = {}, []
    for path in sorted(folder.glob('chunk*.npz')):
        raw = path.read_bytes()
        file_hash = sha(raw)
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        for i, obs in enumerate(z['obs']):
            if not z['has_pol'][i]:
                continue
            dense = np.zeros(4096, np.float64)
            ids, probs = z['pidx'][i], z['pval'][i].astype(np.float64)
            valid = ids >= 0
            if np.any(ids[valid] >= 4096) or len(np.unique(ids[valid])) != valid.sum():
                raise ValueError('Invalid teacher action indices')
            if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs[~valid] != 0):
                raise ValueError('Invalid teacher probabilities')
            dense[ids[valid]] = probs[valid]
            total = dense.sum()
            if total <= 0:
                raise ValueError('Empty teacher target')
            dense /= total
            entry = {'file': path.relative_to(ROOT).as_posix(), 'file_sha256': file_hash,
                     'row': i, 'probability': dense, 'stored_probability_sum': float(total)}
            index.setdefault(obs_key(obs), []).append(entry)
        files.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': file_hash,
                      'rows': len(z['obs'])})
    return index, {'manifest': manifest, 'manifest_sha256': sha(manifest_data), 'files': files,
                   'model_sha256_current_verified': BASE_SHA,
                   'historical_model_hash_in_teacher_manifest': manifest.get('model_sha256'),
                   'identity_limit': 'Teacher manifest declares model path; current bytes verified. Historical generation hash was not recorded unless shown above.'}

def chunk(path, arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise ValueError('Would mutate immutable training chunk: ' + str(path))
    else:
        temp = path.with_suffix('.npz.tmp')
        with temp.open('wb') as out:
            np.savez_compressed(out, **arrays)
        with np.load(temp, allow_pickle=False) as verify:
            if any(not np.array_equal(verify[k], v) for k, v in arrays.items()):
                raise ValueError('Atomic chunk verification failed')
        temp.replace(path)
    return sha(path.read_bytes())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', type=Path, default=ROOT / 'data/ext_site_v1_rs')
    p.add_argument('--out', type=Path, default=ROOT / 'data/ext_site_tactics_v1')
    p.add_argument('--hard', type=Path, nargs='*')
    args = p.parse_args()
    args.teacher = args.teacher.resolve()
    args.out = args.out.resolve()
    if args.hard:
        args.hard = [path.resolve() for path in args.hard]
    baseline = ROOT / 'release/gen02/model.onnx'
    if sha(baseline.read_bytes()) != BASE_SHA:
        raise ValueError('Authoritative GEN02 bytes changed; review before integration')
    sources = args.hard or sorted((ROOT / 'codex/matches').glob('site-gen02-loss-hard-v*.npz'), reverse=True) + sorted((ROOT / 'codex/matches').glob('site-gen02-wins-hard-v*.npz'), reverse=True)
    if any('holdout' in str(s).lower() for s in sources):
        raise ValueError('Held-out input prohibited')
    index, teacher = teacher_index(args.teacher)
    raw_manifest_bytes = (ROOT / 'data/ext_site_v1/manifest.json').read_bytes()
    episodes = {x['game_id']: x for x in json.loads(raw_manifest_bytes)['episodes']}
    args.out.mkdir(parents=True, exist_ok=True)
    prior_chunks = []
    if (args.out / 'manifest.json').exists():
        prior_manifest = json.loads((args.out / 'manifest.json').read_text())
        for row in prior_manifest['published']:
            digest = sha((args.out / row['chunk']).read_bytes())
            if digest != row['chunk_sha256']:
                raise ValueError('Existing published chunk hash changed')
            prior_chunks.append({'chunk': row['chunk'], 'sha256': digest})
    rows, pending, skipped, proof_records, source_files = [], [], [], [], []
    seen, games = set(), {}
    for source in sources:
        raw = source.read_bytes()
        source_files.append({'path': source.relative_to(ROOT).as_posix(), 'sha256': sha(raw)})
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        for i, (gid, ply, obs) in enumerate(zip(z['match'], z['ply'], z['obs'])):
            gid, ply = str(gid), int(ply)
            if (gid, ply) in seen:
                continue
            seen.add((gid, ply))
            if gid not in games:
                protocol = ROOT / 'codex/matches/site-all-v1-derived' / ('match-' + gid + '.txt')
                game, states = parse(protocol, None)
                # Windows write_text translates LF to CRLF in derived files;
                # the exporter records the original protocol string's UTF-8.
                if sha(protocol.read_text(encoding='utf-8-sig').encode()) != episodes[gid]['protocol_sha256']:
                    raise ValueError('Protocol lineage mismatch: ' + gid)
                if sha(Path(episodes[gid]['source']).read_bytes()) != episodes[gid]['source_sha256']:
                    raise ValueError('Raw source game file lineage mismatch: ' + gid)
                games[gid] = (game, states)
            state = games[gid][1][ply - 1]
            if state.action != ply or not np.array_equal(state.observation(), obs):
                raise ValueError('Hard snapshot differs from full legal protocol replay')
            legal, win, safe, certificate = prove(state)
            if 'legal_mask' in z and not np.array_equal(np.flatnonzero(z['legal_mask'][i]), np.sort(legal)):
                raise ValueError('Provided legal mask differs from independent replay')
            if 'safe_start_mask' in z and not np.array_equal(safe, z['safe_start_mask'][i]):
                raise ValueError('Provided safe-start mask disagrees with independent proof')
            identity = {'game_id': gid, 'ply': ply, 'group': 'site-game-' + gid,
                        'team': state.current_team, 'actions_left': state.actions_left,
                        'round': state.round_number, 'observation_float16_sha256': obs_key(obs),
                        'source_npz': source.relative_to(ROOT).as_posix(), 'source_npz_sha256': sha(raw),
                        'source_npz_row': i, 'protocol_sha256': episodes[gid]['protocol_sha256'],
                        'derived_protocol_file_sha256': games[gid][0]['sha256'],
                        'source_game_json_sha256': episodes[gid]['source_sha256'],
                        'model_sha256': BASE_SHA, 'legal_indices': legal.tolist(),
                        'own_turn_winning_indices': np.flatnonzero(win).tolist(),
                        'bounded_safe_indices': np.flatnonzero(safe).tolist()}
            proof = dict(identity, proof=certificate)
            proof_hash = sha(json_bytes(proof))
            proof_records.append(dict(proof, proof_sha256=proof_hash))
            selected = win if win.any() else safe
            rec = dict(identity, proof_sha256=proof_hash,
                       certificate_kind='exact_own_turn_terminal_win' if win.any() else 'bounded_next_opponent_turn_safe')
            if not selected.any():
                skipped.append(dict(rec, reason='No admissible action within the certified bound; earlier recoverable hard snapshot processed separately'))
                print(json.dumps({'game': gid, 'ply': ply, 'result': 'SKIPPED_NO_ADMISSIBLE_ACTION'}), flush=True)
                continue
            teachers = index.get(obs_key(obs), [])
            legal_mask = np.zeros(4096, bool)
            legal_mask[legal] = True
            projected = np.zeros(4096, np.float64)
            if teachers:
                dense = teachers[0]['probability'].copy()
                if any(not np.allclose(dense, t['probability'], rtol=0, atol=1e-7) for t in teachers[1:]):
                    raise ValueError('Conflicting teacher targets for identical observation')
                if np.any(dense[~legal_mask] != 0):
                    raise ValueError('Teacher assigns mass outside complete legal set')
                mass = float(dense[selected].sum())
                fallback = mass == 0
                projected[selected] = 1 / selected.sum() if fallback else dense[selected] / mass
                teacher_receipt = {k: v for k, v in teachers[0].items() if k != 'probability'}
                teacher_source = 'EXISTING_800SIM'
                target_source = 'TEACHER_ZERO_MASS_UNIFORM_CERTIFIED_SET' if fallback else 'TEACHER_CONDITIONAL_PROJECTION'
                before_unsafe = float(dense[legal_mask & ~safe].sum())
                before_excluded = float(dense[legal_mask & ~selected].sum())
            else:
                # Explicitly authorised policy-only supervision. No teacher mass
                # is invented, and this does not assert a full-game value.
                mass = None
                fallback = True
                projected[selected] = 1 / selected.sum()
                teacher_receipt = None
                teacher_source = 'NONE'
                target_source = 'PROOF_ONLY_UNIFORM'
                before_unsafe = before_excluded = None
            if len(legal) > K:
                raise ValueError('Complete legal indices exceed shared128-wide policy contract')
            ids = np.full((1, K), -1, np.int16)
            ids[0, :len(legal)] = legal
            probs = np.zeros((1, K), np.float32)
            probs[0, :len(legal)] = projected[legal]
            arrays = {'obs': np.asarray(obs[None], np.float16), 'pidx': ids, 'pval': probs,
                      'z': np.zeros(1, np.float32), 'q': np.zeros(1, np.float32),
                      'has_pol': np.ones(1, bool), 'has_value': np.zeros(1, bool),
                      'q_valid': np.zeros(1, bool), 'left': np.ones(1, np.float32)}
            assert np.isfinite(probs).all() and abs(float(probs.sum()) - 1) < 1e-6
            assert np.all(projected[~selected] == 0) and set(ids[0][ids[0] >= 0]) == set(legal)
            name = f'chunk-site-tactics-{gid}-ply{ply:03}.npz'
            row = dict(rec, chunk=name, chunk_sha256=chunk(args.out / name, arrays), rows=1,
                       teacher=teacher_receipt, teacher_source=teacher_source, target_source=target_source,
                       matching_teacher_rows=[{k: v for k, v in t.items() if k != 'probability'} for t in teachers],
                       identical_teacher_rows=len(teachers), teacher_mass_on_admissible=mass,
                       unsafe_probability_before=before_unsafe,
                       unsafe_probability_before_status='UNKNOWN' if not teachers else 'MEASURED_EXISTING_800SIM',
                       outside_selected_probability_before=before_excluded,
                       unsafe_probability_after=0., outside_selected_probability_after=0.,
                       fallback='PROOF_ONLY_UNIFORM_MISSING_TEACHER' if not teachers else ('UNIFORM_CERTIFIED_SET_ZERO_TEACHER_MASS' if fallback else None),
                       projection='uniform over independently certified set; no teacher' if not teachers else 'conditional teacher distribution; KL(p||teacher) optimum only when retained mass>0',
                       full_game_value_training=False, approximate_q_training=False)
            rows.append(row)
            print(json.dumps({'game': gid, 'ply': ply, 'result': 'PUBLISHED_POLICY_ONLY',
                              'teacher_unsafe_mass': row['unsafe_probability_before'],
                              'target_source': target_source, 'fallback': fallback}), flush=True)
    dependencies = {'baseline_sha256': BASE_SHA, 'teacher': teacher,
                    'preserved_prior_chunks': prior_chunks,
                    'raw_site_manifest_sha256': sha(raw_manifest_bytes), 'hard_sources': source_files,
                    'engine_sha256': {p.relative_to(ROOT).as_posix(): sha(p.read_bytes())
                        for p in sorted((ROOT / 'env/colony_one').glob('*.py'))},
                    'script_sha256': sha(Path(__file__).read_bytes()),
                    'own_turn_oracle_sha256': sha((ROOT / 'codex/engine_audit/own_turn_oracle.py').read_bytes())}
    run_id = sha(json_bytes(dependencies))[:20]
    proof_name, report_name = f'proofs-{run_id}.json', f'report-{run_id}.json'
    atomic_json(args.out / proof_name, {'dependencies': dependencies, 'states': proof_records}, immutable=True)
    override_entries = []
    for row in rows:
        with np.load(args.out / row['chunk'], allow_pickle=False) as repaired:
            replacement = {'chunk': row['chunk'], 'chunk_sha256': row['chunk_sha256'], 'row': 0,
                           'pidx': repaired['pidx'][0].tolist(), 'pval': repaired['pval'][0].tolist(),
                           'has_pol': True, 'has_value': False, 'q_valid': False,
                           'z': 0., 'q': 0., 'left': 1.}
        override_entries.append({'observation_float16_sha256': row['observation_float16_sha256'],
                                 'source_game_group': row['group'], 'game_id': row['game_id'], 'ply': row['ply'],
                                 'certificate_kind': row['certificate_kind'], 'replacement': replacement,
                                 'proof_file': proof_name, 'proof_file_sha256': sha((args.out / proof_name).read_bytes()),
                                 'proof_record_sha256': row['proof_sha256'],
                                 'teacher_source_rows': row['matching_teacher_rows'],
                                 'teacher_source': row['teacher_source'], 'target_source': row['target_source'],
                                 'teacher_model_sha256_current_verified': BASE_SHA if row['teacher_source'] != 'NONE' else None,
                                 'future_approximate_rows': 'Apply by exact observation hash even when no teacher source row existed at publication',
                                 'full_game_win_claim': False})
    overrides = {'version': 1, 'run_id': run_id, 'authoritative_policy_override': True,
                 'baseline_model_sha256': BASE_SHA, 'entries': override_entries,
                 'key_rule': 'SHA256 of contiguous float16 (16,8,8) observation bytes; exact equality, no fuzzy matching',
                 'apply_to_source_folders': [args.teacher.relative_to(ROOT).as_posix()],
                 'replacement_rule': 'Replace matching approximate search-policy targets, including all matching teacher rows; set q_valid=False and has_value=False. Do not merely sample corrective duplicates alongside conflicting original targets.',
                 'raw_outcome_rule': 'Do not overwrite separate real-game outcome labels in data/ext_site_v1.',
                 'weighting_rule': 'Sampling weights are a separate Opus-owned decision; this file changes no weights or trainer code.'}
    override_name = f'policy-overrides-{run_id}.json'
    atomic_json(args.out / override_name, overrides, immutable=True)
    atomic_json(args.out / 'policy-overrides.json', overrides)
    report = {'version': 1, 'run_id': run_id, 'kind': 'certificate_guided_policy_only_projection',
              'rows': len(rows), 'pending_teacher_rows': len(pending), 'skipped_no_admissible_rows': len(skipped),
              'teacher_backed_rows': sum(row['teacher_source'] != 'NONE' for row in rows),
              'proof_only_uniform_rows': sum(row['target_source'] == 'PROOF_ONLY_UNIFORM' for row in rows),
              'preserved_prior_chunks': prior_chunks,
              'published': rows, 'pending': pending, 'skipped': skipped, 'proof_file': proof_name,
              'proof_file_sha256': sha((args.out / proof_name).read_bytes()), 'dependencies': dependencies,
              'policy_override_file': override_name,
              'policy_override_sha256': sha((args.out / override_name).read_bytes()),
              'baseline': 'GEN02 authoritative; no models, searches or training runs created',
              'value_contract': 'All rows has_value=False,q_valid=False,z=q=0,left=1,has_pol=True',
              'scope': 'Exact own-turn wins take priority; otherwise bounded next-turn safety only. No full-game optimality claim.',
              'holdout_policy': 'No holdout_v1 or synthetic held-out data read',
              'split_rule': 'Keep whole source game group together across every split; do not treat repaired plies as independent games.'}
    atomic_json(args.out / report_name, report, immutable=True)
    manifest = dict(report, report_file=report_name, report_sha256=sha((args.out / report_name).read_bytes()))
    atomic_json(args.out / 'manifest.json', manifest)
    atomic_json(args.out / 'summary.json', {k: v for k, v in manifest.items() if k not in ('published', 'pending', 'skipped', 'dependencies')})
    print(json.dumps({'complete': True, 'rows': len(rows), 'pending': len(pending),
                      'skipped': len(skipped), 'manifest': str(args.out / 'manifest.json')}), flush=True)

if __name__ == '__main__':
    main()
