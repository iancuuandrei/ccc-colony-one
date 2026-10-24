"""Analyse features.npz + final_positions.json -> results.json.

python analysis/analyze.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
F = dict(np.load(OUT / 'features.npz'))
finals = json.loads((OUT / 'final_positions.json').read_text())
R: dict = {}
PIECES = ['ping', 'daemon', 'exploit', 'trace', 'ice']

z = F['z']; q = F['q']; gid = F['gid']; y = (z > 0).astype(float)
n_games = int(gid.max()) + 1
mag = F['magenta_to_move'] > 0.5

# ---------------------------------------------------------------- per game
last = np.where(F['left'] == 1)[0]
first = np.r_[0, last[:-1] + 1]
g_len = last - first + 1
g_round = F['round'][last]
g_winner_cyan = np.where(mag[last], z[last] < 0, z[last] > 0)


def pool_assignment(pp):
    rng = np.random.default_rng(12345)
    out = []
    for _ in range(n_games):
        out.append(int(rng.integers(2)) if rng.random() < pp else -1)  # 0 model-l3, 1 random
    return np.array(out)


learner_cyan = (np.arange(n_games) % 2) == 1
pool_check = {}
for pp in (0.15, 0.2):
    pa = pool_assignment(pp)
    lw = g_winner_cyan == learner_cyan
    pool_check[str(pp)] = {
        'n_pool': int((pa >= 0).sum()),
        'learner_win_vs_random': float(lw[pa == 1].mean()),
        'learner_win_vs_l3': float(lw[pa == 0].mean()),
        'learner_win_in_assumed_selfplay_cyan_rate': float(g_winner_cyan[pa < 0].mean()),
        'mean_len_vs_random': float(g_len[pa == 1].mean()),
        'mean_len_selfplay': float(g_len[pa < 0].mean()),
    }
R['pool_reconstruction_check'] = pool_check
PP = max(pool_check, key=lambda k: pool_check[k]['learner_win_vs_random'])
pool = pool_assignment(float(PP))
R['pool_prob_used'] = float(PP)
selfplay = pool < 0

# ---------------------------------------------------------------- Q1 colour balance
def rate(mask):
    return {'n': int(mask.sum()), 'cyan_win': round(float(g_winner_cyan[mask].mean()), 4) if mask.any() else None}


rb = {'1-8': (1, 8), '9-16': (9, 16), '17-24': (17, 24), '25-26': (25, 26)}
q1 = {'all': rate(np.ones(n_games, bool)), 'selfplay_only': rate(selfplay), 'pool_games': rate(~selfplay)}
q1['by_final_round'] = {k: rate((g_round >= a) & (g_round <= b)) for k, (a, b) in rb.items()}
q1['by_final_round_selfplay'] = {k: rate(selfplay & (g_round >= a) & (g_round <= b)) for k, (a, b) in rb.items()}
qs = np.quantile(g_len, [0, .25, .5, .75, 1])
q1['by_length_quartile_actions'] = {f'{int(qs[i])}-{int(qs[i+1])}': rate((g_len >= qs[i]) & (g_len <= qs[i + 1]) if i == 3 else (g_len >= qs[i]) & (g_len < qs[i + 1])) for i in range(4)}
q1['final_round_histogram'] = {int(k): int(v) for k, v in sorted(Counter(g_round.tolist()).items())}
R['q1_colour_balance'] = q1

# ---------------------------------------------------------------- Q2 how games end
cats = np.array([f['category'] for f in finals])
fin_team = np.array([f['team'] for f in finals])
q2 = {'overall': dict(Counter(cats.tolist()))}
q2['overall_frac'] = {k: round(v / n_games, 4) for k, v in q2['overall'].items()}
for k, (a, b) in rb.items():
    m = (g_round >= a) & (g_round <= b)
    q2[f'round_{k}'] = dict(Counter(cats[m].tolist()))
q2['by_category_cyan_win'] = {c: rate(cats == c) for c in set(cats.tolist())}
q2['firewall_by_round'] = {int(k): int(v) for k, v in sorted(Counter(g_round[cats == 'firewall'].tolist()).items())}
q2['mover_won_last_action'] = {c: round(float(np.mean([f['mover_won'] for f in finals if f['category'] == c])), 4) for c in set(cats.tolist())}
q2['multi_category_ambiguous'] = int(sum(len(f['candidates']) > 1 for f in finals))
# winning side's kernel-capture: did the attacker need two actions (guard killed by action 1)?
q2['kernel_capture_on_second_action_of_turn'] = int(sum(1 for f in finals if f['category'] == 'kernel_capture' and f['actions_left'] == 1))
q2['selfplay_only'] = dict(Counter(cats[selfplay].tolist()))
R['q2_game_endings'] = q2

# ---------------------------------------------------------------- Q3 logistic regression
def fit_logit(X, y, l2=1e-3, iters=50):
    X1 = np.c_[np.ones(len(X)), X]
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X1 @ w, -30, 30)))
        g = X1.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 / len(y) + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), X] @ w, -30, 30)))


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


F['mob_diff'] = F['mob_own'] - F['mob_en']
F['guard_diff'] = F['own_guards'] - F['en_guards']
F['kring_diff'] = F['own_kring'] - F['en_kring']
F['q_feat'] = q
BASIC = ['mat_count_diff', 'mat_value_diff', 'own_guards', 'en_guards', 'own_kring', 'en_kring',
         'magenta_to_move', 'actions_left', 'round']
TACT = BASIC + ['own_win_this_turn', 'en_win_next_turn_static', 'own_k_on_next_lock', 'en_k_on_next_lock',
                'mob_diff', 'n_atk_own_kernel', 'n_atk_en_kernel']
test_g = (gid % 5) == 0
phases = {'r1-8': (1, 8), 'r9-16': (9, 16), 'r17+': (17, 26)}
q3 = {}
for ph, (a, b) in phases.items():
    m = (F['round'] >= a) & (F['round'] <= b)
    tr, te = m & ~test_g, m & test_g
    res = {'n_train': int(tr.sum()), 'n_test': int(te.sum()), 'base_rate_mover_win': round(float(y[m].mean()), 4)}
    res['logloss_constant'] = logloss(np.full(te.sum(), y[tr].mean()), y[te])
    qp = (1 + q) / 2
    res['logloss_q_raw'] = logloss(qp[te], y[te])
    wq = fit_logit(q[tr, None], y[tr])
    res['logloss_q_platt'] = logloss(predict(wq, q[te, None]), y[te])
    res['q_platt_slope'] = round(float(wq[1]), 3)
    for name, cols in [('basic', BASIC), ('tactical', TACT), ('tactical_plus_q', TACT + ['q_feat'])]:
        X = np.stack([F[c].astype(float) for c in cols], 1)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xs = (X - mu) / sd
        w = fit_logit(Xs[tr], y[tr])
        res[f'logloss_{name}'] = logloss(predict(w, Xs[te]), y[te])
        res[f'coef_std_{name}'] = {c: round(float(v), 3) for c, v in zip(cols, w[1:])}
        if name == 'basic':
            res['coef_raw_basic'] = {c: round(float(v / s), 4) for c, v, s in zip(cols, w[1:], sd)}
    # drop-one importance on the basic model (test log-loss increase)
    X = np.stack([F[c].astype(float) for c in BASIC], 1)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = (X - mu) / sd
    full = res['logloss_basic']
    imp = {}
    for j, c in enumerate(BASIC):
        keep = [k for k in range(len(BASIC)) if k != j]
        w = fit_logit(Xs[tr][:, keep], y[tr])
        imp[c] = round(logloss(predict(w, Xs[te][:, keep]), y[te]) - full, 5)
    res['drop_one_logloss_increase_basic'] = imp
    # single-feature log-loss
    single = {}
    for c in TACT:
        x = F[c].astype(float)[:, None]
        w = fit_logit(x[tr], y[tr])
        single[c] = round(logloss(predict(w, x[te]), y[te]), 4)
    res['single_feature_logloss'] = single
    # calibration of q
    bins = np.array([-1.01, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.01])
    cal = []
    for i in range(len(bins) - 1):
        mm = m & (q >= bins[i]) & (q < bins[i + 1])
        if mm.sum() >= 30:
            cal.append({'q_bin': f'{bins[i]:.1f}..{bins[i+1]:.1f}', 'n': int(mm.sum()),
                        'mean_q': round(float(q[mm].mean()), 3), 'mean_z': round(float(z[mm].mean()), 3)})
    res['q_calibration'] = cal
    res['q_sign_accuracy'] = round(float(((q > 0) == (z > 0))[m].mean()), 4)
    res['mean_q'] = round(float(q[m].mean()), 3); res['mean_z'] = round(float(z[m].mean()), 3)
    # q calibration split by colour to move
    res['q_bias_by_side'] = {s: {'mean_q': round(float(q[m & mk].mean()), 3), 'mean_z': round(float(z[m & mk].mean()), 3)}
                             for s, mk in (('cyan_to_move', ~mag), ('magenta_to_move', mag))}
    q3[ph] = res
# q failure modes: tactical flags where q disagrees
fm = {}
for c in ['own_win_this_turn', 'en_win_next_turn_static', 'own_k_on_next_lock', 'en_k_on_next_lock']:
    mk = F[c] > 0
    fm[c] = {'n': int(mk.sum()), 'mean_q': round(float(q[mk].mean()), 3), 'mean_z': round(float(z[mk].mean()), 3)}
mk = (F['en_win_next_turn_static'] > 0) & (F['actions_left'] == 1) & (F['own_win_this_turn'] == 0)
fm['enemy_threat_and_mover_has_1_action_no_win'] = {'n': int(mk.sum()), 'mean_q': round(float(q[mk].mean()), 3), 'mean_z': round(float(z[mk].mean()), 3)}
for d in range(-4, 5, 2):
    mk = np.rint(F['mat_value_diff']) == d
    fm[f'mat_value_diff~{d}'] = {'n': int(mk.sum()), 'mean_q': round(float(q[mk].mean()), 3), 'mean_z': round(float(z[mk].mean()), 3)}
# q vs z by left-to-end bucket
for lo, hi in [(1, 4), (5, 12), (13, 24), (25, 999)]:
    mk = (F['left'] >= lo) & (F['left'] <= hi)
    fm[f'left_{lo}-{hi}'] = {'n': int(mk.sum()), 'mean_abs_q': round(float(np.abs(q[mk]).mean()), 3),
                             'mse_q_z': round(float(((q - z)[mk] ** 2).mean()), 3), 'sign_acc': round(float(((q > 0) == (z > 0))[mk].mean()), 3)}
R['q3_logistic'] = q3
R['q3_q_failure_modes'] = fm

# ---------------------------------------------------------------- Q4 piece losses
# absolute colour counts: own planes belong to the mover
own = np.stack([F[f'own_{p}'] for p in PIECES], 1); en = np.stack([F[f'en_{p}'] for p in PIECES], 1)
cy = np.where(mag[:, None], en, own); mg = np.where(mag[:, None], own, en)
stage = F['stage']
first_loss = defaultdict(lambda: [0, 0])   # (loser colour-agnostic) type -> [n, loser_wins]
first_loss_cause = Counter()
race = defaultdict(lambda: [0, 0])  # type -> [n games where one side lost it strictly first, that side's wins]
ever_lost = {p: {'lost': [0, 0], 'kept': [0, 0]} for p in PIECES}
first_loss_round = defaultdict(list)
for g in range(n_games):
    s, e = first[g], last[g]
    cwin = g_winner_cyan[g]
    dc = np.diff(cy[s:e + 1], axis=0) < 0; dm = np.diff(mg[s:e + 1], axis=0) < 0
    fw = np.diff(stage[s:e + 1]) > 0
    ev = np.where(dc.any(1) | dm.any(1))[0]
    if len(ev):
        t = ev[0]
        losers = [(True, j) for j in np.where(dc[t])[0]] + [(False, j) for j in np.where(dm[t])[0]]
        cause = 'firewall' if fw[t] else 'capture'
        first_loss_cause[cause] += 1
        if len(losers) == 1 and cause == 'capture':
            is_c, j = losers[0]
            won = cwin if is_c else not cwin
            first_loss[PIECES[j]][0] += 1; first_loss[PIECES[j]][1] += int(won)
            first_loss_round[PIECES[j]].append(int(F['round'][s + t]))
    for j, p in enumerate(PIECES):
        tc = np.where(dc[:, j])[0]; tm = np.where(dm[:, j])[0]
        tc0 = tc[0] if len(tc) else 10 ** 6; tm0 = tm[0] if len(tm) else 10 ** 6
        if tc0 != tm0:
            c_first = tc0 < tm0
            race[p][0] += 1; race[p][1] += int(cwin if c_first else not cwin)
        for is_c, lost in ((True, len(tc) > 0), (False, len(tm) > 0)):
            won = cwin if is_c else not cwin
            k = 'lost' if lost else 'kept'
            ever_lost[p][k][0] += 1; ever_lost[p][k][1] += int(won)
q4 = {
    'first_capture_loss_of_game': {p: {'n': v[0], 'loser_win_rate': round(v[1] / v[0], 4),
                                       'median_round': float(np.median(first_loss_round[p]))} for p, v in first_loss.items()},
    'first_loss_cause': dict(first_loss_cause),
    'lost_type_first_race': {p: {'n': v[0], 'win_rate_of_side_losing_it_first': round(v[1] / v[0], 4)} for p, v in race.items()},
    'ever_lost_vs_kept': {p: {k: {'n': v[0], 'win_rate': round(v[1] / max(v[0], 1), 4)} for k, v in d.items()} for p, d in ever_lost.items()},
}
R['q4_piece_losses'] = q4

(OUT / 'results.json').write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))

# ---------------------------------------------------------------- extras
# empirical piece values: logistic on per-type count differences (+ side, guards, rings) per phase
ex = {}
for ph, (a, b) in phases.items():
    m = (F['round'] >= a) & (F['round'] <= b)
    tr, te = m & ~test_g, m & test_g
    cols = [f'd_{p}' for p in PIECES] + ['magenta_to_move', 'own_guards', 'en_guards', 'own_kring', 'en_kring']
    for p in PIECES:
        F[f'd_{p}'] = F[f'own_{p}'] - F[f'en_{p}']
    X = np.stack([F[c].astype(float) for c in cols], 1)
    w = fit_logit(X[tr], y[tr])
    raw = dict(zip(cols, w[1:]))
    ping = raw['d_ping']
    ex[ph] = {'logit_per_piece': {p: round(float(raw[f'd_{p}']), 3) for p in PIECES},
              'relative_to_ping': {p: round(float(raw[f'd_{p}'] / ping), 2) for p in PIECES},
              'logloss_test': logloss(predict(w, X[te]), y[te])}
R['extra_empirical_piece_values'] = ex
# firewall endings: whose kernel burned (last position is Magenta's final action of the round)
fwm = cats == 'firewall'
li = last[fwm]
cy_k_on = np.where(mag[li], F['en_k_on_next_lock'][li], F['own_k_on_next_lock'][li])
mg_k_on = np.where(mag[li], F['own_k_on_next_lock'][li], F['en_k_on_next_lock'][li])
R['extra_firewall_endings'] = {
    'n': int(fwm.sum()), 'mover_magenta_frac': round(float(mag[li].mean()), 3),
    'cyan_kernel_on_locking_ring': int(cy_k_on.sum()), 'magenta_kernel_on_locking_ring': int(mg_k_on.sum()),
    'both_on_ring': int((cy_k_on & mg_k_on).sum()),
}
# how many earlier positions in those games already had the losing kernel on the next-lock ring at round 8
R['extra_kernel_capture_second_action_frac'] = round(q2['kernel_capture_on_second_action_of_turn'] / q2['overall']['kernel_capture'], 4)
(OUT / 'results.json').write_text(json.dumps(R, indent=1))
print(json.dumps({k: R[k] for k in R if k.startswith('extra')}, indent=1))
