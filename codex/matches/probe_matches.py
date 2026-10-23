"""Small exact-engine match capture and replay diagnostics; no training or promotion."""
from __future__ import annotations
import argparse, collections, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'env'))
from colony_one import constants, encoding, mcts
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator
from colony_one.pieces import PieceType
from colony_one.rules import legal_actions_for_team


def pieces(board):
    return [(r, c, p) for r in range(8) for c in range(8) if (p := board.grid[r][c])]


def feature(env, ev=None):
    board = env.board
    ps = pieces(board)
    out = dict(action=env.action, round=env.round_number, team=env.current_team,
               actions_left=env.actions_left, next_ring=board.next_open_ring())
    for team in (constants.CYAN, constants.MAGENTA):
        own = [(r,c,p) for r,c,p in ps if p.team == team]
        ker = next(((r,c) for r,c,p in own if p.piece_type == PieceType.KERNEL), None)
        guards = [] if ker is None else [(r,c,p.piece_type.value) for r,c,p in own
            if p.piece_type != PieceType.KERNEL and max(abs(r-ker[0]),abs(c-ker[1])) == 1]
        enemy = constants.MAGENTA if team == constants.CYAN else constants.CYAN
        enemy_legal = legal_actions_for_team(board, enemy)
        attacks = [a for a in enemy_legal if a.to_pos == ker]
        out[team] = dict(kernel=ker, guards=guards,
            ring=None if ker is None else min(*ker,7-ker[0],7-ker[1]),
            pieces={p.piece_type.value:[r,c] for r,c,p in own},
            enemy_attacks_kernel=[dict(fr=a.from_pos,to=a.to_pos,kind=a.kind) for a in attacks],
            legal_count=len(legal_actions_for_team(board,team)))
    if ev:
        out['value'] = float(ev(env.observation()[None])[1][0])
    return out


def apply(env, action, ev=None):
    before = feature(env, ev)
    board = env.board
    rec = board.apply_action(action)
    event = dict(team=env.current_team,fr=list(action.from_pos),to=list(action.to_pos),kind=action.kind,
        piece=rec.moved_piece.piece_type.value,moved=rec.moved,
        capture=None if rec.captured_piece is None else rec.captured_piece.piece_type.value,
        actual_target=None if rec.actual_target_pos is None else list(rec.actual_target_pos))
    board.undo_action(rec)
    before_pieces = {(r,c,p.team,p.piece_type.value) for r,c,p in pieces(board)}
    env.step(action)
    after_pieces = {(r,c,p.team,p.piece_type.value) for r,c,p in pieces(board)}
    if before['next_ring'] != board.next_open_ring():
        event['firewall_killed'] = [list(x) for x in sorted(before_pieces-after_pieces)
            if board.is_locked(x[0], x[1])]
    cause = None
    if env.terminated:
        cause = 'kernel_capture' if event['capture'] == PieceType.KERNEL.value else (
            'bare_kernels' if board.only_kernels_remain() else 'firewall')
    return dict(before=before,event=event,terminal=env.terminated,winner=env.winner,cause=cause)


def play(seed, model):
    ev = OnnxEvaluator(model)
    env = ColonyOneEnv()
    rng = np.random.default_rng(seed)
    hist=[]
    start=time.monotonic()
    while not env.terminated:
        action = mcts.search_action(env.board,env.current_team,env.actions_left,env.round_number,
            ev,mcts.MCTSConfig(simulations=100),rng=rng)
        if action is None:
            raise RuntimeError('No action; engine needs no-action referee handling')
        hist.append(apply(env,action,ev))
    return dict(seed=seed,model=str(model),sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),
        winner=env.winner,actions=len(hist),seconds=time.monotonic()-start,history=hist)


def summarize(g):
    rows=g['history']; last=rows[-1]; loser='CYAN' if g['winner']=='MAGENTA' else 'MAGENTA'
    finals=last['before']
    guarded_hits=[r for r in rows if r['event']['capture'] and r['event']['actual_target'] != r['event']['to']]
    threats=[]
    for r in rows:
        s=r['before']; en='MAGENTA' if s['team']=='CYAN' else 'CYAN'
        if s[en]['enemy_attacks_kernel'] and len(s[en]['guards']) < s['actions_left']:
            threats.append(dict(action=s['action'],team=s['team'],guards=len(s[en]['guards']),
                left=s['actions_left'],value=s.get('value'),chosen_capture=r['event']['capture'],
                chosen_target=r['event']['to'],kernel=s[en]['kernel']))
    last_loser_turn=[r for r in rows if r['before']['team']==loser][-2:]
    return dict(seed=g.get('seed'),winner=g['winner'],actions=len(rows),cause=last['cause'],
        losing_kernel_before=finals[loser],guard_redirects=len(guarded_hits),
        provable_kernel_win_opportunities=threats,last_loser_turn=last_loser_turn,tail=rows[-6:])


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default=str(ROOT/'model-h1.onnx'))
    p.add_argument('--seeds',type=int,nargs='*',default=[12001,12002,12003,12004]);p.add_argument('--replay')
    p.add_argument('--out',default=str(Path(__file__).parent/'selfplay-h1.json'));o=p.parse_args()
    if o.replay:
        data=json.loads(Path(o.replay).read_text(encoding='utf-8'))
        games=data['games']
        results=[]
        for g in games:
            env=ColonyOneEnv(); rows=[]
            for r in g['history']:
                # Captured probe records or legacy arena.py [side, from, to].
                if isinstance(r,dict):
                    e=r.get('event',r);fr,to=e['fr'],e['to']
                else: _,fr,to=r
                legal=env.legal_actions()
                action=next((a for a in legal if list(a.from_pos)==list(fr) and list(a.to_pos)==list(to)),None)
                if action is None: raise ValueError(f'Illegal replay action {env.action}: {fr}->{to}')
                rows.append(apply(env,action))
            if not env.terminated: raise ValueError('Incomplete replay')
            if env.winner!=g['winner']: raise ValueError(f'Winner mismatch: engine={env.winner} record={g["winner"]}')
            results.append(summarize(dict(g,history=rows)))
        Path(o.out).write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(json.dumps([dict(seed=x['seed'],winner=x['winner'],actions=x['actions'],cause=x['cause'],guard_redirects=x['guard_redirects']) for x in results]))
        return
    if len(o.seeds)>4: raise ValueError('Bounded probe permits at most four seeds')
    games=[]
    for seed in o.seeds:
        g=play(seed,o.model); games.append(g)
        Path(o.out).write_text(json.dumps(dict(games=games),indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in summarize(g).items() if k!='tail'}),flush=True)
    Path(o.out).with_suffix('.summary.json').write_text(json.dumps([summarize(g) for g in games],indent=2),encoding='utf-8')

if __name__=='__main__': main()
