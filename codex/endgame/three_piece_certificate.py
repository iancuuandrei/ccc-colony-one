"""Sufficient exact own-turn win certificate for a narrow three-piece family.

No model/export/training code. All actions are in real board coordinates.
The proof CLI validates every eligible geometry for one representative enemy
non-Kernel type, with explicit crosschecks of the other four types and rounds.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'env'))
from colony_one import constants, encoding
from colony_one.env import ColonyOneEnv
from colony_one.board import Board
from colony_one.pieces import Piece,PieceType


def distance(a,b):return max(abs(a[0]-b[0]),abs(a[1]-b[1]))
def inward_ring(p):return min(p[0],p[1],7-p[0],7-p[1])


def certificate(env):
    """Return proved winning paths/first indices, or None (UNKNOWN, not loss).

    Current side has exactly one Kernel. Opponent has exactly one Kernel and
    one non-Kernel. Capture the last non-Kernel on the LAST own action, then
    win exact bare-Kernel adjudication. With two actions require distance2 and
    an explicitly empty/unlocked intermediate, excluding quiet out-and-back
    families from this intentionally sufficient rule.
    """
    if env.terminated or env.actions_left not in (1,2):return None
    team=env.current_team
    if team not in (constants.CYAN,constants.MAGENTA):return None
    own=[];enemy=[]
    for r,row in enumerate(env.board.grid):
        for c,p in enumerate(row):
            if p is not None:
                (own if p.team==team else enemy).append(((r,c),p))
    if len(own)!=1 or own[0][1].piece_type is not PieceType.KERNEL or len(enemy)!=2:return None
    ek=[p for p in enemy if p[1].piece_type is PieceType.KERNEL]
    other=[p for p in enemy if p[1].piece_type is not PieceType.KERNEL]
    if len(ek)!=1 or len(other)!=1:return None
    s,e,t=own[0][0],ek[0][0],other[0][0]
    if any(env.board.is_locked(*p) for p in (s,e,t)):return None
    if distance(t,e)<=2:return None
    if not(inward_ring(t)>inward_ring(e) or (inward_ring(t)==inward_ring(e) and team==constants.MAGENTA)):return None
    if distance(s,t)!=env.actions_left:return None
    paths=[]
    if env.actions_left==1:
        paths=[[(s,t)]]
    else:
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if (dr,dc)==(0,0):continue
                m=(s[0]+dr,s[1]+dc)
                if distance(m,t)!=1 or env.board.is_locked(*m):continue
                if env.board.piece_at(*m) is not None:continue
                paths.append([(s,m),(m,t)])
    if not paths:return None
    # Confirm only geometrical first-action legality, independently from the
    # later clone/step verification used by the operational proof.
    legal={(a.from_pos,a.to_pos):a for a in env.legal_actions()}
    if any(path[0] not in legal for path in paths):raise AssertionError('Geometric first step disagrees with engine')
    indices=sorted({encoding.action_index(legal[path[0]],team) for path in paths})
    return dict(team=team,actions_left=env.actions_left,source=s,enemy_kernel=e,last_enemy=t,
                enemy_type=other[0][1].piece_type.value,paths=paths,winning_first_indices=indices,value=1,
                terminal_cause='bare_kernels',scope='sufficient own-turn win; None is UNKNOWN')


def verify_paths(env,cert):
    """Independent engine clone/step execution of EVERY advertised path."""
    original_locked=np.asarray(env.board.locked).copy();n=0
    for path in cert['paths']:
        child=env.clone()
        for i,(source,target) in enumerate(path):
            assert child.current_team==cert['team'] and not child.terminated
            choices=[a for a in child.legal_actions() if a.from_pos==source and a.to_pos==target]
            assert len(choices)==1
            before=child.board.piece_at(*target)
            if i==len(path)-1:assert before is not None and before.team!=cert['team'] and before.piece_type is not PieceType.KERNEL
            else:assert before is None
            child.step(choices[0])
            if i<len(path)-1:assert not child.terminated and child.current_team==cert['team']
        assert child.terminated and child.winner==cert['team']
        assert child.board.only_kernels_remain()
        remaining=[p for row in child.board.grid for p in row if p is not None]
        assert len(remaining)==2 and all(p.piece_type is PieceType.KERNEL for p in remaining)
        assert np.array_equal(original_locked,child.board.locked),'Firewall unexpectedly ran before bare adjudication'
        n+=1
    return n


def make_env(s,e,t,team,al,rnd,typ=PieceType.PING):
    env=ColonyOneEnv();env.board.grid=[[None]*8 for _ in range(8)]
    enemy=constants.MAGENTA if team==constants.CYAN else constants.CYAN
    for pos,piece in [(s,Piece(PieceType.KERNEL,team)),(e,Piece(PieceType.KERNEL,enemy)),(t,Piece(typ,enemy))]:
        env.board.grid[pos[0]][pos[1]]=piece
    env.board.locked=Board.locked_cells_for_round(rnd)
    env.current_team=team;env.actions_left=al;env.round_number=rnd;env.action=1;env.terminated=False;env.winner=None
    return env


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path(__file__).with_name('three-piece-report.json'))
    args=parser.parse_args();start=time.perf_counter();digest=hashlib.sha256();strata=[];cross=[]
    total_states=total_paths=0
    # One round per lock stage, chosen at its shrink deadline. Capture-to-bare
    # ends before that deadline is applied; other clocks are checked below.
    for ring,rnd in [(0,8),(1,16),(2,24),(3,26)]:
        squares=[(r,c) for r in range(ring,8-ring) for c in range(ring,8-ring)]
        for team in (constants.CYAN,constants.MAGENTA):
            for al in (1,2):
                states=paths=0;samples=[]
                for s in squares:
                    for t in squares:
                        if distance(s,t)!=al:continue
                        for e in squares:
                            if e in (s,t) or distance(t,e)<=2:continue
                            if not(inward_ring(t)>inward_ring(e) or (inward_ring(t)==inward_ring(e) and team==constants.MAGENTA)):continue
                            env=make_env(s,e,t,team,al,rnd);cert=certificate(env)
                            if cert is None:raise AssertionError('Expected geometry not admitted')
                            n=verify_paths(env,cert);states+=1;paths+=n
                            digest.update(json.dumps([ring,team,al,s,e,t,cert['paths']],separators=(',',':')).encode())
                            # Spread representative type/clock crosschecks across
                            # state order without storing every board.
                            if states%127==1 and len(samples)<64:samples.append((s,e,t))
                total_states+=states;total_paths+=paths
                entry=dict(ring=ring,round=rnd,team=team,actions_left=al,eligible_geometry_states=states,verified_paths=paths)
                strata.append(entry);print(json.dumps(entry),flush=True)
                for typ in (PieceType.DAEMON,PieceType.EXPLOIT,PieceType.TRACE,PieceType.ICE):
                    cp=0
                    for s,e,t in samples:
                        env=make_env(s,e,t,team,al,rnd,typ);cert=certificate(env);assert cert;cp+=verify_paths(env,cert)
                    cross.append(dict(kind='other_enemy_type',ring=ring,round=rnd,team=team,actions_left=al,enemy_type=typ.value,states=len(samples),paths=cp))
                # Validate every legal round at this lock stage on spread cases,
                # rather than repeatedly checking all equivalent geometries.
                for rr in range(1,27):
                    if rr==rnd or sum(x<rr for x in constants.SHRINK_ROUNDS)!=ring:continue
                    cp=0
                    for s,e,t in samples[:8]:
                        env=make_env(s,e,t,team,al,rr);cert=certificate(env);assert cert;cp+=verify_paths(env,cert)
                    cross.append(dict(kind='other_legal_clock',ring=ring,round=rr,team=team,actions_left=al,states=min(8,len(samples)),paths=cp))
    # Real critical leaf witnesses, replayed from immutable actual protocols.
    sys.path.insert(0,str(ROOT/'codex/matches'))
    from site_replay import parse,move
    observed=[]
    for gid,ply,label in [('9928',28,'D5->B3'),('9366',52,'E6->E4')]:
        path=ROOT/'codex/matches/site-all-v1-derived'/('match-'+gid+'.txt')
        _,positions=parse(path,None);env=positions[ply-1]
        env.step(next(a for a in env.legal_actions() if move(a)['notation']==label))
        cert=certificate(env);assert cert;count=verify_paths(env,cert)
        observed.append(dict(game=gid,after_action=ply,action=label,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),certificate=cert,verified_paths=count))
    source_files=[Path(__file__),ROOT/'env/colony_one/env.py',ROOT/'env/colony_one/board.py',ROOT/'env/colony_one/rules.py',ROOT/'env/colony_one/encoding.py']
    result=dict(status='PASS_ELIGIBLE_GEOMETRY_AND_REDUCED_TYPES',scope='Exhaustive all admitted positive geometries for PING at every live lock stage, both teams, both remaining-action counts; every advertised path engine-executed. Other four types and every legal clock crosschecked; extension to all enemy non-Kernel types/clocks uses documented semantic irrelevance argument. Not a complete three-piece solver or a reachable-from-start claim.',
                eligible_geometry_states=total_states,verified_geometry_paths=total_paths,strata=strata,crosschecks=cross,
                crosscheck_states=sum(x['states'] for x in cross),crosscheck_paths=sum(x['paths'] for x in cross),
                observed=observed,geometry_validation_digest=digest.hexdigest(),seconds=time.perf_counter()-start,
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    args.out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('strata','crosschecks','observed','source_sha256')}),flush=True)


if __name__=='__main__':main()
