"""Exact bounded own-turn terminal-win oracle using supplied engine transitions.

No opponent response is searched. An uncertified state is UNKNOWN, not a loss.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'env'))
from colony_one import constants as C, encoding as E
from colony_one.board import Board
from colony_one.action import Action
from colony_one.pieces import PieceType as PT
from colony_one.rules import legal_actions_for_team

def opponent(team): return C.MAGENTA if team==C.CYAN else C.CYAN

def outcome_after_action(board,record,acting_team,remaining,round_number):
    """Terminal winner/reason after an already-applied legal action; restores shrink."""
    if record.captured_piece is not None and record.captured_piece.piece_type is PT.KERNEL:
        return acting_team,'kernel_capture'
    if remaining>0: return None,None
    if board.only_kernels_remain():
        return board.kernel_endgame_winner(acting_team),'bare_kernels'
    if acting_team==C.MAGENTA and round_number in C.SHRINK_ROUNDS:
        shrink=board.apply_firewall_shrink()
        try:
            killed=[p for _,_,p in shrink.killed if p.piece_type is PT.KERNEL] if shrink else []
            if len(killed)==2: return C.MAGENTA,'firewall_both'
            if len(killed)==1: return opponent(killed[0].team),'firewall_one'
        finally:
            if shrink is not None: board.undo_firewall_shrink(shrink)
    return None,None

@dataclass
class OwnTurnResult:
    first_action_win_mask: np.ndarray
    witnesses: dict[int,dict]
    first_actions: int
    second_actions: int

    @property
    def any_win(self): return bool(self.first_action_win_mask.any())

def own_turn_wins(board:Board,team:str,actions_left:int,round_number:int)->OwnTurnResult:
    """All first actions with at least one terminal-winning own-turn continuation.

    Mutates/restores the supplied board with engine undo. Thread callers must
    not share that board concurrently. Accepts only actions_left 1 or 2.
    Witness indices always use the initial team's canonical coordinates,
    which stay unchanged until a terminal end-of-turn result.
    """
    if actions_left not in (1,2): raise ValueError('actions_left must be 1 or 2')
    mask=np.zeros(E.NUM_ACTIONS,dtype=bool); witnesses={}; second_count=0
    first=legal_actions_for_team(board,team)
    for action in first:
        idx=E.action_index(action,team)
        record=board.apply_action(action)
        try:
            winner,reason=outcome_after_action(board,record,team,actions_left-1,round_number)
            if winner is not None:
                if winner==team:
                    mask[idx]=True; witnesses[idx]={'actions':[idx],'reason':reason}
                continue
            if actions_left==1: continue
            for followup in legal_actions_for_team(board,team):
                second_count+=1
                rec2=board.apply_action(followup)
                try:
                    winner,reason=outcome_after_action(board,rec2,team,0,round_number)
                    if winner==team:
                        mask[idx]=True
                        # Enumerate all second actions; keep first witness deterministically.
                        witnesses.setdefault(idx,{'actions':[idx,E.action_index(followup,team)],'reason':reason})
                finally: board.undo_action(rec2)
        finally: board.undo_action(record)
    return OwnTurnResult(mask,witnesses,len(first),second_count)

def recover_live_clock(observation):
    obs=np.asarray(observation)
    team=C.MAGENTA if obs[15,0,0]>.5 else C.CYAN
    actions_left=int(round(2*float(obs[13,0,0])))
    rings={0:0,28:1,48:2,60:3}[int(round(float(obs[12].sum()))) ]
    rnd=round(C.SHRINK_ROUNDS[rings]-8*(1-float(obs[14,0,0])))
    return team,actions_left,rnd

def oracle_observation(observation):
    team,left,rnd=recover_live_clock(observation)
    return own_turn_wins(Board.from_observation(observation,team),team,left,rnd)
