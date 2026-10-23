"""Bounded independent env.clone/step check of next-turn loss certificates."""
from pathlib import Path
import json
import numpy as np
from audit_engine import board,env
from oracle import check_forced_loss_by_next_turn
from colony_one import constants as C
from colony_one.pieces import PieceType as PT

def direct(e,own,own_turn=True):
    legal=e.legal_actions()
    if not legal:return None
    outcomes=[]
    for action in legal:
        nxt=e.clone();nxt.step(action)
        if nxt.terminated: lose=nxt.winner!=own
        elif own_turn:
            lose=direct(nxt,own,nxt.current_team==own)
        elif nxt.current_team==own:lose=False
        else:lose=direct(nxt,own,False)
        outcomes.append(lose)
    if own_turn:
        return False if False in outcomes else None if None in outcomes else True
    return True if True in outcomes else None if None in outcomes else False

def main():
    fixtures=[
        ('bare_forced_loss',env(board([(C.CYAN,PT.KERNEL,(0,0)),(C.MAGENTA,PT.KERNEL,(3,3))]),C.CYAN,1)),
        ('bare_escape',env(board([(C.CYAN,PT.KERNEL,(0,0)),(C.MAGENTA,PT.KERNEL,(7,7))]),C.CYAN,1)),
        ('opponent_two_action_threat',env(board([(C.CYAN,PT.KERNEL,(0,0)),(C.CYAN,PT.PING,(7,7)),
            (C.MAGENTA,PT.KERNEL,(3,3)),(C.MAGENTA,PT.ICE,(2,2))]),C.CYAN,1)),
        ('small_board',env(board([(C.CYAN,PT.KERNEL,(3,3)),(C.MAGENTA,PT.KERNEL,(4,4)),
            (C.CYAN,PT.DAEMON,(4,3)),(C.MAGENTA,PT.PING,(3,4))],25),C.CYAN,2,25)),
        ('firewall_win',env(board([(C.CYAN,PT.KERNEL,(0,0)),(C.MAGENTA,PT.KERNEL,(0,7)),
            (C.MAGENTA,PT.PING,(4,4))],8),C.MAGENTA,1,8))]
    rows=[]
    for name,e in fixtures:
        before=e.observation().copy()
        want=direct(e,e.current_team)
        result=check_forced_loss_by_next_turn(e,100000)
        expected='VERIFIED' if want is True else 'DISPROVED' if want is False else 'UNKNOWN'
        assert result['status']==expected,(name,result,expected)
        assert np.array_equal(before,e.observation())
        limited=check_forced_loss_by_next_turn(e,0)
        assert limited['status']=='UNKNOWN' and limited['budget_exhausted']
        assert np.array_equal(before,e.observation())
        rows.append({'name':name,**result})
    dest=Path(__file__).with_name('loss_oracle_validation.json')
    report={'status':'PASS','independent_env_tree_fixtures':len(rows),
            'budget_unknown_and_board_restore_checks':len(rows),'cases':rows}
    dest.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
