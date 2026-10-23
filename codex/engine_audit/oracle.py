"""Small exact-oracle integration API; bounded certificates never guess."""
from own_turn_oracle import own_turn_wins,outcome_after_action,opponent,C,E
from colony_one.rules import legal_actions_for_team

def solve_this_turn(env,team=None):
    team=env.current_team if team is None else team
    if env.terminated or team!=env.current_team:
        return {'status':'UNKNOWN','reason':'requires live env and actual moving team'}
    result=own_turn_wins(env.board,team,env.actions_left,env.round_number)
    return {'status':'VERIFIED' if result.any_win else 'DISPROVED',
            'any_win':result.any_win,'witnesses':result.witnesses,
            'winning_first_actions':result.first_action_win_mask.nonzero()[0].tolist(),
            'meaning':'existence of terminal win within remaining own actions only'}

class _Budget(Exception): pass

def check_forced_loss_by_next_turn(env,max_nodes=10000):
    """All own continuations lose, against some enemy winning continuation.

    VERIFIED means every legal current-turn choice can be defeated before
    next own turn. DISPROVED means an own continuation avoids defeat through
    this horizon. UNKNOWN means cap/no-action/unsupported input prevented proof.
    No-action positions are deliberately UNKNOWN: env has no terminal rule.
    """
    if env.terminated or env.actions_left not in (1,2):
        return {'status':'UNKNOWN','reason':'requires live env with 1 or 2 actions','nodes':0,'per_first_action':{}}
    b=env.board; own=env.current_team; enemy=opponent(own); nodes=0
    per_first={}; memo={}
    def state_key(team,left,rnd):
        cells=tuple((r,c,p.team,p.piece_type.value) for r,row in enumerate(b.grid)
                    for c,p in enumerate(row) if p is not None)
        return team,left,rnd,b.next_open_ring(),cells
    def charge():
        nonlocal nodes
        if nodes>=max_nodes: raise _Budget
        nodes+=1
    def next_turn_enemy_can_win(rnd):
        # Caller has excluded terminal end-of-own-turn outcomes. Materialize
        # surviving firewall changes exactly as env._end_turn does.
        shrink=None
        if own==C.MAGENTA and rnd in C.SHRINK_ROUNDS:
            shrink=b.apply_firewall_shrink()
        try:
            next_round=rnd+int(own==C.MAGENTA)
            return enemy_can_win(2,next_round)
        finally:
            if shrink is not None: b.undo_firewall_shrink(shrink)
    def enemy_can_win(left,rnd):
        key=state_key(enemy,left,rnd)
        if key in memo:return memo[key]
        legal=legal_actions_for_team(b,enemy)
        if not legal:return None
        unknown=False
        for action in legal:
            charge();record=b.apply_action(action)
            try:
                winner,_=outcome_after_action(b,record,enemy,left-1,rnd)
                if winner is not None: win=winner==enemy
                elif left==1:win=False
                else:win=enemy_can_win(left-1,rnd)
                if win is True:
                    memo[key]=True;return True
                unknown |= win is None
            finally:b.undo_action(record)
        result=None if unknown else False
        memo[key]=result;return result
    def own_action_loses(action,left,rnd):
        charge();record=b.apply_action(action)
        try:
            winner,_=outcome_after_action(b,record,own,left-1,rnd)
            if winner is not None:return winner==enemy
            if left==1:return next_turn_enemy_can_win(rnd)
            legal=legal_actions_for_team(b,own)
            if not legal:return None
            unknown=False
            for second in legal:
                loss=own_action_loses(second,left-1,rnd)
                if loss is False:return False
                unknown |= loss is None
            return None if unknown else True
        finally:b.undo_action(record)
    legal=legal_actions_for_team(b,own)
    exhausted=False
    for first in legal:
        idx=E.action_index(first,own)
        if exhausted:loss=None
        else:
            try:loss=own_action_loses(first,env.actions_left,env.round_number)
            except _Budget:loss=None;exhausted=True
        per_first[idx]='VERIFIED' if loss is True else 'DISPROVED' if loss is False else 'UNKNOWN'
    values=list(per_first.values())
    if 'DISPROVED' in values: status='DISPROVED'
    elif values and all(s=='VERIFIED' for s in values): status='VERIFIED'
    else: status='UNKNOWN'
    return {'status':status,'nodes':nodes,'max_nodes':max_nodes,'budget_exhausted':exhausted,
            'per_first_action':per_first,
            'meaning':'every own continuation admits enemy terminal win before next own turn'}
