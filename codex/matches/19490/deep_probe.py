"""Compose two-enemy-turn bounded certificates using complete engine turns."""
from analyze import *

memo={};stats=dict(enemy_completions=0,leaf_checks=0,leaf_nodes=0)
def endpoint_loss(env,own):
    if env.terminated:return dict(status='VERIFIED' if env.winner!=own else 'DISPROVED',terminal=True,witness=[])
    assert env.current_team!=own
    unknown=False;checked=0
    # Immediate enemy wins are both proof and efficient leaf short circuit.
    w=own_turn_wins(env.board,env.current_team,env.actions_left,env.round_number)
    if w.any_win:return dict(status='VERIFIED',terminal=True,witness=line(env,next(iter(w.witnesses.values()))['actions']))
    for a in env.legal_actions():
        e=env.clone();t=typed(e,a);e.step(a)
        seconds=[None] if e.terminated else e.legal_actions()
        for b in seconds:
            child=e.clone();path=[t]
            if b is not None:path.append(typed(child,b));child.step(b)
            checked+=1;stats['enemy_completions']+=1
            if child.terminated:
                if child.winner!=own:return dict(status='VERIFIED',terminal=True,witness=path,checked_enemy_completions=checked)
                continue
            assert child.current_team==own
            key=(child.observation().tobytes(),child.round_number)
            if key not in memo:
                ownwin=own_turn_wins(child.board,own,child.actions_left,child.round_number)
                if ownwin.any_win:
                    memo[key]=dict(status='DISPROVED',reason='defender can win this turn',winning_witness=line(child,next(iter(ownwin.witnesses.values()))['actions']))
                else:
                    p=check_forced_loss_by_next_turn(child,max_nodes=250000)
                    stats['leaf_checks']+=1;stats['leaf_nodes']+=p['nodes'];memo[key]=p
            p=memo[key]
            if p['status']=='VERIFIED':return dict(status='VERIFIED',terminal=False,witness=path,next_own_state=sparse(child),endpoint_certificate=p,checked_enemy_completions=checked)
            unknown|=p['status']=='UNKNOWN'
    return dict(status='UNKNOWN' if unknown else 'DISPROVED',checked_enemy_completions=checked)

def analyze_position(env,row):
    owner=env.current_team;alternatives=[]
    for a in env.legal_actions():
        first=env.clone();ta=typed(first,a);first.step(a);completions=[]
        seconds=[None] if first.terminated or first.current_team!=owner else first.legal_actions()
        for b in seconds:
            child=first.clone();path=[ta]
            if b is not None:path.append(typed(child,b));child.step(b)
            p=endpoint_loss(child,owner);completions.append(dict(own_line=path,proof=p))
            if p['status']=='DISPROVED':break
        statuses=[r['proof']['status'] for r in completions]
        status='DISPROVED' if 'DISPROVED' in statuses else 'UNKNOWN' if 'UNKNOWN' in statuses else 'VERIFIED'
        alternatives.append(dict(**ta,bounded_loss=status,own_completions=completions))
        print(json.dumps(dict(action=env.action,choice=ta['move'],status=status,own_completions=len(completions),stats=stats)),flush=True)
    return dict(action_number=env.action,state=sparse(env),recorded=typed(env,chosen(env,row)),
        horizon='current own remainder, enemy turn, own turn, second enemy turn; bounded survival is not eventual win',
        alternatives=alternatives,stats=dict(stats))

if __name__=='__main__':
    game,states=parse(HERE/'match-19490.txt',None)
    output=[]
    for n in map(int,sys.argv[1:]):
        d=analyze_position(states[n-1],game['history'][n-1]);output.append(d)
        write('deep-proofs.json',dict(source_sha256=game['sha256'],helper_sha256=sha(__file__),positions=output))
