"""Read-only exact bounded diagnostics for the assigned match, no model/search."""
from pathlib import Path
import argparse,json,sys,hashlib

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'env'));sys.path.insert(0,str(ROOT/'codex/matches'));sys.path.insert(0,str(ROOT/'codex/engine_audit'))
from site_replay import parse,notation,apply
from colony_one import encoding
from oracle import check_forced_loss_by_next_turn
from own_turn_oracle import own_turn_wins


def typed(env,a):
    p=env.board.piece_at(*a.from_pos)
    return dict(index=int(encoding.action_index(a,env.current_team)),piece=p.piece_type.value,
                side=p.team,move=notation(a.from_pos)+'->'+notation(a.to_pos),kind=a.kind)


def sparse(env):
    return dict(side=env.current_team,actions_left=env.actions_left,round=env.round_number,action=env.action,
                pieces=[dict(side=p.team,type=p.piece_type.value,square=notation((r,c)),rc=[r,c])
                        for r,row in enumerate(env.board.grid) for c,p in enumerate(row) if p is not None],
                locked=__import__('numpy').flatnonzero(env.observation()[12].reshape(-1)).tolist())


def extended_after_own_turn(env,own,budget,memo):
    """Exists an enemy turn leading to terminal win or exact next-turn loss.

    Composes only supplied two-action engine/oracle certificates. A negative
    result is no forced loss through the SECOND enemy turn, not eventual win.
    """
    if env.terminated:return dict(status='VERIFIED' if env.winner!=own else 'DISPROVED',witness=[])
    if env.current_team==own:raise ValueError('Expected completed own turn')
    enemy=env.current_team;unknown=False;checked=0
    # Investigate the actual opponent witness first, without changing proof
    # quantifiers or skipping any required branch for a negative result.
    first_actions=sorted(env.legal_actions(),key=lambda a:not(a.from_pos==(3,2) and a.to_pos==(3,5)))
    for a in first_actions:
        t1=typed(env,a);first=env.clone();first.step(a)
        followups=[None] if first.terminated else sorted(first.legal_actions(),key=lambda a:not(a.from_pos==(3,5) and a.to_pos==(2,6)))
        for b in followups:
            if budget['nodes']>=budget['max']:return dict(status='UNKNOWN',checked_enemy_completions=checked,reason='global exact node budget')
            second=first.clone();line=[t1]
            if b is not None:line.append(typed(first,b));second.step(b)
            checked+=1
            if second.terminated:
                if second.winner==enemy:return dict(status='VERIFIED',witness=line,terminal=True,checked_enemy_completions=checked)
                continue
            if second.current_team!=own:raise AssertionError('Turn clock mismatch')
            key=(second.observation().tobytes(),second.round_number,second.actions_left)
            if key not in memo:
                proof=check_forced_loss_by_next_turn(second,max_nodes=min(100000,budget['max']-budget['nodes']))
                budget['nodes']+=proof['nodes'];memo[key]=proof
            proof=memo[key]
            if proof['status']=='VERIFIED':
                return dict(status='VERIFIED',witness=line,next_own_turn_state=sparse(second),
                            endpoint_certificate=proof,checked_enemy_completions=checked)
            unknown |= proof['status']=='UNKNOWN'
    return dict(status='UNKNOWN' if unknown else 'DISPROVED',checked_enemy_completions=checked)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--actions',default='39,40,35,36,31,32,43,44')
    parser.add_argument('--budget',type=int,default=100000)
    parser.add_argument('--extended40',action='store_true')
    parser.add_argument('--extended39-first',type=int)
    parser.add_argument('--counterfactual',action='store_true');args=parser.parse_args()
    path=ROOT/'codex/matches/site-all-v1-derived/match-10910.txt'
    game,states=parse(path,None)
    lookup={env.action:(env,row) for env,row in zip(states,game['history'])}
    print(json.dumps(dict(replay='PASS',actions=game['actions'],winner=game['winner'],source_sha256=game['sha256'])),flush=True)
    for n in map(int,args.actions.split(',')):
        env,row=lookup[n];snapshot=env.observation().copy()
        result=check_forced_loss_by_next_turn(env,max_nodes=args.budget)
        win=own_turn_wins(env.board,env.current_team,env.actions_left,env.round_number)
        legal=env.legal_actions();chosen=next(a for a in legal if list(a.from_pos)==row['event']['fr'] and list(a.to_pos)==row['event']['to'])
        choices=[]
        for a in legal:
            t=typed(env,a);t['bounded_loss']=result['per_first_action'][t['index']];t['winning_first']=bool(win.first_action_win_mask[t['index']])
            choices.append(t)
        assert __import__('numpy').array_equal(snapshot,env.observation())
        output=dict(action=n,state=sparse(env),played=typed(env,chosen),status=result['status'],nodes=result['nodes'],budget_exhausted=result['budget_exhausted'],
                    safe_count=sum(v=='DISPROVED' for v in result['per_first_action'].values()),
                    loss_count=sum(v=='VERIFIED' for v in result['per_first_action'].values()),
                    unknown_count=sum(v=='UNKNOWN' for v in result['per_first_action'].values()),
                    total=len(legal),winning_first_count=int(win.first_action_win_mask.sum()),choices=choices,win_witnesses=win.witnesses)
        print(json.dumps(output),flush=True)
    if args.extended40:
        env,row=lookup[40];budget={'nodes':0,'max':1000000};memo={};results=[]
        for a in env.legal_actions():
            child=env.clone();child.step(a)
            result=extended_after_own_turn(child,env.current_team,budget,memo)
            item=dict(action=typed(env,a),**result);results.append(item)
            print(json.dumps(dict(extended40=item,cumulative_nodes=budget['nodes'])),flush=True)
        print(json.dumps(dict(extended_summary=dict(nodes=budget['nodes'],endpoints=len(memo),
              classifications={s:sum(r['status']==s for r in results) for s in ('VERIFIED','DISPROVED','UNKNOWN')}))),flush=True)
    if args.extended39_first is not None:
        env,row=lookup[39];first=next(a for a in env.legal_actions() if encoding.action_index(a,env.current_team)==args.extended39_first)
        mid=env.clone();mid.step(first);budget={'nodes':0,'max':2000000};memo={};results=[]
        # Prioritize moving the Kernel farther from the opposing Kernel,
        # while still requiring exact universal checks for a horizon escape.
        seconds=sorted(mid.legal_actions(),key=lambda a:not(a.from_pos==(3,6) and a.to_pos==(4,6)))
        for a in seconds:
            child=mid.clone();child.step(a)
            result=extended_after_own_turn(child,env.current_team,budget,memo)
            item=dict(first=typed(env,first),second=typed(mid,a),**result);results.append(item)
            print(json.dumps(dict(extended39=item,cumulative_nodes=budget['nodes'])),flush=True)
            if result['status']=='DISPROVED':break
        print(json.dumps(dict(extended39_summary=dict(first=typed(env,first),nodes=budget['nodes'],endpoints=len(memo),
              checked_seconds=len(results),legal_seconds=len(seconds),
              classifications={s:sum(r['status']==s for r in results) for s in ('VERIFIED','DISPROVED','UNKNOWN')}))),flush=True)
    if args.counterfactual:
        e=lookup[39][0].clone();moves=[((2,5),(1,5)),((1,6),(4,3)),((3,2),(3,5)),((3,5),(1,5)),((1,5),(2,4)),((2,4),(2,3))]
        rows=[]
        for fr,to in moves:
            a=next(a for a in e.legal_actions() if a.from_pos==fr and a.to_pos==to)
            action=e.action;t=typed(e,a);event=apply(e,a,None)['event'];rows.append(dict(action=action,typed=t,event=event))
        assert e.terminated and e.winner=='MAGENTA'
        print(json.dumps(dict(counterfactual=dict(status='ENGINE_REPLAY_PASS',winner=e.winner,line=rows,
              qualification='This concrete Cyan continuation loses; universal two-enemy-turn survival is checked separately'))),flush=True)
        e=lookup[44][0];out=[]
        for a in e.legal_actions():
            child=e.clone();child.step(a);assert child.terminated and child.winner=='CYAN'
            d=max(abs(a.to_pos[0]-2),abs(a.to_pos[1]-3))
            ring=int(max(abs(a.to_pos[0]-3.5),abs(a.to_pos[1]-3.5)))
            out.append(dict(action=typed(e,a),winner=child.winner,chebyshev_distance=d,own_centre_ring=ring,
                            enemy_centre_ring=1,decisive_rule='distance<=2' if d<=2 else 'Cyan closer to centre'))
        print(json.dumps(dict(final_bare_adjudications=out)),flush=True)


if __name__=='__main__':main()
