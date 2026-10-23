"""Build bounded diagnostic/reachable own-turn oracle corpus; no model training."""
from pathlib import Path
import json,sys,hashlib,time
import numpy as np
from own_turn_oracle import own_turn_wins,oracle_observation,ROOT
from audit_engine import board,env
from colony_one import constants as C,encoding as E
from colony_one.pieces import PieceType as PT
from colony_one.env import ColonyOneEnv
from colony_one.action import Action

def main():
    started=time.perf_counter(); cases=[]
    def add(name,items,team=C.CYAN,left=2,rnd=1):
        cases.append((name,env(board(items,rnd),team,left,rnd),'diagnostic'))
    for team in [C.CYAN,C.MAGENTA]:
        enemy=C.MAGENTA if team==C.CYAN else C.CYAN
        # Empty first-move approaches for all movement/capture geometries.
        for kind,src,target in [(PT.PING,(5,1),(2,3)),(PT.DAEMON,(5,1),(2,3)),
                (PT.EXPLOIT,(6,1),(2,1)),(PT.TRACE,(5,1),(2,3)),
                (PT.ICE,(6,1),(2,3)),(PT.KERNEL,(5,1),(3,3))]:
            items=[(team,kind,src),(enemy,PT.KERNEL,target)]
            if kind!=PT.KERNEL: items.append((team,PT.KERNEL,(7,7)))
            for left in [1,2]: add(f'{team}_{kind.name}_approach_left{left}',items,team,left)
        for left in [1,2]:
            add(f'{team}_guardredirect_left{left}',[(team,PT.PING,(1,1)),(team,PT.KERNEL,(7,7)),
                (enemy,PT.KERNEL,(3,3)),(enemy,PT.ICE,(4,4))],team,left)
            # Daemon first shoots adjacent guard without relocating; second shoots Kernel.
            add(f'{team}_stationary_guard_shot_left{left}',[(team,PT.DAEMON,(3,2)),(team,PT.KERNEL,(7,7)),
                (enemy,PT.KERNEL,(3,3)),(enemy,PT.PING,(2,2))],team,left)
            add(f'{team}_trace_adjacent_block_left{left}',[(team,PT.TRACE,(0,0)),(team,PT.KERNEL,(7,7)),
                (enemy,PT.PING,(1,1)),(enemy,PT.KERNEL,(3,3))],team,left)
            add(f'{team}_vacated_source_ray_left{left}',[(team,PT.DAEMON,(3,3)),(team,PT.KERNEL,(7,7)),
                (enemy,PT.KERNEL,(3,0))],team,left)
            add(f'{team}_locked_2x2_left{left}',[(team,PT.KERNEL,(3,3)),(enemy,PT.KERNEL,(4,4)),
                (team,PT.DAEMON,(4,3)),(enemy,PT.PING,(3,4))],team,left,25)
            add(f'{team}_bare_close_left{left}',[(team,PT.KERNEL,(3,3)),(enemy,PT.KERNEL,(1,1))],team,left)
            add(f'{team}_bare_far_left{left}',[(team,PT.KERNEL,(3,3)),(enemy,PT.KERNEL,(0,7))],team,left)
    add('Magenta_simultaneous_burn',[(C.CYAN,PT.KERNEL,(0,0)),(C.MAGENTA,PT.KERNEL,(0,7)),
        (C.MAGENTA,PT.PING,(4,4))],C.MAGENTA,1,8)
    add('Magenta_only_own_burn',[(C.CYAN,PT.KERNEL,(3,3)),(C.MAGENTA,PT.KERNEL,(0,7)),
        (C.MAGENTA,PT.PING,(4,4))],C.MAGENTA,1,8)
    add('bare_preempts_both_burn',[(C.CYAN,PT.KERNEL,(0,0)),(C.MAGENTA,PT.KERNEL,(0,3))],C.MAGENTA,1,8)
    # Fixed small set of random trajectories is solely a reachable-state sampler.
    # It does not estimate playing strength or compare agents.
    transitions=0
    for seed in range(6):
        e=ColonyOneEnv(seed=7300+seed)
        for step in range(104):
            if e.terminated: break
            if step%3==0: cases.append((f'reachable_seed{seed}_step{step}',e.clone(),'random_reachable'))
            legal=e.legal_actions()
            if not legal: break
            e.step(e.sample_legal_action()); transitions+=1
    observations=[]; masks=[]; rows=[]; branches=0; witnesses_checked=0
    for name,e,origin in cases:
        obs=e.observation(); before=obs.copy()
        result=own_turn_wins(e.board,e.current_team,e.actions_left,e.round_number)
        assert np.array_equal(before,e.observation()),'oracle failed board restoration'
        # Every positive has an actual env.step witness; all negatives come from
        # complete enumeration, not heuristic rejection. Independently replay
        # every legal second-action branch for diagnostic positions below.
        for idx,witness in result.witnesses.items():
            replay=e.clone(); acting=e.current_team
            for action_idx in witness['actions']:
                replay.step(action_idx)
            assert replay.terminated and replay.winner==acting,(name,idx,witness)
            witnesses_checked+=1
        if origin=='diagnostic':
            exact=np.zeros(4096,bool)
            for first in e.legal_actions():
                after=e.clone(); acting=e.current_team; after.step(first)
                idx=E.action_index(first,acting)
                if after.terminated:
                    exact[idx]=after.winner==acting
                elif e.actions_left==2:
                    for second in after.legal_actions():
                        terminal=after.clone(); terminal.step(second)
                        if terminal.terminated and terminal.winner==acting: exact[idx]=True
            assert np.array_equal(exact,result.first_action_win_mask),(name,'direct_env_disagreement')
        recovered=oracle_observation(obs)
        assert np.array_equal(recovered.first_action_win_mask,result.first_action_win_mask)
        observations.append(obs); masks.append(result.first_action_win_mask)
        branches+=result.first_actions+result.second_actions
        rows.append({'name':name,'origin':origin,'team':e.current_team,'actions_left':e.actions_left,
            'round_number':e.round_number,'any_win':result.any_win,
            'winning_first_actions':np.flatnonzero(result.first_action_win_mask).tolist(),
            'witnesses':result.witnesses,'first_actions':result.first_actions,'second_actions':result.second_actions})
    here=Path(__file__).parent
    np.savez_compressed(here/'own_turn_corpus.npz',observations=np.stack(observations),
        first_action_win_masks=np.stack(masks),any_win=np.array([r['any_win'] for r in rows],bool),
        names=np.array([r['name'] for r in rows]),origins=np.array([r['origin'] for r in rows]))
    report={'status':'PASS','states':len(rows),'diagnostic_states':sum(r['origin']=='diagnostic' for r in rows),
        'reachable_states':sum(r['origin']=='random_reachable' for r in rows),'random_sampling_transitions':transitions,
        'any_win_states':sum(r['any_win'] for r in rows),'enumerated_action_branches':branches,
        'env_replayed_positive_witnesses':witnesses_checked,
        'direct_env_all_branches_checked':'all diagnostic states',
        'elapsed_seconds':round(time.perf_counter()-started,3),
        'meaning':'Exact existence of terminal win within remaining own actions; false means UNKNOWN beyond this turn.',
        'cases':rows}
    (here/'own_turn_corpus.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))

if __name__=='__main__': main()
