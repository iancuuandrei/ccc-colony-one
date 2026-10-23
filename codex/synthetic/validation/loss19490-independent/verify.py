"""Independent horizon-separated 19490 verification. No release or inference."""
from pathlib import Path
import sys,importlib.util,json,re,hashlib,time
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BASE=ROOT/'codex'/'synthetic'/'validation'/'deep10910'/'verify.py'
spec=importlib.util.spec_from_file_location('independent_full_turn_verifier',BASE);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
INPUT=ROOT/'codex'/'matches'/'19490';SOURCE=INPUT/'match-19490.txt'
EXPECTED={'match-19490.txt':'d52cc466fa8e9ed53b91fe561022ff6056973d6f89f0a7035bb9e44e063e97cb','deep-proofs.json':'7809f9b9cb306d2499f53ea9fba14679ef2d8daa7dfe9fffdc1f5fcdc3943132','critical-states.npz':'333dbc28bfa7a06967c5267a0dc97e039ef9b0fb9c9a7e2d807d971b5a47495d','critical-states.json':'381f5b4132de5a0bfaf39d581e8b42a784baf1ce3c3c3444be2cc99a58ed4cc3'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replay():
    e=v.ColonyOneEnv();states={};moves={};typed=[]
    regex=re.compile(r'^\s*(\d+)\.\s+(CYAN|MAGENTA)\s+(\w+)\s+([A-H][1-8])\s+->\s+([A-H][1-8])\s+(MOVE|SHOOT)(.*)$')
    for line in SOURCE.read_text(encoding='utf-8-sig').splitlines():
        m=regex.match(line)
        if not m:assert not re.match(r'^\s*\d+\.',line);continue
        n,side,piece,fr,to,kind,extra=m.groups();assert e.action==int(n) and e.current_team==side
        rc=lambda s:(8-int(s[1]),ord(s[0])-65)
        a=next(a for a in e.legal_actions() if a.from_pos==rc(fr) and a.to_pos==rc(to));assert e.board.piece_at(*a.from_pos).piece_type.abbr==piece
        record=e.board.apply_action(a)
        try:
            redirect=record.actual_target_pos is not None and record.actual_target_pos!=a.to_pos and not record.moved
            assert a.kind==kind.lower() or (kind=='SHOOT' and redirect)
            capture=re.search(r'x(PI|DE|EX|TR|IC|KE)(?:\s*\(([A-H][1-8])\))?',extra);assert bool(capture)==bool(record.captured_piece)
            if capture:
                assert record.captured_piece.piece_type.abbr==capture[1]
                if capture[2]:assert record.actual_target_pos==rc(capture[2])
            event=dict(captured=record.captured_piece.piece_type.value if record.captured_piece else None,actual_target=record.actual_target_pos,moved=record.moved)
        finally:e.board.undo_action(record)
        states[e.action]=e.clone();idx=int(v.E.action_index(a,side));moves[e.action]=idx;t,e=v.typed(e,[idx]);t[0]['capture_event']=event;typed+=t
    assert len(typed)==74 and e.terminated and e.winner=='CYAN';return states,moves,typed
def complete_after_first(e,first):
    f=v.step(e,first)
    if f.terminated or f.current_team!=e.current_team:return [([first],f)]
    return [([first]+seq,g) for seq,g in v.turns(f)]
def one_enemy(e,own):
    if e.terminated:return dict(forced_loss=e.winner!=own,terminal_winner=e.winner)
    assert e.current_team!=own and e.actions_left==2
    win=v.own_turn_wins(e.board,e.current_team,e.actions_left,e.round_number)
    if win.any_win:
        witness=win.witnesses[min(win.witnesses)]['actions'];t,end=v.typed(e,witness);assert end.terminated and end.winner==e.current_team
        return dict(forced_loss=True,enemy_terminal_witness=t)
    assert v.actions(e),'No-action state requires UNKNOWN'
    return dict(forced_loss=False,complete_own_turn_oracle=True)
def two_enemy(e,own):
    first=one_enemy(e,own)
    if first['forced_loss'] or e.terminated:return first
    return v.enemy_can_force_by_second_turn(e,own)
def classify(e,horizon):
    results=[]
    for first in v.actions(e):
        checked=[];safe=False
        for seq,f in complete_after_first(e,first):
            proof=(one_enemy if horizon==1 else two_enemy)(f,e.current_team);checked.append(dict(own_sequence=seq,proof=proof))
            if not proof['forced_loss']:safe=True;break
        results.append(dict(first_action=first,bounded_safe=safe,checked_own_completions=checked))
    return results
def mask(results):
    m=np.zeros(4096,bool)
    for r in results:m[r['first_action']]=r['bounded_safe']
    return m
def main():
    start=time.monotonic()
    for name,h in EXPECTED.items():assert sha(INPUT/name)==h
    receipt=json.loads((INPUT/'source-receipt.json').read_text())
    for name,h in receipt['file_hashes'].items():assert sha(ROOT/name)==h
    deep=json.loads((INPUT/'deep-proofs.json').read_text());assert sha(INPUT/'deep_probe.py')==deep['helper_sha256']
    z=np.load(INPUT/'critical-states.npz');states,moves,typed=replay();assert z['action_number'].tolist()==[67,68,71,72]
    output=[]
    for i,n in enumerate(z['action_number']):
        n=int(n);e=states[n];legal=v.actions(e);assert e.current_team=='MAGENTA' and e.actions_left==(2 if n%4==3 else 1)
        assert e.round_number==((n-1)//4+1);assert np.array_equal(e.observation(),z['obs'][i])
        assert set(legal)==set(z['pidx'][i][z['pidx'][i]>=0].tolist())==set(np.flatnonzero(z['legal_mask'][i]).tolist());assert moves[n]==int(z['recorded_action'][i])
        w=v.own_turn_wins(e.board,e.current_team,e.actions_left,e.round_number);assert np.array_equal(w.first_action_win_mask,z['winning_mask'][i]) and not w.any_win
        one=classify(e,1);ms=mask(one);assert np.array_equal(ms,z['safe_mask'][i]);assert np.array_equal(z['legal_mask'][i]&~ms,z['unsafe_mask'][i]);assert not z['unknown_mask'][i].any()
        two=classify(e,2) if n in (67,68) else one;mt=mask(two);assert np.array_equal(mt,z['two_enemy_safe_mask'][i]);assert np.array_equal(z['legal_mask'][i]&~mt,z['two_enemy_unsafe_mask'][i]);assert not z['two_enemy_unknown_mask'][i].any()
        record=dict(action=n,round=e.round_number,actions_left=e.actions_left,legal_actions=legal,own_turn_wins=0,one_enemy_safe=np.flatnonzero(ms).tolist(),two_enemy_safe=np.flatnonzero(mt).tolist(),one_enemy_proofs=one,two_enemy_proofs=two,two_enemy_inferred_from_stronger_one_enemy_loss=n in (71,72))
        output.append(record);print(json.dumps({k:x for k,x in record.items() if 'proofs' not in k}),flush=True)
    assert output[0]['two_enemy_safe']==[2211,2218,2787,2788]
    assert output[1]['two_enemy_safe']==[2786] and not output[2]['one_enemy_safe'] and not output[3]['one_enemy_safe']
    safe=next(r for r in output[1]['two_enemy_proofs'] if r['first_action']==2786);survival=safe['checked_own_completions'][0]['proof'];assert survival['checked_enemy_completions']==59
    # Bind the recoverable actual first action and the exact recorded losing line.
    assert moves[67]==2218 and moves[68]==2722
    assert np.array_equal(v.step(states[67],2218).observation(),states[68].observation())
    actual=[moves[k] for k in (68,69,70)];t,end=v.typed(states[68],actual);assert np.array_equal(end.observation(),states[71].observation())
    assert not v.can_survive_one_enemy_turn(end)['survives']
    assert z['source_group'].tolist()==['site:19490']*4
    assert not z['has_value'].any() and not z['has_pol'].any() and not z['q_valid'].any() and np.isfinite(z['value']).all()
    # Optional quarantine check strengthens future packet handling without release.
    hold=ROOT/'codex'/'synthetic'/'holdout_v1';frozen=json.loads((hold/'FROZEN.json').read_text())
    for name,h in frozen['files'].items():assert sha(hold/name)==h
    orbits=set().union(*(v.orbit(o) for o in np.load(hold/'holdout_states.npz')['obs']));assert not any(o.astype(np.float32).tobytes() in orbits for o in z['obs'])
    for name,h in EXPECTED.items():assert sha(INPUT/name)==h
    report=dict(status='PASS',scope='Independent exact classification of one- and two-following-enemy-turn loss horizons; safe does not mean eventual win.',source_model='UNKNOWN',user_side='UNKNOWN',losing_actor='MAGENTA',winner='CYAN',source_files=EXPECTED,script_sha256=sha(__file__),independent_verifier_dependency_sha256=sha(BASE),engine_and_oracle_bindings=receipt['file_hashes'],replayed_actions=typed,positions=output,actual_losing_continuation=t,safe68_enemy_completions=59,resources=dict(seconds=time.monotonic()-start,endpoint_states=len(v.memo),complete_defender_turns=v.endpoint_turns,leaf_oracle_checks=v.leaf_checks),holdout=dict(frozen_sha256=sha(hold/'FROZEN.json'),hashes_unchanged=True,d4_orbit_intersections=0),training_release=False,qualification='No earliest-fullgame-error or eventual-win assertion. Complete legal policy sets verified for stated horizons only. Packet supervision remains disabled.')
    (HERE/'proof.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:x for k,x in report.items() if k not in ['positions','replayed_actions','actual_losing_continuation','engine_and_oracle_bindings']}),flush=True)
if __name__=='__main__':main()
