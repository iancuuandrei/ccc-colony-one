"""Independently verify two value-only 14958 rows, with no ext release."""
from pathlib import Path
import sys,importlib.util,json,re,hashlib,time,ast
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BASE=ROOT/'codex'/'synthetic'/'validation'/'deep10910'/'verify.py'
spec=importlib.util.spec_from_file_location('independent_deep_turn_verifier',BASE);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
SOURCE=Path('D:/Downloads/match-14958.txt');PACKET=ROOT/'codex'/'research'/'loss-14958-training.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replay():
    e=v.ColonyOneEnv();states={};typed=[]
    regex=re.compile(r'^\s*(\d+)\.\s+(CYAN|MAGENTA)\s+(\w+)\s+([A-H][1-8])\s+->\s+([A-H][1-8])\s+(MOVE|SHOOT)(.*)$')
    for line in SOURCE.read_text(encoding='utf-8-sig').splitlines():
        m=regex.match(line)
        if not m:
            assert not re.match(r'^\s*\d+\.',line);continue
        n,side,piece,fr,to,kind,extra=m.groups();assert int(n)==e.action and side==e.current_team
        rc=lambda s:(8-int(s[1]),ord(s[0])-65)
        a=next(a for a in e.legal_actions() if a.from_pos==rc(fr) and a.to_pos==rc(to));assert e.board.piece_at(*a.from_pos).piece_type.abbr==piece
        record=e.board.apply_action(a)
        try:
            redirected=record.actual_target_pos is not None and record.actual_target_pos!=a.to_pos and not record.moved
            assert kind.lower()==a.kind or (kind=='SHOOT' and redirected)
            capture=re.search(r'x(PI|DE|EX|TR|IC|KE)(?:\s*\(([A-H][1-8])\))?',extra)
            assert bool(capture)==bool(record.captured_piece)
            if capture:
                assert record.captured_piece.piece_type.abbr==capture[1]
                if capture[2]:assert record.actual_target_pos==rc(capture[2])
        finally:e.board.undo_action(record)
        states[e.action]=e.clone();seq,e=v.typed(e,[int(v.E.action_index(a,side))]);typed+=seq
    assert len(typed)==44 and e.terminated and e.winner=='MAGENTA';return states,typed
def assess(e,own):
    if e.terminated:return dict(forced_loss=e.winner!=own,terminal_winner=e.winner)
    assert e.current_team!=own
    ow=v.own_turn_wins(e.board,e.current_team,e.actions_left,e.round_number)
    if ow.any_win:
        witness=ow.witnesses[min(ow.witnesses)]['actions'];typed,end=v.typed(e,witness);assert end.terminated and end.winner==e.current_team
        return dict(forced_loss=True,immediate_enemy_win=True,witness=typed)
    return v.enemy_can_force_by_second_turn(e,own)
def main():
    start=time.monotonic();p=json.loads(PACKET.read_text());assert sha(SOURCE)==p['source']['sha256'];npz=Path(p['packet']['path']);assert sha(npz)==p['packet']['sha256']
    z=np.load(npz);states,typed=replay();results=[]
    for i,meta in enumerate(p['rows']):
        given=json.loads(Path(meta['proof_path']).read_text());assert sha(meta['proof_path'])==meta['proof_sha256']
        for name,h in given['engine_sha256'].items():assert sha(ROOT/'env'/'colony_one'/name)==h
        for name,h in given['oracle_sha256'].items():assert sha(ROOT/'codex'/'engine_audit'/name)==h
        e=states[meta['action']];assert e.current_team=='CYAN';assert np.array_equal(e.observation(),z['obs'][i]);assert z['left'][i]==45-e.action
        legal=v.actions(e);assert set(legal)==set(z['pidx'][i][z['pidx'][i]>=0].tolist())==set(np.flatnonzero(z['legal_mask'][i]).tolist())
        assert len(legal)==meta['legal_actions'];classified=[]
        for first in legal:
            f=v.step(e,first);completions=[([],f)] if f.terminated or f.current_team!=e.current_team else list(v.turns(f))
            proofs=[]
            for seq,g in completions:
                proof=assess(g,e.current_team);assert proof['forced_loss'],(e.action,first,seq,proof)
                proofs.append(dict(own_sequence=[first]+seq,proof=proof))
            classified.append(dict(first_action=first,remaining_completions=len(completions),all_completions_force_loss=True,completions=proofs))
            print(json.dumps(dict(action=e.action,first=first,completed=len(completions),forced_loss=True,endpoint_states=len(v.memo),leaf_checks=v.leaf_checks)),flush=True)
        results.append(dict(position_id=meta['id'],state=v.q.encode_state(e),legal=legal,classification=classified))
    assert z['z'].tolist()==[-1.,-1.] and z['has_value'].all() and z['has_length'].all() and not z['has_pol'].any() and not z['q_valid'].any()
    assert not z['pval'].any() and not z['unknown_action_mask'].any() and not z['demonstrated_safe_mask'].any()
    assert np.array_equal(z['proved_unsafe_mask'],z['legal_mask']) and z['source_group'].tolist()==['site:14958']*2
    assert z['left'].tolist()==[8.,7.]
    hold=ROOT/'codex'/'synthetic'/'holdout_v1';frozen=json.loads((hold/'FROZEN.json').read_text())
    for name,h in frozen['files'].items():assert sha(hold/name)==h
    orbits=set().union(*(v.orbit(o) for o in np.load(hold/'holdout_states.npz')['obs']))
    assert not any(o.astype(np.float32).tobytes() in orbits for o in z['obs'])
    train=ROOT/'train.py';tree=ast.parse(train.read_text(encoding='utf-8'));key=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='KEYS' for t in n.targets));load=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load');ns={'np':np,'Path':Path};exec(compile(ast.Module(body=[key,load],type_ignores=[]),str(train),'exec'),ns)
    loaded=ns['load']([npz]);assert loaded['has_value'].all() and not loaded['has_pol'].any() and not loaded['q_valid'].any();assert loaded['left'].tolist()==[8.,7.]
    report=dict(status='PASS_INDEPENDENT_ENGINE_VERIFICATION',scope='TERMINAL_LOSS_THROUGH_SECOND_FOLLOWING_OPPONENT_TURN',method='Independent full-env turn composition from prior independently validated verifier; no research minimax helper imported. Every positive leaf win replayed.',source_sha256=sha(SOURCE),packet_json_sha256=sha(PACKET),packet_npz_sha256=sha(npz),script_sha256=sha(__file__),independent_dependency_sha256=sha(BASE),source_game='site:14958',source_group_alias='site-game-14958',source_model='UNKNOWN',replayed_actions=typed,proof=results,resources=dict(seconds=time.monotonic()-start,endpoint_states=len(v.memo),complete_defender_turns=v.endpoint_turns,leaf_checks=v.leaf_checks),masks=dict(rows=2,z=[-1,-1],has_value=[True,True],has_pol=[False,False],q_valid=[False,False],has_length=[True,True],recorded_remaining_actions=[8,7]),holdout=dict(frozen_sha256=sha(hold/'FROZEN.json'),hashes_unchanged=True,d4_orbit_intersections=0),loader=dict(train_sha256=sha(train),exact_loader_function_executed=True),qualification='Both positions already forced losses; no positive action preference or earliest recoverable mistake established. No ext release by this verifier.')
    (HERE/'proof.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:x for k,x in report.items() if k not in ['proof','replayed_actions']}),flush=True)
if __name__=='__main__':main()
