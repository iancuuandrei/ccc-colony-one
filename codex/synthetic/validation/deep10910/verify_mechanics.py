"""Additional typed capture and stationary-bodyguard checks for the proof."""
from verify import *
states,records=replay();checks=[]
for line in SOURCE.read_text().splitlines():
    match=re.match(r'^\s*(\d+)\..*?([A-H][1-8])\s+->\s+([A-H][1-8])\s+(?:MOVE|SHOOT)(.*)$',line)
    if not match:continue
    number,fr,to,extra=match.groups();e=states[int(number)];rc=lambda s:(8-int(s[1]),ord(s[0])-65)
    a=next(a for a in e.legal_actions() if a.from_pos==rc(fr) and a.to_pos==rc(to));record=e.board.apply_action(a)
    try:
        expected=re.search(r'x(PI|DE|EX|TR|IC|KE)(?:\s*\(([A-H][1-8])\))?',extra)
        assert bool(expected)==bool(record.captured_piece)
        if expected:
            assert record.captured_piece.piece_type.abbr==expected[1]
            if expected[2]:assert record.actual_target_pos==rc(expected[2])
        checks.append(dict(action=int(number),captured=record.captured_piece.piece_type.value if record.captured_piece else None,actual_target=record.actual_target_pos,moved=record.moved))
    finally:e.board.undo_action(record)
e=step(states[39],2738);e=step(e,3164)
assert e.board.piece_at(1,6).piece_type.value=='TRACE' and e.board.piece_at(4,3) is None
for fr,to in [((3,2),(3,5)),((3,5),(1,5))]:
    a=next(a for a in e.legal_actions() if a.from_pos==fr and a.to_pos==to);e.step(int(E.action_index(a,e.current_team)))
assert e.board.piece_at(3,5).piece_type.value=='DAEMON' and e.board.piece_at(1,6) is None and e.board.piece_at(1,5).piece_type.value=='KERNEL'
assert e.round_number==11 and e.current_team=='MAGENTA' and e.actions_left==2
report=dict(status='PASS',source_sha256=sha(SOURCE),typed_capture_checks=checks,trace_shot_stays_G7=True,daemon_redirect_stays_F5=True,guard_removed_G7=True,kernel_survives_F7=True,post_redirect_clock=dict(round=e.round_number,side=e.current_team,actions_left=e.actions_left),main_proof_sha256=sha(HERE/'proof.json'))
(HERE/'mechanics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(status='PASS',typed_actions=len(checks),guard_and_clock_checks=True)))
