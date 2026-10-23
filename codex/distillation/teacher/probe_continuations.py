"""Each first-action survival tier: exists own continuation avoiding next-turn loss."""
exec(__import__('pathlib').Path(__file__).with_name('probe_packet.py').read_text().split('packet_id=')[0])
packet_id=sys.argv[1] if len(sys.argv)>1 else 'packet0001'
packet=json.loads((DIST/'packets'/(packet_id+'.json')).read_text())
out=[]
for p in packet['positions']:
    env=restore(p['state']);team=env.current_team
    if env.actions_left!=2:continue
    rows=[]
    for a in env.legal_actions():
        ix=action_index(a,team);child=env.clone();child.step(a)
        d=dict(action_index=ix,safe_continuation=False,checked=0)
        if child.terminated:
            d['safe_continuation']=child.winner==team
        else:
            for b in child.legal_actions():
                leaf=child.clone();leaf.step(b);d['checked']+=1
                if leaf.terminated:safe=leaf.winner==team
                else:safe=not own_turn_wins(leaf.board,leaf.current_team,leaf.actions_left,leaf.round_number).any_win
                if safe:
                    d['safe_continuation']=True;d['witness']=[ix,action_index(b,team)];break
        rows.append(d)
    out.append(dict(position_id=p['position_id'],actions=rows))
    print(json.dumps(out[-1]),flush=True)
(HERE/(packet_id+'_continuations.json')).write_text(json.dumps(out,indent=2))
