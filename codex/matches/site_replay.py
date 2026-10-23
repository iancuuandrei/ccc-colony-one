"""Parse website text matches strictly as data; validate every move with the engine."""
import argparse, datetime, hashlib, json, re
from pathlib import Path
from probe_matches import *

HERE=Path(__file__).parent
MOVE=re.compile(r'^\s*(\d+)\.\s+(CYAN|MAGENTA)\s+(PI|DE|EX|TR|IC|KE)\s+([A-H][1-8])\s+->\s+([A-H][1-8])\s+(MOVE|SHOOT)(?:\s+x(PI|DE|EX|TR|IC|KE)(?:\s+\([A-H][1-8]\))?)?\s*$')

def pos(s):return (8-int(s[1]),ord(s[0])-65)
def notation(p):return chr(65+p[1])+str(8-p[0])
def move(a):return dict(fr=list(a.from_pos),to=list(a.to_pos),notation=notation(a.from_pos)+'->'+notation(a.to_pos),kind=a.kind)

def winning_line(env,team):
    if env.terminated:return [] if env.winner==team else None
    if env.current_team!=team:return None
    for a in env.legal_actions():
        e=env.clone(); e.step(a)
        if e.terminated:
            if e.winner==team:return [move(a)]
        elif e.current_team==team:
            rest=winning_line(e,team)
            if rest is not None:return [move(a)]+rest
    return None

def parse(path,ev):
    lines=path.read_text(encoding='utf-8-sig').splitlines()
    env=ColonyOneEnv();rows=[];states=[]
    scene=next((s.split('SCENE ',1)[1] for s in lines if s.startswith('# SCENE ')),None)
    final=next((re.search(r'\*\*\* (CYAN|MAGENTA) WINS',s).group(1) for s in lines if re.search(r'\*\*\* (CYAN|MAGENTA) WINS',s)),None)
    for line in lines:
        m=MOVE.fullmatch(line)
        if not m:
            if re.match(r'^\s*\d+\.',line):raise ValueError('Unparsed move: '+line)
            continue
        n,team,abbr,fr,to,kind,cap=m.groups()
        if int(n)!=env.action or team!=env.current_team:raise ValueError('Action order/team mismatch')
        legal=env.legal_actions();a=next((a for a in legal if a.from_pos==pos(fr) and a.to_pos==pos(to)),None)
        if a is None:raise ValueError('Illegal move: '+line)
        if env.board.piece_at(*a.from_pos).piece_type.abbr!=abbr:raise ValueError('Piece metadata mismatch: '+line)
        if a.kind!=kind.lower():
            record=env.board.apply_action(a)
            redirected=record.actual_target_pos is not None and record.actual_target_pos != a.to_pos and not record.moved
            env.board.undo_action(record)
            if not (kind=='SHOOT' and redirected):raise ValueError('Move metadata mismatch: '+line)
        states.append(env.clone());row=apply(env,a,ev)
        if (row['event']['capture'] is None)!=(cap is None):raise ValueError('Capture mismatch: '+line)
        if cap and PieceType(row['event']['capture']).abbr!=cap:raise ValueError('Capture type mismatch: '+line)
        row['notation']=fr+'->'+to;rows.append(row)
    if not env.terminated or env.winner!=final:raise ValueError('Winner/termination mismatch')
    game=dict(id=path.stem,source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scene=scene,
        winner=final,history=rows,actions=len(rows),model_identity='UNKNOWN: text record contains no model hash, player identity, logits, values, or search seed')
    return game,states

def main():
    ev=OnnxEvaluator(ROOT/'model-h1.onnx');games=[];probes=[];obs=[];ids=[];plies=[];acts=[];zs=[];exact=[]
    for path in sorted((HERE/'site-raw').glob('*.txt')):
        g,states=parse(path,ev);games.append(g);loser='CYAN' if g['winner']=='MAGENTA' else 'MAGENTA'
        # Exact tactical check at every loser's second action, plus the final two turns.
        for state,row in zip(states,g['history']):
            obs.append(state.observation());ids.append(g['id']);plies.append(state.action)
            action=next(a for a in state.legal_actions() if list(a.from_pos)==row['event']['fr'] and list(a.to_pos)==row['event']['to'])
            acts.append(encoding.action_index(action,state.current_team));zs.append(1. if state.current_team==g['winner'] else -1.)
            exact.append(1. if winning_line(state,state.current_team) is not None else np.nan)
            if state.current_team!=loser or state.actions_left!=1:continue
            chosen_state=state.clone();chosen_state.step(action)
            witness=winning_line(chosen_state,g['winner'])
            if witness is None and state.action < len(g['history'])-10:continue
            logits,_=ev(state.observation()[None]);alternatives=[]
            for a in state.legal_actions():
                child=state.clone();child.step(a);line=winning_line(child,g['winner'])
                alternatives.append(dict(**move(a),allows_opponent_win_this_turn=line is not None,witness=line,
                    prior=float(logits[0,encoding.action_index(a,state.current_team)])))
            p=dict(match=g['id'],action=state.action,chosen=row['notation'],chosen_allows_win=witness is not None,
                witness=witness,safe_count=sum(not a['allows_opponent_win_this_turn'] for a in alternatives),total=len(alternatives),alternatives=alternatives)
            probes.append(p);print(json.dumps({k:v for k,v in p.items() if k!='alternatives'}),flush=True)
        (HERE/(g['id']+'.replay.json')).write_text(json.dumps(g,indent=2),encoding='utf-8')
    (HERE/'site-probes.json').write_text(json.dumps(probes,indent=2),encoding='utf-8')
    np.savez_compressed(HERE/'site-hard-positions.npz',obs=np.stack(obs),match=np.array(ids),ply=np.array(plies),recorded_action=np.array(acts),observed_outcome=np.array(zs),exact_current_turn_win=np.array(exact))
    provenance=dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),h1_sha256=hashlib.sha256((ROOT/'model-h1.onnx').read_bytes()).hexdigest(),
        scene_sha256=hashlib.sha256((ROOT/'env/colony_one/scene.json').read_bytes()).hexdigest(),matches=[{k:v for k,v in g.items() if k!='history'} for g in games],
        policy_comparison='NOT AVAILABLE: match text contains actions but no recorded policy/value or model metadata; move similarity cannot identify model')
    (HERE/'site-provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')

if __name__=='__main__':main()
