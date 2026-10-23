"""Compare observed actions to candidate search; never infer identity from resemblance."""
import json
from site_replay import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--v2',action='store_true');o=p.parse_args()
    models=({'v2':ROOT/'codex/tactical/model-tactical-v2.onnx'} if o.v2 else
        {'h1':ROOT/'model-h1.onnx','tactical':ROOT/'codex/tactical/model-tactical.onnx'})
    out=[]
    for name,model in models.items():
        ev=OnnxEvaluator(model)
        for path in sorted((HERE/'site-raw').glob('*.txt')):
            g,states=parse(path,ev);loser='CYAN' if g['winner']=='MAGENTA' else 'MAGENTA';rows=[]
            for e,row in zip(states,g['history']):
                if e.action<=8 or e.current_team!=loser:continue
                root=mcts.run(e.board,e.current_team,e.actions_left,e.round_number,ev,mcts.MCTSConfig(simulations=100),rng=np.random.default_rng(0))
                idx=mcts.select_action_index(root,0);a=encoding.decode_action(idx,e.current_team,e.board)
                recorded=next(a for a in e.legal_actions() if list(a.from_pos)==row['event']['fr'] and list(a.to_pos)==row['event']['to'])
                ridx=encoding.action_index(recorded,e.current_team)
                rows.append(dict(action=e.action,recorded=row['notation'],selected=move(a)['notation'],matches=idx==ridx,
                    recorded_visits=root.N[ridx],selected_visits=root.N[idx],recorded_prior=root.P[ridx],value=row['before']['value']))
            result=dict(model=name,sha256=hashlib.sha256(model.read_bytes()).hexdigest(),match=g['id'],losing_team=loser,
                matches=sum(r['matches'] for r in rows),total=len(rows),rows=rows)
            out.append(result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    (HERE/('site-v2-action-check.json' if o.v2 else 'site-candidate-action-check.json')).write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__':main()
