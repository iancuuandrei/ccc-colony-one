"""Deterministic position packets and validated teacher-label ingestion (no training)."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'env'))
from colony_one import constants,encoding,rules
from colony_one.env import ColonyOneEnv
from colony_one.board import Board

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def coord(pos):return chr(65+pos[1])+str(8-pos[0])
def encode_state(env):
    o=env.observation().astype(np.float32)
    return dict(piece_squares=[np.flatnonzero(p.reshape(-1)).tolist() for p in o[:12]],
        locked_squares=np.flatnonzero(o[12].reshape(-1)).tolist(),
        global_planes=[float(o[i,0,0]) for i in (13,14,15)],
        side=env.current_team,actions_left=env.actions_left,round_number=env.round_number,action_number=env.action)
def observation(state):
    o=np.zeros((16,64),np.float32)
    for i,squares in enumerate(state['piece_squares']):o[i,squares]=1
    o[12,state['locked_squares']]=1
    for i,val in enumerate(state['global_planes'],13):o[i,:]=val
    return o.reshape(16,8,8)
def restore(state):
    env=ColonyOneEnv();env.board=Board.from_observation(observation(state),state['side'])
    env.current_team=state['side'];env.actions_left=state['actions_left'];env.round_number=state['round_number'];env.action=state['action_number'];env.terminated=False;env.winner=None
    return env
def state_hash(state):return hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def package(env,source,category):
    state=encode_state(env);digest=state_hash(state)
    ps=[]
    for r in range(8):
        for c in range(8):
            p=env.board.grid[r][c]
            if p:ps.append(dict(team=p.team,piece=p.piece_type.value,rc=[r,c],square=coord((r,c))))
    legal=[]
    for a in env.legal_actions():
        legal.append(dict(action_index=encoding.action_index(a,env.current_team),piece=env.board.piece_at(*a.from_pos).piece_type.value,
            kind=a.kind,from_rc=list(a.from_pos),to_rc=list(a.to_pos),move=coord(a.from_pos)+'-'+coord(a.to_pos)))
    return dict(position_id='p-'+digest[:16],state_sha256=digest,state=state,source=source,category=category,
        pieces=ps,legal_actions=legal,burn_urgency=encoding.firewall_urgency(env.round_number),
        firewall_after_this_turn=env.current_team==constants.MAGENTA and env.round_number in constants.SHRINK_ROUNDS,
        board_rows=[' '.join('##' if env.board.is_locked(r,c) else '..' if env.board.grid[r][c] is None else env.board.grid[r][c].team[0]+env.board.grid[r][c].piece_type.abbr for c in range(8)) for r in range(8)])

def build(out,count=20):
    matches=ROOT/'codex'/'matches'
    history_path=matches/'selfplay-h1.corrected.json'
    games=json.loads(history_path.read_text())['games']
    avoid={g['seed']:g for g in json.loads((matches/'last-turn-avoidability.json').read_text())}
    selected=[];seen=set()
    def add(env,source,category):
        p=package(env,source,category)
        if p['state_sha256'] not in seen:seen.add(p['state_sha256']);selected.append(p)
    candidates=[]
    for g in games:
        env=ColonyOneEnv();av=avoid.get(g['seed'])
        for row in g['history']:
            e=row['event'];n=env.action
            source=dict(path=str(history_path.relative_to(ROOT)),sha256=sha(history_path),seed=g['seed'],action_number=n)
            if av and n in (av['action']-1,av['action']):
                cat='avoidable_final_action' if n==av['action'] and av['safe_count']>0 else 'last_turn_tactical'
                candidates.append((0 if cat=='avoidable_final_action' else 2,env.clone(),source,cat))
            elif e.get('capture') and e.get('actual_target')!=e.get('to'):
                candidates.append((1,env.clone(),source,'guard_redirection'))
            elif env.round_number in (8,16,24,26):
                candidates.append((3,env.clone(),source,'firewall_timing'))
            legal=env.legal_actions();a=next(a for a in legal if list(a.from_pos)==e['fr'] and list(a.to_pos)==e['to'])
            env.step(a)
    for _,env,src,cat in sorted(candidates,key=lambda x:(x[0],x[2]['seed'],x[2]['action_number'])):
        if sum(p['category']==cat for p in selected)>=4:continue
        add(env,src,cat)
        if len(selected)>=min(14,count):break
    # Fresh tactical/ambiguous examples from gen00, preserving complete-game provenance.
    for f in sorted((ROOT/'data'/'gen00').glob('chunk*.npz')):
        with np.load(f) as d:
            for i in np.linspace(5,len(d['z'])-2,45,dtype=int):
                o=d['obs'][i].astype(np.float32);side=constants.MAGENTA if o[15,0,0]>.5 else constants.CYAN
                lock=int(o[12].sum());completed={0:0,28:1,48:2,60:3}.get(lock)
                if completed is None:continue
                rnd=round((8,16,24,26)[completed]-8*(1-float(o[14,0,0])))
                env=ColonyOneEnv();env.board=Board.from_observation(o,side);env.current_team=side;env.actions_left=round(float(o[13,0,0])*2);env.round_number=rnd
                env.action=(rnd-1)*4+(1 if side==constants.CYAN else 3)+(2-env.actions_left)
                q=float(d['q'][i]);gap=abs(q-float(d['z'][i]))
                if abs(q)<.5 or gap>.8:
                    add(env,dict(path=str(f.relative_to(ROOT)),sha256=sha(f),row=int(i),game_index=int(np.count_nonzero(d['left'][:i]==1))),
                        'uncertain_search' if abs(q)<.5 else 'search_outcome_disagreement')
                if len(selected)>=count:break
        if len(selected)>=count:break
    packet=dict(packet_id='packet0001',schema_version=1,teacher_model='gpt-6-astra',
        observation_contract='Sparse lossless canonical float32 (16,8,8); piece_squares planes0..11, locked_squares plane12, global_planes13..15; action_index=canonical_from*64+canonical_to. Readable board/coordinates are ABSOLUTE real board; row0 top, A8=(0,0).',
        instructions='Rank legal actions, give side-to-move value [-1,1], confidence[0,1], and short factual evidence. Do not produce private reasoning. Treat strategic evaluations as prediction. Exact forced_win_this_turn/forced_loss_by_next_turn claims require engine verification; safe from next-turn tactical loss does not imply game-theoretic win. Never negate value between consecutive same-team actions. Magenta wins ties; adjudication priority: capture, bare kernels at turn end, then firewall after Magenta.',
        engine_sha256={f:sha(ROOT/'env'/'colony_one'/f) for f in ['env.py','board.py','rules.py','encoding.py','mcts.py']},positions=selected)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(packet,indent=1),encoding='utf-8')
    print(json.dumps(dict(packet=str(out.resolve()),positions=len(selected),categories={c:sum(p['category']==c for p in selected) for c in sorted({p['category'] for p in selected})})))

def consume(packet,labels,out):
    import importlib.util
    vspec=importlib.util.spec_from_file_location('codex_distillation_label_validation',HERE/'validation.py');vmod=importlib.util.module_from_spec(vspec);sys.modules[vspec.name]=vmod;vspec.loader.exec_module(vmod);validate_file=vmod.validate_file
    out=Path(out);manifest_path=out/'student_ready.json';old=None
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['packet_sha256']!=sha(packet) or old['labels_sha256']!=sha(labels):raise ValueError('Output already binds different packet/labels; choose a fresh output directory')
    result=validate_file(Path(packet),Path(labels))
    if result.get('synthetic_fixture'):raise ValueError('Synthetic verification fixtures cannot enter the student queue')
    if old and result['accepted'] and Path(old['targets']).exists() and old.get('targets_sha256'):
        if sha(old['targets'])!=old['targets_sha256']:raise ValueError('Existing validated targets failed content hash check')
        print(json.dumps(dict(old,ingest='UNCHANGED_VERIFIED')));return
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report=out/'validation.json';report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not result['accepted']:raise SystemExit('Rejected labels; see '+str(report))
    p=json.loads(Path(packet).read_text());byid={x['position_id']:x for x in p['positions']};labs=result['labels']
    obs=np.stack([observation(byid[l['position_id']]['state']) for l in labs]);k=max(len(l['policy_rankings']) for l in labs)
    idx=np.full((len(labs),k),-1,np.int16);val=np.zeros((len(labs),k),np.float32)
    for i,l in enumerate(labs):
        pol=l['policy_rankings'];w=np.array([x['weight'] for x in pol],np.float32);w/=w.sum()
        idx[i,:len(pol)]=[x['action_index'] for x in pol];val[i,:len(pol)]=w
    groups=[]
    for l in labs:
        pos=byid[l['position_id']];src=pos['source'];groups.append(pos.get('group_id',src.get('game_group',src['path']+'::'+str(src.get('seed',src.get('match',src.get('game_index')))))))
    legal_mask=np.stack([encoding.legal_action_mask(restore(byid[l['position_id']]['state']).board,byid[l['position_id']]['state']['side']) for l in labs])
    target=out/'teacher_targets.npz';np.savez_compressed(target,obs=obs,pidx=idx,pval=val,legal_mask=legal_mask,value=np.array([x['value'] for x in labs],np.float32),value_valid=np.array([x['value_valid'] for x in labs]),confidence=np.array([x['confidence'] for x in labs],np.float32),value_confidence=np.array([x['value_confidence'] for x in labs],np.float32),position_id=np.array([x['position_id'] for x in labs]),game_group=np.array(groups),label_source=np.array([x['label_source'] for x in labs]),exact_claim=np.array([x['claim'] in ('forced_win_this_turn','forced_loss_by_next_turn') for x in labs]))
    manifest=dict(packet=str(Path(packet).resolve()),packet_sha256=sha(packet),labels=str(Path(labels).resolve()),labels_sha256=sha(labels),targets=str(target.resolve()),targets_sha256=sha(target),count=len(labs),game_groups=len(set(groups)),validator_sha256=result['validator_sha256'],semantics='Teacher estimates are predictions except independently verified exact claims; targets are a distinct distillation corpus, not self-play outcomes.',status='READY_FOR_STUDENT')
    (out/'student_ready.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest))
def build_site(out):
    origin=ROOT/'codex'/'matches'/'site-hard-positions.npz';digest=sha(origin)
    f=HERE/'sources'/('site-hard-positions-'+digest[:16]+'.npz');f.parent.mkdir(exist_ok=True)
    if not f.exists():f.write_bytes(origin.read_bytes())
    if sha(f)!=digest:raise ValueError('Frozen source content hash mismatch')
    positions=[]
    wanted={'match-4650':{27,28,29,30},'match-4674':{33,34,35,36,37,38,39,40}}
    with np.load(f) as d:
        for i in range(len(d['obs'])):
            match=str(d['match'][i]);ply=int(d['ply'][i])
            if ply not in wanted.get(match,set()):continue
            o=d['obs'][i];side=constants.MAGENTA if o[15,0,0]>.5 else constants.CYAN
            completed={0:0,28:1,48:2,60:3}[int(o[12].sum())];rnd=round((8,16,24,26)[completed]-8*(1-float(o[14,0,0])))
            env=ColonyOneEnv();env.board=Board.from_observation(o,side);env.current_team=side;env.actions_left=round(float(o[13,0,0])*2);env.round_number=rnd;env.action=ply
            category='observed_avoidable_final_action' if (match,ply) in {('match-4650',28),('match-4674',34)} else 'observed_blunder_context'
            positions.append(package(env,dict(path=str(f.relative_to(ROOT)),origin_path=str(origin.relative_to(ROOT)),sha256=digest,row=i,match=match,game_group='site:'+match,ply=ply,recorded_action=int(d['recorded_action'][i])),category))
    template=json.loads((HERE/'packets'/'packet0001.json').read_text());template.update(packet_id=Path(out).stem,positions=positions)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise ValueError('Preserve existing packet; use a new packet path')
    out.write_text(json.dumps(template,indent=1),encoding='utf-8');print(json.dumps(dict(packet=str(out.resolve()),positions=len(positions))))
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    b=sub.add_parser('build');b.add_argument('--out',default=str(HERE/'packets'/'packet0001.json'));b.add_argument('--count',type=int,default=20)
    s=sub.add_parser('build-site');s.add_argument('--out',default=str(HERE/'packets'/'packet0002.json'))
    c=sub.add_parser('consume');c.add_argument('--packet',required=True);c.add_argument('--labels',required=True);c.add_argument('--out',required=True)
    q=sub.add_parser('ingest');q.add_argument('--packet-dir',default=str(HERE/'packets'));q.add_argument('--labels-dir',default=str(HERE/'teacher_labels'));q.add_argument('--out',default=str(HERE/'validated'))
    a=p.parse_args()
    if a.cmd=='build':build(a.out,a.count)
    elif a.cmd=='build-site':build_site(a.out)
    elif a.cmd=='ingest':
        for packet in sorted(Path(a.packet_dir).glob('packet*.json')):
            labels=Path(a.labels_dir)/('astra_'+packet.stem+'.json')
            if labels.exists():consume(packet,labels,Path(a.out)/packet.stem)
    else:consume(a.packet,a.labels,a.out)
if __name__=='__main__':main()
