"""Build a compact report from completed counterfactual runs; no inference."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def main():
    rows=[]
    for p in sorted(HERE.glob('*-results.jsonl')):
        rows.extend(json.loads(x) for x in p.read_text(encoding='utf-8').splitlines())
    queried_count=len(rows)
    alias_file=HERE/'reused-alias-results.json'
    if alias_file.exists():rows.extend(json.loads(alias_file.read_text(encoding='utf-8')))
    byid={}
    for r in rows:
        assert r['case_id'] not in byid
        byid[r['case_id']]=r
    rows=sorted(rows,key=lambda r:(int(r['game_id']),r['ply']))
    repeated={}
    for r in rows:repeated.setdefault(r['observation_sha256'],[]).append(r['case_id'])
    aliases=[{'observation_sha256':h,'case_ids':ids} for h,ids in repeated.items() if len(ids)>1]
    actionable=[];forced=[]
    for r in rows:
        count=r['certificate_counts'];member=r['certificate_membership']
        if count.get('two_enemy_safe',0)>0:
            if not member.get('two_enemy_safe',False):actionable.append(r['case_id'])
        elif count.get('safe',0)>0:
            if not member.get('safe',False):actionable.append(r['case_id'])
        elif count.get('safe')==0:forced.append(r['case_id'])
    summary={'model_sha256':'8b23b6a7f80089aef590beb69a829738e3c8c66dcd5ccbb652820142dbabcebd',
             'queried_case_count':queried_count,'reported_case_count_including_reused_aliases':len(rows),
             'distinct_queried_observations':len(repeated),
             'queried_game_ids':sorted({r['game_id'] for r in rows}),
             'duplicate_observation_groups':aliases,'missed_nonempty_certified_support_cases':actionable,
             'distinct_missed_nonempty_support_observations':len({byid[k]['observation_sha256'] for k in actionable}),
             'empty_support_forced_cases':forced,
             'limitation':'State counterfactuals only. Existing safe first actions may still be followed by an unsafe actual second action. Bounded survival is not eventual victory. Historical model UNKNOWN.'}
    (HERE/'combined-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Current GEN07 six-game counterfactual audit','',
           'Fixed current checkpoint `ckpt/gen07-endgame.onnx`, SHA256 `'+summary['model_sha256']+'`, matches `release/gen07/model.onnx`. Referee MCTS100 defaults, RNG seed 0 per decision, one CPU process / one ORT thread. Historical source-model identities remain UNKNOWN. No old-model inference, strength games, training or model export.','',
           '| Case | Actor / user | GEN07 first | Actual GEN07 second | Supplied certificate / actual result |',
           '|---|---|---|---|---|']
    for r in rows:
        count=r['certificate_counts'];member=r['certificate_membership']
        if 'two_enemy_safe' in count:
            status=('two-enemy-turn safe start' if member['two_enemy_safe'] else 'two-enemy-turn unsafe start')
            completion=r.get('saved_two_enemy_actual_completion_certificate')
            if completion:status+='; actual completion bounded-loss '+completion['proof']['status']
        else:
            status='safe start' if member.get('safe') else ('missed safe set' if count.get('safe',0)>0 else 'already forced: empty safe support')
        if r['gen07_actual_turn_win']:status+='; actual terminal win'
        elif r['gen07_actual_turn_allows_enemy_terminal_win']:status+='; terminal enemy reply replayed'
        lines.append(f"| {r['case_id']} | {r['actor']} / {r['user_team']} | {r['gen07_action']} | {r['gen07_actual_second_action'] or 'n/a'} | {status} |")
    lines+=['','Corrections still needed:', '',
            '- **18587/36:** unique safe Kernel E7->F6; GEN07 keeps the unsafe Trace G5->D2.',
            '- **18972/62:** unique safe C4->B4; GEN07 chooses C4->C3.',
            '- **19255/37:** B3->C3 then C3->C2, or Kernel B2->C2 then B3->B2, avoids the checked terminal loss. GEN07 chooses B3->B4 then B4->C4.',
            '- **19490/68 and exact alias 19101/68:** unique two-enemy-turn-safe Kernel E6->F5 (canonical index 2786); GEN07 chooses Trace F6->F5 (2722). At pre67 it chooses F5->F6, which has a safe possible completion, but its actual second action is the same unsafe Trace return. A safe-first-action membership alone would conceal this failure.',
            '', 'Independent deterministic-state equivalence PASS covers 19101/19490 actions63,65,67,68,71,72. The earlier67/68 outputs are reused with zero new inference; final71/72 also exactly duplicate. These are two source games with one repeated ending, not separate tactical evidence. The final71/72 positions are already forced under the supplied horizon. The final 19040/93-94 positions likewise have no safe action target.',
            '', 'The separate 19040 miner followup reports every choice89/90 loses within two enemy turns and every own completion from85 plus every choice86 loses within three enemy turns. The actual86 Trace move survives the shorter two-turn horizon. No further nonempty corrective support was supplied, so no additional GEN07 query was needed. Those longer proofs were not independently re-enumerated by this inference audit; the earliest avoidable commitment before85 remains UNKNOWN. Sources are pinned in additional-proof-notes.json.',
            '', 'All exact packet observations and legal masks matched complete engine replays. Actual GEN07 followthrough was executed on every same-actor second action. Any available immediate enemy-terminal reply was replayed to its terminal outcome. No supplied winning-first-action certificate was selected or used to claim an actual model win.',
            '', 'Provenance is the captured `loop-state-start.json`, per-run loop-state snapshots, actual artifact hashes and per-run manifests. No unverified ablation label is asserted: the captured state records main arguments `--ext-frac 0.06`, variant arguments including `--policy-mix 0.5`, and chosen `gen07`. Those configuration records do not reconstruct the exact arguments loaded by a historical training process.',
            '', 'Every mask is used only at its documented horizon. Eventual values for the 19490 critical packet are UNKNOWN and masked. Forced-state choices are not avoidable tactical errors merely because a terminal enemy reply exists.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    pins={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='artifact-hashes.json'}
    (HERE/'artifact-hashes.json').write_text(json.dumps(pins,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
