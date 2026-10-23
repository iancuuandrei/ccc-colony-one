"""Reuse exact verified deterministic-state aliases; no new inference."""
import copy
import json
from pathlib import Path
from query import HERE,ROOT,sha

def main():
    receipt=ROOT/'codex/matches/export6/user-five-v1/19101-19490-state-equivalence.json'
    r=json.loads(receipt.read_text(encoding='utf-8'));assert r['proof_reuse']=='PASS'
    original=HERE/'game19490-results.jsonl'
    rows={x['ply']:x for x in map(json.loads,original.read_text(encoding='utf-8').splitlines())}
    aliases=[]
    for ply in [67,68]:
        proof=next(s for s in r['states'] if s['action']==ply)
        assert all(proof[k] for k in ['full_deterministic_state_equal','observation_equal','legal_indices_equal'])
        assert proof['observation_sha256']==rows[ply]['observation_sha256']
        alias=copy.deepcopy(rows[ply]);alias.update(case_id=f'19101/{ply}',game_id='19101',user_team='MAGENTA',
            group='site-game-19101',equivalence_group=r['equivalence_group'],
            query_kind='REUSED_EXACT_DETERMINISTIC_STATE_ALIAS_NO_NEW_INFERENCE',
            reused_from_case_id=rows[ply]['case_id'],alias_receipt_sha256=sha(receipt))
        aliases.append(alias)
    (HERE/'reused-alias-results.json').write_text(json.dumps(aliases,indent=2),encoding='utf-8')
    manifest={'receipt_path':str(receipt),'receipt_sha256':sha(receipt),'source_result_path':str(original),
              'source_result_sha256':sha(original),'equivalence_group':r['equivalence_group'],
              'harness_sha256':sha(__file__),'new_inference_calls':0,
              'claim':'Two source games, one repeated ending; count identical states once.'}
    (HERE/'reused-alias-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps([{'case_id':x['case_id'],'reused_from':x['reused_from_case_id'],'gen07_action':x['gen07_action'],
                      'second':x['gen07_actual_second_action']} for x in aliases]))

if __name__=='__main__':main()
