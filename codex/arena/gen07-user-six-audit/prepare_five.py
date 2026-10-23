"""Normalize immutable first-five proof packets; never modify their source files."""
import argparse
import json
import numpy as np
from query import HERE,ROOT,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ids',nargs='+',required=True);ap.add_argument('--label',required=True)
    args=ap.parse_args();base=ROOT/'codex/matches/export6/user-five-v1';cases=[]
    for gid in args.ids:
        proof=base/f'{gid}-proof.json';meta=json.loads(proof.read_text(encoding='utf-8'));packet=base/f'{gid}-hard.npz'
        assert meta['replay']=='PASS_COMPLETE_EPISODE'
        assert meta['supplied_txt_equals_export_protocol']=='PASS_LINE_EXACT'
        with np.load(packet,allow_pickle=False) as data:
            for i,ply in enumerate(data['ply']):
                kind=str(data['mask_kind'][i])
                classification='winning' if 'WIN' in kind and 'SAFE' not in kind else 'safe'
                cases.append({'case_id':f'{gid}/{int(ply)}','game_id':gid,'ply':int(ply),
                              'actor':meta['user_team'],'user_team':meta['user_team'],
                              'source_path':meta['raw_supplied'],'source_sha256':meta['raw_sha256'],
                              'packet_path':str(packet.resolve()),'packet_sha256':sha(packet),'packet_row':i,
                              'mask_fields':{'legal':'legal_mask',classification:'safe_start_mask'},
                              'proof_files':[str(proof.resolve())],
                              'certificate_note':kind+'; empty support supplies no corrective action; historical model UNKNOWN'})
    out=HERE/f'cases-{args.label}.json';out.write_text(json.dumps(cases,indent=2),encoding='utf-8')
    print(str(out))

if __name__=='__main__':main()
