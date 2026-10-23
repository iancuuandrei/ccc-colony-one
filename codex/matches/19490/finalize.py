"""Combine horizon-separated diagnostics; not an external training release."""
from analyze import *

def main():
    packet=json.loads((HERE/'critical-states.json').read_text(encoding='utf-8'))
    deep=json.loads((HERE/'deep-proofs.json').read_text(encoding='utf-8'))
    by_action={p['action_number']:p for p in deep['positions']}
    loaded=np.load(HERE/'critical-states.npz');arrays={k:loaded[k] for k in loaded.files};loaded.close()
    ds=np.zeros_like(arrays['legal_mask']);du=ds.copy();dx=ds.copy()
    for j,row in enumerate(packet['rows']):
        n=row['action_number']
        if n in by_action:
            d=by_action[n];row['two_enemy_turn_certificate']=d
            for a in d['alternatives']:
                {'DISPROVED':ds,'VERIFIED':du,'UNKNOWN':dx}[a['bounded_loss']][j,a['index']]=True
        elif row['safety']['status']=='VERIFIED':
            du[j]=arrays['legal_mask'][j]
            row['two_enemy_turn_certificate']=dict(status='VERIFIED',reason='already forced within the first enemy turn')
        else:dx[j]=arrays['legal_mask'][j]
        row['observation_sha256']=hashlib.sha256(arrays['obs'][j].tobytes()).hexdigest()
    assert np.array_equal(ds|du|dx,arrays['legal_mask'])
    arrays.update(two_enemy_safe_mask=ds,two_enemy_unsafe_mask=du,two_enemy_unknown_mask=dx)
    np.savez_compressed(HERE/'critical-states.npz',**arrays)
    packet.update(npz_sha256=sha(HERE/'critical-states.npz'),deep_proof_sha256=sha(HERE/'deep-proofs.json'),
        two_enemy_scope='complete all-legal classification for rows67/68; rows71/72 already forced by first enemy turn; survival is bounded, eventual value UNKNOWN',
        training_release='NONE; diagnostic packet, all value/policy/Q supervision disabled',
        reproduction=['python codex/matches/19490/analyze.py','python codex/matches/19490/deep_probe.py 68 67','python codex/matches/19490/finalize.py'])
    write('critical-states.json',packet)
    print(json.dumps(dict(actions=arrays['action_number'].tolist(),npz_sha256=packet['npz_sha256'],
        json_sha256=sha(HERE/'critical-states.json'),deep_sha256=packet['deep_proof_sha256'])))

if __name__=='__main__':main()
