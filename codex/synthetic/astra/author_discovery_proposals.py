"""Refined multiple-actor seeds exclude unintended single-piece approach wins."""
from pathlib import Path
import json,copy
HERE=Path(__file__).resolve().parent
raw=json.loads((HERE/'seed_proposals.json').read_text())
pieces=[['C','KERNEL',7,6],['C','DAEMON',3,0],['C','TRACE',3,2],['C','PING',7,4],['C','EXPLOIT',7,7],['C','ICE',7,0],['M','KERNEL',3,3],['M','PING',0,7],['M','DAEMON',6,1],['M','EXPLOIT',6,5],['M','TRACE',5,1],['M','ICE',5,6]]
seeds=[]
for i,changes in enumerate([{}, {('M','DAEMON'):(5,0)}, {('M','TRACE'):(5,2)}, {('M','DAEMON'):(5,0),('M','TRACE'):(5,2)}]):
    ps=copy.deepcopy(pieces)
    for p in ps:
        if tuple(p[:2]) in changes:p[2:]=changes[tuple(p[:2])]
    seeds.append(dict(id='strict_discovery_'+str(i),family='strict_other_actor_discovery',side='C',left=2,round=6,pieces=ps,proposal='Teacher deliberately positions own Ice, Exploit and Ping outside their two-action capture reach to enemy Kernel. Trace must vacate the Daemon line; the same piece cannot finish the win. Enemy Ice threatens the guarded own Kernel and capture decoys compete with the quiet discovery. Verify exact winning paths before asserting this intended mechanism.'))
(HERE/'discovery_seed_proposals.json').write_text(json.dumps(dict(schema_version=1,provenance=raw['provenance'],seeds=seeds),indent=2))
