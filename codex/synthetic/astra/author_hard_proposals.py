"""Materialize explicit teacher-designed counterfactuals; never infer labels here."""
from pathlib import Path
import json,hashlib,copy
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
raw=json.loads((HERE/'seed_proposals.json').read_text());seeds=[]
def add(id,family,pieces,proposal,side='C',left=2,rnd=6):
    seeds.append(dict(id=id,family=family,side=side,left=left,round=rnd,pieces=pieces,proposal=proposal))
# Five cluttered boards increase branching and offer material-capture decoys.
crowded=[['C','KERNEL',7,7],['C','DAEMON',3,0],['C','TRACE',3,2],['C','PING',6,5],['C','EXPLOIT',7,1],['C','ICE',5,5],['M','KERNEL',3,3],['M','PING',0,7],['M','DAEMON',6,4],['M','EXPLOIT',6,7],['M','TRACE',5,3],['M','ICE',5,7]]
for name,changes in [('base',{}),('daemon_bait',{('M','DAEMON'):(4,6)}),('trace_bait',{('M','TRACE'):(6,6)}),('king_lure',{('M','EXPLOIT'):(6,6)}),('double_lure',{('M','DAEMON'):(4,4),('M','TRACE'):(6,6)})]:
    pieces=copy.deepcopy(crowded)
    for p in pieces:
        if tuple(p[:2]) in changes:p[2:]=changes[tuple(p[:2])]
    add('cluttered_discovery_'+name,'cluttered_other_actor_discovery',pieces,'Teacher adds capture decoys and a vulnerable own Kernel around a Trace-unblocks-Daemon capture. Check whether 100 simulations spend visits on high-material distractions instead of the different-actor winning setup.')
# A poisoned first-action Kernel capture observed in match5435 inspires new
# counterfactual boards; remote piece relocations below were explicitly chosen.
poison=[['C','KERNEL',4,3],['C','DAEMON',7,2],['C','EXPLOIT',6,3],['M','DAEMON',3,1],['M','EXPLOIT',0,2],['M','TRACE',2,7],['M','ICE',3,4],['M','KERNEL',0,7]]
edits=[(('C','DAEMON'),(7,1)),(('C','DAEMON'),(7,3)),(('C','DAEMON'),(7,4)),(('M','TRACE'),(2,6)),(('M','TRACE'),(1,7)),(('M','EXPLOIT'),(0,4)),(('C','EXPLOIT'),(6,1)),(('C','EXPLOIT'),(7,5))]
for i,(piece,pos) in enumerate(edits):
    ps=copy.deepcopy(poison)
    for p in ps:
        if tuple(p[:2])==piece:p[2:]=pos
    add('poisoned_capture_counterfactual_'+str(i),'poisoned_first_kernel_capture',ps,'Teacher counterfactual inspired by match5435: relocate one remote piece, keep the tempting Kernel D4-E5 Ice capture, and exhaust all second-action repairs. This constructed state is distinct from the observed game.',rnd=5)
for i,pos in enumerate([(2,5),(2,6),(1,4),(3,5),(2,3)]):
    add('ping_cage_counterfactual_'+str(i),'guarded_kernel_approach_cage',[['C','KERNEL',6,2],['M','KERNEL',3,4],['M','PING',*pos]],'Teacher changes the guarding Ping location around the E5 Kernel. Test whether Cyan C2-D3 walks into a two-step Kernel capture, and whether avoiding immediate loss actually establishes anything beyond the bounded horizon.',left=1,rnd=9)
for i,pos in enumerate([(3,1),(2,2),(4,2)]):
    add('opposing_kernel_cage_'+str(i),'two_kernel_ping_cage',[['M','KERNEL',2,5],['C','KERNEL',3,3],['C','PING',*pos]],'Teacher perturbs the Ping in the match5644-style cage. Magenta F6-F5 permits a two-step opposing-Kernel attack; some retreats may only delay defeat. Label next-turn safety separately from eventual value.',side='M',left=1,rnd=9)
for i,pos in enumerate([(2,4),(2,3),(3,4)]):
    add('firewall_last_piece_'+str(i),'firewall_then_last_piece_adjudication',[['M','KERNEL',1,6],['M','TRACE',1,7],['M','ICE',*pos],['C','KERNEL',5,2]],'Teacher counterfactual inspired by match5530: Trace burns on the firewall this action. A central Ice can become the final non-Kernel piece, capturable by the opposing Kernel, causing immediate bare-Kernel center adjudication. Analyze the post-burn inventory.',side='M',left=1,rnd=8)
out=dict(schema_version=1,provenance=raw['provenance'],curriculum=['multiple-actor discovered attacks under capture distractions','first-action poisoned Kernel captures with no second-action repair','opposing-Kernel approach cages','firewall changing last-piece endgame semantics'],inspiration_sources=[dict(path='codex/matches/site-v2-first-action-oracle.npz',sha256=hashlib.sha256((ROOT/'codex'/'matches'/'site-v2-first-action-oracle.npz').read_bytes()).hexdigest(),use='Geometric inspiration only; every seed explicitly modifies or reconstructs a counterfactual and remains reachability-unproven')],seeds=seeds)
(HERE/'hard_seed_proposals.json').write_text(json.dumps(out,indent=2));print(len(seeds))
