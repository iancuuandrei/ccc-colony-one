"""Check every live bare configuration through the composed two-output graph."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import onnxruntime as ort

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
    composed=ort.InferenceSession(a.model,sess_options=opt,providers=['CPUExecutionProvider'])
    proof=ort.InferenceSession(str(Path(a.model).with_suffix('.proof.onnx')),sess_options=opt,providers=['CPUExecutionProvider'])
    coords=np.array([(s//8,s%8) for s in range(64)]);rings=np.minimum.reduce([coords[:,0],coords[:,1],7-coords[:,0],7-coords[:,1]])
    buf=[];n=0;t=time.time()
    def flush():
        nonlocal n
        if not buf:return
        x=np.stack(buf);expected,value,enabled=proof.run(None,{'observation':x});policy,v=composed.run(None,{'observation':x})
        assert enabled.all();assert np.array_equal(value,v)
        want=np.broadcast_to((40*expected-20)[:,None,:],(len(buf),64,64)).reshape(-1,4096)
        assert np.array_equal(want,policy)
        n+=len(buf);buf.clear()
    for ring in range(4):
        squares=np.flatnonzero(rings>=ring)
        for side in [0.,1.]:
            for acts in [.5,1.]:
                for own in squares:
                    for enemy in squares:
                        if own==enemy:continue
                        x=np.zeros((16,64),np.float32);x[5,own]=1;x[11,enemy]=1;x[12]=(rings<ring);x[13]=acts;x[14]=1;x[15]=side
                        buf.append(x.reshape(16,8,8))
                        if len(buf)>=128:flush()
    flush();result=dict(status='PASS',states=n,policy_exact=True,value_exact=True,batch='dynamic (including final partial)',seconds=time.time()-t,sha256=hashlib.sha256(Path(a.model).read_bytes()).hexdigest())
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
