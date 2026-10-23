"""Independent exact-oracle subset validation of tactical-v2 certificates."""
from pathlib import Path
import json,hashlib
import numpy as np
import onnxruntime as ort

HERE=Path(__file__).parent
def main():
    corpus=np.load(HERE/'own_turn_corpus.npz')
    model=HERE.parent/'tactical'/'approach-diagnostic.onnx'
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    sess=ort.InferenceSession(str(model),opts,providers=['CPUExecutionProvider'])
    policy,values,approach=sess.run(None,{'observation':corpus['observations']})
    predicted=approach.reshape(-1,4096)>.5
    truth=corpus['first_action_win_masks']
    fp=np.argwhere(predicted & ~truth)
    value_cert=values>=1-1e-7
    value_fp=np.flatnonzero(value_cert & ~corpus['any_win'])
    report={'status':'PASS' if len(fp)==len(value_fp)==0 else 'FAIL',
        'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
        'states':len(values),'oracle_positive_states':int(corpus['any_win'].sum()),
        'predicted_approach_actions':int(predicted.sum()),'approach_false_positive_count':len(fp),
        'value_one_states':int(value_cert.sum()),'value_one_false_positive_count':len(value_fp),
        'oracle_winning_actions':int(truth.sum()),
        'false_positive_actions':[{'name':str(corpus['names'][n]),'index':int(a)} for n,a in fp],
        'false_positive_value_states':[str(corpus['names'][n]) for n in value_fp],
        'scope':'Bounded corpus; subset soundness only, not completeness or playing strength.'}
    (HERE/'tactical_v2_oracle_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    assert report['status']=='PASS'

if __name__=='__main__':main()
