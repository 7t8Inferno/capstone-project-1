"""Generate public-demo CSVs with exactly 350 current-model Review predictions.

WARNING: These rows are synthetically generated and deliberately SELECTED by
the model's own predictions. They are not fraud ground truth or valid test metrics.
"""
from pathlib import Path
import hashlib
import pickle
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
MODEL_HASH="232ad91f3ea13dc956662b3718f2381b667590565c4ddd39c7cf5bd5792f6d5f"
rng=np.random.default_rng(20261111)
REVIEW_N,CLEAR_N=350,650

def scores(model,arr):
    p=np.asarray(model.predict_proba(np.asarray(arr,dtype=np.float32)))[:,1]
    assert np.isfinite(p).all()
    return p

def find_review_seed(model,data,threshold):
    """Only used if the app's synthetic examples contain no Review rows."""
    i=int(np.argmax(scores(model,data)))
    best=data[i:i+1].copy()
    candidate=np.array([50000]+[-5,4,-7,5,-4,-1,-5,1,-2,-6,3,-6,0,-7,0,-5,-7,-2,0,1,1,-1,0,0,0,0,0,0]+[80],dtype=float)
    best=np.vstack([best,candidate])
    for k in range(28):
        picks=best[rng.integers(0,len(best),500)].copy()
        noise=rng.normal(0,[2.5,1.2,.6,.25][min(k//7,3)],size=picks.shape)
        noise[:,0]*=300
        noise[:,-1]*=3
        choices=picks+noise
        choices[:,0]=np.clip(choices[:,0],0,172800)
        choices[:,-1]=np.clip(choices[:,-1],0,5000)
        values=scores(model,choices)
        best=choices[np.argsort(values)[-25:]]
        if float(np.max(values))>=threshold+.02:
            break
    found=best[scores(model,best)>=threshold]
    if len(found)==0:
        raise RuntimeError("Unable to synthesize any Review case at this threshold")
    return found

def generate_group(model,seeds,n,want_review,threshold):
    batches=[]
    total=0
    for attempt in range(50):
        if total>=n:
            break
        count=max(100,2*(n-total))
        choices=seeds[rng.integers(0,len(seeds),count)].astype(np.float64).copy()
        choices[:,1:-1]+=rng.normal(0,.045,size=(count,28))
        choices[:,0]+=rng.normal(0,4,size=count)
        choices[:,-1]+=rng.normal(0,.65,size=count)
        choices[:,0]=np.clip(choices[:,0],0,172800)
        choices[:,-1]=np.clip(choices[:,-1],0,5000)
        # Score rounded values, because those exact values will be exported.
        choices=np.round(choices,6)
        mask=(scores(model,choices)>=threshold)==want_review
        accepted=choices[mask][:n-total]
        if len(accepted):
            batches.append(accepted)
            total+=len(accepted)
    if total<n:
        base=np.round(seeds[rng.integers(0,len(seeds),n-total)],6)
        assert np.all((scores(model,base)>=threshold)==want_review)
        batches.append(base)
    out=np.vstack(batches)
    assert len(out)==n and np.all((scores(model,out)>=threshold)==want_review)
    return out

def main():
    raw=(ROOT/"fraud_model.pkl").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==MODEL_HASH
    bundle=pickle.loads(raw)
    model=bundle["model"]
    threshold=float(bundle["threshold"])
    cols=list(bundle["features"])
    assert cols==["Time"]+[f"V{k}" for k in range(1,29)]+["Amount"]
    data=pd.read_csv(ROOT/"demo_stream.csv").loc[:,cols].to_numpy(dtype=float)
    p=scores(model,data)
    high=data[p>=threshold]
    low=data[p<threshold]
    print(f"Initial synthetic samples: {len(high)} Review, {len(low)} Clear; threshold {threshold:.6f}",flush=True)
    if not len(high):
        high=find_review_seed(model,data,threshold)
    assert len(low)>0
    a=generate_group(model,high,REVIEW_N,True,threshold)
    b=generate_group(model,low,CLEAR_N,False,threshold)
    output=np.vstack([a,b])
    shuffled=output[rng.permutation(len(output))]
    dest=ROOT/"samples"
    dest.mkdir(exist_ok=True)
    test=dest/"mentor_test_1000_rows_350_review.csv"
    pd.DataFrame(shuffled,columns=cols).to_csv(test,index=False,float_format="%.6f")
    reloaded=pd.read_csv(test)
    prediction=scores(model,reloaded.loc[:,cols].to_numpy(dtype=float))
    is_review=prediction>=threshold
    assert len(reloaded)==1000
    assert int(is_review.sum())==350, f"Expected 350 Review; got {int(is_review.sum())}"
    assert int((~is_review).sum())==650
    assert test.stat().st_size<=1024*1024
    assert reloaded.shape==(1000,30)
    assert np.isfinite(reloaded.to_numpy(dtype=float)).all()
    assert (reloaded["Time"]>=0).all() and (reloaded["Amount"]>=0).all()
    key=pd.DataFrame({"Row":np.arange(1,1001),"Expected_Decision":np.where(is_review,"Review","Clear"),"Model_score":np.round(prediction,6)})
    key.to_csv(dest/"mentor_test_1000_expected_decisions.csv",index=False)
    print(f"Verified using exact exported CSV: {int(is_review.sum())} Review, {int((~is_review).sum())} Clear",flush=True)
    print(f"CSV file size: {test.stat().st_size} bytes, 30 feature columns; no Class column",flush=True)
    print("NOT independent fraud data: Review labels were used to select synthetic rows.",flush=True)

if __name__=="__main__":
    main()
