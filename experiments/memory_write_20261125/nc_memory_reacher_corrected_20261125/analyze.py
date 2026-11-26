from pathlib import Path
import json, hashlib
import numpy as np

BASE = Path(__file__).resolve().parent.parent
SRC = BASE/'nc_memory_reacher_20261125'
OUT = Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name):
    with (SRC/name).open(encoding='utf-8-sig') as f:return [json.loads(line) for line in f if line.strip()]
quota=load('quota_evaluation.jsonl'); robust=load('robust.jsonl'); baseline=load('evaluation.jsonl')
scenes=sorted({q['case_seed'] for q in quota}); fits=sorted({q['seed'] for q in quota})
assert len(scenes)==24 and len(fits)==3 and len(quota)==1296 and len(robust)==144
conditions=[(r,d) for r in ('q_up','r_up') for d in (0,2,4)]
metrics=['tracking_rmse','control_energy']
comparators=['age','innovation','keep','huber_kf']
family=len(conditions)*len(metrics)*len(comparators)
assert family==48

def index(rows,mode):
    out={}
    for q in rows:
        if q['mode']==mode:
            key=(q['regime'],q['delay'],q['case_seed'],q['seed'])
            if key in out:raise RuntimeError('duplicate key')
            out[key]=q
    return out
items={m:index(quota,m) for m in ['learned','age','innovation']}
items['keep']=index(baseline,'keep'); items['huber_kf']=index(robust,'huber_kf')
rng=np.random.default_rng(9814951); ndraw=100000
scene_index=rng.integers(0,len(scenes),(ndraw,len(scenes)))
fit_index=rng.integers(0,len(fits),(ndraw,len(fits)))
alpha=0.05/family
out=[]
for r,d in conditions:
    for comp in comparators:
        matrix=np.empty((len(scenes),len(fits),len(metrics)))
        for i,scene in enumerate(scenes):
            for j,fit in enumerate(fits):
                a=items['learned'][(r,d,scene,fit)]
                k=(r,d,scene,None if comp=='huber_kf' else fit)
                b=items[comp][k]
                assert a['exogenous_sha256']==b['exogenous_sha256'],k
                assert not a['failed'] and not b['failed']
                for m,key in enumerate(metrics):matrix[i,j,m]=a['metrics'][key]-b['metrics'][key]
        for mi,metric in enumerate(metrics):
            v=matrix[:,:,mi]
            draws=np.empty(ndraw)
            for start in range(0,ndraw,5000):
                stop=min(ndraw,start+5000)
                draws[start:stop]=v[scene_index[start:stop,:,None],fit_index[start:stop,None,:]].mean(axis=(1,2))
            low,high=np.quantile(draws,[alpha/2,1-alpha/2])
            lo95,hi95=np.quantile(draws,[.025,.975])
            avg=float(v.mean())
            out.append(dict(regime=r,delay_ms=d*20,comparator=comp,metric=metric,
                estimate=avg,pointwise_95=[float(lo95),float(hi95)],bonferroni_familywise_95=[float(low),float(high)],
                excludes_zero=bool(low>0 or high<0),unit='m' if metric=='tracking_rmse' else 'squared torque integral',
                scene_count=len(scenes),fit_count=len(fits),rollouts_paired=len(scenes)*len(fits)))
result={
 'scope':'Post hoc exploratory reanalysis. The 48-test family was declared after the data had been observed. Do not call this preregistered or confirmatory.',
 'family':'2 noise conditions x 3 delays x 4 learned-minus-comparator contrasts (age, innovation, keep, Huber Kalman) x 2 endpoints (tracking RMSE, squared-torque integral) = 48; total 95% familywise error control targeted by Bonferroni adjusted crossed-bootstrap percentile intervals.',
 'method':'100000 crossed bootstrap draws resample the 24 shared scenario IDs and 3 fitted observer seeds as clusters. The same independent Huber rollout for a scenario is repeated across seeds and resampled by scenario. Bootstrap percentile intervals are approximate, particularly with only 3 fitted seeds sharing training data.',
 'confidence_level_per_interval':1-alpha,'draws':ndraw,'random_seed':9814951,'source_sha256':{n:sha(SRC/n) for n in ('quota_evaluation.jsonl','evaluation.jsonl','robust.jsonl','quota_protocol.json')},
 'physical_scope':'Simulated Reacher, known fixed 90-event post-switch horizon, 45 writes. Descriptive observed forced-phase write proportions in original quota_summary.json.',
 'contrasts':out}
(OUT/'corrected_analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
for metric in metrics:
  for comp in ('innovation','huber_kf','age','keep'):
    row=[v for v in out if v['metric']==metric and v['comparator']==comp]
    print(metric,comp,'familywise excludes zero',sum(v['excludes_zero'] for v in row),'of',len(row))
    for v in row:print(' ',v['regime'],v['delay_ms'],f"{v['estimate']:.7g}",[round(x,7) for x in v['bonferroni_familywise_95']])
