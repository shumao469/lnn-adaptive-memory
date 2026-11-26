from pathlib import Path
import json
import time
import csv
from collections import defaultdict
import numpy as np
root=Path('nc_memory_gate_20261125')
while not (root/'period_quota_analysis_progress.json').exists():
 time.sleep(1)
status=json.loads((root/'period_quota_analysis_progress.json').read_text(encoding='utf8'))
if status.get('status')!='completed':raise RuntimeError(str(status))
allruns=defaultdict(dict)
for f in (root/'period_quota_cells').glob('seq_*.csv'):
 for r in csv.DictReader(f.open(newline='',encoding='utf8')):
  allruns[(float(r['strength']),r['regime'],int(r['delay_ms']),int(r['scene']),int(r['fit']))][r['policy']]=r
rng=np.random.default_rng(947516)
data=[]
for strength in (.5,1.,1.5):
 for regime in ('q_up','r_up'):
  for delay in (0,40,80):
   subset={k:v for k,v in allruns.items() if k[:3]==(strength,regime,delay)}
   scenes=sorted({k[3] for k in subset}); fits=sorted({k[4] for k in subset})
   for comparator in ('age','innovation','random'):
    out={'strength':strength,'regime':regime,'delay_ms':delay,'comparison':'learned_minus_'+comparator}
    for metric in ('tracking_rmse','control_integral','saturation_fraction','write_fraction','forced_fraction_full'):
     a=np.array([[float(subset[(strength,regime,delay,scene,fit)]['learned'][metric])-float(subset[(strength,regime,delay,scene,fit)][comparator][metric]) for scene in scenes] for fit in fits])
     fi=rng.integers(0,len(fits),(4000,len(fits),1));si=rng.integers(0,len(scenes),(4000,1,len(scenes)))
     ci=np.quantile(a[fi,si].mean(axis=(1,2)),[.025,.975])
     out[metric]={'mean':float(a.mean()),'descriptive_95':ci.tolist()}
    data.append(out)
(root/'period_quota_direct_comparisons.json').write_text(json.dumps({'status':'completed','interval':'descriptive 95% crossed scenario x fit bootstrap; not corrected for multiple comparisons','comparisons':data},indent=2),encoding='utf8')
print('completed direct comparisons',len(data))
