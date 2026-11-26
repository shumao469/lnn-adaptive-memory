import json,time,os,sys,itertools,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import importlib.util
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'observer_core'))
spec=importlib.util.spec_from_file_location('reacher_gate_runner',HERE/'run.py')
base=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=base
spec.loader.exec_module(base)

class QuotaGate(base.Controlled):
 def __init__(self,*a,**k):
  super().__init__(*a,**k)
  self.eligible=0
  self.post_writes=0
  self.forced=0
 def step(self,raw,kf,event,carry):
  accepted=bool((event[0,:2]>.5).any())
  pre=self.tick<60
  orig=self.thresholds[self.mode+'_threshold']
  force=False
  if accepted:
   if pre:
    self.thresholds[self.mode+'_threshold']=-1e99
   else:
    remaining=90-self.eligible
    need=45-self.post_writes
    if remaining<1:raise RuntimeError('More post-shift events than the protocol fixed budget')
    if need>=remaining:
     self.thresholds[self.mode+'_threshold']=-1e99
     force=True
    elif need<=0:self.thresholds[self.mode+'_threshold']=1e99
  prev=self.written
  try:result,new=super().step(raw,kf,event,carry)
  finally:self.thresholds[self.mode+'_threshold']=orig
  if accepted and not pre:
   self.eligible+=1
   self.post_writes+=self.written-prev
   self.forced+=int(force and self.written>prev)
  return result,new

def boot():
 base.boot()

def task(task):
 r,d,c,seed,mode,threshold=task
 if c not in base.tables:base.tables.update(base.ff.build_tables(base.robots,base.plant,base.nomplant,[c]))
 body=base.ff.FeedforwardPlant(base.plant,base.tables[c])
 body=base.ff.speed.DiagnosticSimulator(body,base.local,retention=.70)
 m=QuotaGate(base.models[seed],mode,c,80+d,dict(threshold))
 row=base.prior.rollout(base.robots,body,base.nominal,m,r,d,c)
 row.update(seed=seed,mode=mode,post_writes=m.post_writes,post_events=m.eligible,
            forced_writes=m.forced,post_rate=m.post_writes/max(1,m.eligible),
            forced_fraction=m.forced/max(1,m.post_writes))
 if not row['failed'] and (m.eligible!=90 or m.post_writes!=45):
  raise RuntimeError(f'Quota events/write count mismatch {r},{d},{c},{seed},{mode}: {m.eligible}, {m.post_writes}')
 return row

def main():
 while not (HERE/'summary.json').exists():
  q=json.loads((HERE/'progress.json').read_text(encoding='utf-8-sig'))
  if q.get('status')=='failed':raise RuntimeError('Primary Reacher process failed, refusing automatic quota')
  time.sleep(2)
 threshold=json.loads((HERE/'gate_calibration.json').read_text(encoding='utf-8-sig'))['thresholds']
 protocol={'scope':'exploratory fixed evaluation-window write-budget comparator',
  'design':'before tick 60 always write, then exactly 45 writes in 90 post-shift packet-event ticks for each gate',
  'availability':'simulation uses known remaining 90-event horizon; force if remaining events equals remaining budget, stop writes once budget exhausted',
  'gate_thresholds':'development-calibrated threshold from gate_calibration.json; no outcomes or noise condition used online',
  'features':'same learned/age/innovation policies, PD controller, velocity filtering and exogenous pairing as original evaluation',
  'metrics':'forced tail writes, post-shift events, write rate, tracking, squared actuator command integral, saturation, failure'}
 base.write('quota_protocol.json',protocol)
 tasks=[(r,d,c,s,m,threshold) for r,d in base.CELLS for c in base.HELD for s in base.SEEDS for m in ('age','innovation','learned')]
 with ProcessPoolExecutor(max_workers=4,initializer=boot) as pool:
  rows=base.run_stage(pool,'reacher_fixed_quota_evaluation','quota_evaluation.jsonl',tasks,task,base.key)
 refs=base.readrows('evaluation.jsonl')
 robust=base.readrows('robust.jsonl')
 rng=np.random.default_rng(981370)
 out={}
 for r,d in base.CELLS:
  cohort=[q for q in rows if (q['regime'],q['delay'])==(r,d)]
  anchors={(q['case_seed'],q['seed']):q for q in refs if (q['regime'],q['delay'],q['mode'])==(r,d,'keep')}
  h={q['case_seed']:q for q in robust if (q['regime'],q['delay'])==(r,d)}
  units={m:{(q['case_seed'],q['seed']):q for q in cohort if q['mode']==m} for m in ('age','innovation','learned')}
  entry={}
  for m in units:
   values=list(units[m].values())
   entry[m]={'n_rollouts':len(values),'write_rate':sum(v['post_writes'] for v in values)/sum(v['post_events'] for v in values),
             'forced_phase_write_fraction':sum(v['forced_writes'] for v in values)/max(1,sum(v['post_writes'] for v in values)),
             'failures':sum(v['failed'] for v in values)}
  for comparator in ('age','innovation','keep','huber_kf'):
   diffs=[]
   for scene in base.HELD:
    one=[]
    for seed in base.SEEDS:
     a=units['learned'][(scene,seed)]
     b=units[comparator][(scene,seed)] if comparator in units else anchors[(scene,seed)] if comparator=='keep' else h[scene]
     assert a['exogenous_sha256']==b['exogenous_sha256']
     one.append([a['metrics']['tracking_rmse']-b['metrics']['tracking_rmse'],
                 a['metrics']['control_energy']-b['metrics']['control_energy'],
                 a['metrics']['saturation_fraction']-b['metrics']['saturation_fraction']])
    diffs.append(one)
   v=np.array(diffs);draw=[]
   for i in range(5000):
    ids=rng.integers(0,len(base.HELD),len(base.HELD))
    fits=rng.integers(0,len(base.SEEDS),len(base.SEEDS))
    draw.append(v[ids][:,fits].mean(axis=(0,1)))
   entry['learned_minus_'+comparator]={'mean':v.mean(axis=(0,1)).tolist(),
      'interval95':np.quantile(draw,[.025,.975],axis=0).tolist(),
      'units':['m tracking RMSE','squared torque integral','saturation fraction'],
      'interval':'descriptive crossed bootstrap scene and seed, 5000 draws'}
  out[f'{r}_{d*20}ms']=entry
 base.write('quota_summary.json',dict(status='completed',conditions=out,forced_phase_definition='Share of writes after the remaining quota equals remaining events, including writes that the unconstrained gate could also have selected.'))
 base.write('progress.json',dict(status='completed',stage='reacher_fixed_quota_completed',
  completed=len(rows),total=len(tasks),updated_at=base.datetime.now(base.timezone.utc).isoformat(),eta_seconds=0))
if __name__=='__main__':
 try:main()
 except Exception as e:
  import traceback;traceback.print_exc()
  base.write('progress.json',dict(status='failed',stage='quota_exception',completed=0,total=None,
   updated_at=base.datetime.now(base.timezone.utc).isoformat(),eta_seconds=None,error=str(e)))
  raise
