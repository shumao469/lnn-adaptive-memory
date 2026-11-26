import os, sys, json, time, hashlib, io, traceback, itertools
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'nc_reacher_objective'),str(ROOT/'observer_core'),str(ROOT/'observer_algorithm')]
import numpy as np
import torch
import evaluate_objectives as ev
from adaptive_observer import AdaptiveObserver
from reacher_retest_residual import event_features
torch.set_num_threads(1)
SEEDS=(8361001,8361002,8361003)
DEVELOP=tuple(range(8950001,8950017))
CAL=tuple(range(8950201,8950209))
HELD=tuple(range(8951001,8951025))
CELLS=tuple((r,d) for r in ('q_up','r_up') for d in (0,2,4))
START=time.time()
np_rng=np.random.default_rng(946373)

def write(name,obj):
 p=HERE/name
 t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 for attempt in range(12):
  try:
   os.replace(t,p);return
  except PermissionError:
   time.sleep(.025*(attempt+1))
 if name!='progress.json':raise PermissionError(f'Unable to write {p}')

def status(stage,done,total,**other):
 dt=time.time()-START
 write('progress.json',dict(status='running',stage=stage,completed=done,total=total,
  updated_at=datetime.now(timezone.utc).isoformat(),
  eta_seconds=round(dt*(total-done)/done) if done else None,elapsed_seconds=round(dt,1),**other))

class Controlled:
 def __init__(self,model,mode,seed,trigger=None,thresholds=None):
  self.model=model
  self.mode=mode
  self.tick=0
  self.random=np.random.default_rng(seed+7183)
  self.trigger=trigger
  self.thresholds=thresholds
  self.feature=None
  self.features=[]
  self.total=0
  self.written=0
 def initial(self,n):return self.model.initial(n)
 def gated(self,*args):return self.model.gated(*args)
 def step(self,raw,kf,event,carry):
  accepted=bool((event[0,:2]>.5).any())
  mask=(event[0,:2]>.5)
  z=event[0,2:4][mask].numpy()
  a=event[0,4:6][mask].numpy()
  hist=carry[2][0].numpy()
  action=raw[0,9:11].numpy()
  feature=np.array([np.mean(a) if len(a) else 0., np.linalg.norm(z),
    np.log(max(hist[0],.01)),np.log(max(hist[1],.01)),hist[2],
    hist[3],np.linalg.norm(action)],dtype=float)
  if accepted:
   self.features.append(feature.tolist());self.total+=1
  if accepted and self.tick==self.trigger:self.feature=feature.tolist()
  keep=True
  if accepted:
   if self.mode=='skip_once' and self.tick==self.trigger:keep=False
   elif self.mode in ('age','innovation','learned'):
    if self.mode=='age':score=-feature[0]
    elif self.mode=='innovation':score=-feature[1]
    else:
     x=(feature-self.thresholds['mu'])/self.thresholds['sd']
     pred=np.r_[1.,x]@self.thresholds['weights']
     score=float(-pred[0]/self.thresholds['target_scale'][0]-pred[1]/self.thresholds['target_scale'][1])
    threshold=self.thresholds[self.mode+'_threshold']
    # Ties are broken using a scenario-seeded PRNG, without access to future.
    keep=score>threshold or (score==threshold and self.random.random()<self.thresholds[self.mode+'_tie'])
   self.written+=int(keep)
  result,new=self.model.step(raw,kf,event,carry)
  if accepted and not keep:
   null=event.clone();null[:,2:4]=0.
   _,no_write=self.model.step(raw,kf,null,carry)
   new=(no_write[0],*new[1:])
   result=torch.tanh(self.model.readout(new[0]))*self.model.residual_cap
  self.tick+=1
  return result,new

def boot():
 global prior,robots,nominal,adaptive,local,plant,nomplant,models,ff,tables
 prior=ev.prior
 _,_,robots,nominal,adaptive,_,_,_=ev.margin.load()
 _,_,_,_,scales=prior.load()
 models={}
 receipt=json.loads((ROOT/'observer_algorithm/outputs/training_summary.json').read_text())
 plan=ev.digest(ROOT/'observer_algorithm/outputs/protocol.json')
 for seed in SEEDS:
  path=ROOT/f'observer_algorithm/outputs/adaptive_gain_{seed}.pt'
  assert ev.digest(path)==receipt[path.name]['sha256']
  payload=torch.load(path,map_location='cpu',weights_only=True)
  assert payload['protocol_sha256']==plan
  model=AdaptiveObserver(prior.MODEL_CFG,scales,True)
  model.load_state_dict(payload['state_dict'])
  models[seed]=model.eval()
 local=robots.nominal_local_model(prior.reacher.TASK)
 plant=robots._Simulator(prior.reacher.TASK)
 nomplant=robots._Simulator(prior.reacher.TASK)
 ff=ev.ff;tables={}
 torch.set_num_threads(1)

def run_task(task):
 reg,delay,case,seed,mode,thresholds=task
 if case not in tables:tables.update(ff.build_tables(robots,plant,nomplant,[case]))
 body=ff.FeedforwardPlant(plant,tables[case])
 body=ff.speed.DiagnosticSimulator(body,local,retention=.70)
 model=Controlled(models[seed],mode,case,80+delay,thresholds)
 row=prior.rollout(robots,body,nominal,model,reg,delay,case)
 row.update(seed=seed,mode=mode,writes=model.written,events=model.total,
  event_features=model.feature,features=model.features)
 return row

def robust_task(task):
 reg,delay,case=task
 if case not in tables:tables.update(ff.build_tables(robots,plant,nomplant,[case]))
 factory=nominal
 class Huber(factory):
  def _assimilate(self,mean,covariance,measurements):
   # Bound standardized capture-aligned innovations before normal KF updates.
   clean={}
   for channel,value in measurements.items():
    sd=np.sqrt(max(covariance[channel,channel]+self.r[channel],1e-12))
    clean[channel]=mean[channel]+np.clip(value-mean[channel],-2.5*sd,2.5*sd)
   return super()._assimilate(mean,covariance,clean)
 body=ff.FeedforwardPlant(plant,tables[case])
 body=ff.speed.DiagnosticSimulator(body,local,retention=.70)
 row=prior.rollout(robots,body,Huber,None,reg,delay,case)
 row.update(seed=None,mode='huber_kf',writes=None,events=None)
 return row

def readrows(name):
 p=HERE/name
 if not p.exists():return []
 with p.open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]

def run_stage(pool,stage,name,tasks,fn,key):
 old=readrows(name);done={key(r) for r in old}
 pending=[t for t in tasks if key(t) not in done]
 total=len(tasks)
 status(stage,len(old),total)
 with (HERE/name).open('a',encoding='utf-8',buffering=1) as f:
  for row in pool.map(fn,pending,chunksize=1):
   f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
   old.append(row)
   status(stage,len(old),total)
 return old

def key(x):
 if isinstance(x,tuple):return tuple(x[:5])
 return (x['regime'],x['delay'],x['case_seed'],x['seed'],x['mode'])

def keyrob(x):
 if isinstance(x,tuple):return tuple(x)
 return (x['regime'],x['delay'],x['case_seed'])

def main():
 protocol={'scope':'exploratory Reacher simulation with unseen scene splits',
  'development_scenarios':DEVELOP,'calibration_scenarios':CAL,'evaluation_scenarios':HELD,'cells':CELLS,'fit_seeds':SEEDS,
  'controller':'same PD and reference acceleration feedforward, common velocity retention .70',
  'single_event_training':'At tick 80+delay, keep versus skip one accepted neural innovation update; preserve KF update and innovation history; full post-shift trajectory objective.',
  'model':'7 deployment-visible summary features, standardized ridge regression of keep-minus-skip tracking RMSE and applied torque integral, dual-output aggregate score; trained solely on dev cases; each arriving event gated online.',
  'thresholds':'global target .5, empirical development data, randomized deterministic tie handling; actual rate reported; branch feedback can shift test rates.',
  'comparators':'age and innovation thresholds on identical dev set and target .5; always write; Huber capture-aligned KF clipping at 2.5 normalized standard deviations; same PD and filter, exogenous paired.',
  'data_separation':'test cases not used for training or calibration; regime not passed to model; no physical robot; no claim matching rates exactly in test.',
  'seed':946373,'updated_at':datetime.now(timezone.utc).isoformat()}
 path=HERE/'protocol.json'
 if not path.exists():write('protocol.json',protocol)
 receipt_path=HERE/'run_identity.json'
 files=[Path(__file__),ROOT/'observer_algorithm/adaptive_observer.py',
        ROOT/'observer_core/run_reacher_retest.py',
        ROOT/'observer_core/reacher_retest_kalman.py']
 identity={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
 if receipt_path.exists():
  if json.loads(receipt_path.read_text(encoding='utf-8-sig'))!=identity:raise RuntimeError('Code changed since last run')
 else:write('run_identity.json',identity)
 print('Starting Reacher closed-loop independent experiments',flush=True)
 with ProcessPoolExecutor(max_workers=4,initializer=boot) as pool:
  tasks=[(r,d,c,s,m,None) for r,d in CELLS for c in DEVELOP for s in SEEDS for m in ('keep','skip_once')]
  dev=run_stage(pool,'reacher_single_write_labels','labels.jsonl',tasks,run_task,key)
  grouped={(q['regime'],q['delay'],q['case_seed'],q['seed'],q['mode']):q for q in dev}
  x=[];y=[];invalid=[]
  for r,d,c,s in itertools.product(('q_up','r_up'),(0,2,4),DEVELOP,SEEDS):
   a=grouped[(r,d,c,s,'keep')];b=grouped[(r,d,c,s,'skip_once')]
   assert a['exogenous_sha256']==b['exogenous_sha256']
   if a['event_features'] is None or b['event_features'] is None:
    invalid.append([r,d,c,s]);continue
   x.append(a['event_features'])
   y.append([a['metrics']['tracking_rmse']-b['metrics']['tracking_rmse'],
      a['metrics']['control_energy']-b['metrics']['control_energy']])
  x=np.asarray(x);y=np.asarray(y)
  if len(x)<50:raise RuntimeError(f'Insufficient accepted events: {len(x)}')
  mu=x.mean(0);sd=np.maximum(x.std(0),1e-6);D=np.c_[np.ones(len(x)),(x-mu)/sd]
  weights=np.linalg.solve(D.T@D+10*np.diag([0,1,1,1,1,1,1,1]),D.T@y)
  scale=np.maximum(np.std(y,axis=0),1e-8)
  np.savez(HERE/'gate_reacher.npz',mu=mu,sd=sd,weights=weights,target_scale=scale)
  collect_tasks=[(r,d,c,s,'collect',None) for r,d in CELLS for c in CAL for s in SEEDS]
  collected=run_stage(pool,'calibrate_event_gates','calibration.jsonl',collect_tasks,run_task,key)
  features=np.asarray([f for row in collected for f in row['features']])
  scores={'age':-features[:,0],'innovation':-features[:,1],
    'learned':-(((features-mu)/sd)@weights[1:]+weights[0])[:,0]/scale[0]
              -(((features-mu)/sd)@weights[1:]+weights[0])[:,1]/scale[1]}
  threshold={'mu':mu.tolist(),'sd':sd.tolist(),'weights':weights.tolist(),'target_scale':scale.tolist()}
  for name,values in scores.items():
   t=float(np.median(values));up=float(np.mean(values>t));eq=float(np.mean(values==t))
   threshold[name+'_threshold']=t;threshold[name+'_tie']=float(np.clip((.5-up)/max(eq,1e-12),0,1))
  for round_id in range(2):
   stage=f'calibrate_online_write_rate_{round_id+1}'
   ctasks=[(r,d,c,s,m,threshold) for r,d in CELLS for c in CAL for s in SEEDS for m in ('age','innovation','learned')]
   crows=run_stage(pool,stage,f'online_calibration_{round_id+1}.jsonl',ctasks,run_task,key)
   for mode in ('age','innovation','learned'):
    subset=[v for v in crows if v['mode']==mode]
    feature=np.asarray([v for row in subset for v in row['features']])
    if mode=='age':values=-feature[:,0]
    elif mode=='innovation':values=-feature[:,1]
    else:
     pred=((feature-mu)/sd)@weights[1:]+weights[0]
     values=-pred[:,0]/scale[0]-pred[:,1]/scale[1]
    t=float(np.median(values));up=float(np.mean(values>t));eq=float(np.mean(values==t))
    threshold[mode+'_threshold']=t;threshold[mode+'_tie']=float(np.clip((.5-up)/max(eq,1e-12),0,1))
  write('gate_calibration.json',dict(thresholds=threshold,n_training=len(x),invalid_events=invalid,calibration_event_count=len(features),
       online_calibration_rounds=2,online_calibration_scenarios=list(CAL)))
  tasks=[(r,d,c,s,m,threshold) for r,d in CELLS for c in HELD for s in SEEDS for m in ('keep','age','innovation','learned')]
  rows=run_stage(pool,'reacher_online_paired_evaluation','evaluation.jsonl',tasks,run_task,key)
  tasks=[(r,d,c) for r,d in CELLS for c in HELD]
  robust=run_stage(pool,'robust_kalman_paired_evaluation','robust.jsonl',tasks,robust_task,keyrob)
  by={}
  for regime,delay in CELLS:
   cell=[q for q in rows if (q['regime'],q['delay'])==(regime,delay)]
   refs={(q['case_seed'],q['seed']):q for q in cell if q['mode']=='keep'}
   result={}
   for mode in ('age','innovation','learned'):
    subset=[q for q in cell if q['mode']==mode]
    diffs=[]
    for q in subset:
     ref=refs[q['case_seed'],q['seed']]
     assert ref['exogenous_sha256']==q['exogenous_sha256']
     diffs.append([q['metrics']['tracking_rmse']-ref['metrics']['tracking_rmse'],q['metrics']['control_energy']-ref['metrics']['control_energy']])
    result[mode]={'n':len(subset),'mean_tracking_diff_m':float(np.mean(diffs,axis=0)[0]),
      'mean_energy_diff':float(np.mean(diffs,axis=0)[1]),'write_rate':sum(q['writes'] for q in subset)/max(1,sum(q['events'] for q in subset)),
      'failures':sum(q['failed'] for q in subset)}
   rb=[q for q in robust if (q['regime'],q['delay'])==(regime,delay)]
   for q in rb:assert q['exogenous_sha256']==refs[q['case_seed'],SEEDS[0]]['exogenous_sha256']
   result['huber_kf']={'n':len(rb),'mean_tracking_rmse':float(np.mean([q['metrics']['tracking_rmse'] for q in rb])),
    'mean_energy':float(np.mean([q['metrics']['control_energy'] for q in rb])),'failures':sum(q['failed'] for q in rb)}
   by[f'{regime}_{delay*20}ms']=result
  write('summary.json',dict(status='completed',conditions=by,elapsed_seconds=round(time.time()-START,1)))
  status('completed',1,1);p=json.loads((HERE/'progress.json').read_text());p['status']='completed';p['eta_seconds']=0;write('progress.json',p)

if __name__=='__main__':
 try:main()
 except Exception as e:
  traceback.print_exc()
  write('progress.json',dict(status='failed',stage='exception',completed=0,total=None,
   updated_at=datetime.now(timezone.utc).isoformat(),eta_seconds=None,error=str(e)))
  raise

