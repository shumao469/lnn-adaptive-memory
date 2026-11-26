"""Sequential neural write-gate closed-loop pilot, paired scenarios."""
import csv, hashlib, itertools, json, os, time
from pathlib import Path
from datetime import datetime, timezone
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np
import torch
import run as pipeline
import pilot_engine as eng

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"sequential_cells"
OUT.mkdir(exist_ok=True)
START=time.monotonic()
MODE=("keep","learned","age","innovation","random")

def state(stage,complete,total,detail="",status="running"):
    pipeline.save("sequential_progress.json",{"status":status,"stage":stage,"completed":complete,
     "total":total,"updated_at":datetime.now(timezone.utc).isoformat(),
     "elapsed_seconds":round(time.monotonic()-START,2),"eta_seconds":None if complete==0 else round((time.monotonic()-START)*(total-complete)/complete),
     "detail":detail})

def calibrate():
    rows=pipeline.dataset()
    cal=[r for r in rows if r["split"]=="validation" and float(r["strength"])<1.5]
    inp=np.array([[float(r[k]) for k in pipeline.COLUMNS] for r in cal],np.float32)
    chk=eng.GATE_STATS
    with torch.inference_mode():
        score=eng.GATE_MODEL(torch.tensor((inp-chk["xmean"])/chk["xstd"],dtype=torch.float32)).numpy()[:,0]*chk["ystd"][0]+chk["ymean"][0]
    keep=float(np.mean(score<=chk["threshold"]))
    # Ages have mass points: tie fractions are randomized using a separate,
    # reproducible, per-scenario arrival sequence.
    def cut(a,upper):
        cutoff=float(np.quantile(a,1-keep if upper else keep))
        strict=float(np.mean(a>cutoff if upper else a<cutoff))
        equal=float(np.mean(a==cutoff))
        return cutoff,float(np.clip((keep-strict)/equal,0,1)) if equal else 0.
    agecut,agetie=cut(inp[:,0],False)
    zcut,ztie=cut(abs(inp[:,1]),True)
    cfg={"calibration_scope":"validation scenes at multipliers 0.5 and 1.0; single-event feature distribution",
      "learned_retention_rate":keep,"age_cut":agecut,"age_tie_probability":agetie,
      "innovation_absolute_cut":zcut,"innovation_tie_probability":ztie,
      "random_retention_probability":keep,"caveat":"matched in calibration expectation; report actual per-policy rates in closed loop"}
    pipeline.save("sequential_gate_calibration.json",cfg)
    return cfg

def rollout(tapes,model,delay,cal):
    cfg=eng.CFG;n=len(tapes);h=cfg.horizon;B=n*len(MODE)
    mult=list(tapes)*len(MODE)
    sensors=[eng.events.CausalSensor(cfg,t) for t in mult]
    dyn,bmat,noise,ref=[np.stack([getattr(t,k)[:h] for t in mult]) for k in ("f","b","process_increments","reference")]
    truth=np.stack([t.truth[0] for t in mult]);past=np.zeros(B)
    carry=model.initial(B); err=np.zeros((len(MODE),n,h-cfg.change_tick))
    cost=np.zeros((len(MODE),n));sat=np.zeros((len(MODE),n));write=np.zeros((len(MODE),n))
    arrivals=np.zeros((len(MODE),n))
    rng=np.random.default_rng(88700+delay)
    for tick in range(h):
        if tick:truth=np.einsum("bij,bj->bi",dyn[:,tick],truth)+bmat[:,tick]*past[:,None]+noise[:,tick]
        packet=[sensor.step(tick,s,u) for sensor,s,u in zip(sensors,truth,past)]
        raw,nom,event=[np.asarray(x,dtype=np.float32) for x in zip(*packet)]
        accepted=event[:,0]>.5
        innovation=event[:,1]/np.sqrt(np.maximum(event[:,2],1e-9))
        hist=carry[2].numpy()
        X=np.column_stack((event[:,3],innovation,hist[:,0],hist[:,1],hist[:,2],hist[:,3],past))
        inp=(X-eng.GATE_STATS["xmean"])/eng.GATE_STATS["xstd"]
        with torch.inference_mode():
            pred=eng.GATE_MODEL(torch.tensor(inp,dtype=torch.float32)).numpy()[:,0]
        score=pred*eng.GATE_STATS["ystd"][0]+eng.GATE_STATS["ymean"][0]
        toss=rng.random((len(MODE),n)).reshape(-1)
        keep=np.ones(B,dtype=bool)
        for k in range(1,len(MODE)):
            part=slice(k*n,(k+1)*n)
            if k==1:keep[part]=score[part]<=eng.GATE_THRESHOLD
            elif k==2:keep[part]=(X[part,0]<cal["age_cut"])|((X[part,0]==cal["age_cut"])&(toss[part]<cal["age_tie_probability"]))
            elif k==3:keep[part]=(abs(innovation[part])>cal["innovation_absolute_cut"])|((abs(innovation[part])==cal["innovation_absolute_cut"])&(toss[part]<cal["innovation_tie_probability"]))
            else:keep[part]=toss[part]<cal["random_retention_probability"]
        blocked=~keep & accepted
        def block(_module,_args,output):
            out=output.clone();out[torch.from_numpy(blocked)]=0.;return out
        handle=model.innovation_direction.register_forward_hook(block)
        try:
            with torch.inference_mode(): residual,carry,_=model.step(torch.from_numpy(raw),torch.from_numpy(nom),torch.from_numpy(event),carry)
        finally:handle.remove()
        est=np.asarray([p[1] for p in packet],dtype=float)+model.gate(residual,torch.from_numpy(raw),tick).detach().numpy()
        delta=ref[:,tick]-est
        past=np.clip(cfg.kp*delta[:,0]+cfg.kd*delta[:,1],-cfg.max_abs_action,cfg.max_abs_action)
        if tick>=cfg.change_tick:
            err[:,:,tick-cfg.change_tick]=(truth[:,0]-ref[:,tick,0]).reshape(len(MODE),n)
            cost+=cfg.dt*past.reshape(len(MODE),n)**2
            sat+=(np.abs(past.reshape(len(MODE),n))>=cfg.max_abs_action)
            write+=(keep&accepted).reshape(len(MODE),n)
            arrivals+=accepted.reshape(len(MODE),n)
    return np.sqrt(np.mean(err**2,axis=2)),cost,sat/(h-cfg.change_tick),write/np.maximum(arrivals,1)

def main():
    state("waiting_for_checkpoint",0,54)
    while True:
        f=ROOT/"progress.json"
        status=json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        if status.get("status")=="failed":raise RuntimeError("Upstream failed: "+str(status))
        if status.get("status")=="completed" and (ROOT/"gate_checkpoint.pt").exists():break
        time.sleep(3)
    eng.torch.set_num_threads(1)
    bank,_=eng.models();pipeline.online();cal=calibrate()
    tasks=list(itertools.product(pipeline.STRENGTHS,("q_up","r_up"),(0,2,4),tuple(bank)))
    completed=0;start=time.monotonic()
    for strength,regime,delay,fit in tasks:
        target=OUT/f"seq_{strength:g}_{regime}_{delay}_{fit}.csv"
        if not target.exists():
            eng.CFG=pipeline.cfg_for(strength)
            tapes,identities=eng.events.novel_tape_bank(eng.CFG,regime,delay,pipeline.TEST)
            rmse,cost,sat,rates=rollout(tapes,bank[fit],delay,cal)
            rows=[]
            for j,mode in enumerate(MODE):
                for i,scene in enumerate(pipeline.TEST):
                    rows.append({"strength":strength,"regime":regime,"delay_ms":delay*20,"fit":fit,
                      "scene":scene,"tape_sha256":identities[i],"policy":mode,
                      "tracking_rmse":float(rmse[j,i]),"control_integral":float(cost[j,i]),
                      "saturation_fraction":float(sat[j,i]),"write_fraction":float(rates[j,i])})
            pipeline.write_csv(target,rows)
        completed+=1;state("sequential_closed_loop",completed,len(tasks),f"{regime} strength {strength} delay {delay*20} fit {fit}")
    rows=[r for p in sorted(OUT.glob("seq_*.csv")) for r in pipeline.read_csv(p)]
    pipeline.save("sequential_summary.json",{"status":"completed","rows":len(rows),"modes":MODE,
      "duration_seconds":time.monotonic()-start,"calibration":cal,"scope":"exploratory multiple arriving packets; policies independently close the loop. Equivalent write rate is in validation expectation, verified rates are recorded per run."})
    state("complete",54,54,status="completed")

if __name__=="__main__":
    try:main()
    except Exception as exc:
        pipeline.save("sequential_progress.json",{"status":"failed","stage":"exception","completed":0,
         "total":54,"updated_at":datetime.now(timezone.utc).isoformat(),"eta_seconds":None,"error":repr(exc)})
        raise
