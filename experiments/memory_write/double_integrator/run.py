"""Exploratory held-out evaluation and single-arrival neural write gate."""
import csv
import itertools
import json
import os
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np
import torch
from scipy.stats import spearmanr
import pilot_engine as engine

ROOT = Path(__file__).resolve().parent
CELLS = ROOT / "cells"
CELLS.mkdir(exist_ok=True)
SCENES = tuple(range(9471001, 9471097))
TRAIN, VALIDATION, TEST = SCENES[:64], SCENES[64:80], SCENES[80:]
STRENGTHS = (.5, 1., 1.5)
COLUMNS = ("age_s", "normalized_innovation", "fast_innovation_power",
           "slow_innovation_power", "innovation_lag_product", "prior_innovation", "prior_action")
TARGETS = ("tracking_rmse_delta", "control_integral_delta")
STEPS = 54 + 1 + 54
START = time.monotonic()

def save(name, data):
    path = ROOT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)

def progress(stage, completed, detail=""):
    elapsed = time.monotonic() - START
    save("progress.json", {"status":"running","stage":stage,"completed":completed,"total":STEPS,
         "updated_at":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":round(elapsed,1),
         "eta_seconds":round(elapsed * (STEPS-completed)/completed) if completed else None,"detail":detail})

def cfg_for(strength):
    return replace(engine.linear.Config(), q_add_position_sd=.070*strength, q_add_velocity_sd=.070*strength)

def rows_for(regime, delay, strength, fit, model, scenelist):
    cfg = cfg_for(strength)
    engine.CFG = cfg
    tapes, identities = engine.events.novel_tape_bank(cfg, regime, delay, scenelist)
    f, rmse, effort = engine.rollout(tapes, model, delay)
    result=[]
    for i, (scene, identity) in enumerate(zip(scenelist, identities)):
        split = "train" if scene in TRAIN else "validation" if scene in VALIDATION else "test"
        row={"strength":strength,"regime":regime,"delay_ms":int(delay*cfg.dt*1000),"scene":scene,
             "split":split,"fit":fit,"tape_sha256":identity,
             "tracking_rmse_keep":float(rmse[0,i]),"tracking_rmse_gate":float(rmse[2,i]),
             "tracking_rmse_delta":float(rmse[0,i]-rmse[2,i]),
             "control_integral_keep":float(effort[0,i]),"control_integral_gate":float(effort[2,i]),
             "control_integral_delta":float(effort[0,i]-effort[2,i])}
        row.update({k:float(v[i]) for k,v in f.items()})
        result.append(row)
    return result

def write_csv(path, rows):
    tmp=path.with_suffix(".tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    os.replace(tmp,path)

def read_csv(path):
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))

def dataset():
    return [r for path in sorted(CELLS.glob("labels_*.csv")) for r in read_csv(path)]

def train(rows):
    torch.manual_seed(94750)
    x=np.array([[float(r[k]) for k in COLUMNS] for r in rows],np.float64)
    y=np.array([[float(r[k]) for k in TARGETS] for r in rows],np.float64)
    trainidx=np.array([r["split"]=="train" and float(r["strength"])<1.5 for r in rows])
    validx=np.array([r["split"]=="validation" and float(r["strength"])<1.5 for r in rows])
    testidx=np.array([r["split"]=="test" for r in rows])
    means=x[trainidx].mean(0); std=x[trainidx].std(0);std[std<1e-9]=1.
    ym=y[trainidx].mean(0); ys=y[trainidx].std(0);ys[ys<1e-9]=1.
    xx=torch.tensor((x-means)/std,dtype=torch.float32); yy=torch.tensor((y-ym)/ys,dtype=torch.float32)
    net=torch.nn.Sequential(torch.nn.Linear(7,16),torch.nn.Tanh(),torch.nn.Linear(16,2))
    opt=torch.optim.Adam(net.parameters(),lr=.005,weight_decay=1e-3)
    best=None; bestloss=float("inf");bestepoch=0
    for epoch in range(401):
        net.train();opt.zero_grad(); loss=((net(xx[trainidx])-yy[trainidx])**2).mean();loss.backward();opt.step()
        with torch.no_grad(): v=float(((net(xx[validx])-yy[validx])**2).mean())
        if v<bestloss:
            bestloss=v;best={k:t.detach().clone() for k,t in net.state_dict().items()};bestepoch=epoch
    net.load_state_dict(best);net.eval()
    with torch.no_grad(): pred=net(xx).numpy()*ys+ym
    def measures(mask,col):
        truth=y[mask,col]; estimate=pred[mask,col]
        corr=spearmanr(truth,estimate).statistic
        return {"n_rows":int(mask.sum()),"mae":float(np.mean(abs(truth-estimate))),
                "direction_accuracy":float(np.mean((truth<0)==(estimate<0))),
                "constant_direction_accuracy":float(np.mean((truth<0)==(ym[col]<0))),
                "spearman":float(corr) if np.isfinite(corr) else None}
    metrics={}
    for strength in STRENGTHS:
        m=testidx & np.array([float(r["strength"])==strength for r in rows])
        metrics[str(strength)]={k:measures(m,i) for i,k in enumerate(TARGETS)}
    # Accept the regular neural write only when the predicted tracking effect
    # is below the validation median. This selects a fixed, pre-evaluated
    # threshold, rather than tuning it on held-out performance.
    threshold=float(np.median(pred[validx,0]))
    checkpoint={"state_dict":best,"xmean":means,"xstd":std,"ymean":ym,"ystd":ys,
                "input_features":COLUMNS,"targets":TARGETS,"threshold":threshold,
                "best_epoch":bestepoch,"validation_mse_standardized":bestloss}
    torch.save(checkpoint,ROOT/"gate_checkpoint.pt")
    save("predictive_summary.json",{"status":"complete","test_metrics_by_strength":metrics,
         "held_out_strength_1_5":True,"single_packet_utility_only":True,"best_epoch":bestepoch,
         "threshold":threshold,"n_train":int(trainidx.sum()),"n_validation":int(validx.sum()),"n_test":int(testidx.sum())})
    engine.GATE_MODEL=net;engine.GATE_STATS=checkpoint;engine.GATE_THRESHOLD=threshold

def online():
    path=ROOT/"gate_checkpoint.pt"
    chk=torch.load(path,map_location="cpu",weights_only=False)
    net=torch.nn.Sequential(torch.nn.Linear(7,16),torch.nn.Tanh(),torch.nn.Linear(16,2))
    net.load_state_dict(chk["state_dict"]);net.eval()
    engine.GATE_MODEL=net;engine.GATE_STATS=chk;engine.GATE_THRESHOLD=chk["threshold"]

def main():
    torch.set_num_threads(1)
    bank, hashes=engine.models()
    proto={"scope":"Exploratory single-packet write-gating experiment; one intervention at a fixed event, full closed-loop response afterwards",
       "scene_splits":{"train":list(TRAIN),"validation":list(VALIDATION),"test":list(TEST)},
       "strengths":list(STRENGTHS),"strength_definition":"multiplier on post-change added process-noise standard deviations; R-increase variance increment scales with squared multiplier",
       "training_strengths":[.5,1.],"test_strengths":list(STRENGTHS),"regimes":["q_up","r_up"],"delay_ms":[0,40,80],
       "features":COLUMNS,"targets":TARGETS,"selected_checkpoint_sha256":hashes,"architecture":"7-16-2 tanh MLP",
       "labels":"keep minus skip with one target residual write, Kalman and history unchanged",
       "selection":"median predicted tracking delta on validation strengths .5 and 1.0; no tuning on test outcomes",
       "notes":"No independent model retraining; three frozen base observers; learned gate predicts local 1-second outcome. No sequential multi-write deployment claim."}
    pp=ROOT/"protocol.json"
    if pp.exists():
        if json.loads(pp.read_text(encoding="utf-8")) != json.loads(json.dumps(proto)):
            raise RuntimeError("Existing protocol differs, refusing to reuse results")
    else:save("protocol.json",proto)
    tasks=list(itertools.product(STRENGTHS,("q_up","r_up"),(0,2,4),tuple(bank)))
    progress("single_write_labels",0)
    for n,(strength,regime,delay,fit) in enumerate(tasks,1):
        target=CELLS/f"labels_{strength:g}_{regime}_{delay}_{fit}.csv"
        if not target.exists():write_csv(target,rows_for(regime,delay,strength,fit,bank[fit],SCENES))
        progress("single_write_labels",n,f"{regime}, noise multiplier {strength:g}, delay {delay*20} ms, fit {fit}")
    data=dataset()
    if len(data)!=len(tasks)*len(SCENES):raise RuntimeError("Missing or duplicated label cells")
    train(data)
    progress("predictor_fit",55)
    online()
    for j,(strength,regime,delay,fit) in enumerate(tasks,1):
        target=CELLS/f"online_{strength:g}_{regime}_{delay}_{fit}.csv"
        if not target.exists():write_csv(target,rows_for(regime,delay,strength,fit,bank[fit],TEST))
        progress("single_event_gate_closed_loop",55+j,f"{regime}, noise multiplier {strength:g}, delay {delay*20} ms, fit {fit}")
    rows=[r for p in sorted(CELLS.glob("online_*.csv")) for r in read_csv(p)]
    save("online_summary.json",{"status":"complete","number_pairs":len(rows),
       "tracking_keep_minus_gate":float(np.mean([float(r["tracking_rmse_delta"]) for r in rows])),
       "control_keep_minus_gate":float(np.mean([float(r["control_integral_delta"]) for r in rows])),
       "caveat":"Exploratory test of one gated packet per rollout, no multiplicity-adjusted statistical test."})
    save("progress.json",{"status":"completed","stage":"complete","completed":STEPS,"total":STEPS,
         "updated_at":datetime.now(timezone.utc).isoformat(),"eta_seconds":0,
         "elapsed_seconds":round(time.monotonic()-START,1)})

if __name__=="__main__":
    try:main()
    except Exception as exc:
        save("progress.json",{"status":"failed","stage":"exception","completed":0,"total":STEPS,
             "updated_at":datetime.now(timezone.utc).isoformat(),"eta_seconds":None,"error":repr(exc)})
        raise
