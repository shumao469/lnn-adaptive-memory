"""Summarize paired sequential-gate rollouts after upstream completion."""
import csv,json,time
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
import run
ROOT=Path(__file__).resolve().parent

def main():
    while True:
        p=ROOT/"period_quota_progress.json"
        status=json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if status.get("status")=="failed": raise RuntimeError("Sequential stage failed: "+str(status))
        if status.get("status")=="completed":break
        time.sleep(3)
    bank=defaultdict(dict)
    for f in sorted((ROOT/"period_quota_cells").glob("seq_*.csv")):
        with f.open(newline="",encoding="utf-8") as inp:
            for r in csv.DictReader(inp):
                key=(float(r["strength"]),r["regime"],int(r["delay_ms"]),int(r["scene"]),int(r["fit"]))
                bank[key][r["policy"]]=r
    allout=[]
    rng=np.random.default_rng(947511)
    for strength in run.STRENGTHS:
        for regime in ("q_up","r_up"):
            for delay in (0,40,80):
                for policy in ("learned","age","innovation","random"):
                    batch=[(key,item) for key,item in bank.items() if key[:3]==(strength,regime,delay)]
                    if len(batch)!=48 or any(set(item)!={"keep","learned","age","innovation","random"} for _,item in batch):
                        raise ValueError(f"Missing paired runs in {strength,regime,delay,policy}")
                    fits=sorted({k[4] for k,_ in batch});scenes=sorted({k[3] for k,_ in batch})
                    d={(k[3],k[4]):v for k,v in batch}
                    rec={"strength":strength,"regime":regime,"delay_ms":delay,"policy":policy,"scenes":len(scenes),"fits":len(fits)}
                    for metric in ("tracking_rmse","control_integral","saturation_fraction","write_fraction","write_fraction_full","forced_fraction_full"):
                        arr=np.array([[float(d[(sc,fit)][policy][metric])-float(d[(sc,fit)]["keep"][metric]) for sc in scenes] for fit in fits])
                        fi=rng.integers(0,len(fits),(4000,len(fits),1))
                        si=rng.integers(0,len(scenes),(4000,1,len(scenes)))
                        draw=arr[fi,si].mean(axis=(1,2))
                        rec[metric+"_minus_keep"]={"mean":float(arr.mean()),"descriptive_95":np.quantile(draw,[.025,.975]).tolist()}
                        rec[metric+"_mean"]=float(np.mean([float(v[policy][metric]) for _,v in batch]))
                    allout.append(rec)
    run.save("period_quota_paired_analysis.json",{"scope":"Exploratory fixed post-switch quota, identical pre-switch actions, with forced late writes. 16 held-out scenes x 3 fixed observer fits; crossed bootstrap descriptive 95%, unadjusted.","contrasts":allout})
    run.save("period_quota_analysis_progress.json",{"status":"completed","stage":"paired_summary","completed":len(allout),"total":len(allout),"updated_at":datetime.now(timezone.utc).isoformat(),"eta_seconds":0})
if __name__=="__main__":
    try:main()
    except Exception as e:
        run.save("period_quota_analysis_progress.json",{"status":"failed","stage":"exception","completed":0,"total":72,"updated_at":datetime.now(timezone.utc).isoformat(),"eta_seconds":None,"error":repr(e)})
        raise
