"""Timestamp-aligned paired tapes for the linear factorial observer."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch

import linear


EVENT_FEATURES = ("accepted", "innovation", "innovation_var", "age_seconds", "quality", "quality_present")


def tape_sha(tape):
    h = hashlib.sha256()
    for name in ("truth", "action", "reference", "f", "b", "q", "r",
                 "process_increments", "measurement_increments"):
        value = np.asarray(getattr(tape, name), dtype="<f8", order="C")
        h.update(name.encode("ascii") + repr(value.shape).encode("ascii"))
        h.update(value.tobytes())
    h.update(json.dumps([(p.source_tick, p.arrival_tick, p.quality, p.quality_present)
                         for p in tape.packets], allow_nan=False).encode("ascii"))
    return h.hexdigest()


def open_history(cfg, tape, method="fixed_kf"):
    row = linear.rollout(cfg, tape, method=method, closed_loop=False)
    event = np.zeros((cfg.horizon, len(EVENT_FEATURES)), dtype=np.float32)
    used = set()
    for p in row["event_records"]:
        arrival, capture = p["arrival_tick"], p["source_tick"]
        if not 0 <= capture <= arrival < cfg.horizon or arrival in used:
            raise ValueError("Expected a single causal packet per tick under fixed delay")
        used.add(arrival)
        event[arrival] = [1., p["innovation"], p["innovation_var"], (arrival-capture)*cfg.dt,
                          p["quality"], p["quality_present"]]
    if not np.array_equal(event[:, 0] != 0, row["fresh"]):
        raise ValueError("Event lane differs from nominal causal freshness")
    return {**row, "event": event}


def novel_tape_bank(cfg, regime, delay, case_ids):
    tapes = [linear.generate_tape(cfg, regime, delay, seed, quality_available=False)
             for seed in case_ids]
    if not tapes or any(p.arrival_tick-p.source_tick != delay for t in tapes for p in t.packets):
        raise ValueError("Frozen fixed-delay grid differs")
    return tapes, [tape_sha(t) for t in tapes]


class CausalSensor:
    def __init__(self, cfg, tape):
        self.dt = cfg.dt
        self.noise = tape.measurement_increments
        self.captures = {p.source_tick:p for p in tape.packets}
        self.queue = {}
        self.latest = None
        self.observer = linear.DelayedKalman(cfg)

    def step(self, tick, state, previous_action):
        if tick in self.captures:
            spec = self.captures[tick]
            value = float(state[0] + self.noise[tick])
            packet = linear.Packet(tick,spec.arrival_tick,value,spec.quality,spec.quality_present)
            if packet.arrival_tick in self.queue:
                raise ValueError("Multiple arrivals outside fixed-delay protocol")
            self.queue[packet.arrival_tick] = packet
        arrived = self.queue.pop(tick,None)
        current, scores = self.observer.step(tick,float(previous_action),[arrived] if arrived else ())
        event = (0.,) * len(EVENT_FEATURES)
        if arrived:
            if arrived.source_tick > tick or (self.latest and arrived.source_tick <= self.latest.source_tick):
                raise ValueError("Noncausal or reordered packet outside protocol")
            self.latest = arrived
            innovation, variance, _ = scores[arrived.source_tick]
            event = (1., innovation, variance,(tick-arrived.source_tick)*self.dt,
                     arrived.quality,arrived.quality_present)
        last = self.latest
        age = (tick-last.source_tick)*self.dt if last else (tick+1)*self.dt
        raw = (last.value if last else 0.,last is not None,arrived is not None,age,self.dt,previous_action,
               last.quality if last else 0.,last.quality_present if last else 0.)
        return raw,current,event


def summary_metrics(cfg, truth, estimate, reference, action):
    truth,estimate,reference,action = map(np.asarray,(truth,estimate,reference,action))
    if (truth.shape != (cfg.horizon,2) or estimate.shape != truth.shape or reference.shape != truth.shape
            or action.shape != (cfg.horizon,) or any(not np.isfinite(x).all() for x in (truth,estimate,reference,action))):
        raise ValueError("Nonfinite or misaligned closed-loop trajectory")
    t = cfg.change_tick
    xerr = np.sqrt(np.mean((estimate[t:]-truth[t:])**2,axis=0))
    terr = truth[:,0]-reference[:,0]
    return {"position_rmse":float(xerr[0]),"velocity_rmse":float(xerr[1]),
            "tracking_rmse":float(np.sqrt(np.mean(terr[t:]**2))),
            "whole_tracking_rmse":float(np.sqrt(np.mean(terr**2))),
            "command_rms":float(np.sqrt(np.mean(action[t:]**2))),
            "saturated_ticks":int(np.count_nonzero(abs(action) >= cfg.max_abs_action)),
            "failed":False}


def feedback_batch(cfg, tapes, model, *, return_traces=False):
    """Advance each plant under its own action, batching only the neural operator."""
    if not tapes or any(t.cfg != cfg for t in tapes):
        raise ValueError("Expected paired plants with one configuration")
    n,horizon = len(tapes),cfg.horizon
    sensors = [CausalSensor(cfg,t) for t in tapes]
    f,b,w,ref = (np.stack([getattr(t,k) for t in tapes]) for k in ("f","b","process_increments","reference"))
    truth,estimate = (np.empty((n,horizon,2)) for _ in range(2))
    commands = np.empty((n,horizon))
    truth[:,0] = np.stack([t.truth[0] for t in tapes])
    raw_batch = np.empty((n,len(linear.FEATURES)),dtype=np.float32)
    kf_batch = np.empty((n,2),dtype=np.float32)
    event_batch = np.empty((n,len(EVENT_FEATURES)),dtype=np.float32)
    # Torch tensors share the reusable input arrays; all consumers run synchronously.
    raw_tensor,kf_tensor,event_tensor = map(torch.from_numpy,(raw_batch,kf_batch,event_batch))
    exact_kf = np.empty((n,2))
    past = np.zeros(n)
    hist_raw,hist_kf,hist_event = (np.empty((n,horizon,k),dtype=np.float32) for k in
                                  (len(linear.FEATURES),2,len(EVENT_FEATURES))) if return_traces else (None,)*3
    sums = {k:np.zeros(n) for k in ("tau","gain","innovation_update")}
    sq = {k:np.zeros(n) for k in ("tau","gain")}
    event_count = np.zeros(n)
    tau_count = 0
    with torch.inference_mode():
        carry = model.initial(n) if model is not None else None
        for tick in range(horizon):
            if tick:
                truth[:,tick] = np.einsum("bij,bj->bi",f[:,tick],truth[:,tick-1])+b[:,tick]*past[:,None]+w[:,tick]
            for i,sensor in enumerate(sensors):
                raw_batch[i],exact_kf[i],event_batch[i] = sensor.step(tick,truth[i,tick],past[i])
            kf_batch[:] = exact_kf
            if return_traces:
                hist_raw[:,tick],hist_kf[:,tick],hist_event[:,tick] = raw_batch,kf_batch,event_batch
            if model is None:
                estimate[:,tick] = exact_kf
            else:
                delta,carry,diagnostics = model.step(raw_tensor,kf_tensor,event_tensor,carry)
                estimate[:,tick] = exact_kf + model.gate(delta,raw_tensor,tick).numpy()
                if tick >= cfg.change_tick:
                    tau = diagnostics["tau"].detach().numpy().reshape(n,-1).mean(axis=1,dtype=np.float64)
                    gain = diagnostics["gain"].detach().numpy().reshape(n,-1).mean(axis=1,dtype=np.float64)
                    upd = diagnostics["innovation_update"].detach().numpy().reshape(n,-1)
                    mask = event_batch[:,0].astype(float)
                    sums["tau"] += tau
                    sums["gain"] += gain * mask
                    sums["innovation_update"] += np.linalg.norm(upd,axis=1)*mask
                    sq["tau"] += tau**2
                    sq["gain"] += gain**2*mask
                    event_count += mask
                    tau_count += 1
            error = ref[:,tick] - estimate[:,tick]
            past = np.clip(cfg.kp*error[:,0]+cfg.kd*error[:,1],-cfg.max_abs_action,cfg.max_abs_action)
            commands[:,tick] = past
    rows = []
    for i in range(n):
        item = summary_metrics(cfg,truth[i],estimate[i],ref[i],commands[i])
        if model is not None:
            if tau_count != horizon-cfg.change_tick or event_count[i] <= 0:
                raise ValueError("Expected post-change observation diagnostics")
            for key,count in (("tau",tau_count),("gain",event_count[i])):
                avg = sums[key][i]/count
                variance = max(0.,sq[key][i]/count-avg**2)
                item.update({f"post_{key}_mean":float(avg),f"post_{key}_temporal_sd":math.sqrt(variance)})
            item["post_innovation_update_l2_mean"] = float(sums["innovation_update"][i]/event_count[i])
            item["post_event_count"] = int(event_count[i])
        rows.append(item)
    if return_traces:
        return rows,{"truth":truth,"estimate":estimate,"action":commands,"raw":hist_raw,"kf":hist_kf,"event":hist_event}
    return rows
