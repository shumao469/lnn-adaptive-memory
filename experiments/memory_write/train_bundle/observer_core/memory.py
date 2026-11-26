"""Matched causal fixed/adaptive tau residuals and delivered-packet sensitivity.

Uses only the supplied causal features and fixed-KF estimate. True states are
training labels, never model or kernel inputs. In the controlled observation
perturbation, open-loop actions and the entire exogenous tape are held fixed;
the causal KF and neural hidden state are replayed from the same initial prior.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import numbers
import time

import numpy as np
import torch
from torch import nn


STATIC = 'learned_static'
ADAPTIVE = 'state_age'
ARMS = (STATIC, ADAPTIVE)
FEATURES = ('observed_position', 'present', 'fresh', 'age_seconds', 'dt_seconds',
            'previous_applied_action', 'quality', 'quality_present')
RAW_DIM = len(FEATURES)
X_DIM = 2


@dataclass(frozen=True)
class MemoryConfig:
    hidden_size: int = 24
    optimizer_steps: int = 192
    batch_size: int = 12
    sequence_length: int = 40
    burn_in: int = 8
    startup_ticks: int = 24
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    tau_min_seconds: float = .015
    tau_initial_seconds: float = .15
    tau_max_seconds: float = 1.5
    max_observation_age_seconds: float = .65
    residual_cap_state_sd_fraction: float = .15
    residual_cap_fixed_kf_error_rms_multiple: float = 2.5

    def __post_init__(self):
        for name in ('hidden_size','optimizer_steps','batch_size','sequence_length','burn_in','startup_ticks'):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in ('burn_in','startup_ticks') else 1):
                raise ValueError(f'Invalid {name}')
        if not 0 <= self.burn_in < self.sequence_length:
            raise ValueError('Burn-in must leave at least one scored sample')
        for name in ('learning_rate','tau_min_seconds','tau_initial_seconds','tau_max_seconds',
                     'max_observation_age_seconds','residual_cap_state_sd_fraction','residual_cap_fixed_kf_error_rms_multiple'):
            value = getattr(self,name)
            if not isinstance(value, numbers.Real) or isinstance(value,bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid {name}')
        if not 0 < self.tau_min_seconds < self.tau_initial_seconds < self.tau_max_seconds:
            raise ValueError('Tau bounds and initial value are not ordered')
        if not isinstance(self.weight_decay,numbers.Real) or isinstance(self.weight_decay,bool) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Invalid weight decay')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _coerce_cfg(cfg):
    if cfg is None:
        return MemoryConfig()
    if isinstance(cfg, MemoryConfig):
        return cfg
    if isinstance(cfg, dict):
        return MemoryConfig(**cfg)
    raise TypeError('Expected MemoryConfig or equivalent mapping')


def _train_arrays(rows, cfg):
    if not isinstance(rows, (tuple,list)) or not rows:
        raise ValueError('Provide one or more training-only linear rollout mappings')
    raw = np.stack([np.asarray(row['raw'], dtype=np.float32) for row in rows])
    kf = np.stack([np.asarray(row['kf'], dtype=np.float32) for row in rows])
    truth = np.stack([np.asarray(row['truth'], dtype=np.float32) for row in rows])
    _require(raw.ndim == 3 and raw.shape[-1] == RAW_DIM and kf.shape == truth.shape == (*raw.shape[:2],X_DIM),
             'Training raw/fixed-KF/truth array shape mismatch')
    _require(raw.shape[1] >= cfg.sequence_length and raw.shape[1] > cfg.startup_ticks and
             all(np.isfinite(x).all() for x in (raw,kf,truth)), 'Training trajectories too short or nonfinite')
    _require(np.all(raw[...,4] > 0) and np.all(raw[...,3] >= 0) and
             np.isin(raw[...,[1,2,7]],(0,1)).all() and np.all(raw[...,2] <= raw[...,1]),
             'Invalid positive rollout dt or causal sensor-present/fresh flags')
    return np.ascontiguousarray(raw), np.ascontiguousarray(kf), np.ascontiguousarray(truth)


def _train_norm(raw, kf, truth, cfg):
    residual = truth - kf
    scale = np.ones(RAW_DIM,dtype=np.float32)
    scale[0] = max(.05, float(np.std(truth[...,0])))
    scale[3], scale[4] = .20, .02
    scale[5] = max(.05, float(np.sqrt(np.mean(raw[...,5]**2))))
    norm_state = np.maximum(np.std(truth,axis=(0,1)), .01)
    residual_rms = np.maximum(np.sqrt(np.mean(residual**2,axis=(0,1))), 1e-4)
    cap = np.minimum(cfg.residual_cap_state_sd_fraction*norm_state,
                     cfg.residual_cap_fixed_kf_error_rms_multiple*residual_rms)
    return {'raw_scale':scale, 'kf_mean':np.mean(kf,axis=(0,1)), 'kf_scale':norm_state,
            'residual_rms':residual_rms, 'residual_cap':cap}


class MemoryObserver(nn.Module):
    def __init__(self, arm, cfg, normalization):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unregistered memory model')
        self.arm, self.cfg, self.width = arm, cfg, cfg.hidden_size
        for key, values in normalization.items():
            self.register_buffer(key, torch.as_tensor(values,dtype=torch.float32).clone())
        p = (cfg.tau_initial_seconds-cfg.tau_min_seconds)/(cfg.tau_max_seconds-cfg.tau_min_seconds)
        self.rate = nn.Parameter(torch.full((self.width,), math.log(p/(1-p)), dtype=torch.float32))
        self.drive = nn.Sequential(nn.Linear(RAW_DIM+X_DIM+self.width,self.width),nn.Tanh(),
                                   nn.Linear(self.width,self.width),nn.Tanh())
        self.readout = nn.Linear(self.width,X_DIM)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.rate_head = nn.Linear(2,self.width,bias=False) if arm == ADAPTIVE else None
        if self.rate_head is not None:
            nn.init.zeros_(self.rate_head.weight)

    def initial(self,batch):
        return self.raw_scale.new_zeros(batch,self.width)

    def step(self,raw,kf,h):
        elapsed = raw[:,4:5]
        x = raw/self.raw_scale
        k = (kf-self.kf_mean)/self.kf_scale
        pre = self.rate.unsqueeze(0).expand_as(h)
        if self.rate_head is not None:
            state_context = torch.tanh((k.square().mean(-1,keepdim=True)+1e-8).sqrt())
            age_context = torch.tanh(raw[:,3:4]*raw[:,1:2]/.20)
            pre = pre+self.rate_head(torch.cat((state_context,age_context), dim=-1))
        tau = self.cfg.tau_min_seconds+(self.cfg.tau_max_seconds-self.cfg.tau_min_seconds)*torch.sigmoid(pre)
        decay = torch.exp(-elapsed/tau)
        drive = self.drive(torch.cat((x,k,h),dim=-1))
        h = decay*h+(1-decay)*drive
        residual = torch.tanh(self.readout(h))*self.residual_cap
        return residual,h,tau

    def forward(self,raw,kf,*,start_tick=None):
        h = self.initial(raw.shape[0]); out = []
        if start_tick is None:
            start_tick = torch.zeros(raw.shape[0],dtype=torch.int64,device=raw.device)
        for i in range(raw.shape[1]):
            event = raw[:,i]
            residual,h,_ = self.step(event,kf[:,i],h)
            available = ((event[:,1:2] > .5) & (event[:,3:4] <= self.cfg.max_observation_age_seconds) &
                         (start_tick[:,None]+i >= self.cfg.startup_ticks))
            out.append(residual*available)
        return torch.stack(out,dim=1)


def _params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _train_digest(arrays):
    m = hashlib.sha256()
    for x in arrays:
        m.update(str(x.shape).encode('ascii')+b':'+str(x.dtype).encode('ascii')+b'\0')
        m.update(memoryview(x).cast('B'))
    return m.hexdigest()


def train_pair(train_rows, cfg=None, seed=20261116, callback=None):
    """Fit two observers on the same training-only histories and supervised targets.

    `callback(step, total)`, if supplied, runs after paired steps. Each step
    corresponds to one optimizer update per arm (two total model updates).
    No tuning/selection, future samples or changes to the control law.
    """
    cfg = _coerce_cfg(cfg)
    _require(type(seed) is int and 0 <= seed < 2**63, 'Invalid training seed')
    torch.set_num_threads(1)
    raw, kf, truth = _train_arrays(train_rows, cfg)
    norm = _train_norm(raw,kf,truth,cfg)
    with torch.random.fork_rng(devices=[]):
        models = {}
        for arm in ARMS:
            torch.random.default_generator.manual_seed(seed)
            models[arm] = MemoryObserver(arm,cfg,norm).to('cpu')
    a,b = (models[m].state_dict() for m in ARMS)
    _require(all(torch.equal(a[k],b[k]) for k in a), 'Shared weights did not start identically')
    params = {arm:_params(model) for arm,model in models.items()}
    _require((params[ADAPTIVE]-params[STATIC])/params[STATIC] <= .05, 'Trainable model capacities differ by more than 5%')
    optimizers = {arm:torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
                  for arm, model in models.items()}
    generator = np.random.default_rng(np.random.SeedSequence([seed, 0x4D454D]))
    time_ids = np.arange(cfg.sequence_length)
    start = time.perf_counter()
    elapsed_first = None
    batch_digest = hashlib.sha256()
    last_loss = {}
    for step in range(1,cfg.optimizer_steps+1):
        ids = generator.integers(len(raw),size=cfg.batch_size,dtype=np.int64)
        offsets = generator.integers(raw.shape[1]-cfg.sequence_length+1,size=cfg.batch_size,dtype=np.int64)
        selected = (ids[:,None],offsets[:,None]+time_ids[None])
        x, k, y = (torch.from_numpy(t[selected]) for t in (raw,kf,truth))
        t0 = torch.from_numpy(offsets)
        delta_truth = y - k
        batch_digest.update(ids.tobytes()+offsets.tobytes())
        for arm in ARMS:
            model,opt = models[arm],optimizers[arm]
            estimate = model(x,k,start_tick=t0)
            residual = (estimate[:,cfg.burn_in:] - delta_truth[:,cfg.burn_in:])/model.residual_rms
            loss = torch.mean(residual.square())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite paired observer training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            last_loss[arm] = float(loss.detach())
        elapsed = time.perf_counter()-start
        if step == 1:
            elapsed_first = elapsed
        if callback is not None:
            callback(step,cfg.optimizer_steps)
    for model in models.values():
        model.eval()
    receipt = {'scope':'development_only', 'seed':seed, 'config':asdict(cfg), 'device':'cpu', 'cpu_threads':1,
               'trajectories':len(raw), 'ticks_per_trajectory':raw.shape[1], 'features':list(FEATURES),
               'optimizer_steps_each_arm':cfg.optimizer_steps, 'optimizer_updates_total':2*cfg.optimizer_steps,
               'optimizer_batch_index_sha256':batch_digest.hexdigest(), 'training_array_sha256':_train_digest((raw,kf,truth)),
               'active_parameter_count':params, 'same_initial_shared_parameters':True, 'same_causal_inputs_and_examples':True,
               'selected_checkpoint':'final_fixed_step', 'no_validation_selection':True,
               'final_batch_normalized_mse':last_loss, 'training_only_normalization':{k:v.tolist() for k,v in norm.items()},
               'first_pair_step_seconds':elapsed_first, 'elapsed_seconds':time.perf_counter()-start}
    return models,receipt


class _StreamingResidual:
    def __init__(self, model):
        self.model = model.eval()
        self.h = model.initial(1)
        self.tick = 0
        self.last_tau = None
        self.tau_history = []

    def __call__(self,raw,kf):
        x = np.asarray(raw,dtype=np.float32)
        k = np.asarray(kf,dtype=np.float32)
        _require(x.shape == (RAW_DIM,) and k.shape == (X_DIM,) and np.isfinite(x).all() and np.isfinite(k).all(),
                 'Causal streaming input shape or values invalid')
        with torch.inference_mode():
            residual, self.h, tau = self.model.step(torch.from_numpy(x)[None],torch.from_numpy(k)[None],self.h)
            self.last_tau = tau[0].numpy().astype(np.float64,copy=True)
            self.tau_history.append(self.last_tau)
            active = self.tick >= self.model.cfg.startup_ticks and x[1] > .5 and x[3] <= self.model.cfg.max_observation_age_seconds
            self.tick += 1
            return residual[0].numpy().astype(np.float64,copy=True) if active else np.zeros(X_DIM,dtype=np.float64)


def new_observer(model):
    """Return a fresh residual `(raw_row, kf_row) -> [position, velocity]`."""
    return _StreamingResidual(model)


def checkpoint(model):
    """Return a CPU, weights-only-compatible snapshot of one learned observer."""
    norm_keys = ('raw_scale', 'kf_mean', 'kf_scale', 'residual_rms', 'residual_cap')
    return {'schema_version':1, 'scope':'paired_linear_observer_development_only', 'arm':model.arm,
            'features':list(FEATURES), 'config':asdict(model.cfg),
            'normalization':{k:getattr(model,k).detach().cpu().tolist() for k in norm_keys},
            'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}


def restore_checkpoint(payload):
    """Restore a previously verified local snapshot (files are not opened)."""
    _require(payload['schema_version'] == 1 and payload['features'] == list(FEATURES) and
             payload['scope'] == 'paired_linear_observer_development_only', 'Wrong linear observer checkpoint identity')
    model = MemoryObserver(payload['arm'],_coerce_cfg(payload['config']),payload['normalization'])
    model.load_state_dict(payload['state_dict'],strict=True)
    _require(all(torch.isfinite(x).all() for x in model.state_dict().values()), 'Nonfinite model parameters')
    return model.eval()


def predict_many(model, rows, *, batch_size=32):
    """Estimate `N×T×2` from fully frozen open-loop histories, without τ logging.

    The world histories and previous actions are not mutated. No true state is
    requested or read. Each independent history starts from zero hidden state.
    """
    _require(isinstance(rows,(list,tuple)) and bool(rows) and type(batch_size) is int and batch_size >= 1,
             'At least one immutable history and a positive prediction batch size required')
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            batch = rows[start:start+batch_size]
            raw64 = np.stack([np.asarray(r['raw'],dtype=np.float64) for r in batch])
            kf64 = np.stack([np.asarray(r['kf'],dtype=np.float64) for r in batch])
            _require(raw64.ndim == 3 and raw64.shape[-1] == RAW_DIM and kf64.shape == (*raw64.shape[:2],X_DIM) and
                     np.isfinite(raw64).all() and np.isfinite(kf64).all(), 'Invalid frozen causal histories')
            raw, kf = (torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)) for x in (raw64,kf64))
            residual = model(raw,kf).numpy().astype(np.float64)
            outputs.append(kf64+residual)
    return np.concatenate(outputs)


def tau_summary(residual, *, ticks=None):
    values = np.asarray(residual.tau_history,dtype=float)
    if ticks is not None:
        values = values[ticks]
    return _tau_summary(values)


def _tau_summary(values):
    _require(values.size > 0 and values.ndim == 2 and np.isfinite(values).all(), 'No valid measured tau history')
    return {'tau_seconds_mean':float(values.mean()), 'tau_seconds_median':float(np.median(values)),
            'tau_seconds_q05':float(np.quantile(values,.05)), 'tau_seconds_q95':float(np.quantile(values,.95)),
            'minimum_seconds':float(values.min()), 'maximum_seconds':float(values.max()),
            'ticks':int(values.shape[0]), 'units':int(values.shape[1])}


def _kernel_model_estimates(model, histories, *, tau_start, tau_stop):
    raw_np = np.stack([row['raw'] for row in histories])
    kf_np = np.stack([row['kf'] for row in histories])
    estimates = kf_np.astype(np.float64)
    tau_trace = []
    model.eval()
    with torch.inference_mode():
        raw = torch.from_numpy(np.ascontiguousarray(raw_np,dtype=np.float32))
        kf = torch.from_numpy(np.ascontiguousarray(kf_np,dtype=np.float32))
        # The three causal histories are byte-identical before the delivery.
        # Evaluate that shared prefix once then branch, avoiding tiny BLAS
        # row-dependent rounding discrepancies falsely resembling leakage.
        _require(all(np.array_equal(part[0,:tau_start],x[:tau_start]) for part in (raw_np,kf_np) for x in part[1:]),
                 'Perturbation changed a causal learned-model history before delivery')
        h = model.initial(1)
        for i in range(tau_start):
            x = raw[:1,i]
            residual,h,_ = model.step(x,kf[:1,i],h)
            if i >= model.cfg.startup_ticks and x[0,1] > .5 and x[0,3] <= model.cfg.max_observation_age_seconds:
                estimates[:,i] += residual.numpy()[0]
        h = h.expand(len(histories),-1).clone()
        for i in range(tau_start,tau_stop):
            x = raw[:,i]
            residual,h,tau = model.step(x,kf[:,i],h)
            if i >= model.cfg.startup_ticks:
                active = ((x[:,1] > .5) & (x[:,3] <= model.cfg.max_observation_age_seconds)).numpy()
                estimates[active,i] += residual.numpy()[active]
            tau_trace.append(tau[0].numpy().astype(np.float64,copy=True))
    return estimates,np.array(tau_trace)


def effective_memory(linear_cfg, tape, models, source_tick, *, horizon_ticks=60, delta=.001):
    """Local central difference for one packet on a frozen open-loop tape.

    This is a finite-window sensitivity, not a claim about a system's global
    time constant, closed-loop recovery, monotone half-life, or performance.
    The position and velocity sensitivities retain their own units.
    """
    from linear import rollout

    _require(set(models) == set(ARMS) and type(source_tick) is int and source_tick >= 0 and
             type(horizon_ticks) is int and horizon_ticks >= 1 and math.isfinite(delta) and delta > 0,
             'Invalid measurement kernel inputs')
    packets = [packet for packet in tape.packets if packet.source_tick == source_tick]
    _require(len(packets) == 1, 'Requested source does not identify exactly one delivered packet')
    packet = packets[0]
    arrival = int(packet.arrival_tick)
    dt = float(linear_cfg.dt)
    _require(source_tick <= arrival and dt > 0, 'Noncausal delivery identity')
    summary = {'scope':'open_loop_frozen_action_history_local_sensitivity', 'source_tick':source_tick,
               'arrival_tick':arrival, 'arrival_age_seconds':float((arrival-source_tick)*dt),
               'observation_perturbation_magnitude':float(delta), 'requested_lag_ticks':horizon_ticks,
               'age_definition':'(current control tick - acquisition tick) × dt', 'methods':{}}
    base = rollout(linear_cfg,tape,method='fixed_kf',closed_loop=False)
    perturb = dict(method='fixed_kf',closed_loop=False,perturb_source_tick=source_tick)
    plus = rollout(linear_cfg,tape,**perturb,perturb_delta=delta)
    minus = rollout(linear_cfg,tape,**perturb,perturb_delta=-delta)
    histories = (base,plus,minus)
    _require(all(np.array_equal(np.asarray(base[k]),np.asarray(x[k])) for x in (plus,minus)
                 for k in ('action','truth','present','fresh','source_tick')), 'Perturbed replay changed action, truth or the exogenous delivery tape')
    deliveries = [[e for e in trace['event_records'] if e['source_tick'] == source_tick] for trace in histories]
    _require(all(len(d) == 1 and d[0]['arrival_tick'] == arrival for d in deliveries), 'Target acquisition was not delivered once')
    y0,yp,ym = (d[0]['value'] for d in deliveries)
    _require(math.isclose(yp-y0,delta,rel_tol=1e-9,abs_tol=1e-12) and math.isclose(y0-ym,delta,rel_tol=1e-9,abs_tol=1e-12),
             'Perturbation is not symmetric on the chosen delivered measurement')
    stop = arrival + horizon_ticks
    _require(stop <= len(base['estimate']), 'Requested fixed post-arrival kernel would fall outside the rollout')
    _require(np.array_equal(base['raw'][:arrival],plus['raw'][:arrival]) and np.array_equal(base['raw'][:arrival],minus['raw'][:arrival]),
             'Future perturbation changed a pre-arrival input')
    _require(not np.array_equal(plus['estimate'][arrival],minus['estimate'][arrival]), 'Fixed-KF arrival response is zero; invalid probe')
    summary['target_packet_is_latest_observation_at_arrival'] = bool(base['source_tick'][arrival] == source_tick)
    # Fixed filtering is also measured: otherwise the common estimator's
    # contribution could wrongly be attributed to the recurrent time scale.
    for arm in ('fixed_kf',*ARMS):
        if arm == 'fixed_kf':
            b,p,m = (np.asarray(row['estimate'],dtype=float) for row in histories)
            tau_trace = None
        else:
            (b,p,m),tau_trace = _kernel_model_estimates(models[arm],histories,tau_start=arrival,tau_stop=stop)
        _require(b.ndim == 2 and b.shape == p.shape == m.shape and b.shape[1] == X_DIM and
                 all(np.isfinite(x).all() for x in (b,p,m)), 'Invalid estimation trace during measurement replay')
        _require(np.array_equal(b[:arrival],p[:arrival]) and np.array_equal(b[:arrival],m[:arrival]),
                 'Perturbation changed an estimate before packet delivery')
        signed = (p[arrival:stop]-m[arrival:stop])/(2*delta)
        magnitude = np.abs(signed[:,0]); maximum = float(magnitude.max())
        lags = np.arange(horizon_ticks)
        ages = (lags+arrival-source_tick)*dt
        total = float(magnitude.sum())
        tail_recovery = None
        # Peak-based suffix definition avoids calling a transient zero crossing
        # a return to baseline. This is not physical tracking recovery.
        if maximum > 0:
            suffix = np.maximum.accumulate(magnitude[::-1])[::-1]
            candidates = np.flatnonzero((suffix <= .05*maximum) & (lags <= horizon_ticks-5))
            if len(candidates):
                tail_recovery = int(candidates[0])
        result = {'lag_ticks':lags.tolist(),'physical_age_seconds':ages.tolist(),
                  'd_position_estimate_d_observation':signed[:,0].tolist(),
                  'd_velocity_estimate_d_observation_per_second':signed[:,1].tolist(),
                  'arrival_position_sensitivity':float(signed[0,0]), 'arrival_velocity_sensitivity_per_second':float(signed[0,1]),
                  'peak_abs_position_sensitivity':maximum,
                  'abs_position_kernel_area_seconds':float(total*dt),
                  'sensitivity_weighted_physical_age_seconds':float(np.dot(magnitude,ages)/total) if total else None,
                  'tail_below_5pct_peak_from_lag_tick':tail_recovery,
                  'tail_not_observed_in_fixed_window':tail_recovery is None,
                  'future_actions_fixed':True,'prearrival_unchanged':True}
        if tau_trace is not None:
            result['baseline_internal_tau'] = _tau_summary(tau_trace)
            result['baseline_mean_tau_seconds_by_lag'] = tau_trace.mean(1).tolist()
        else:
            result['baseline_internal_tau'] = None
        summary['methods'][arm] = result
    return summary
