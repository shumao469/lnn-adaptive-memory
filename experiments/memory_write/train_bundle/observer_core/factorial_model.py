"""Causal bounded-residual 2x2 factorial of memory rate and innovation gain.

The common delayed KF is not modified. Its capture-aligned innovation is a
feature for a *separate learned residual*, not a second KF correction. Between
packets, hidden memory evolves from pre-event context; only actually accepted
packets can write. A timestamp-dependent attenuation acts on the residual's
innovation feature, not as an exact out-of-sequence KF transport equation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np
import torch
from torch import nn

from memory import MemoryConfig, FEATURES, RAW_DIM, X_DIM, _train_arrays, _train_norm, _train_digest


ARMS = (
    "fixed_tau_fixed_gain",
    "adaptive_tau_fixed_gain",
    "fixed_tau_adaptive_gain",
    "adaptive_tau_adaptive_gain",
)
EVENT_FEATURES = ("accepted", "innovation", "innovation_var", "age_seconds", "quality", "quality_present")
EVENT_DIM = len(EVENT_FEATURES)
_RATE_ARMS = (ARMS[1], ARMS[3])
_GAIN_ARMS = (ARMS[2], ARMS[3])


@dataclass(frozen=True)
class FactorialConfig(MemoryConfig):
    gain_min: float = .015
    gain_initial: float = .16
    gain_max: float = .60
    innovation_clip: float = 8.0
    transport_seconds: float = .20
    fast_history_seconds: float = .10
    slow_history_seconds: float = .60

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.gain_min < self.gain_initial < self.gain_max < 1:
            raise ValueError("Gain limits and initial value must be ordered within (0,1)")
        if not all(math.isfinite(x) and x > 0 for x in (
                self.innovation_clip, self.transport_seconds,
                self.fast_history_seconds, self.slow_history_seconds,
        )):
            raise ValueError("Invalid time scales or innovation clip")
        if not self.fast_history_seconds < self.slow_history_seconds:
            raise ValueError("Fast and slow summaries must be distinct and ordered")


def _validate_events(ev, raw):
    if ev.shape != (*raw.shape[:-1], EVENT_DIM) or not np.isfinite(ev).all():
        raise ValueError("Malformed accepted-packet feature grid")
    accepted = ev[..., 0] == 1.
    if (not np.isin(ev[..., [0, 5]], (0., 1.)).all() or
            np.any(ev[~accepted] != 0.) or np.any(ev[..., 2][accepted] <= 0.) or
            np.any(ev[..., 3] < 0.) or np.any((ev[..., 5] == 0.) & (ev[..., 4] != 0.)) or
            np.any(accepted & (raw[..., 1] != 1.))):
        raise ValueError("Noncausal/missing packet event or nonpositive innovation variance")


def _options(opts):
    if isinstance(opts, FactorialConfig):
        return opts
    if isinstance(opts, MemoryConfig):
        return FactorialConfig(**asdict(opts))
    return FactorialConfig(**({} if opts is None else opts))


def _logit_between(value, low, high):
    p = (value - low) / (high - low)
    return math.log(p / (1. - p))


class FactorialObserver(nn.Module):
    """The only ablated components are multiplicative context-to-rate/gain heads."""

    def __init__(self, arm, cfg, normalization):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Invalid factorial arm")
        self.arm, self.cfg, self.width = arm, cfg, cfg.hidden_size
        for key, value in normalization.items():
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float32).clone())
        h = self.width
        self.rate = nn.Parameter(torch.full((h,), _logit_between(
            cfg.tau_initial_seconds, cfg.tau_min_seconds, cfg.tau_max_seconds)))
        self.gain = nn.Parameter(torch.full((h,), _logit_between(
            cfg.gain_initial, cfg.gain_min, cfg.gain_max)))
        # Same process-drive and event readout sizes for all four arms.
        self.drive = nn.Sequential(nn.Linear(h + 2 + 1 + 5, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.innovation_direction = nn.Linear(1, h, bias=False)
        self.readout = nn.Linear(h, X_DIM)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        # Low-rank heads retain active capacity matching without dummy weights.
        # Rates/gains have learned per-unit bases, plus an optional scalar context shift.
        self.rate_context_head = nn.Linear(5, 1, bias=False) if arm in _RATE_ARMS else None
        self.gain_context_head = nn.Linear(9, 1, bias=False) if arm in _GAIN_ARMS else None
        for head in (self.rate_context_head, self.gain_context_head):
            if head is not None:
                nn.init.zeros_(head.weight)

    def initial(self, batch):
        h = self.raw_scale.new_zeros(batch, self.width)
        previous_kf = self.raw_scale.new_zeros(batch, X_DIM)
        # fast/slow squared innovation, lag product, previous innovation, elapsed since event
        hist = self.raw_scale.new_zeros(batch, 5)
        hist[:, :2] = 1.
        flags = self.raw_scale.new_zeros(batch, 2)  # initialised; ever accepted event
        return h, previous_kf, hist, flags

    def step(self, raw, kf, event, carry):
        """Perform free evolution and then at most one accepted event jump.

        All arguments are batched float tensors: raw[B,8], kf[B,2], event[B,6].
        The propagation branch reads the *previous* posterior and the action
        applied to reach this tick; it never reads the current packet or KF
        posterior conditioned on that packet.
        """
        h, previous_kf, hist, flags = carry
        cfg = self.cfg
        elapsed = raw[:, 4:5] * flags[:, 0:1]
        fast, slow, corr, last, since = hist.split(1, dim=-1)
        gap = since + elapsed
        history_context = torch.cat((
            torch.tanh(torch.log(fast.clamp_min(.01))),
            torch.tanh(torch.log(slow.clamp_min(.01))),
            torch.tanh(corr / slow.clamp_min(.10)),
            torch.tanh(gap / cfg.transport_seconds),
            torch.tanh(previous_kf.norm(dim=-1, keepdim=True) / math.sqrt(2)),
        ), dim=-1)
        rate = self.rate.unsqueeze(0).expand_as(h)
        if self.rate_context_head is not None:
            rate = rate + self.rate_context_head(history_context)
        tau = cfg.tau_min_seconds + (cfg.tau_max_seconds - cfg.tau_min_seconds) * torch.sigmoid(rate)
        retention = torch.exp(-elapsed / tau)
        known_action = raw[:, 5:6] / self.raw_scale[5]
        drive = self.drive(torch.cat((h, previous_kf, known_action, history_context), dim=-1))
        h_minus = retention * h + (1. - retention) * drive

        # Accepted is a pulse, not presence/age of the last held observation.
        accepted = event[:, 0:1] > .5
        variance = torch.where(accepted, event[:, 2:3], torch.ones_like(elapsed)).clamp_min(1.e-9)
        z = (event[:, 1:2] / variance.sqrt()).clamp(-cfg.innovation_clip, cfg.innovation_clip)
        age = event[:, 3:4]
        transported = z * torch.exp(-age / cfg.transport_seconds)
        quality = event[:, 4:5] * event[:, 5:6]
        event_context = torch.cat((
            history_context, torch.tanh(z.abs()), torch.tanh(age / cfg.transport_seconds),
            torch.tanh(quality), event[:, 5:6],
        ), dim=-1)
        gain_logit = self.gain.unsqueeze(0).expand_as(h)
        if self.gain_context_head is not None:
            gain_logit = gain_logit + self.gain_context_head(event_context)
        gain = cfg.gain_min + (cfg.gain_max - cfg.gain_min) * torch.sigmoid(gain_logit)
        write_value = torch.tanh(self.innovation_direction(transported))
        # Additive innovation write: zero innovation means exactly zero write.
        # Making the correction a convex reset would also delete past h and
        # conflate gain adaptation with extra memory forgetting.
        jump = torch.where(accepted, gain * write_value, torch.zeros_like(h))
        h_plus = h_minus + jump

        # History updates after applying the event; today's observation may
        # affect future propagation, never the propagation into its own arrival.
        a_fast = torch.exp(-gap / cfg.fast_history_seconds)
        a_slow = torch.exp(-gap / cfg.slow_history_seconds)
        observed_corr = last * z * flags[:, 1:2]
        hist = torch.cat((
            torch.where(accepted, a_fast * fast + (1. - a_fast) * z.square(), fast),
            torch.where(accepted, a_slow * slow + (1. - a_slow) * z.square(), slow),
            torch.where(accepted, a_slow * corr + (1. - a_slow) * observed_corr, corr),
            torch.where(accepted, z, last),
            torch.where(accepted, torch.zeros_like(gap), gap),
        ), dim=-1)
        flags = torch.cat((torch.ones_like(elapsed), torch.maximum(flags[:, 1:2], accepted.to(h.dtype))), dim=-1)
        new_carry = (h_plus, (kf - self.kf_mean) / self.kf_scale, hist, flags)
        delta = torch.tanh(self.readout(h_plus)) * self.residual_cap
        return delta, new_carry, {
            "tau": tau, "gain": gain, "transported_innovation": transported, "innovation_update": jump,
        }

    def gate(self, residual, raw, tick):
        tick = torch.as_tensor(tick, device=raw.device).reshape(-1, 1)
        present = (raw[:, 1:2] > .5) & (raw[:, 3:4] <= self.cfg.max_observation_age_seconds)
        return residual * (present & (tick >= self.cfg.startup_ticks))

    def forward(self, raw, kf, event, *, start_tick=None):
        carry = self.initial(raw.shape[0])
        offset = torch.zeros(raw.shape[0], device=raw.device, dtype=torch.int64) if start_tick is None else start_tick
        parts = []
        for i in range(raw.shape[1]):
            correction, carry, _ = self.step(raw[:, i], kf[:, i], event[:, i], carry)
            parts.append(self.gate(correction, raw[:, i], offset + i))
        return torch.stack(parts, dim=1)


def train_four(rows, seed, opts=None, callback=None):
    """Fit paired models using identical initial shared tensors and sample windows.

    Callback, if supplied, receives (step,total), one step per arm per callback.
    All arrays must come from training-only rollouts, not evaluator outcomes.
    """
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Invalid optimizer seed")
    cfg = _options(opts)
    torch.set_num_threads(1)
    raw, kf, truth = _train_arrays(rows, cfg)
    events = np.ascontiguousarray(np.stack([np.asarray(r["event"], dtype=np.float32) for r in rows]))
    _validate_events(events, raw)
    norm = _train_norm(raw, kf, truth, cfg)
    with torch.random.fork_rng(devices=[]):
        models = {}
        for arm in ARMS:
            torch.random.default_generator.manual_seed(seed)
            models[arm] = FactorialObserver(arm, cfg, norm).to("cpu")
    common = models[ARMS[0]].state_dict()
    if not all(all(torch.equal(v, models[a].state_dict()[k]) for k, v in common.items()) for a in ARMS[1:]):
        raise RuntimeError("The four arms did not start from exactly matched common weights")
    counts = {arm: sum(p.numel() for p in model.parameters() if p.requires_grad) for arm, model in models.items()}
    if max(counts.values()) / min(counts.values()) > 1.05:
        raise RuntimeError("Four-arm active capacity gap exceeds 5%")
    optim = {a:torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
             for a, model in models.items()}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x46414354]))
    digest, losses, first = _Digest(), {}, None
    step_indices = np.arange(cfg.sequence_length)
    started = time.perf_counter()
    for step in range(1, cfg.optimizer_steps + 1):
        traj = rng.integers(len(raw), size=cfg.batch_size, dtype=np.int64)
        offsets = rng.integers(raw.shape[1] - cfg.sequence_length + 1, size=cfg.batch_size, dtype=np.int64)
        ind = (traj[:, None], offsets[:, None] + step_indices)
        x,k,y,ev = (torch.from_numpy(t[ind]) for t in (raw,kf,truth,events))
        t0 = torch.from_numpy(offsets)
        target = y - k
        digest.update(traj, offsets)
        for arm, model in models.items():
            residual = model(x,k,ev,start_tick=t0)
            error = (residual[:, cfg.burn_in:] - target[:, cfg.burn_in:]) / model.residual_rms
            loss = error.square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite factorial loss: {arm}")
            opt = optim[arm]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            losses[arm] = float(loss.detach())
        if step == 1:
            first = time.perf_counter() - started
        if callback is not None:
            callback(step, cfg.optimizer_steps)
    for model in models.values():
        model.eval()
    return models, {
        "schema_version": 1, "scope": "development_only", "seed": seed, "config": asdict(cfg),
        "arm_names": list(ARMS), "raw_features": list(FEATURES), "event_features": list(EVENT_FEATURES),
        "trajectories": len(raw), "ticks_per_trajectory": raw.shape[1], "optimizer_steps_each_arm": cfg.optimizer_steps,
        "optimizer_updates_total": 4 * cfg.optimizer_steps, "optimizer_batch_index_sha256": digest.hexdigest(),
        "training_array_sha256": _train_digest((raw,kf,truth,events)), "active_parameter_count": counts,
        "same_initial_shared_parameters": True, "same_causal_inputs_and_examples": True,
        "first_pair_step_seconds": first, "elapsed_seconds": time.perf_counter() - started,
        "no_validation_selection": True, "selected_checkpoint": "final_fixed_step",
        "device": "cpu", "cpu_threads": 1, "final_batch_normalized_mse": losses,
        "training_only_normalization": {key: val.tolist() for key,val in norm.items()},
    }


class _Digest:
    def __init__(self):
        import hashlib
        self.h = hashlib.sha256()

    def update(self, *values):
        for v in values:
            self.h.update(v.tobytes())

    def hexdigest(self):
        return self.h.hexdigest()


def new_residual(model):
    """Return one fresh causal (raw, KF state, accepted event) -> residual."""
    model.eval()
    carry = model.initial(1)
    tick = 0

    def predict(raw, kf, event):
        nonlocal carry, tick
        x,k,ev = (np.asarray(t, dtype=np.float32) for t in (raw,kf,event))
        if x.shape != (RAW_DIM,) or k.shape != (X_DIM,) or not np.isfinite(x).all() or not np.isfinite(k).all():
            raise ValueError("Invalid rollout feature shape/value")
        _validate_events(ev, x)
        with torch.inference_mode():
            tx,tk,te = (torch.from_numpy(np.array(t, copy=True))[None] for t in (x,k,ev))
            delta, carry, _ = model.step(tx, tk, te, carry)
            output = model.gate(delta, tx, tick)[0].numpy().astype(np.float64)
            tick += 1
            return output
    return predict


def predict_many(model, rows):
    """Return fixed-action state estimates [trajectory,tick,position/velocity]."""
    if not rows:
        raise ValueError("No causal rollout rows")
    raw,kf,event = (np.stack([np.asarray(r[key], dtype=np.float32) for r in rows])
                    for key in ("raw","kf","event"))
    if raw.ndim != 3 or raw.shape[-1] != RAW_DIM or kf.shape != (*raw.shape[:2], X_DIM):
        raise ValueError("Invalid evaluation array shape")
    if not np.isfinite(raw).all() or not np.isfinite(kf).all():
        raise ValueError("Invalid nonfinite evaluation input")
    _validate_events(event,raw)
    model.eval()
    with torch.inference_mode():
        residual = model(*(torch.from_numpy(t) for t in (raw,kf,event))).numpy()
    return kf + residual


def checkpoint(model):
    """Serializable only; the caller determines provenance and destination."""
    return {"arm": model.arm, "config": asdict(model.cfg), "state_dict": model.state_dict(),
            "architecture": "linear_causal_adaptive_memory_gain_factorial_v1"}
