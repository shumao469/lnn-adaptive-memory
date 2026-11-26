"""Paired single-write deletion with frozen observers and independent plant feedback."""
from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
ROOT = Path(__file__).resolve().parent
import torch
GATE_MODEL = None
GATE_STATS = None
GATE_THRESHOLD = 0.0
BASE = ROOT.parent / "nc_adaptive_memory_20261116"
sys.path.insert(0, str(BASE))

import numpy as np
import torch
import factorial_events as events
import factorial_model as factorial
import linear

CFG = linear.Config()
ARM = "fixed_tau_adaptive_gain"
FITS = (8210001, 8210002, 8210003)
SCENES = tuple(range(9432001, 9432025))
MODES = ("keep", "sham", "skip")
REGIMES = ("q_up", "r_up")
DELAYS = (0, 2, 4)
CAPTURE = CFG.change_tick + 20
WINDOW = 50
START = perf_counter()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    path = ROOT / name
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def models():
    run = BASE / "runs/factorial_20261116"
    manifest = json.loads((run / "summary.json").read_bytes())
    result = {}
    hashes = {}
    for seed in FITS:
        path = run / f"seed_{seed}.pt"
        identity = sha(path)
        if identity != manifest["checkpoint_files"][path.name]:
            raise RuntimeError(f"Frozen checkpoint mismatch: {path}")
        item = torch.load(io.BytesIO(path.read_bytes()), weights_only=True, map_location="cpu")[ARM]
        state = item["state_dict"]
        norm = {k: state[k] for k in ("raw_scale", "kf_mean", "kf_scale", "residual_rms", "residual_cap")}
        model = factorial.FactorialObserver(ARM, factorial.FactorialConfig(**item["config"]), norm)
        model.load_state_dict(state, strict=True)
        result[seed] = model.eval()
        hashes[path.name] = identity
    return result, hashes


def rollout(tapes, model, delay):
    arrival = CAPTURE + delay
    end = arrival + WINDOW
    count = len(tapes)
    matrix_tapes = list(tapes) * len(MODES)
    shape = (len(MODES), count, end)
    sensors = [events.CausalSensor(CFG, t) for t in matrix_tapes]
    f, b, disturbance, reference = [
        np.stack([getattr(t, k)[:end] for t in matrix_tapes])
        for k in ("f", "b", "process_increments", "reference")
    ]
    truth = np.stack([t.truth[0] for t in matrix_tapes])
    past = np.zeros(len(matrix_tapes))
    action = np.empty(shape)
    position = np.empty(shape)
    estimated = np.empty((*shape, 2))
    kalman = np.empty((*shape[:2], 2))
    features = {}
    carry = model.initial(len(matrix_tapes))

    def drop_write_projection(module, arguments, output):
        modified = output.clone()
        if GATE_MODEL is None:
            modified[2 * count:] = 0.
        else:
            columns = ("age_s", "normalized_innovation", "fast_innovation_power",
                       "slow_innovation_power", "innovation_lag_product",
                       "prior_innovation", "prior_action")
            arr = np.column_stack([features[k] for k in columns])
            arr = (arr - GATE_STATS["xmean"]) / GATE_STATS["xstd"]
            predictions = GATE_MODEL(torch.from_numpy(arr.astype(np.float32))).numpy()
            score = predictions[:, 0] * GATE_STATS["ystd"][0] + GATE_STATS["ymean"][0]
            mask = torch.from_numpy(score > GATE_THRESHOLD)
            features["gate_skip"] = mask.numpy().astype(np.float32)
            features["predicted_tracking_delta"] = score
            modified[2 * count:][mask] = 0.
        return modified

    with torch.inference_mode():
        for tick in range(end):
            if tick:
                truth = np.einsum("bij,bj->bi", f[:, tick], truth) + b[:, tick] * past[:, None] + disturbance[:, tick]
            packet_rows = [sensor.step(tick, state, control) for sensor, state, control in zip(sensors, truth, past)]
            raw, nominal, event = [np.asarray(x, dtype=np.float32) for x in zip(*packet_rows)]
            if tick == arrival:
                if not np.all(event[:, 0] > .5):
                    raise RuntimeError("Scheduled target packet was not accepted in all paired rollouts")
                if not np.allclose(event[:, 3], delay * CFG.dt, atol=1e-8, rtol=0):
                    raise RuntimeError("Target age differs from frozen schedule")
                hist = carry[2].numpy().copy()
                features = {
                    "innovation": event[:count, 1].copy(),
                    "normalized_innovation": event[:count, 1] / np.sqrt(event[:count, 2]),
                    "innovation_variance": event[:count, 2].copy(),
                    "age_s": event[:count, 3].copy(),
                    "quality_flag_available": event[:count, 5].copy(),
                    "fast_innovation_power": hist[:count, 0],
                    "slow_innovation_power": hist[:count, 1],
                    "innovation_lag_product": hist[:count, 2],
                    "prior_innovation": hist[:count, 3],
                    "prior_action": past[:count].copy(),
                }
                kalman = nominal.reshape(len(MODES), count, 2)
                for group in (raw, nominal, event, hist):
                    grid = group.reshape(len(MODES), count, -1)
                    np.testing.assert_allclose(grid[0], grid[2], rtol=0, atol=1e-10)
                handle = model.innovation_direction.register_forward_hook(drop_write_projection)
            else:
                handle = None
            try:
                raw_tensor = torch.from_numpy(raw)
                residual, carry, diagnosis = model.step(raw_tensor, torch.from_numpy(nominal), torch.from_numpy(event), carry)
            finally:
                if handle is not None:
                    handle.remove()
            if tick == arrival:
                features["mean_neural_gain"] = diagnosis["gain"][:count].mean(dim=1).numpy()
                features["full_write_l2"] = diagnosis["innovation_update"][:count].norm(dim=1).numpy()
                if GATE_MODEL is None:
                    mask = torch.ones(count, dtype=torch.bool)
                else:
                    mask = torch.from_numpy(features["gate_skip"].astype(bool))
                if torch.any(diagnosis["innovation_update"][2 * count:][mask] != 0):
                    raise RuntimeError("The skipped neural writes were not blocked")
                if not torch.allclose(carry[2][:count], carry[2][2 * count:], rtol=0, atol=1e-10):
                    raise RuntimeError("Target skip changed the accepted-packet history")
            nominal64 = np.asarray([record[1] for record in packet_rows], dtype=np.float64)
            estimate = nominal64 + model.gate(residual, raw_tensor, tick).numpy()
            error = reference[:, tick] - estimate
            past = np.clip(CFG.kp * error[:, 0] + CFG.kd * error[:, 1], -CFG.max_abs_action, CFG.max_abs_action)
            action[:, :, tick] = past.reshape(len(MODES), count)
            position[:, :, tick] = truth[:, 0].reshape(len(MODES), count)
            estimated[:, :, tick] = estimate.reshape(len(MODES), count, 2)

    for variable in (action, position, estimated):
        np.testing.assert_allclose(variable[0], variable[1], atol=1e-10, rtol=0)
        np.testing.assert_allclose(variable[0, :, :arrival], variable[2, :, :arrival], atol=1e-10, rtol=0)
    tracks = position[:, :, arrival:] - reference[:, arrival:, 0].reshape(len(MODES), count, -1)
    rmse = np.sqrt(np.mean(tracks**2, axis=2))
    cost = CFG.dt * np.sum(action[:, :, arrival:]**2, axis=2)
    return features, rmse, cost


def main():
    torch.set_num_threads(1)
    bank, hashes = models()
    protocol = {
        "scope": "Exploratory post-development simulation on new paired scenarios",
        "arm": ARM, "fit_seeds": FITS, "scenario_seeds": SCENES, "modes": MODES,
        "regimes": REGIMES, "delay_ms": [int(round(d * CFG.dt * 1000)) for d in DELAYS],
        "post_change_capture_ticks": CAPTURE - CFG.change_tick, "capture_tick": CAPTURE,
        "definition": "Keep or zero a single accepted packet's neural innovation projection on the original arrival tick; delayed Kalman and accepted-event history remain intact. Each observer drives its own trajectory with the same controller, initial condition and exogenous tape.",
        "negative_control": "Identity sham in the same batch, verified over all trajectory states and actions",
        "window_s": WINDOW * CFG.dt, "contrast": "keep minus skip; negative favors keeping the neural write",
        "sources": {name: sha(BASE / name) for name in ("factorial_model.py", "factorial_events.py", "linear.py")},
        "checkpoint_sha256": hashes, "script_sha256": sha(Path(__file__)),
        "uncertainty": "Crossed descriptive 95% bootstrap, 10,000 draws for all scenario/fit combinations, unadjusted",
    }
    write("protocol.json", protocol)
    rows = []
    summary = []
    completed = 0
    for regime, delay in itertools.product(REGIMES, DELAYS):
        tapes, identities = events.novel_tape_bank(CFG, regime, delay, SCENES)
        cell = []
        for fit, model in bank.items():
            features, rmse, cost = rollout(tapes, model, delay)
            for i, (scene, identity) in enumerate(zip(SCENES, identities)):
                row = {
                    "regime": regime, "delay_ms": int(round(delay * CFG.dt * 1000)),
                    "scenario_seed": scene, "fit_seed": fit, "tape_sha256": identity,
                    "capture_tick": CAPTURE, "arrival_tick": CAPTURE + delay,
                    "tracking_rmse_keep": float(rmse[0, i]), "tracking_rmse_skip": float(rmse[2, i]),
                    "tracking_rmse_delta": float(rmse[0, i] - rmse[2, i]),
                    "control_integral_keep": float(cost[0, i]), "control_integral_skip": float(cost[2, i]),
                    "control_integral_delta": float(cost[0, i] - cost[2, i]),
                    "control_integral_delta_pct": float(100 * (cost[0, i] - cost[2, i]) / cost[2, i]),
                }
                row.update({k: float(v[i]) for k, v in features.items()})
                cell.append(row)
            completed += len(SCENES) * len(MODES)
            print(f"{regime} {delay} ticks fit {fit}: {completed} rollouts", flush=True)
        rows.extend(cell)
        rng = np.random.default_rng(9432030)
        draw_scenes = rng.integers(0, len(SCENES), size=(10000, 1, len(SCENES)))
        draw_fits = rng.integers(0, len(FITS), size=(10000, len(FITS), 1))
        entry = {"regime": regime, "delay_ms": cell[0]["delay_ms"], "scenes": len(SCENES), "fits": len(FITS)}
        for key in ("tracking_rmse_delta", "control_integral_delta", "control_integral_delta_pct"):
            matrix = np.asarray([r[key] for r in cell]).reshape(len(FITS), len(SCENES))
            draw = matrix[draw_fits, draw_scenes].mean(axis=(1, 2))
            entry[key] = {
                "mean": float(matrix.mean()), "ci_95": np.quantile(draw, [.025, .975]).tolist(),
                "fraction_scene_means_favor_keep": float(np.mean(matrix.mean(axis=0) < 0)),
            }
        summary.append(entry)
    with (ROOT / "paired_write_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write("summary.json", {"status":"completed", "rollouts": completed, "paired_rows": len(rows), "sham": "all traces identical to keep within 1e-10", "elapsed_s": perf_counter() - START, "cells": summary})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
