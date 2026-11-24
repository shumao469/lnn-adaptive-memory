"""Train four causal residual observers and run a small paired simulation.

The short run verifies the workflow. Its metrics are illustrative and should
not be used as the paper's benchmark results.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

import factorial_events
import factorial_model
import linear


def run(steps: int, evaluation_scenes: int) -> dict:
    if steps < 1 or evaluation_scenes < 1:
        raise ValueError("steps and evaluation scenes must be positive")

    torch.set_num_threads(1)
    config = linear.Config(horizon=120, change_tick=60)
    training = []
    # Disjoint seeds ensure that the sample tapes are not reused for evaluation.
    for index, regime in enumerate(linear.REGIMES):
        for replicate in range(2):
            tape = linear.generate_tape(config, regime, delay=2,
                                        seed=1000 + index * 10 + replicate)
            training.append(factorial_events.open_history(config, tape))

    options = factorial_model.FactorialConfig(
        optimizer_steps=steps, batch_size=4, sequence_length=30, burn_in=4
    )
    models, training_receipt = factorial_model.train_four(
        training, seed=12345, opts=options
    )

    results = {}
    for regime in ("q_up", "r_up"):
        tapes = [linear.generate_tape(config, regime, delay=2, seed=5000 + i)
                 for i in range(evaluation_scenes)]
        pairs = {"fixed_kf": factorial_events.feedback_batch(config, tapes, None)}
        for name, model in models.items():
            pairs[name] = factorial_events.feedback_batch(config, tapes, model)
        # This adaptive Kalman baseline estimates noise parameters online.
        pairs["adaptive_kf"] = [
            linear.rollout(config, tape, method="adaptive_kf", closed_loop=True)
            for tape in tapes
        ]
        results[regime] = {}
        for name, rows in pairs.items():
            tracking_key = "post_change_tracking_rmse" if name == "adaptive_kf" else "tracking_rmse"
            results[regime][name] = {
                "mean_tracking_rmse": float(np.mean([r[tracking_key] for r in rows])),
                "scenes": len(rows),
            }

    return {
        "scope": "small synthetic demonstration, not a published benchmark",
        "delay_ms": config.dt * 2 * 1000,
        "simulation": asdict(config),
        "training": {
            "trajectories": len(training),
            "steps_per_model": steps,
            "fit_seed": training_receipt["seed"],
            "training_array_sha256": training_receipt["training_array_sha256"],
            "shared_initial_parameters": training_receipt["same_initial_shared_parameters"],
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3, help="optimizer steps per model")
    parser.add_argument("--scenes", type=int, default=2, help="paired evaluation scenes per regime")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_results.json"))
    args = parser.parse_args()
    result = run(args.steps, args.scenes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}. Tracking RMSE is in simulator model units.")
    for regime, rows in result["results"].items():
        print(regime)
        for name, metrics in rows.items():
            print(f"  {name:30} {metrics['mean_tracking_rmse']:.6f}")


if __name__ == "__main__":
    main()
