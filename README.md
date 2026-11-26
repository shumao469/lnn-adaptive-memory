# LNN Adaptive Memory

Research code for studying continuous-time-inspired memory and event-driven
innovation correction in closed-loop state estimation. The repository component
implements a bounded neural residual on top of a delayed Kalman filter and a
paired double-integrator simulation.

## What is included

- A double-integrator plant, asynchronous observations, process and measurement
  perturbations, and fixed/adaptive Kalman baselines.
- Four matched residual observers that cross fixed or dynamic time constant
  with fixed or adaptive innovation gain.
- A small reproducible training and paired closed-loop evaluation example.
- Tests for simulation, causal processing and the four observer variants.

This component supports the broader Liquid AI research project through a
continuous-time-inspired residual observer and an engineering demonstration.
The experiment package includes memory-write training code, simulation data,
and statistical reanalysis of paired Reacher results.

## Setup

Python 3.10 or later is recommended. Install PyTorch suitable for your system;
the example runs on CPU. The tested environment used Python 3.13, NumPy 1.26,
SciPy 1.16 and a development build of PyTorch 2.10.

```bash
git clone https://github.com/shumao469/lnn-adaptive-memory.git
cd lnn-adaptive-memory
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
python -m pip install -r requirements.txt
```

## Quick start

```bash
python demo.py --steps 3 --scenes 2
```

The command trains the four models for three optimizer steps each, then tests
each observer and two Kalman baselines on new paired Q↑ and R↑ simulation
scenarios at 40 ms observation delay. It prints the mean post-change tracking
RMSE and writes `outputs/demo_results.json`. Tracking values are in
simulation model units. Override the destination with `--output PATH`.

For a longer demonstration, change `--steps` and `--scenes`. These parameters
control the optimization steps and the number of paired evaluation scenarios.

## Verify the installation

```bash
python -m unittest -v test_linear test_memory test_factorial_model
```

The tests check properties including causal packet handling, deterministic
training, the four-arm parameterization and innovation-gated updates.

## Code guide

| File | Role |
| --- | --- |
| `linear.py` | Simulation, delayed observation handling and Kalman baselines |
| `memory.py` | Base bounded-residual memory model and shared utilities |
| `factorial_model.py` | Four fixed/adaptive time-constant and gain observers |
| `factorial_events.py` | Training histories and paired closed-loop feedback |
| `demo.py` | Minimal train-and-evaluate workflow |

## Interpreting the results

The experiments distinguish process noise Q from observation noise R and
compare matched scenarios. Interpret dynamic time constants and adaptive gains
in the context of the controller, noise conditions, evaluation protocol and
model checkpoint. Report uncertainty using independent paired scenarios.

## Memory-write and Reacher gate experiments

The [experiment package](experiments/memory_write/README.md) adds
single-write and sequential gating code, a double-integrator training bundle
with frozen observer checkpoints, and a standalone reanalysis of paired
simulated Reacher results. The mechanical-arm image is illustrative artwork.
Restore the released data and run the smoke test:

```bash
cd experiments/memory_write
python -m pip install numpy scipy torch
python restore_data.py
python train_bundle/smoke_test.py
python verify_results.py
```

The Reacher analysis reports paired results and post hoc intervals adjusted
across 48 comparisons. Its simulation protocol specifies the post-switch write
budget and event horizon. Reacher retraining uses the original simulator and
observer dependencies.

![Illustrative two-joint robotic arm](experiments/memory_write/assets/robot_arm.svg)

## License and citation

The project authors have not yet specified a license or citation metadata for
this release. Contact the repository owner before reuse beyond applicable
default copyright permissions.
