# Double-integrator memory-write training bundle

This bundle preserves the experimental code for the paired keep-versus-skip
single-observation intervention, the fitted gate and exploratory repeated-write
policies. It includes the three frozen observer checkpoints used to generate
the labels. They are fixed inputs to the gate experiment, not newly trained
observers. The checkpoint manifest lists their original SHA-256 digests.

## Setup and smoke test

From `experiments/memory_write_20261125`, install `numpy`, `scipy` and `torch`; CPU is sufficient.
The original release was tested on Python 3.13, PyTorch 2.10 development build,
and SciPy 1.16. The following smoke test verifies the checkpoint digests and
simulates a paired Q-increase cell at 40 ms:

```sh
python train_bundle/smoke_test.py
```

The smoke test uses all 96 original scene IDs for one fitted observer. This
matches the original training-cell batch shape. On one tested CPU, a single
scene instead of the original batch produced round-off differences of up to
5.4e-8 between identically initialized branches, exceeding the original
strict 1e-10 check; the original check is intentionally unchanged here.

## Run the original experiment

```sh
python train_bundle/nc_memory_gate_20261125/run.py
```

The run generates paired labels for 54 cells (three noise strengths, two noise
sources, three delays, three frozen observer fits), fits the gate on the
training split, chooses its checkpoint and threshold on validation scenes and
evaluates one gated write on test scenes. Outputs are written under
`train_bundle/nc_memory_gate_20261125/`: `cells/`, `protocol.json`,
`gate_checkpoint.pt`, `predictive_summary.json`, `online_summary.json` and
`progress.json`. Do not mix outputs from different experiments: the runner
reuses existing cell files and checks a fixed protocol. Use a fresh checkout
or an empty output directory for a new run.

For repeated-write experiments the additional `sequential.py`, `quota.py`
and `period_quota.py` runners depend on completed gate-training outputs; their
matching `*_finalize.py` scripts perform the corresponding summaries. Review
the files and their protocol before executing these longer exploratory runs.
The serialized gate checkpoint is generated locally by the runner and loaded
with PyTorch pickle support; only load checkpoints from trusted sources.

The bundled code represents simulation studies. Single-write supervision and
repeated gating answer different questions. An event-budget experiment that
uses a known future event count is not evidence of deployable unknown-horizon
performance. Reacher code and its post hoc multiple-comparison reanalysis are
documented separately.
