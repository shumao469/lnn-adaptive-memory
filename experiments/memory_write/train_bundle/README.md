# Double-integrator memory-write training

This bundle contains the shared observer modules and the three frozen observer checkpoints used by the paired single-write experiment. The checkpoints are fixed inputs to gate training; they are not newly trained observers. Their SHA-256 digests are checked against the included manifest.

From the repository root, install NumPy, SciPy and PyTorch, then run the smoke test:

```sh
cd experiments/memory_write
python -m pip install numpy scipy torch
python restore_data.py
python train_bundle/smoke_test.py
```

The smoke test verifies the checkpoints and runs one paired 96-scene Q-increase condition at 40 ms. It follows the original batch shape so the strict numerical identity check remains unchanged.

The experiment runners live in `double_integrator/`. After restoring the simulation tables, `python double_integrator/run.py` regenerates single-write labels, fits the gate on the training split, selects its checkpoint and threshold on validation scenes, and evaluates test scenes. Existing cell tables are reused when their protocol matches. The `sequential.py`, `quota.py` and `period_quota.py` runners extend this analysis to repeated writes and fixed event budgets; consult the protocol files before running them.

The event-budget simulation uses a known post-switch horizon. Its results describe that controlled comparison and do not establish operation under unknown event arrivals.
