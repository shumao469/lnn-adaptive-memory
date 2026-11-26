"""Check frozen model integrity and simulate one paired observation write."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'nc_memory_gate_20261125'))
import torch
import run

torch.set_num_threads(1)
models, hashes = run.engine.models()
assert len(models) == 3 and len(hashes) == 3
fit = 8210001
rows = run.rows_for('q_up', 2, 1.0, fit, models[fit], run.SCENES)
assert len(rows) == 96 and rows[0]['fit'] == fit
for field in ('tracking_rmse_delta', 'control_integral_delta', 'tape_sha256'):
    assert field in rows[0], field
print('PASS: three checkpoint hashes and 96 paired 40 ms Q-up single-write replays')
print('tracking keep-minus-skip =', rows[0]['tracking_rmse_delta'])
print('control keep-minus-skip =', rows[0]['control_integral_delta'])
