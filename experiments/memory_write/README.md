# Memory writing and closed-loop gate comparisons

This research package contains the original experiment scripts, selected simulated data and a runnable reanalysis for the adaptive-memory observer study. Simulations include a double-integrator and a MuJoCo Reacher. It is an exploratory research release, not physical robot validation.

![Illustrative two-joint robotic arm](assets/robot_arm.svg)

*Illustrative mechanical-arm artwork; it is not an experimental result.*

## Contents

- `double_integrator/`: original double-integrator single-write labelling, predictor training, online-policy and quota-control scripts, their fitted gate checkpoint, protocols and summaries; the per-cell CSV tables are archived in `double_integrator_data.zip`. Training labels compare retaining versus skipping one neural write under paired conditions.
- `reacher_gate/`: original Reacher label and gate scripts, frozen ridge-gate parameters, protocol, summary, source labels, quota results and robust comparator results. `evaluation.jsonl` is reduced to the always-write reference rows with feature sequences removed; all fields used by reanalysis, including `exogenous_sha256`, are preserved.
- `reacher_gate_analysis/`: post hoc crossed-bootstrap multiplicity analysis and both the original and regenerated outputs.
- `train_bundle/`: the double-integrator training scripts, dependencies and three verified frozen observer checkpoints; see its README for smoke testing and training.
- `PROVENANCE.json`: SHA-256 fingerprints of original files and corresponding released files, with transformations identified.

## Reproduce the Reacher multiplicity analysis

The supported, standalone calculation requires Python 3 and NumPy:

```bash
cd experiments/memory_write
python -m pip install numpy
python restore_data.py
python verify_results.py
```

The verifier reruns `analyze.py` and compares all 48 complete contrast records (point estimates, ordinary and adjusted confidence intervals and decisions) to the original saved reference. It validates the matching `exogenous_sha256` values when pairing each scenario. The output file `corrected_analysis.json` is regenerated in place. The restore_data.py command expands the archived simulation inputs to their expected relative paths. Source file hashes differ from the original run because `evaluation.jsonl` was reduced; they are traced in `PROVENANCE.json`.

The analysis combines 2 noise regimes × 3 delays × 4 comparators × 2 outcomes, with 100,000 bootstrap draws resampling 24 scenarios and 3 model fits and a Bonferroni-adjusted percentile interval for each contrast. This 48-endpoint family was defined after observing results; intervals are exploratory and approximate. Huber-Kalman trajectories are reused across the three fits for each scenario. Each gate has a 45-of-90 post-switch write budget; the known switch and remaining-event count are used to enforce this budget. Forced-stage writes include events the gate might have selected anyway.

## Training scripts and scope

The [double-integrator training bundle](train_bundle/README.md) includes the three frozen observer checkpoints and original modules required to generate single-write labels and train the gate. Its 96-scene smoke test verifies the original checkpoints and runs one paired condition. The gate was independently refitted from 5,184 stored labels and reproduced the reported tracking-direction accuracies; that check does not mean the entire simulation grid was rerun. Run the instructions from this experiment folder.

The archived Reacher training and rollout scripts preserve the original implementation for methods inspection; they require additional original sibling projects, trained observers and simulation dependencies that are not supplied here. A clean checkout can reproduce the Reacher post hoc analysis from the paired inputs, but not the full Reacher training or simulation. The simulations are exploratory and do not constitute physical robot validation.

The learned Reacher gate had lower mean control integral than the innovation-threshold gate in all six tested conditions, while the adjusted tracking-difference intervals included zero. This controlled equal-quota result does not establish deployment performance under an unknown observation horizon.



