"""Recompute the post hoc comparisons and compare every result to the original."""
from pathlib import Path
import json
import subprocess
import sys

here = Path(__file__).resolve().parent / "reacher_gate_analysis"
subprocess.run([sys.executable, str(here / "analyze.py")], check=True)
reference = json.loads((here / "reference_corrected_analysis.json").read_text(encoding="utf-8"))
result = json.loads((here / "corrected_analysis.json").read_text(encoding="utf-8"))
assert len(reference["contrasts"]) == len(result["contrasts"]) == 48
for i, (expected, actual) in enumerate(zip(reference["contrasts"], result["contrasts"])):
    if expected != actual:
        raise AssertionError(f"Contrast {i} differs from the original output")
print("Verified: all 48 contrast records match the original exactly.")
