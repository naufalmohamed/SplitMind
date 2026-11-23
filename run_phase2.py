"""Reproduce every Phase 2 result (Phase 1 outputs are not touched).

    python run_phase2.py          # tests + 2A, 2A-X, 2D, 2C, 2B, 2E + analyses (about 7 minutes)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ["-m", "unittest", "discover", "-s", "tests", "-t", "."],
    ["experiments/p2a_volatility_noise.py"],
    ["experiments/p2a_volatility_noise.py", "--config", "configs/phase2/volatility_noise_accumulating_memory.yaml"],
    ["analysis/p2a_volatility_noise.py"],
    ["analysis/p2a_volatility_noise.py", "results/phase2/volatility_noise_accumulating_memory"],
    ["experiments/p2d_monitor_prior.py"],
    ["analysis/p2d_monitor_prior.py"],
    ["experiments/p2c_coherence_failure.py"],
    ["analysis/p2c_coherence_failure.py"],
    ["experiments/p2b_multi_source.py", "--stage", "all"],
    ["analysis/p2b_multi_source.py"],
    ["experiments/p2e_complex_environment.py"],
    ["analysis/p2e_complex_environment.py"],
]


def main():
    for step in STEPS:
        cmd = [sys.executable, *step]
        print(">", " ".join(step), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
