"""Reproduce every result: unit tests, all experiments, figures and tables.

    python run_all.py            # full run (~1 minute)
    python run_all.py --seeds 5  # quick smoke run (overwrites results/)
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ["experiments/latency_sweep.py", "experiments/conflict_sweep.py",
           "experiments/metacognition_eval.py", "experiments/sensitivity_sweep.py",
           "experiments/node_failure.py"]


def run(cmd):
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=None)
    args = p.parse_args()
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."])
    extra = ["--seeds", str(args.seeds)] if args.seeds else []
    for s in SCRIPTS:
        run([sys.executable, s, *extra])
    run([sys.executable, "analysis/plots.py"])


if __name__ == "__main__":
    main()
