#!/usr/bin/env python3
"""Generate all plots. Usage: python -m cnfmt.scripts.generate_plots"""
import subprocess, sys
from pathlib import Path
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
def main():
    print("CNFMT - Generate All Plots")
    commands = [
        ("Bulk", [sys.executable, "-m", "cnfmt.scripts.train_bulk"]),
        ("Compare", [sys.executable, "-m", "cnfmt.scripts.train_bulk", "--compare"]),
        ("LJ", [sys.executable, "-m", "cnfmt.lj.phase_diagram"]),
    ]
    for name, cmd in commands:
        print(f">>> {name}")
        subprocess.run(cmd)
    print("Done - plots in ./outputs/")
if __name__ == "__main__": main()
