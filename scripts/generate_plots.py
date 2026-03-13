#!/usr/bin/env python3
"""Generate all plots. Usage: python -m cnfmt.scripts.generate_plots"""
import subprocess, sys
from pathlib import Path
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
def main():
    print("CNFMT - Generate All Plots")
    for name, mod in [("Bulk", "cnfmt.scripts.train_bulk"), ("Compare", "cnfmt.scripts.compare_methods"), ("LJ", "cnfmt.lj.phase_diagram")]:
        print(f">>> {name}")
        subprocess.run([sys.executable, "-m", mod])
    print("Done - plots in ./outputs/")
if __name__ == "__main__": main()
