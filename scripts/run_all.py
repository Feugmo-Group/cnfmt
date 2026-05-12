#!/usr/bin/env python3
"""
Master script to run all CNFMT analyses and generate plots.

Usage:
    python -m cnfmt.scripts.run_all [--mode MODE]

Modes:
    all      - Run everything (default)
    bulk     - Train on bulk thermodynamics only
    compare  - Compare Optimized vs CNN
    lj       - Run LJ phase diagram
    plots    - Generate all plots from saved models
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run CNFMT analyses')
    parser.add_argument('--mode', choices=['all', 'bulk', 'compare', 'lj', 'plots'],
                        default='all', help='What to run')
    args = parser.parse_args()
    
    print("="*70)
    print("CNFMT - Conditional Neural Fundamental Measure Theory")
    print("="*70)
    
    if args.mode in ['all', 'bulk']:
        print("\n>>> Training on bulk thermodynamics...")
        subprocess.run([sys.executable, '-m', 'cnfmt.scripts.train_bulk'])
    
    if args.mode in ['all', 'compare']:
        print("\n>>> Comparing Optimized vs CNN...")
        subprocess.run([sys.executable, '-m', 'cnfmt.scripts.train_bulk', '--compare'])
    
    if args.mode in ['all', 'lj']:
        print("\n>>> Running LJ phase diagram...")
        subprocess.run([sys.executable, '-m', 'cnfmt.lj.phase_diagram'])
    
    if args.mode in ['all', 'plots']:
        print("\n>>> Generating all plots...")
        subprocess.run([sys.executable, '-m', 'cnfmt.scripts.generate_plots'])
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
