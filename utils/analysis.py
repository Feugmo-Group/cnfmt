"""
Analysis Utilities
==================

Functions for evaluating and comparing functionals.
"""

import numpy as np
from typing import Dict, List, Optional
from core.thermodynamics import BulkThermodynamics


def evaluate_network(network, eta_values: Optional[np.ndarray] = None,
                    verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Evaluate trained network performance.
    
    Parameters
    ----------
    network : ConditionalNetwork
        Trained network
    eta_values : array, optional
        Packing fractions to evaluate
    verbose : bool
        Print results
    
    Returns
    -------
    results : dict
        Evaluation metrics
    """
    if eta_values is None:
        eta_values = np.linspace(0.1, 0.48, 10)
    
    results = {
        'eta': eta_values,
        'A': [], 'B': [], 'C': [],
        'Z_err_pct': [], 'mu_err_pct': [],
        'Z_learned': [], 'Z_CS': [],
        'mu_learned': [], 'mu_CS': []
    }
    
    if verbose:
        print(f"\n{'η':>6} {'A':>8} {'B':>8} {'C':>8} {'Z_err%':>8} {'μ_err%':>8}")
        print("-"*55)
    
    for eta in eta_values:
        A, B = network.from_eta(eta)
        A_val, B_val = float(A), float(B)
        C_val = 8*A_val + 2*B_val - 9
        
        Z_l = float(BulkThermodynamics.Z_lutsko(eta, A, B))
        Z_cs = float(BulkThermodynamics.Z_CS(eta))
        mu_l = float(BulkThermodynamics.mu_ex_bulk_lutsko(eta, A, B))
        mu_cs = float(BulkThermodynamics.mu_ex_CS(eta))
        
        Z_err = abs(Z_l - Z_cs) / Z_cs * 100
        mu_err = abs(mu_l - mu_cs) / (abs(mu_cs) + 0.1) * 100
        
        results['A'].append(A_val)
        results['B'].append(B_val)
        results['C'].append(C_val)
        results['Z_err_pct'].append(Z_err)
        results['mu_err_pct'].append(mu_err)
        results['Z_learned'].append(Z_l)
        results['Z_CS'].append(Z_cs)
        results['mu_learned'].append(mu_l)
        results['mu_CS'].append(mu_cs)
        
        if verbose:
            print(f"{eta:6.3f} {A_val:8.4f} {B_val:8.4f} {C_val:8.4f} "
                  f"{Z_err:8.2f} {mu_err:8.2f}")
    
    # Convert to arrays
    for key in results:
        if key != 'eta':
            results[key] = np.array(results[key])
    
    if verbose:
        print("-"*55)
        mean_Z_err = np.mean(results['Z_err_pct'])
        mean_mu_err = np.mean(results['mu_err_pct'])
        print(f"{'Mean':>6} {'':>8} {'':>8} {'':>8} {mean_Z_err:8.2f} {mean_mu_err:8.2f}")
    
    return results


def compare_functionals(eta_values: Optional[np.ndarray] = None,
                       verbose: bool = True) -> Dict[str, Dict]:
    """
    Compare different Lutsko parameter choices.
    
    Returns comparison of Rosenfeld, Lutsko, Optimal, and White Bear.
    """
    if eta_values is None:
        eta_values = np.linspace(0.1, 0.48, 10)
    
    functionals = {
        'Rosenfeld': (1.5, 0.0),
        'Lutsko': (1.0, 0.0),
        'Optimal': (1.3, -1.0),
        'White Bear': (1.125, -1.125),
    }
    
    results = {}
    
    for name, (A, B) in functionals.items():
        C = 8*A + 2*B - 9
        
        Z_err = []
        mu_err = []
        
        for eta in eta_values:
            Z_l = float(BulkThermodynamics.Z_lutsko(eta, A, B))
            Z_cs = float(BulkThermodynamics.Z_CS(eta))
            mu_l = float(BulkThermodynamics.mu_ex_bulk_lutsko(eta, A, B))
            mu_cs = float(BulkThermodynamics.mu_ex_CS(eta))
            
            Z_err.append(abs(Z_l - Z_cs) / Z_cs * 100)
            mu_err.append(abs(mu_l - mu_cs) / (abs(mu_cs) + 0.1) * 100)
        
        results[name] = {
            'A': A, 'B': B, 'C': C,
            'Z_err_mean': np.mean(Z_err),
            'mu_err_mean': np.mean(mu_err),
            'Z_err_max': np.max(Z_err),
            'mu_err_max': np.max(mu_err),
        }
    
    if verbose:
        print("\nFunctional Comparison:")
        print(f"{'Name':>12} {'A':>6} {'B':>7} {'C':>6} {'Z_err%':>8} {'μ_err%':>8}")
        print("-"*55)
        for name, r in results.items():
            print(f"{name:>12} {r['A']:6.3f} {r['B']:7.3f} {r['C']:6.2f} "
                  f"{r['Z_err_mean']:8.2f} {r['mu_err_mean']:8.2f}")
    
    return results


def compute_errors(network, eta_values: np.ndarray) -> Dict[str, float]:
    """
    Compute summary error statistics.
    """
    results = evaluate_network(network, eta_values, verbose=False)
    
    return {
        'Z_err_mean': float(np.mean(results['Z_err_pct'])),
        'Z_err_max': float(np.max(results['Z_err_pct'])),
        'mu_err_mean': float(np.mean(results['mu_err_pct'])),
        'mu_err_max': float(np.max(results['mu_err_pct'])),
        'A_mean': float(np.mean(results['A'])),
        'B_mean': float(np.mean(results['B'])),
        'C_mean': float(np.mean(results['C'])),
    }
