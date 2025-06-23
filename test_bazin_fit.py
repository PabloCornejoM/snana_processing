#!/usr/bin/env python3
"""
Test script to fit Bazin function to actual supernova data
"""

import sys
sys.path.append('src')

from fit_bazin import fit_lightcurve_from_file, fit_bazin, plot_fit_results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_with_actual_data():
    """Test Bazin fitting with actual supernova data"""
    
    # Path to your data file
    data_file = "notebooks/symbolic_regression_data/SNIa-SALT2/sample_0_band_i _sr.csv"
    
    print("Testing Bazin fit with actual supernova data...")
    print(f"Loading data from: {data_file}")
    
    # Load and examine the data
    df = pd.read_csv(data_file)
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Time range: {df['time'].min():.1f} to {df['time'].max():.1f} days")
    print(f"Flux range: {df['flux'].min():.6f} to {df['flux'].max():.6f}")
    print(f"Normalized: {df['normalized'].iloc[0]}")
    
    if df['normalized'].iloc[0]:
        print(f"Original flux range: {df['flux_min'].iloc[0]:.6f} to {df['flux_min'].iloc[0] + df['flux_range'].iloc[0]:.6f}")
    
    # Extract data
    time = df['time'].values
    flux = df['flux'].values
    flux_err = df['flux_err'].values
    
    print(f"\nNumber of data points: {len(time)}")
    print(f"Peak flux: {np.max(flux):.6f} at time {time[np.argmax(flux)]:.1f} days")
    
    # Fit Bazin function with different methods
    methods = ['lm', 'differential_evolution']
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Fitting with method: {method}")
        print('='*50)
        
        # Fit without baseline
        print("\n--- Without baseline ---")
        try:
            results_no_baseline = fit_bazin(time, flux, flux_err=flux_err, 
                                           with_baseline=False, method=method)
            
            if results_no_baseline['success']:
                print("Fit successful!")
                print("Parameters:")
                for name, value in results_no_baseline['parameters'].items():
                    if results_no_baseline['parameter_errors'] is not None:
                        error = results_no_baseline['parameter_errors'][name]
                        print(f"  {name} = {value:.6f} ± {error:.6f}")
                    else:
                        print(f"  {name} = {value:.6f}")
                
                print(f"χ²/dof = {results_no_baseline['chi2_reduced']:.3f}")
                print(f"R² = {results_no_baseline['r_squared']:.6f}")
                
                # Plot results
                plot_fit_results(time, flux, results_no_baseline, flux_err=flux_err, 
                               with_baseline=False, 
                               title=f"Bazin Fit - No Baseline ({method})",
                               save_path=f"bazin_fit_no_baseline_{method}.png")
            else:
                print("Fit failed!")
        except Exception as e:
            print(f"Fit failed with exception: {e}")
        
        # Fit with baseline
        print("\n--- With baseline ---")
        try:
            results_with_baseline = fit_bazin(time, flux, flux_err=flux_err, 
                                             with_baseline=True, method=method)
            
            if results_with_baseline['success']:
                print("Fit successful!")
                print("Parameters:")
                for name, value in results_with_baseline['parameters'].items():
                    if results_with_baseline['parameter_errors'] is not None:
                        error = results_with_baseline['parameter_errors'][name]
                        print(f"  {name} = {value:.6f} ± {error:.6f}")
                    else:
                        print(f"  {name} = {value:.6f}")
                
                print(f"χ²/dof = {results_with_baseline['chi2_reduced']:.3f}")
                print(f"R² = {results_with_baseline['r_squared']:.6f}")
                
                # Plot results
                plot_fit_results(time, flux, results_with_baseline, flux_err=flux_err, 
                               with_baseline=True, 
                               title=f"Bazin Fit - With Baseline ({method})",
                               save_path=f"bazin_fit_with_baseline_{method}.png")
            else:
                print("Fit failed!")
        except Exception as e:
            print(f"Fit failed with exception: {e}")

def denormalize_parameters(results, flux_min, flux_range):
    """
    Convert normalized Bazin parameters back to original flux scale
    
    Parameters:
    -----------
    results : dict
        Fitting results from normalized data
    flux_min : float
        Minimum flux value from original data
    flux_range : float
        Range of original flux data
    
    Returns:
    --------
    dict
        Parameters in original flux scale
    """
    if not results['success']:
        return results
    
    # Copy results
    denorm_results = results.copy()
    denorm_params = results['parameters'].copy()
    
    # Denormalize amplitude
    denorm_params['A'] = denorm_params['A'] * flux_range
    
    # Denormalize baseline if present
    if 'baseline' in denorm_params:
        denorm_params['baseline'] = denorm_params['baseline'] * flux_range + flux_min
    else:
        # Add the minimum flux to account for the shift
        denorm_params['A'] = denorm_params['A'] + flux_min
    
    # Time parameters (t0, tau_rise, tau_fall) don't need denormalization
    # as time was only shifted, not scaled
    
    denorm_results['parameters'] = denorm_params
    
    # Denormalize parameter errors if available
    if results['parameter_errors'] is not None:
        denorm_errors = results['parameter_errors'].copy()
        denorm_errors['A'] = denorm_errors['A'] * flux_range
        if 'baseline' in denorm_errors:
            denorm_errors['baseline'] = denorm_errors['baseline'] * flux_range
        denorm_results['parameter_errors'] = denorm_errors
    
    return denorm_results

def test_denormalization():
    """Test parameter denormalization"""
    
    data_file = "notebooks/symbolic_regression_data/SNIa-SALT2/sample_0_band_i _sr.csv"
    df = pd.read_csv(data_file)
    
    # Get normalization parameters
    flux_min = df['flux_min'].iloc[0]
    flux_range = df['flux_range'].iloc[0]
    
    print(f"\nTesting parameter denormalization...")
    print(f"Original flux_min: {flux_min:.6f}")
    print(f"Original flux_range: {flux_range:.6f}")
    
    # Fit normalized data
    time = df['time'].values
    flux = df['flux'].values
    flux_err = df['flux_err'].values
    
    results = fit_bazin(time, flux, flux_err=flux_err, with_baseline=True, method='leastsq')
    
    if results['success']:
        print(f"\nNormalized parameters:")
        for name, value in results['parameters'].items():
            print(f"  {name} = {value:.6f}")
        
        # Denormalize
        denorm_results = denormalize_parameters(results, flux_min, flux_range)
        
        print(f"\nDenormalized parameters (original flux scale):")
        for name, value in denorm_results['parameters'].items():
            print(f"  {name} = {value:.6f}")
        
        # The denormalized amplitude should be much larger
        print(f"\nAmplitude scaling factor: {denorm_results['parameters']['A'] / results['parameters']['A']:.1f}")

if __name__ == "__main__":
    # Test with actual data
    test_with_actual_data()
    
    # Test denormalization
    test_denormalization() 