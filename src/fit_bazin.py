import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
import warnings
from typing import Tuple, Optional, Dict, Any

def bazin(t, A, t0, tau_rise, tau_fall):
    """
    Bazin function for supernova light curves.
    
    Parameters:
    -----------
    t : array-like
        Time values
    A : float
        Amplitude parameter
    t0 : float
        Time of maximum (peak time)
    tau_rise : float
        Rise time constant
    tau_fall : float
        Fall time constant
    
    Returns:
    --------
    array-like
        Bazin function values
    """
    return A * np.exp(-t / tau_fall) / (1 + np.exp(-(t - t0) / tau_rise))

def bazin_with_baseline(t, A, t0, tau_rise, tau_fall, baseline):
    """
    Bazin function with a constant baseline offset.
    
    Parameters:
    -----------
    t : array-like
        Time values
    A : float
        Amplitude parameter
    t0 : float
        Time of maximum (peak time)
    tau_rise : float
        Rise time constant
    tau_fall : float
        Fall time constant
    baseline : float
        Constant baseline offset
    
    Returns:
    --------
    array-like
        Bazin function values with baseline
    """
    return bazin(t, A, t0, tau_rise, tau_fall) + baseline

def estimate_initial_parameters(t, flux, with_baseline=False):
    """
    Estimate initial parameters for Bazin function fitting.
    
    Parameters:
    -----------
    t : array-like
        Time values
    flux : array-like
        Flux values
    with_baseline : bool
        Whether to include baseline parameter
    
    Returns:
    --------
    tuple
        Initial parameter estimates
    """
    # Find peak
    max_idx = np.argmax(flux)
    t0_init = t[max_idx]
    A_init = np.max(flux)
    
    # Estimate baseline
    baseline_init = 0.0
    if with_baseline:
        # Use median of first and last few points as baseline estimate
        n_edge = max(1, len(flux) // 10)
        baseline_init = np.median(np.concatenate([flux[:n_edge], flux[-n_edge:]]))
        A_init = A_init - baseline_init
    
    # Estimate time constants
    # Rise time: time from 10% to 90% of peak before maximum
    pre_peak = flux[:max_idx] if max_idx > 0 else flux
    if len(pre_peak) > 1:
        peak_val = flux[max_idx] - baseline_init
        t_10 = np.interp(0.1 * peak_val + baseline_init, pre_peak, t[:len(pre_peak)])
        t_90 = np.interp(0.9 * peak_val + baseline_init, pre_peak, t[:len(pre_peak)])
        tau_rise_init = (t_90 - t_10) / np.log(9)  # ln(0.9/0.1) ≈ 2.2
    else:
        tau_rise_init = (t[-1] - t[0]) / 10
    
    # Fall time: time from peak to 1/e of peak after maximum
    post_peak = flux[max_idx:] if max_idx < len(flux) - 1 else flux
    if len(post_peak) > 1:
        peak_val = flux[max_idx] - baseline_init
        try:
            t_1e = np.interp(peak_val / np.e + baseline_init, 
                           post_peak[::-1], t[max_idx:][::-1])
            tau_fall_init = t_1e - t0_init
        except:
            tau_fall_init = (t[-1] - t0_init) / 2
    else:
        tau_fall_init = (t[-1] - t0_init) / 2
    
    # Ensure positive time constants
    tau_rise_init = max(tau_rise_init, 1.0)
    tau_fall_init = max(tau_fall_init, 1.0)
    
    if with_baseline:
        return (A_init, t0_init, tau_rise_init, tau_fall_init, baseline_init)
    else:
        return (A_init, t0_init, tau_rise_init, tau_fall_init)

def fit_bazin(t, flux, flux_err=None, with_baseline=False, method='lm', 
              bounds=None, maxfev=5000):
    """
    Fit Bazin function to supernova light curve data.
    
    Parameters:
    -----------
    t : array-like
        Time values
    flux : array-like
        Flux values
    flux_err : array-like, optional
        Flux uncertainties
    with_baseline : bool
        Whether to include baseline parameter
    method : str
        Fitting method ('lm', 'trf', 'dogbox', or 'differential_evolution')
    bounds : tuple, optional
        Parameter bounds as (lower_bounds, upper_bounds)
    maxfev : int
        Maximum number of function evaluations
    
    Returns:
    --------
    dict
        Fitting results containing parameters, uncertainties, and fit statistics
    """
    # Convert to numpy arrays
    t = np.asarray(t)
    flux = np.asarray(flux)
    if flux_err is not None:
        flux_err = np.asarray(flux_err)
        # Avoid division by zero
        flux_err = np.where(flux_err <= 0, np.median(flux_err[flux_err > 0]), flux_err)
    
    # Choose function
    func = bazin_with_baseline if with_baseline else bazin
    
    # Get initial parameters
    p0 = estimate_initial_parameters(t, flux, with_baseline)
    
    # Set default bounds if not provided
    if bounds is None:
        if with_baseline:
            # (A, t0, tau_rise, tau_fall, baseline)
            t_span = t[-1] - t[0]
            flux_span = np.max(flux) - np.min(flux)
            flux_min = np.min(flux)
            flux_max = np.max(flux)
            lower_bounds = [0, t[0] - t_span, 0.1, 0.1, flux_min - flux_span]
            upper_bounds = [10 * flux_span, t[-1] + t_span, t_span, 5 * t_span, flux_max + flux_span]
        else:
            # (A, t0, tau_rise, tau_fall)
            t_span = t[-1] - t[0]
            flux_span = np.max(flux) - np.min(flux)
            lower_bounds = [0, t[0] - t_span, 0.1, 0.1]
            upper_bounds = [10 * flux_span, t[-1] + t_span, t_span, 5 * t_span]
        bounds = (lower_bounds, upper_bounds)
    
    # Fit using different methods
    if method == 'differential_evolution':
        # Global optimization
        def objective(params):
            try:
                model = func(t, *params)
                if not np.all(np.isfinite(model)):
                    return 1e10
                if flux_err is not None:
                    residuals = (flux - model) / flux_err
                else:
                    residuals = flux - model
                return np.sum(residuals**2)
            except:
                return 1e10
        
        result = differential_evolution(objective, bounds=list(zip(*bounds)), 
                                      seed=42, maxiter=1000)
        popt = result.x
        success = result.success
        
        # Calculate parameter uncertainties using Hessian approximation
        try:
            # Compute Jacobian numerically
            eps = np.sqrt(np.finfo(float).eps)
            jac = np.zeros((len(t), len(popt)))
            for i in range(len(popt)):
                params_plus = popt.copy()
                params_minus = popt.copy()
                params_plus[i] += eps
                params_minus[i] -= eps
                try:
                    model_plus = func(t, *params_plus)
                    model_minus = func(t, *params_minus)
                    jac[:, i] = (model_plus - model_minus) / (2 * eps)
                except:
                    jac[:, i] = 0
            
            # Weight by uncertainties
            if flux_err is not None:
                jac = jac / flux_err[:, np.newaxis]
            
            # Compute covariance matrix
            try:
                pcov = np.linalg.inv(jac.T @ jac)
                perr = np.sqrt(np.diag(pcov))
            except:
                pcov = None
                perr = None
        except:
            pcov = None
            perr = None
            
    else:
        # Local optimization using curve_fit
        try:
            popt, pcov = curve_fit(func, t, flux, p0=p0, sigma=flux_err,
                                 bounds=bounds, method=method, maxfev=maxfev)
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else None
            success = True
        except Exception as e:
            print(f"Fitting failed: {e}")
            popt = np.array(p0)
            pcov = None
            perr = None
            success = False
    
    # Calculate fit statistics
    try:
        model = func(t, *popt)
        residuals = flux - model
        
        if flux_err is not None:
            chi2 = np.sum((residuals / flux_err)**2)
            chi2_reduced = chi2 / (len(t) - len(popt))
        else:
            chi2 = np.sum(residuals**2)
            chi2_reduced = chi2 / (len(t) - len(popt))
        
        # Calculate R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((flux - np.mean(flux))**2)
        r_squared = 1 - (ss_res / ss_tot)
    except:
        model = np.full_like(flux, np.nan)
        residuals = np.full_like(flux, np.nan)
        chi2 = np.inf
        chi2_reduced = np.inf
        r_squared = -np.inf
        success = False
    
    # Organize results
    param_names = ['A', 't0', 'tau_rise', 'tau_fall']
    if with_baseline:
        param_names.append('baseline')
    
    results = {
        'success': success,
        'parameters': dict(zip(param_names, popt)),
        'parameter_errors': dict(zip(param_names, perr)) if perr is not None else None,
        'covariance_matrix': pcov,
        'model': model,
        'residuals': residuals,
        'chi2': chi2,
        'chi2_reduced': chi2_reduced,
        'r_squared': r_squared,
        'n_data': len(t),
        'n_params': len(popt)
    }
    
    return results

def plot_fit_results(t, flux, fit_results, flux_err=None, with_baseline=False,
                    title=None, save_path=None, show_residuals=True):
    """
    Plot the Bazin fit results.
    
    Parameters:
    -----------
    t : array-like
        Time values
    flux : array-like
        Flux values
    fit_results : dict
        Results from fit_bazin function
    flux_err : array-like, optional
        Flux uncertainties
    with_baseline : bool
        Whether baseline was included in fit
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save plot
    show_residuals : bool
        Whether to show residual plot
    """
    if show_residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                      gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Main plot
    if flux_err is not None:
        ax1.errorbar(t, flux, yerr=flux_err, fmt='o', alpha=0.7, 
                    label='Data', capsize=3)
    else:
        ax1.plot(t, flux, 'o', alpha=0.7, label='Data')
    
    # Plot fit
    t_fine = np.linspace(t.min(), t.max(), 200)
    func = bazin_with_baseline if with_baseline else bazin
    params = list(fit_results['parameters'].values())
    model_fine = func(t_fine, *params)
    
    ax1.plot(t_fine, model_fine, '-', color='red', linewidth=2, label='Bazin fit')
    ax1.plot(t, fit_results['model'], 'x', color='red', markersize=8, 
            alpha=0.7, label='Fit at data points')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Flux')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add fit statistics to plot
    stats_text = f"χ²/dof = {fit_results['chi2_reduced']:.3f}\n"
    stats_text += f"R² = {fit_results['r_squared']:.3f}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add parameter values
    param_text = "Parameters:\n"
    for name, value in fit_results['parameters'].items():
        if fit_results['parameter_errors'] is not None:
            error = fit_results['parameter_errors'][name]
            param_text += f"{name} = {value:.3f} ± {error:.3f}\n"
        else:
            param_text += f"{name} = {value:.3f}\n"
    
    ax1.text(0.98, 0.98, param_text, transform=ax1.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if title:
        ax1.set_title(title)
    
    # Residuals plot
    if show_residuals:
        if flux_err is not None:
            ax2.errorbar(t, fit_results['residuals'], yerr=flux_err, 
                        fmt='o', alpha=0.7, capsize=3)
        else:
            ax2.plot(t, fit_results['residuals'], 'o', alpha=0.7)
        
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Residuals')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def fit_lightcurve_from_file(file_path, time_col='time', flux_col='flux', 
                           flux_err_col='flux_err', with_baseline=False,
                           method='leastsq', plot=True, save_plot=None):
    """
    Fit Bazin function to light curve data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file containing light curve data
    time_col : str
        Name of time column
    flux_col : str
        Name of flux column
    flux_err_col : str
        Name of flux error column
    with_baseline : bool
        Whether to include baseline parameter
    method : str
        Fitting method
    plot : bool
        Whether to plot results
    save_plot : str, optional
        Path to save plot
    
    Returns:
    --------
    dict
        Fitting results
    """
    # Load data
    df = pd.read_csv(file_path)
    
    t = df[time_col].values
    flux = df[flux_col].values
    flux_err = df[flux_err_col].values if flux_err_col in df.columns else None
    
    # Fit Bazin function
    results = fit_bazin(t, flux, flux_err=flux_err, with_baseline=with_baseline, method=method)
    
    # Plot results
    if plot:
        title = f"Bazin Fit - {file_path}"
        plot_fit_results(t, flux, results, flux_err=flux_err, 
                        with_baseline=with_baseline, title=title, save_path=save_plot)
    
    return results

# Example usage
if __name__ == "__main__":
    # Generate synthetic supernova data for testing
    np.random.seed(42)
    
    # True parameters
    A_true = 100.0
    t0_true = 20.0
    tau_rise_true = 5.0
    tau_fall_true = 30.0
    baseline_true = 2.0
    
    # Generate time points
    t = np.linspace(0, 80, 40)
    
    # Generate true light curve
    flux_true = bazin_with_baseline(t, A_true, t0_true, tau_rise_true, tau_fall_true, baseline_true)
    
    # Add noise
    noise_level = 5.0
    flux_err = np.full_like(flux_true, noise_level)
    flux = flux_true + np.random.normal(0, noise_level, len(t))
    
    print("Fitting synthetic supernova light curve...")
    print(f"True parameters: A={A_true}, t0={t0_true}, tau_rise={tau_rise_true}, tau_fall={tau_fall_true}, baseline={baseline_true}")
    
    # Fit with different methods
    methods = ['lm', 'differential_evolution']
    
    for method in methods:
        print(f"\nFitting with method: {method}")
        results = fit_bazin(t, flux, flux_err=flux_err, with_baseline=True, method=method)
        
        if results['success']:
            print("Fit successful!")
            print("Fitted parameters:")
            for name, value in results['parameters'].items():
                if results['parameter_errors'] is not None:
                    error = results['parameter_errors'][name]
                    print(f"  {name} = {value:.3f} ± {error:.3f}")
                else:
                    print(f"  {name} = {value:.3f}")
            
            print(f"χ²/dof = {results['chi2_reduced']:.3f}")
            print(f"R² = {results['r_squared']:.3f}")
            
            # Plot results
            plot_fit_results(t, flux, results, flux_err=flux_err, with_baseline=True,
                           title=f"Bazin Fit ({method})", show_residuals=True)
        else:
            print("Fit failed!")
