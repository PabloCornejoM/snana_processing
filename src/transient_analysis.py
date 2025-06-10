# %% [markdown]
# # Transient Lightcurve Analysis
# 
# This script explores and processes transient lightcurves from SNANA simulation data for symbolic regression.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# Create output directories if they don't exist
os.makedirs("processed_data", exist_ok=True)
os.makedirs("processed_data/lightcurves", exist_ok=True)
os.makedirs("processed_data/plots", exist_ok=True)

# %% [markdown]
# ## Data Exploration
# 
# First, let's load and explore the dataset structure.

# %%
# Load the dataset
data_path = "data/sim_samples.npz"
data = np.load(data_path, allow_pickle=True)

# Print available keys in the dataset
print("Available keys in the dataset:")
print("-" * 30)
print(list(data.keys()))
print()

# Print sample counts for each key/model type
print("Sample counts:")
print("-" * 30)
for key in data:
    print(f"{key}: {len(data[key])} samples")
print()

# %% [markdown]
# ## Examining Sample Structure
# 
# Let's look at the structure of one sample from each model type.

# %%
# Select one model type to examine in detail
model_types = ["SNIa-SALT2", "SNIc-Templates", "SLSN-I+host", "TDE", 
               "KN_K17", "ILOT", "PISN-MOSFIT", "SNIax"]

# Find a valid model type from our list
selected_model = None
for model in model_types:
    if model in data and len(data[model]) > 0:
        selected_model = model
        break

if selected_model:
    print(f"Examining sample from model: {selected_model}")
    
    # Get a sample
    sample = data[selected_model][0]
    
    # Display structure
    print(f"Type: {type(sample)}")
    print(f"Shape: {sample.shape}")
    
    if sample.dtype.names:
        print("Fields:", sample.dtype.names)
        
        # Convert to DataFrame for easier viewing
        df = pd.DataFrame(sample)
        print("\nSample data (first 5 rows):")
        display(df.head())
        
        # Check unique bands
        if 'BAND' in df.columns:
            bands = np.unique(df['BAND'])
            print(f"\nUnique bands: {bands}")
else:
    print("No valid model types found in the data.")

# %% [markdown]
# ## Processing Lightcurves
# 
# Let's process the lightcurves for each model type and create plots.

# %%
def extract_lightcurve(sample):
    """
    Extract lightcurve data from a sample, organized by band.
    Returns a dictionary with one DataFrame per band.
    """
    # Convert to DataFrame
    df = pd.DataFrame(sample)
    
    # Get unique bands
    bands = np.unique(df['BAND'])
    
    # Create a dictionary to store data for each band
    band_curves = {}
    
    for band in bands:
        # Filter by band
        band_data = df[df['BAND'] == band]
        
        # Sort by time (MJD)
        band_data = band_data.sort_values('MJD')
        
        # Create a DataFrame with relevant columns
        curve_data = pd.DataFrame({
            'MJD': band_data['MJD'],
            'FLUXCAL': band_data['FLUXCAL'],
            'FLUXCALERR': band_data['FLUXCALERR'],
            'MAG': band_data.get('SIM_MAGOBS', np.nan),  # Not all samples might have this
            'PHOTFLAG': band_data['PHOTFLAG']
        })
        
        # Only keep good observations (PHOTFLAG = 0)
        good_data = curve_data[curve_data['PHOTFLAG'] == 0]
        
        # Only include bands with enough good data points (at least 5)
        if len(good_data) >= 5:
            band_curves[band.decode('utf-8') if isinstance(band, bytes) else band] = good_data
    
    return band_curves

# %%
def plot_lightcurve(band_curves, model_name, sample_id, save_path=None):
    """
    Plot the lightcurve data for all bands.
    """
    if not band_curves:
        return
        
    plt.figure(figsize=(12, 8))
    
    # Plot flux
    plt.subplot(2, 1, 1)
    for band_name, band_data in band_curves.items():
        plt.plot(band_data['MJD'], band_data['FLUXCAL'], 'o-', label=f'Band {band_name}')
        plt.fill_between(
            band_data['MJD'], 
            band_data['FLUXCAL'] - band_data['FLUXCALERR'],
            band_data['FLUXCAL'] + band_data['FLUXCALERR'],
            alpha=0.3
        )
    
    plt.title(f"{model_name} - Sample {sample_id}")
    plt.xlabel("Time (MJD)")
    plt.ylabel("Flux")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot magnitude if available
    if not np.all(np.isnan(next(iter(band_curves.values()))['MAG'])):
        plt.subplot(2, 1, 2)
        for band_name, band_data in band_curves.items():
            plt.plot(band_data['MJD'], band_data['MAG'], 'o-', label=f'Band {band_name}')
        
        plt.xlabel("Time (MJD)")
        plt.ylabel("Magnitude")
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Magnitudes decrease as brightness increases
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# %% [markdown]
# ## Process selected model types
# 
# Now let's process a few samples from each model type.

# %%
# Define model types of interest
model_types = ["SNIa-SALT2", "SNIc-Templates", "SLSN-I+host", "TDE", 
               "KN_K17", "ILOT", "PISN-MOSFIT", "SNIax"]

# Process a few samples from each model
num_samples_per_model = 2  # Adjust as needed
processed_data = {}

for model_name in model_types:
    if model_name in data and len(data[model_name]) > 0:
        print(f"Processing {model_name}...")
        
        # Create directory for this model
        model_dir = f"processed_data/lightcurves/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        model_samples = []
        n_samples = min(num_samples_per_model, len(data[model_name]))
        
        for i in range(n_samples):
            sample = data[model_name][i]
            
            # Extract lightcurve
            band_curves = extract_lightcurve(sample)
            
            if band_curves:
                # Save information
                sample_info = {
                    'model': model_name,
                    'sample_id': i,
                    'band_curves': band_curves
                }
                
                model_samples.append(sample_info)
                
                # Plot lightcurve
                plot_path = f"processed_data/plots/{model_name}_sample_{i+1}.png"
                plot_lightcurve(band_curves, model_name, i+1, save_path=plot_path)
                
                # Save each band curve as CSV
                for band_name, band_data in band_curves.items():
                    # Normalize time to start at 0
                    time = band_data['MJD'].values
                    time_normalized = time - time.min()
                    
                    # Create output DataFrame
                    df_out = pd.DataFrame({
                        'time': time_normalized,
                        'flux': band_data['FLUXCAL'].values,
                        'flux_err': band_data['FLUXCALERR'].values,
                        'original_time': time
                    })
                    
                    # Save as CSV
                    df_out.to_csv(f"{model_dir}/sample_{i}_band_{band_name}.csv", index=False)
        
        processed_data[model_name] = model_samples
        print(f"  Processed {len(model_samples)} samples\n")

# %% [markdown]
# ## Create a summary of the processed data

# %%
# Create a summary DataFrame
summary_rows = []

for model_name, samples in processed_data.items():
    for sample in samples:
        sample_id = sample['sample_id']
        bands = list(sample['band_curves'].keys())
        
        # Calculate total data points and time span
        total_points = sum(len(band_data) for band_data in sample['band_curves'].values())
        
        # Get min and max time across all bands
        all_times = np.concatenate([band_data['MJD'].values for band_data in sample['band_curves'].values()])
        time_span = np.max(all_times) - np.min(all_times)
        
        summary_rows.append({
            'model': model_name,
            'sample_id': sample_id,
            'bands': ','.join(bands),
            'num_bands': len(bands),
            'total_points': total_points,
            'time_span': time_span
        })

summary_df = pd.DataFrame(summary_rows)
print("Summary of processed lightcurves:")
display(summary_df)

# Save summary
summary_df.to_csv("processed_data/lightcurve_summary.csv", index=False)

# %% [markdown]
# ## Prepare data for symbolic regression
# 
# Now we'll prepare the data specifically for symbolic regression by:
# 1. Normalizing the flux values
# 2. Aligning the time to start at 0
# 3. Optionally smoothing the data

# %%
def prepare_for_symbolic_regression(band_curves, normalize=True, smooth=False, spline_s=0.1):
    """
    Prepare lightcurve data for symbolic regression.
    
    Parameters:
    -----------
    band_curves : dict
        Dictionary of band data
    normalize : bool
        Whether to normalize flux to [0,1]
    smooth : bool
        Whether to smooth the data using spline interpolation
    spline_s : float
        Smoothing factor for spline interpolation
        
    Returns:
    --------
    dict of DataFrames with processed data
    """
    from scipy.interpolate import UnivariateSpline
    from sklearn.preprocessing import MinMaxScaler
    
    processed_curves = {}
    
    for band_name, band_data in band_curves.items():
        # Extract data
        time = band_data['MJD'].values
        flux = band_data['FLUXCAL'].values
        flux_err = band_data['FLUXCALERR'].values
        
        # Ensure time is sorted
        sort_idx = np.argsort(time)
        time = time[sort_idx]
        flux = flux[sort_idx]
        flux_err = flux_err[sort_idx]
        
        # Normalize time to start at 0
        time_normalized = time - time.min()
        
        # Normalize flux if requested
        if normalize:
            scaler = MinMaxScaler()
            flux = scaler.fit_transform(flux.reshape(-1, 1)).flatten()
            
            # Scale errors proportionally
            flux_range = np.max(band_data['FLUXCAL'].values) - np.min(band_data['FLUXCAL'].values)
            if flux_range > 0:
                flux_err = flux_err / flux_range
        
        # Smooth data if requested
        if smooth and len(time) > 3:
            try:
                # Use inverse of error as weights
                weights = 1.0 / (flux_err + 1e-10)
                spline = UnivariateSpline(time_normalized, flux, w=weights, s=spline_s)
                
                # Generate smoothed flux
                flux_smooth = spline(time_normalized)
                
                # Only use smoothed values if they're valid
                if np.all(np.isfinite(flux_smooth)):
                    flux = flux_smooth
            except Exception as e:
                print(f"Warning: Smoothing failed for band {band_name}: {e}")
        
        # Create DataFrame with processed data
        processed_curves[band_name] = pd.DataFrame({
            'time': time_normalized,
            'flux': flux,
            'flux_err': flux_err,
            'original_time': time
        })
    
    return processed_curves

# %%
# Process data for symbolic regression
for model_name, samples in processed_data.items():
    print(f"Preparing {model_name} for symbolic regression...")
    
    # Create directory
    sr_dir = f"processed_data/symbolic_regression/{model_name}"
    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(f"processed_data/symbolic_regression/plots", exist_ok=True)
    
    for sample in samples:
        sample_id = sample['sample_id']
        band_curves = sample['band_curves']
        
        # Process data
        processed_curves = prepare_for_symbolic_regression(
            band_curves, 
            normalize=True,
            smooth=True,
            spline_s=0.1
        )
        
        # Save each band
        for band_name, processed_data in processed_curves.items():
            # Save as CSV
            processed_data.to_csv(f"{sr_dir}/sample_{sample_id}_band_{band_name}_sr.csv", index=False)
            
            # Plot original vs processed
            plt.figure(figsize=(10, 6))
            
            # Original data with error bars
            original_data = band_curves[band_name]
            time_orig = original_data['MJD'].values - original_data['MJD'].values.min()
            flux_orig = original_data['FLUXCAL'].values
            flux_err_orig = original_data['FLUXCALERR'].values
            
            plt.errorbar(time_orig, flux_orig, yerr=flux_err_orig, 
                        fmt='o', alpha=0.5, color='gray', label='Original data')
            
            # Processed data
            plt.plot(processed_data['time'], processed_data['flux'], 'o-', 
                    color='blue', label='Processed for SR')
            
            plt.title(f"{model_name} - Sample {sample_id} - Band {band_name}")
            plt.xlabel("Time (days from first observation)")
            plt.ylabel("Flux")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.savefig(f"processed_data/symbolic_regression/plots/{model_name}_sample_{sample_id}_band_{band_name}_sr.png")
            plt.close()
    
    print(f"  Completed {len(samples)} samples\n")

print("All data prepared for symbolic regression!") 