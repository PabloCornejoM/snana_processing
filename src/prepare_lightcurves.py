import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import pandas as pd

# Create output directories
os.makedirs("data/processed/lightcurves", exist_ok=True)
os.makedirs("data/processed/plots", exist_ok=True)

# Load the filtered dataset
filtered_data_path = "data/processed/filtered_transients.npz"
data = np.load(filtered_data_path, allow_pickle=True)

# Print basic statistics of the filtered dataset
print("Filtered Dataset Summary:")
print("-" * 45)
for model_name, samples in data.items():
    print(f"{model_name:<25} | {len(samples)} samples")
print("\n")

# Function to extract lightcurve from a sample
def extract_lightcurve(sample, model_name):
    """
    Extract time series data from a sample for each band.
    Returns a dictionary of dataframes, one per band.
    """
    # Convert structured array to DataFrame for easier manipulation
    df = pd.DataFrame(sample)
    
    # Get unique bands
    bands = np.unique(df['BAND'])
    
    # Create a dictionary to store time series for each band
    band_curves = {}
    
    for band in bands:
        # Filter by band
        band_data = df[df['BAND'] == band]
        
        # Sort by time (MJD)
        band_data = band_data.sort_values('MJD')
        
        # Create time series with MJD as time and FLUXCAL as values
        # Also include error information
        time_series = pd.DataFrame({
            'MJD': band_data['MJD'],
            'FLUX': band_data['FLUXCAL'],
            'FLUX_ERR': band_data['FLUXCALERR'],
            'MAG': band_data['SIM_MAGOBS'],
            'PHOTFLAG': band_data['PHOTFLAG']
        })
        
        # Only keep good observations (PHOTFLAG = 0 typically indicates good data)
        good_data = time_series[time_series['PHOTFLAG'] == 0]
        
        # Only include bands with enough good data points
        if len(good_data) >= 5:  # Minimum 5 points to be useful for fitting
            band_curves[band.decode('utf-8')] = good_data
    
    return band_curves

# Process and save all samples
all_lightcurves = defaultdict(list)

for model_name, samples in data.items():
    print(f"Processing {model_name}...")
    
    for i, sample in enumerate(samples):
        # Extract lightcurve
        band_curves = extract_lightcurve(sample, model_name)
        
        # Skip if no valid bands found
        if not band_curves:
            continue
            
        # Add to collection
        curve_info = {
            'model': model_name,
            'sample_id': i,
            'bands': band_curves
        }
        all_lightcurves[model_name].append(curve_info)
        
        # Plot lightcurve
        plt.figure(figsize=(12, 8))
        
        for band_name, band_data in band_curves.items():
            # Plot flux
            plt.subplot(2, 1, 1)
            plt.plot(band_data['MJD'], band_data['FLUX'], 'o-', label=f'Band {band_name}')
            plt.fill_between(
                band_data['MJD'], 
                band_data['FLUX'] - band_data['FLUX_ERR'],
                band_data['FLUX'] + band_data['FLUX_ERR'],
                alpha=0.3
            )
            
            # Plot magnitude
            plt.subplot(2, 1, 2)
            plt.plot(band_data['MJD'], band_data['MAG'], 'o-', label=f'Band {band_name}')
        
        plt.subplot(2, 1, 1)
        plt.title(f"{model_name} - Sample {i+1}")
        plt.xlabel("Time (MJD)")
        plt.ylabel("Flux")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.xlabel("Time (MJD)")
        plt.ylabel("Magnitude")
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Magnitudes decrease as brightness increases
        
        plt.tight_layout()
        plt.savefig(f"data/processed/plots/{model_name}_sample_{i+1}_curve.png")
        plt.close()

# Save all processed lightcurves
print("\nSaving processed lightcurves...")

# Save in a format suitable for symbolic regression
for model_name, curves in all_lightcurves.items():
    # Create directory for this model
    model_dir = f"data/processed/lightcurves/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    for curve in curves:
        sample_id = curve['sample_id']
        
        for band_name, band_data in curve['bands'].items():
            # Normalize time to start at 0
            time = band_data['MJD'].values
            flux = band_data['FLUX'].values
            mag = band_data['MAG'].values
            
            # Normalize time to start from 0
            time_normalized = time - time.min()
            
            # Save as CSV for easy importing into symbolic regression tools
            df_out = pd.DataFrame({
                'time': time_normalized,
                'flux': flux,
                'magnitude': mag,
                'original_time': time
            })
            
            df_out.to_csv(f"{model_dir}/sample_{sample_id}_band_{band_name}.csv", index=False)

# Create a summary file with all curve information
print("\nCreating summary file...")

summary_rows = []
for model_name, curves in all_lightcurves.items():
    for curve in curves:
        sample_id = curve['sample_id']
        bands = list(curve['bands'].keys())
        
        # Count total data points across all bands
        total_points = sum(len(band_data) for band_data in curve['bands'].values())
        
        # Calculate time span
        all_times = np.concatenate([band_data['MJD'].values for band_data in curve['bands'].values()])
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
summary_df.to_csv("data/processed/lightcurve_summary.csv", index=False)

print(f"\nProcessed {len(summary_rows)} lightcurves from {len(all_lightcurves)} transient types")
print("Data saved in data/processed/lightcurves/")
print("Summary file saved as data/processed/lightcurve_summary.csv")
print("Plots saved in data/processed/plots/") 