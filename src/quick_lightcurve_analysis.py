import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output directories
os.makedirs("output", exist_ok=True)

# Load the dataset
data_path = "data/sim_samples.npz"
data = np.load(data_path, allow_pickle=True)

# Print available transient types and their counts
print("Available transient types:")
print("-" * 30)
for key in data:
    print(f"{key}: {len(data[key])} samples")
print()

# Select specific transient types (modify as needed)
selected_types = ["SNIa-SALT2", "SNIc-Templates", "SLSN-I+host", "TDE", 
                 "KN_K17", "ILOT", "PISN-MOSFIT", "SNIax"]

# Function to extract and plot a lightcurve
def extract_and_plot_lightcurve(sample, model_name, sample_id, save=True):
    """Extract and plot a lightcurve from a sample."""
    # Convert to DataFrame
    df = pd.DataFrame(sample)
    
    # Get unique bands
    bands = np.unique(df['BAND'])
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot all bands
    for band in bands:
        # Filter by band
        band_data = df[df['BAND'] == band]
        
        # Filter good observations (PHOTFLAG = 0)
        good_data = band_data[band_data['PHOTFLAG'] == 0]
        
        # Skip if not enough data points
        if len(good_data) < 5:
            continue
            
        # Sort by time
        good_data = good_data.sort_values('MJD')
        
        # Extract data
        time = good_data['MJD'].values
        flux = good_data['FLUXCAL'].values
        flux_err = good_data['FLUXCALERR'].values
        
        # Plot the data
        band_label = band.decode('utf-8') if isinstance(band, bytes) else band
        plt.errorbar(time, flux, yerr=flux_err, fmt='o-', label=f'Band {band_label}')
    
    plt.title(f"{model_name} - Sample {sample_id}")
    plt.xlabel("Time (MJD)")
    plt.ylabel("Flux")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save:
        plt.savefig(f"output/{model_name}_sample_{sample_id}.png")
        plt.close()
    else:
        plt.show()
    
    return df

# Function to save lightcurve data
def save_lightcurve_data(sample, model_name, sample_id):
    """Save lightcurve data to CSV files."""
    # Convert to DataFrame
    df = pd.DataFrame(sample)
    
    # Get unique bands
    bands = np.unique(df['BAND'])
    
    saved_files = []
    
    # Process each band
    for band in bands:
        # Filter by band
        band_data = df[df['BAND'] == band]
        
        # Filter good observations
        good_data = band_data[band_data['PHOTFLAG'] == 0]
        
        # Skip if not enough data points
        if len(good_data) < 5:
            continue
            
        # Sort by time
        good_data = good_data.sort_values('MJD')
        
        # Extract data
        time = good_data['MJD'].values
        flux = good_data['FLUXCAL'].values
        flux_err = good_data['FLUXCALERR'].values
        
        # Normalize time to start at 0
        time_normalized = time - time.min()
        
        # Create output DataFrame
        band_label = band.decode('utf-8') if isinstance(band, bytes) else band
        output_file = f"output/{model_name}_sample_{sample_id}_band_{band_label}.csv"
        
        df_out = pd.DataFrame({
            'time': time_normalized,
            'flux': flux,
            'flux_err': flux_err,
            'original_time': time
        })
        
        # Save to CSV
        df_out.to_csv(output_file, index=False)
        saved_files.append(output_file)
    
    return saved_files

# Process one sample from each selected type
for model_name in selected_types:
    if model_name in data and len(data[model_name]) > 0:
        print(f"Processing {model_name}...")
        
        # Get first sample
        sample = data[model_name][0]
        
        # Plot the lightcurve
        df = extract_and_plot_lightcurve(sample, model_name, 1)
        
        # Save data
        saved_files = save_lightcurve_data(sample, model_name, 1)
        
        print(f"  Saved files: {saved_files}")
        print()

print("Quick analysis completed!")

# Optional: Examine detailed structure of one sample
if selected_types[0] in data and len(data[selected_types[0]]) > 0:
    model = selected_types[0]
    sample = data[model][0]
    
    print(f"\nDetailed examination of a {model} sample:")
    print("-" * 50)
    
    df = pd.DataFrame(sample)
    
    print("Sample shape:", sample.shape)
    print("Available columns:", df.columns.tolist())
    
    # Show the first few rows
    print("\nFirst 5 rows of data:")
    print(df.head())
    
    # Show column types and sample values
    print("\nColumn information:")
    for col in df.columns:
        values = df[col].unique()
        value_sample = values[:3] if len(values) > 3 else values
        print(f"{col}: {len(values)} unique values, examples: {value_sample}") 