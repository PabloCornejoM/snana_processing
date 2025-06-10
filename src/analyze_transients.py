import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Load the data
data_path = "/Users/pablocornejo/Documents/Tesis/SRNetwork/data/raw/sim_samples.npz"
models = np.load(data_path, allow_pickle=True)

# Print basic statistics
print("Model Name".ljust(25) + " | N Lightcurves")
print("-" * 45)
for k, v in models.items():
    print(f"{k:<25} | {len(v)}")
print("\n")

# List of transient types we're particularly interested in
target_types = [
    "SNIa-SALT2", 
    "SNIc-Templates", 
    "SLSN-I+host", 
    "TDE", 
    "KN_K17", 
    "ILOT", 
    "PISN-MOSFIT", 
    "SNIax"
]

# Create a directory for plots
os.makedirs("data/plots", exist_ok=True)

# Analyze structure of the first example for each type
print("Data Structure Analysis:")
print("-" * 45)

# For storing statistics
stats = defaultdict(dict)

for model_name in models.keys():
    if len(models[model_name]) == 0:
        print(f"{model_name}: Empty dataset")
        continue
        
    sample = models[model_name][0]  # Get the first sample
    
    print(f"{model_name}:")
    print(f"  Sample type: {type(sample)}")
    
    if isinstance(sample, dict):
        print("  Dictionary keys:")
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                print(f"    {key}: array of shape {value.shape}, dtype: {value.dtype}")
                # Store statistics
                stats[model_name][key] = {
                    "shape": value.shape,
                    "dtype": str(value.dtype)
                }
            else:
                print(f"    {key}: {type(value)}")
                stats[model_name][key] = {
                    "type": str(type(value))
                }
    elif isinstance(sample, np.ndarray):
        print(f"  Array of shape {sample.shape}, dtype: {sample.dtype}")
        stats[model_name]["array"] = {
            "shape": sample.shape,
            "dtype": str(sample.dtype)
        }
    else:
        print(f"  Unknown structure: {type(sample)}")
    
    print("")

# Plot an example for each target type (focusing on lightcurves)
print("\nGenerating example plots for target transient types...")

for model_name in target_types:
    if model_name not in models or len(models[model_name]) == 0:
        print(f"No data for {model_name}")
        continue
    
    # Get first 3 samples of this type (or fewer if not available)
    n_samples = min(3, len(models[model_name]))
    
    for i in range(n_samples):
        sample = models[model_name][i]
        
        # Attempt to extract time and flux data based on common conventions
        # This part may need adjustment based on actual data structure
        if isinstance(sample, dict):
            # Try to find time and flux arrays
            time_key = next((k for k in sample if 'time' in k.lower()), None)
            flux_key = next((k for k in sample if 'flux' in k.lower() or 'mag' in k.lower()), None)
            
            if time_key and flux_key:
                plt.figure(figsize=(10, 6))
                time = sample[time_key]
                flux = sample[flux_key]
                
                # Check if we have multiple bands/filters
                if len(flux.shape) > 1 and flux.shape[0] > 1:
                    for band_idx in range(flux.shape[0]):
                        plt.plot(time, flux[band_idx], 'o-', label=f'Band {band_idx}')
                    plt.legend()
                else:
                    plt.plot(time, flux, 'o-')
                
                plt.title(f"{model_name} - Sample {i+1}")
                plt.xlabel("Time")
                plt.ylabel("Flux/Magnitude")
                plt.grid(True, alpha=0.3)
                plt.savefig(f"data/plots/{model_name}_sample_{i+1}.png")
                plt.close()
                print(f"Plotted {model_name} sample {i+1}")
        
print("\nAnalysis complete. Check data/plots directory for example plots.")

# Create a filtered dataset with representative samples
print("\nCreating filtered dataset with representative samples...")

filtered_data = {}
n_per_type = 5  # Number of samples to keep per type

for model_name in target_types:
    if model_name in models and len(models[model_name]) > 0:
        # Take up to n_per_type samples
        filtered_data[model_name] = models[model_name][:n_per_type]
        print(f"Added {len(filtered_data[model_name])} samples of {model_name}")

# Save filtered dataset
filtered_path = "data/processed/filtered_transients.npz"
os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
np.savez(filtered_path, **filtered_data)

print(f"\nFiltered dataset saved to {filtered_path}") 