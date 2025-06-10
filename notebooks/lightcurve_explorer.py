# %% [markdown]


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from IPython.display import display
import ipywidgets as widgets

# Create output directories
os.makedirs("selected_lightcurves", exist_ok=True)

# %% [markdown]
# ## Load the Dataset

# %%
# Load the dataset
data_path = "data/sim_samples.npz"
data = np.load(data_path, allow_pickle=True)

# Display available transient types
print("Available transient types:")
print("-" * 30)
for key in data:
    print(f"{key}: {len(data[key])} samples")

# Define model types of interest
model_types = ["SNIa-SALT2", "SNIc-Templates", "SLSN-I+host", "TDE", 
               "KN_K17", "ILOT", "PISN-MOSFIT", "SNIax"]

# Filter to only include available types
available_types = [model for model in model_types if model in data and len(data[model]) > 0]

# %% [markdown]
# ## Helper Functions

# %%
def extract_lightcurve(sample):
    """Extract lightcurve data by band from a sample."""
    # Convert to DataFrame
    df = pd.DataFrame(sample)
    
    # Get unique bands
    bands = np.unique(df['BAND'])
    
    # Create a dictionary to store data for each band
    band_curves = {}
    
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
        band_label = band.decode('utf-8') if isinstance(band, bytes) else band
        
        # Create DataFrame with relevant columns
        band_curves[band_label] = pd.DataFrame({
            'MJD': good_data['MJD'].values,
            'FLUXCAL': good_data['FLUXCAL'].values,
            'FLUXCALERR': good_data['FLUXCALERR'].values,
            'PHOTFLAG': good_data['PHOTFLAG'].values
        })
    
    return band_curves

def plot_lightcurve(band_curves, model_name, sample_id):
    """Plot the lightcurve data for all bands."""
    if not band_curves:
        return
        
    plt.figure(figsize=(12, 6))
    
    for band_name, band_data in band_curves.items():
        plt.errorbar(
            band_data['MJD'], 
            band_data['FLUXCAL'], 
            yerr=band_data['FLUXCALERR'],
            fmt='o-', 
            label=f'Band {band_name}'
        )
    
    plt.title(f"{model_name} - Sample {sample_id}")
    plt.xlabel("Time (MJD)")
    plt.ylabel("Flux")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.show()

def save_selected_lightcurve(band_curves, model_name, sample_id):
    """Save the selected lightcurve for symbolic regression."""
    # Create directory
    os.makedirs(f"selected_lightcurves/{model_name}", exist_ok=True)
    
    saved_files = []
    
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
        output_file = f"selected_lightcurves/{model_name}/sample_{sample_id}_band_{band_name}.csv"
        df_out.to_csv(output_file, index=False)
        saved_files.append(output_file)
    
    print(f"Saved {len(saved_files)} band curves to:")
    for file in saved_files:
        print(f"  - {file}")
    
    return saved_files

# %% [markdown]
# ## Interactive Lightcurve Explorer
# 
# Use the widgets below to browse through transient types and samples.

# %%
# Create dropdown for model selection
model_dropdown = widgets.Dropdown(
    options=available_types,
    description='Model Type:',
    disabled=False,
)

# Function to update sample slider when model changes
def update_sample_slider(change):
    model = change['new']
    num_samples = len(data[model])
    sample_slider.max = num_samples - 1
    sample_slider.value = 0
    
model_dropdown.observe(update_sample_slider, names='value')

# Create slider for sample selection
sample_slider = widgets.IntSlider(
    value=0,
    min=0,
    max=len(data[available_types[0]]) - 1 if available_types else 0,
    step=1,
    description='Sample ID:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True
)

# Function to display the selected lightcurve
def display_lightcurve(model, sample_id):
    sample = data[model][sample_id]
    band_curves = extract_lightcurve(sample)
    
    if band_curves:
        plot_lightcurve(band_curves, model, sample_id)
        
        # Show number of data points per band
        print("Data points per band:")
        for band, curve in band_curves.items():
            print(f"  Band {band}: {len(curve)} points")
    else:
        print("No valid lightcurve data found for this sample.")
    
    return band_curves

# Button to display the selected lightcurve
display_button = widgets.Button(
    description='Display Lightcurve',
    disabled=False,
    button_style='info',
    tooltip='Click to display the selected lightcurve'
)

def on_display_button_clicked(b):
    global current_band_curves
    current_band_curves = display_lightcurve(model_dropdown.value, sample_slider.value)
    
display_button.on_click(on_display_button_clicked)

# Button to save the selected lightcurve
save_button = widgets.Button(
    description='Save Selected',
    disabled=False,
    button_style='success',
    tooltip='Save the selected lightcurve for symbolic regression'
)

def on_save_button_clicked(b):
    if 'current_band_curves' in globals() and current_band_curves:
        save_selected_lightcurve(current_band_curves, model_dropdown.value, sample_slider.value)
    else:
        print("No lightcurve has been displayed yet. Click 'Display Lightcurve' first.")
        
save_button.on_click(on_save_button_clicked)

# Display widgets
display(widgets.VBox([
    model_dropdown,
    sample_slider,
    widgets.HBox([display_button, save_button])
]))

# %% [markdown]
# ## Prepare Selected Lightcurves for Symbolic Regression
# 
# After you've selected some lightcurves, run this cell to prepare them for symbolic regression.

# %%
def prepare_for_symbolic_regression(normalize=True, smooth=False, spline_s=0.1):
    """Prepare selected lightcurves for symbolic regression."""
    from scipy.interpolate import UnivariateSpline
    from sklearn.preprocessing import MinMaxScaler
    
    # Create output directories
    os.makedirs("symbolic_regression_data", exist_ok=True)
    os.makedirs("symbolic_regression_plots", exist_ok=True)
    
    # Get all selected lightcurve files
    selected_models = [d for d in os.listdir("selected_lightcurves") 
                      if os.path.isdir(os.path.join("selected_lightcurves", d))]
    
    for model in selected_models:
        print(f"Processing {model}...")
        os.makedirs(f"symbolic_regression_data/{model}", exist_ok=True)
        
        model_dir = os.path.join("selected_lightcurves", model)
        csv_files = [f for f in os.listdir(model_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            # Load data
            file_path = os.path.join(model_dir, csv_file)
            df = pd.read_csv(file_path)
            
            # Extract info from filename
            parts = os.path.splitext(csv_file)[0].split('_')
            sample_id = parts[1]
            band = '_'.join(parts[3:])
            
            # Extract data
            time = df['time'].values
            flux = df['flux'].values
            flux_err = df['flux_err'].values
            original_time = df['original_time'].values
            
            # Sort by time
            sort_idx = np.argsort(time)
            time = time[sort_idx]
            flux = flux[sort_idx]
            flux_err = flux_err[sort_idx]
            original_time = original_time[sort_idx]
            
            # Normalize flux if requested
            if normalize:
                scaler = MinMaxScaler()
                flux = scaler.fit_transform(flux.reshape(-1, 1)).flatten()
                
                # Scale errors proportionally
                flux_range = np.max(df['flux'].values) - np.min(df['flux'].values)
                if flux_range > 0:
                    flux_err = flux_err / flux_range
            
            # Smooth data if requested
            if smooth and len(time) > 3:
                try:
                    # Use inverse of error as weights
                    weights = 1.0 / (flux_err + 1e-10)
                    spline = UnivariateSpline(time, flux, w=weights, s=spline_s)
                    
                    # Generate smoothed flux
                    flux_smooth = spline(time)
                    
                    # Only use smoothed values if they're valid
                    if np.all(np.isfinite(flux_smooth)):
                        flux = flux_smooth
                except Exception as e:
                    print(f"  Warning: Smoothing failed for {csv_file}: {e}")
            
            # Create output DataFrame
            df_out = pd.DataFrame({
                'time': time,
                'flux': flux,
                'flux_err': flux_err,
                'original_time': original_time
            })
            
            # Save processed data
            output_file = f"symbolic_regression_data/{model}/{os.path.splitext(csv_file)[0]}_sr.csv"
            df_out.to_csv(output_file, index=False)
            
            # Plot original vs processed
            plt.figure(figsize=(10, 6))
            
            # Original data
            plt.errorbar(df['time'], df['flux'], yerr=df['flux_err'], 
                        fmt='o', alpha=0.5, color='gray', label='Original data')
            
            # Processed data
            plt.plot(time, flux, 'o-', color='blue', label='Processed for SR')
            
            plt.title(f"{model} - Sample {sample_id} - Band {band}")
            plt.xlabel("Time (days from first observation)")
            plt.ylabel("Flux")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plot_file = f"symbolic_regression_plots/{model}_sample_{sample_id}_band_{band}_sr.png"
            plt.savefig(plot_file)
            plt.close()
        
        print(f"  Completed processing {len(csv_files)} files for {model}")
    
    print("\nAll selected lightcurves prepared for symbolic regression!")

# Run this cell when you're ready to process the selected lightcurves
# prepare_for_symbolic_regression(normalize=True, smooth=True, spline_s=0.1) 