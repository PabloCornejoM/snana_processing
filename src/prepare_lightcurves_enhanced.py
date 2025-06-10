import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import pandas as pd
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract and visualize transient lightcurves.')
    parser.add_argument('--data_path', type=str, default="/Users/pablocornejo/Documents/Tesis/SRNetwork/data/raw/sim_samples.npz", 
                        help='Path to the raw data file')
    parser.add_argument('--model_types', type=str, nargs='+', 
                        default=["SNIa-SALT2", "SNIc-Templates", "SLSN-I+host", "TDE", "KN_K17", "ILOT", "PISN-MOSFIT", "SNIax"],
                        help='List of transient model types to process')
    parser.add_argument('--specific_ids', type=str, nargs='*', 
                        help='Specific model_type:id pairs to process (e.g., SNIa-SALT2:10 TDE:5)')
    parser.add_argument('--output_dir', type=str, default="data/processed", 
                        help='Directory to save outputs')
    parser.add_argument('--n_per_type', type=int, default=5, 
                        help='Number of samples per type if specific IDs not provided')
    return parser.parse_args()

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

def plot_lightcurve(curve_info, output_dir):
    """Plot the lightcurve with focus on flux and flux error."""
    model_name = curve_info['model']
    sample_id = curve_info['sample_id']
    band_curves = curve_info['bands']
    
    # Plot lightcurve
    plt.figure(figsize=(12, 6))
    
    # Create a subplot for flux
    plt.subplot(1, 1, 1)
    
    for band_name, band_data in band_curves.items():
        plt.errorbar(
            band_data['MJD'], 
            band_data['FLUX'], 
            yerr=band_data['FLUX_ERR'], 
            fmt='o-', 
            label=f'Band {band_name}',
            capsize=3
        )
    
    plt.title(f"{model_name} - Sample {sample_id}")
    plt.xlabel("Time (MJD)")
    plt.ylabel("Flux")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    save_path = f"{output_dir}/plots/{model_name}_sample_{sample_id}_flux.png"
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directories
    os.makedirs(f"{args.output_dir}/lightcurves", exist_ok=True)
    os.makedirs(f"{args.output_dir}/plots", exist_ok=True)
    
    # Load the data
    print(f"Loading data from {args.data_path}...")
    models = np.load(args.data_path, allow_pickle=True)
    
    # Print basic statistics
    print("\nAvailable transient types:")
    print("-" * 45)
    for k, v in models.items():
        if k in args.model_types:
            print(f"{k:<25} | {len(v)} samples")
    print("\n")
    
    # Parse specific IDs if provided
    selected_samples = {}
    if args.specific_ids:
        for id_pair in args.specific_ids:
            parts = id_pair.split(':')
            if len(parts) != 2:
                print(f"Warning: Invalid ID pair format '{id_pair}'. Expected 'model_type:id'")
                continue
                
            model_type, sample_id = parts
            sample_id = int(sample_id)
            
            if model_type not in models:
                print(f"Warning: Model type '{model_type}' not found in data")
                continue
                
            if sample_id >= len(models[model_type]):
                print(f"Warning: Sample ID {sample_id} exceeds available samples for {model_type}")
                continue
                
            if model_type not in selected_samples:
                selected_samples[model_type] = []
                
            selected_samples[model_type].append(sample_id)
    
    # If no specific IDs provided, use n_per_type
    if not selected_samples:
        for model_type in args.model_types:
            if model_type in models and len(models[model_type]) > 0:
                selected_samples[model_type] = list(range(min(args.n_per_type, len(models[model_type]))))
    
    # Process and save selected samples
    all_lightcurves = defaultdict(list)
    for model_name, sample_ids in selected_samples.items():
        print(f"Processing {model_name} (samples: {sample_ids})...")
        
        for sample_id in sample_ids:
            sample = models[model_name][sample_id]
            
            # Extract lightcurve
            band_curves = extract_lightcurve(sample, model_name)
            
            # Skip if no valid bands found
            if not band_curves:
                print(f"  Warning: No valid bands found for {model_name} sample {sample_id}")
                continue
                
            # Add to collection
            curve_info = {
                'model': model_name,
                'sample_id': sample_id,
                'bands': band_curves
            }
            all_lightcurves[model_name].append(curve_info)
            
            # Plot the lightcurve
            plot_path = plot_lightcurve(curve_info, args.output_dir)
            print(f"  Plotted {model_name} sample {sample_id}: {plot_path}")
    
    # Save processed lightcurves
    print("\nSaving processed lightcurves...")
    
    # Save in a format suitable for symbolic regression
    for model_name, curves in all_lightcurves.items():
        # Create directory for this model
        model_dir = f"{args.output_dir}/lightcurves/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        for curve in curves:
            sample_id = curve['sample_id']
            
            for band_name, band_data in curve['bands'].items():
                # Get time and flux data
                time = band_data['MJD'].values
                flux = band_data['FLUX'].values
                flux_err = band_data['FLUX_ERR'].values
                
                # Normalize time to start from 0
                time_normalized = time - time.min()
                
                # Save as CSV
                df_out = pd.DataFrame({
                    'time': time_normalized,
                    'flux': flux,
                    'flux_err': flux_err,
                    'original_time': time
                })
                
                csv_path = f"{model_dir}/sample_{sample_id}_band_{band_name}.csv"
                df_out.to_csv(csv_path, index=False)
                print(f"  Saved {csv_path}")
    
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
            
            # Calculate signal-to-noise ratio
            snr_values = []
            for band_data in curve['bands'].values():
                # Calculate mean SNR for this band
                snr = np.mean(np.abs(band_data['FLUX']) / band_data['FLUX_ERR'])
                snr_values.append(snr)
            
            mean_snr = np.mean(snr_values) if snr_values else 0
            
            summary_rows.append({
                'model': model_name,
                'sample_id': sample_id,
                'bands': ','.join(bands),
                'num_bands': len(bands),
                'total_points': total_points,
                'time_span': time_span,
                'mean_snr': mean_snr
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = f"{args.output_dir}/lightcurve_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nProcessed {len(summary_rows)} lightcurves from {len(all_lightcurves)} transient types")
    print(f"Data saved in {args.output_dir}/lightcurves/")
    print(f"Summary file saved as {summary_path}")
    print(f"Plots saved in {args.output_dir}/plots/")

if __name__ == "__main__":
    main() 