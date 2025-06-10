import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import UnivariateSpline
import json

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare lightcurve data for symbolic regression.')
    parser.add_argument('--data_dir', type=str, default="data/processed/lightcurves", 
                        help='Directory with processed lightcurve CSV files')
    parser.add_argument('--model_types', type=str, nargs='+', 
                        default=["SNIa-SALT2", "SNIc-Templates", "SLSN-I+host", "TDE", "KN_K17", "ILOT", "PISN-MOSFIT", "SNIax"],
                        help='List of transient model types to process')
    parser.add_argument('--specific_files', type=str, nargs='*', 
                        help='Specific CSV files to process (relative to data_dir)')
    parser.add_argument('--output_dir', type=str, default="data/processed/symbolic_regression_input", 
                        help='Directory to save preprocessed data for symbolic regression')
    parser.add_argument('--normalize', action='store_true', 
                        help='Whether to normalize the flux values to [0,1]')
    parser.add_argument('--smooth', action='store_true', 
                        help='Whether to smooth the data using spline interpolation')
    parser.add_argument('--spline_s', type=float, default=0.1, 
                        help='Smoothing factor for spline interpolation (0=no smoothing)')
    return parser.parse_args()

def find_lightcurve_files(data_dir, model_types, specific_files=None):
    """Find all lightcurve CSV files or specific ones if provided."""
    if specific_files:
        return [os.path.join(data_dir, file) for file in specific_files]
    
    all_files = []
    for model_type in model_types:
        model_dir = os.path.join(data_dir, model_type)
        if os.path.exists(model_dir):
            files = glob.glob(os.path.join(model_dir, "*.csv"))
            all_files.extend(files)
    
    return all_files

def preprocess_lightcurve(df, normalize=False, smooth=False, spline_s=0.1):
    """Preprocess a lightcurve dataframe for symbolic regression."""
    # Extract basic data
    time = df['time'].values
    flux = df['flux'].values
    flux_err = df['flux_err'].values
    
    # Only keep finite values
    valid_idx = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    time = time[valid_idx]
    flux = flux[valid_idx]
    flux_err = flux_err[valid_idx]
    
    # Ensure time is sorted
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    flux = flux[sort_idx]
    flux_err = flux_err[sort_idx]
    
    # Normalize flux if requested
    if normalize:
        # Scale flux to [0,1] range
        scaler = MinMaxScaler()
        flux = scaler.fit_transform(flux.reshape(-1, 1)).flatten()
        
        # Scale errors proportionally
        max_orig = np.max(df['flux'].values) - np.min(df['flux'].values)
        flux_err = flux_err / max_orig
    
    # Smooth data if requested
    if smooth and len(time) > 3:
        # Create a spline with appropriate smoothing
        weights = 1.0 / (flux_err + 1e-10)  # Use inverse of error as weights
        spline = UnivariateSpline(time, flux, w=weights, s=spline_s)
        
        # Generate smoothed flux
        flux_smooth = spline(time)
        
        # Only use smoothed values if they make sense
        if np.all(np.isfinite(flux_smooth)):
            flux = flux_smooth
    
    return time, flux, flux_err

def plot_preprocessed(time, flux, flux_err, time_orig=None, flux_orig=None, 
                      title="Preprocessed Lightcurve", save_path=None):
    """Plot the preprocessed lightcurve data."""
    plt.figure(figsize=(12, 6))
    
    # Plot original data if provided
    if time_orig is not None and flux_orig is not None:
        plt.plot(time_orig, flux_orig, 'o', alpha=0.5, color='gray', label='Original data')
    
    # Plot preprocessed data
    plt.errorbar(time, flux, yerr=flux_err, fmt='o-', color='blue', 
                 label='Preprocessed data', capsize=3)
    
    plt.title(title)
    plt.xlabel("Time (days from first observation)")
    plt.ylabel("Flux")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)
    
    # Find all lightcurve files
    lightcurve_files = find_lightcurve_files(args.data_dir, args.model_types, args.specific_files)
    print(f"Found {len(lightcurve_files)} lightcurve files to process")
    
    # Process each file
    metadata = []
    
    for file_path in lightcurve_files:
        # Extract file name and model information
        file_name = os.path.basename(file_path)
        dir_name = os.path.basename(os.path.dirname(file_path))
        
        print(f"Processing {dir_name}/{file_name}...")
        
        # Load the data
        df = pd.read_csv(file_path)
        
        # Preprocess
        time, flux, flux_err = preprocess_lightcurve(
            df, 
            normalize=args.normalize,
            smooth=args.smooth,
            spline_s=args.spline_s
        )
        
        # Create output name
        base_name = os.path.splitext(file_name)[0]
        output_base = f"{dir_name}_{base_name}"
        
        # Save preprocessed data
        output_df = pd.DataFrame({
            'time': time,
            'flux': flux,
            'flux_err': flux_err,
            'original_time': df['original_time'].values[np.argsort(df['time'].values)]
        })
        
        data_path = os.path.join(args.output_dir, "data", f"{output_base}.csv")
        output_df.to_csv(data_path, index=False)
        
        # Plot the preprocessed data
        plot_path = os.path.join(args.output_dir, "plots", f"{output_base}.png")
        
        plot_preprocessed(
            time, flux, flux_err,
            time_orig=df['time'].values, 
            flux_orig=df['flux'].values,
            title=f"{dir_name} - {base_name}",
            save_path=plot_path
        )
        
        # Extract metadata
        num_points = len(time)
        time_span = np.max(time) - np.min(time)
        mean_snr = np.mean(np.abs(flux) / (flux_err + 1e-10))
        
        metadata.append({
            'model_type': dir_name,
            'file_name': file_name,
            'output_file': f"{output_base}.csv",
            'plot_file': f"{output_base}.png",
            'num_points': num_points,
            'time_span': time_span,
            'mean_snr': mean_snr,
            'preprocessing': {
                'normalized': args.normalize,
                'smoothed': args.smooth,
                'spline_s': args.spline_s if args.smooth else None
            }
        })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)
    
    # Save metadata as JSON too for easier reading
    with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nPreprocessing completed for {len(metadata)} lightcurves")
    print(f"Results saved to {args.output_dir}")
    print(f"Metadata saved as {os.path.join(args.output_dir, 'metadata.csv')}")

if __name__ == "__main__":
    main() 