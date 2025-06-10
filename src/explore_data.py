import numpy as np
import pandas as pd

# Load the dataset
data_path = "data/sim_samples.npz"
data = np.load(data_path, allow_pickle=True)

# Print dataset keys
print("Available keys in the dataset:")
print("-" * 30)
print(list(data.keys()))
print()

# Print sample counts for each key
print("Sample counts:")
print("-" * 30)
for key in data:
    print(f"{key}: {len(data[key])} samples")
print()

# Examine the structure of one sample from each key
print("Sample structure examples:")
print("-" * 30)
for key in data:
    if len(data[key]) > 0:
        sample = data[key][0]
        print(f"\nStructure of a {key} sample:")
        
        # Print the data type and shape
        print(f"Type: {type(sample)}")
        
        if isinstance(sample, np.ndarray):
            print(f"Shape: {sample.shape}")
            if sample.dtype.names:
                print("Fields:", sample.dtype.names)
                
                # Print a few rows as an example
                df = pd.DataFrame(sample)
                print("\nSample data (first 5 rows):")
                print(df.head())
        else:
            print("Content:", sample)
            
if __name__ == "__main__":
    pass 