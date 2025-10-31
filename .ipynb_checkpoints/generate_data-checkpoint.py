# generate_data.py

import numpy as np
import pandas as pd
import os

# --- Define the Ground-Truth Parameters for the Data ---
# Based on the statistical analysis for the Monte Carlo histogram.
NUM_SAMPLES = 1000
MU = 25.8  # Mean jitter in fs
SIGMA = 0.9 # Standard deviation in fs

# --- Generate the Data ---
# Set a seed for reproducibility, so the "random" data is always the same.
np.random.seed(42) 
jitter_data = np.random.normal(loc=MU, scale=SIGMA, size=NUM_SAMPLES)

# --- Format and Save to CSV ---
# Create a pandas DataFrame for easy saving
df = pd.DataFrame(jitter_data, columns=['jitter_fs'])

# Define the output path
output_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, 'monte_carlo_jitter.csv')

# Save the DataFrame to a CSV file
df.to_csv(output_path, index=False, float_format='%.4f')

print(f"Successfully generated and saved {len(df)} samples to '{output_path}'")
print("\n--- Data Statistics ---")
print(f"Mean: {df['jitter_fs'].mean():.4f}")
print(f"Std Dev: {df['jitter_fs'].std():.4f}")
