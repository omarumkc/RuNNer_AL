import pickle
import os
import numpy as np

# Define the data directory and structures.pkl file
data_dir = "data"
structures_pkl_file = os.path.join(data_dir, "structures.pkl")

# Load the structures.pkl file
with open(structures_pkl_file, "rb") as f:
    structures_data = pickle.load(f)

# Extract disagreement values for each iteration
iteration_disagreements = {}

for conf_id, data in structures_data.items():
    for key in data:
        if key.startswith("iteration_") and "std_disagreement" in data[key]:
            if key not in iteration_disagreements:
                iteration_disagreements[key] = []
            iteration_disagreements[key].append(float(data[key]["std_disagreement"]))

# Compute the average value for each iteration
average_disagreements = {iteration: np.mean(values) for iteration, values in iteration_disagreements.items()}

# Print the results
for iteration, avg_value in sorted(average_disagreements.items()):
    print(f"{iteration}: {avg_value:.16f}")
