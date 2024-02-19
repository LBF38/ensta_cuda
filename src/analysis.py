import pandas as pd
import matplotlib.pyplot as plt
import sys

# Check command-line arguments
if len(sys.argv) != 3:
    print("Usage: python3 analysis.py <data_file> <output_png>")
    sys.exit(1)

# Load the data
data = pd.read_csv(
    sys.argv[1], header=None, names=["Size", "Operation", "Type", "Time"]
)

# Convert 'Size' to numeric, errors='coerce' will replace non-numeric values with NaN
data["Size"] = pd.to_numeric(data["Size"], errors="coerce")

# Drop rows with NaN values in 'Size' column
data = data.dropna(subset=["Size"])

# Separate the data for the initialization and addition operations
init_data = data[data["Operation"] == "Initialization"]
add_data = data[data["Operation"] == "Addition"]

# Calculate mean and standard deviation for each size and type
init_means = init_data.groupby(["Size", "Type"])["Time"].mean().unstack()
init_stds = init_data.groupby(["Size", "Type"])["Time"].std().unstack()

add_means = add_data.groupby(["Size", "Type"])["Time"].mean().unstack()
add_stds = add_data.groupby(["Size", "Type"])["Time"].std().unstack()

# Plot the results
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

init_means.plot(kind="bar", yerr=init_stds, ax=ax[0])
ax[0].set_ylabel("Initialization time (s)")
ax[0].set_title("Initialization time for different array sizes and types")

add_means.plot(kind="bar", yerr=add_stds, ax=ax[1])
ax[1].set_ylabel("Addition time (s)")
ax[1].set_title("Addition time for different array sizes and types")

# Save the plot to the specified PNG file
plt.savefig(sys.argv[2])
