import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(
    "output.txt", header=None, names=["Size", "Operation", "Array", "Time"]
)

# Convert 'Size' to numeric, errors='coerce' will replace non-numeric values with NaN
data["Size"] = pd.to_numeric(data["Size"], errors='coerce')

# Drop rows with NaN values in 'Size' column
data = data.dropna(subset=['Size'])

# Separate the data for the initialization and addition operations
init_data = data[data["Operation"] == "Initialization"]
add_data = data[data["Operation"] == "Addition"]

# Calculate mean and standard deviation for each size and array
init_means = init_data.groupby(["Size", "Array"])["Time"].mean().unstack()
init_stds = init_data.groupby(["Size", "Array"])["Time"].std().unstack()

add_means = add_data.groupby("Size")["Time"].mean()
add_stds = add_data.groupby("Size")["Time"].std()

# Plot the results
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

init_means.plot(kind="bar", yerr=init_stds, ax=ax[0])
ax[0].set_ylabel("Initialization time (s)")
ax[0].set_title("Initialization time for different array sizes")

add_means.plot(kind="bar", yerr=add_stds, ax=ax[1])
ax[1].set_ylabel("Addition time (s)")
ax[1].set_title("Addition time for different array sizes")

plt.tight_layout()
plt.savefig("output.png")
