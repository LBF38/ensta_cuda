import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

if len(sys.argv) != 3:
    print(f"Usage: python3 {sys.argv[0]} <logs_directory> <output_png>")
    sys.exit(1)

logs_directory = sys.argv[1]
output_png = sys.argv[2]

log_files = glob.glob(os.path.join(logs_directory, "*.csv"))

# all_data = pd.DataFrame()
all_data = []

print("Reading the log files...")
for log_file in tqdm(log_files):
    numbers = re.findall(r"\d+", log_file)
    array_size, grid_size, block_size, exp_num = map(int, numbers)
    array_size = int(array_size)
    grid_size = int(grid_size)
    block_size = int(block_size)
    exp_num = int(exp_num)

    with open(log_file, "r") as f:
        lines = f.readlines()
        headers = lines[4].strip().replace('"', "").split(",")
        units = lines[5].strip().split(",")

    data = pd.read_csv(
        log_file,
        skiprows=5,
    )

    data.columns = [f"{header} ({unit})" for header, unit in zip(headers, units)]

    # Calculate sum of GPU activities
    gpu_data = data[data["Type ()"] == "GPU activities"]
    print(gpu_data)
    gpu_sum = gpu_data["Time (us)"].sum() * 1e-6
    # gpu_sum = data["GPU activities"].sum()

    all_data.append(
        {
            "Array Size": array_size,
            "Grid Size": grid_size,
            "Block Size": block_size,
            "Experiment Number": exp_num,
            "GPU Sum": gpu_sum,
        }
    )

all_data = pd.DataFrame(all_data)

# Calculate mean of GPU Sum for each combination of parameters
mean_data = (
    all_data.groupby(["Array Size", "Grid Size", "Block Size"])["GPU Sum"]
    .mean()
    .reset_index()
)

print("Plotting the results...")
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    mean_data["Block Size"],
    mean_data["Array Size"],
    mean_data["Grid Size"],
    c=mean_data["GPU Sum"],
)
plt.colorbar(sc)
ax.set_xlabel("Block Size")
ax.set_ylabel("Array Size")
ax.set_label("Grid Size")
fig.suptitle("GPU results")

plt.savefig(output_png)
print(f"Plot saved to {output_png}")
