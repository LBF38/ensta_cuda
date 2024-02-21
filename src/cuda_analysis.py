import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

if len(sys.argv) != 3:
    print("Usage: python3 cuda_analysis.py <logs_directory> <output_png>")
    sys.exit(1)

logs_directory = sys.argv[1]
output_png = sys.argv[2]

log_files = glob.glob(os.path.join(logs_directory, "*.csv"))
total_times = {}

print("Reading the log files...")
for log_file in tqdm(log_files):
    with open(log_file, "r") as f:
        lines = f.readlines()
        headers = lines[4].strip().replace('"', "").split(",")
        units = lines[5].strip().split(",")

    data = pd.read_csv(
        log_file,
        skiprows=5,
    )

    data.columns = [f"{header} ({unit})" for header, unit in zip(headers, units)]

    # Filter the GPU activities and sum the time
    gpu_data = data[data["Type ()"] == "GPU activities"]
    total_time = gpu_data["Time (us)"].sum() * 1e-6

    # Get the number of threads/blocks from the file name
    num_threads_blocks = os.path.basename(log_file).split("_")[2]

    # Store the total time in the dictionary
    total_times[num_threads_blocks] = total_time

total_times = {k: total_times[k] for k in sorted(total_times, key=int)}

print("Plotting the results...")
fig, ax = plt.subplots(figsize=(10, 10))
ax.bar(total_times.keys(), total_times.values())
ax.set_ylabel("Total Time (s)")
ax.set_title("Total GPU activities time for different numbers of threads/blocks")

plt.savefig(output_png)
print(f"Plot saved to {output_png}")
