import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import glob

if len(sys.argv) != 3:
    print("Usage: python3 cuda_analysis.py <logs_directory> <output_directory>")
    sys.exit(1)

logs_directory = sys.argv[1]
output_directory = sys.argv[2]

# Get all the log files in the directory
log_files = glob.glob(os.path.join(logs_directory, "*.csv"))

for log_file in log_files:
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
    total_time = gpu_data["Time (s)"].sum()

    # Get the number of threads/blocks from the file name
    num_threads_blocks = os.path.basename(log_file).split("_")[2]

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.bar(num_threads_blocks, total_time)
    ax.set_ylabel("Total Time (s)")
    ax.set_title(f"Total GPU activities time for {num_threads_blocks} threads/blocks")

    # Save the plot to the output directory
    plt.savefig(os.path.join(output_directory, f"{num_threads_blocks}.png"))
