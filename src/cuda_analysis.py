import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 3:
    print("Usage: python3 cuda_analysis.py <data_file> <output_png>")
    sys.exit(1)

with open(sys.argv[1], "r") as f:
    lines = f.readlines()
    headers = lines[4].strip().replace('"', "").split(",")
    units = lines[5].strip().split(",")

data = pd.read_csv(
    sys.argv[1],
    skiprows=5,
)

data.columns = [f"{header} ({unit})" for header, unit in zip(headers, units)]


gpu_data = data[data["Type ()"] == "GPU activities"]


print(data)
print(gpu_data)

# fig, ax = plt.subplots(figsize=(10, 10))

# gpu_data.plot(kind="bar", x="Name", y="Time", ax=ax)

# ax.set_ylabel("Time")
# ax.set_title("GPU activities time for different operations")

# plt.savefig(sys.argv[2])
