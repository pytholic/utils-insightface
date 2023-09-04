import re

import matplotlib.pyplot as plt

# Read log entries from the file
log_filename = "training.log"
with open(log_filename, "r") as log_file:
    log_entries = log_file.read()

# Regular expression patterns to match "Loss" and "Global Step" entries
loss_pattern = re.compile(r"Loss ([\d.]+)")
step_pattern = re.compile(r"Global Step: (\d+)")

# Lists to store extracted data
loss_values = []
step_values = []

# Extract data from log entries
for line in log_entries.split("\n"):
    loss_match = loss_pattern.search(line)
    step_match = step_pattern.search(line)

    if loss_match and step_match:
        loss_values.append(float(loss_match.group(1)))
        step_values.append(int(step_match.group(1)))

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(step_values, loss_values, marker="o", color="blue", linewidth=0.01)
plt.title("Loss vs Global Step")
plt.xlabel("Global Step")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
