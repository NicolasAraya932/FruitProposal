import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Load the saved losses
loss_file = "losses.npy"
losses = np.load(loss_file, allow_pickle=True).item()

# Plot each loss component
plt.figure(figsize=(10, 6))
for key, values in losses.items():
    plt.plot(values, label=key)

plt.xlabel("Training Iterations")
plt.ylabel("Loss Value")
plt.title("Loss Components Over Training")
plt.legend()
plt.grid()
plt.show()
