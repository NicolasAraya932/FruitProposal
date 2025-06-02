import torch
import matplotlib.pyplot as plt
import numpy as np

metrics_history = torch.load("metrics_history.pt")
print(f"Metrics history loaded with {len(metrics_history)} entries.")

# Ensure all values are floats (move to CPU if needed)
accuracies = [float(a.cpu().item() if hasattr(a, "cpu") else a) for a in metrics_history]

steps = np.arange(1, len(accuracies) + 1)

plt.figure(figsize=(8, 5))
plt.plot(steps, accuracies, label="Semantic Accuracy")
plt.xscale("log")
plt.xlabel("Iteration (log scale)")
plt.ylabel("Semantic Accuracy")
plt.title("Semantic Accuracy over Training (Log Scale)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig("semantic_accuracy_logscale.png")
plt.show()

print(accuracies[-1])
