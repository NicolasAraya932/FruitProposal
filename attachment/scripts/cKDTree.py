import torch
from scipy.spatial import cKDTree

# Build a KD-tree for efficient nearest-neighbor search
cherry_tree = cKDTree(points.cpu().numpy())  # Cherry radiance field points

# Define a distance threshold (e.g., 0.05)
distance_threshold = 0.05

# Query the KD-tree for each point in the full scene
distances, _ = cherry_tree.query(_full_scene_points.cpu().numpy(), k=1)

# Assign labels based on the distance threshold
labels = torch.tensor((distances < distance_threshold).astype(int), device=_full_scene_points.device)

# `labels` is a binary tensor where 1 indicates cherry points and 0 indicates non-cherry points
print(f"Number of cherry points: {labels.sum().item()}")
print(f"Number of non-cherry points: {(labels == 0).sum().item()}")