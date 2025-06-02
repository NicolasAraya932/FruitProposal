import open3d as o3d
import viser
import torch
import numpy as np
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from visualize_points import create_color_vector
from sklearn.cluster import DBSCAN

def cluster_and_filter(points: np.ndarray, eps: float, min_samples: int, min_cluster_size: int):
    """
    1) Run DBSCAN on `points` with (eps, min_samples).
    2) Throw away any resulting cluster whose size < min_cluster_size.
    Returns: a list of (cluster_label, indices), and a final boolean mask.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_  # -1 means noise
    unique_labels = set(labels)
    
    cluster_indices = []
    keep_mask = np.zeros(points.shape[0], dtype=bool)
    
    for lbl in unique_labels:
        if lbl == -1:
            continue  # skip DBSCAN noise
        idx = np.where(labels == lbl)[0]
        if len(idx) >= min_cluster_size:
            cluster_indices.append((lbl, idx))
            keep_mask[idx] = True
    
    return cluster_indices, keep_mask

def density_filter(points: np.ndarray, r: float, k: int):
    """
    Keep only points that have at least k neighbors within radius r.
    Returns a boolean mask and the filtered points.
    """
    nbrs = NearestNeighbors(radius=r, algorithm="auto").fit(points)
    idx_radius = nbrs.radius_neighbors(points, return_distance=False)
    # exclude self by subtracting 1
    neighbor_counts = np.array([len(idxs) - 1 for idxs in idx_radius])
    keep_mask = neighbor_counts >= k
    return keep_mask, points[keep_mask]

data = torch.load("/workspace/FruitProposal/attachment/RadianceCloud/semantic_radiance_field_20250529_014047.pt")

origins    = data["origins"]
directions = data["directions"]
labels     = data["semantic_labels"]
depth      = data["depth"]

points = origins + directions * depth

# Detele the 0 label points
mask = labels != 0
points = points[mask]
labels = labels[mask]

# 1) Compute nearest neighbor distances (k=5)
k = 5  # check k nearest neighbors
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)
distances, _ = nbrs.kneighbors(points)
# distances[:, 0] is zero (distance to itself), so take dist[:,1]...dist[:,k]
nearest_distances = distances[:, 1]  # nearest neighbor distance

# 2) Plot distribution of nearest neighbor distances
plt.figure(figsize=(8, 4))
plt.hist(nearest_distances, bins=50, color='green', alpha=0.7)
plt.title("Nearest Neighbor Distance Distribution")
plt.xlabel("Distance to 1st Nearest Neighbor (meters)")
plt.ylabel("Number of Points")
plt.show()

# Compute a candidate r as, for example, the 95th percentile of nearest distances
r_candidate = np.percentile(nearest_distances, 95)
print(f"Suggested radius r ≈ {r_candidate:.4f} (90th percentile)")

# 3) For a chosen radius r, compute neighbor counts for each point
r = r_candidate  # use the candidate radius
nbrs_radius = NearestNeighbors(radius=r, algorithm='auto').fit(points)
# radius_neighbors returns a list of neighbor indices for each point
indices_within_r = nbrs_radius.radius_neighbors(points, return_distance=False)
neighbor_counts = np.array([len(idxs) - 1 for idxs in indices_within_r])  # subtract self

# 4) Plot histogram of neighbor counts at radius r
plt.figure(figsize=(8, 4))
plt.hist(neighbor_counts, bins=50, color='blue', alpha=0.7)
plt.title(f"Neighbor Count Distribution within Radius r={r:.4f}")
plt.xlabel("Number of Neighbors (within r)")
plt.ylabel("Number of Points")
plt.show()

# 5) Suggest k as a small threshold above which points are likely in clusters
# For example, choose k as 10th percentile of neighbor counts
k_candidate = int(np.percentile(neighbor_counts, 10))
print(f"Suggested k ≈ {k_candidate} (10th percentile of neighbor counts)")

# Filter out points with fewer than k neighbors
min_neighbors = k_candidate
mask_keep = neighbor_counts >= min_neighbors
filtered_points = points[mask_keep]

print(f"Filtered out {np.sum(~mask_keep)} noise points; kept {np.sum(mask_keep)} points in clusters.")


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points.cpu().numpy())
colors = create_color_vector(labels[mask_keep].cpu().numpy())
pcd.colors = o3d.utility.Vector3dVector(colors)
# Visualize with viser
ViserServer = viser.ViserServer(port=9009)

# Add the point cloud to the viser server
ViserServer.scene.add_point_cloud(
    name="radiance_field",
    points=np.asarray(pcd.points),
    colors=np.asarray(pcd.colors),
    point_size=0.001,
)

while(True):
    x = input("Press Enter to continue...")
    if x == "":
        break
