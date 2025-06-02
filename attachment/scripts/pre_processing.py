import open3d as o3d
import viser
import torch
import numpy as np
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from visualize_points import create_color_vector, VisualizePoints
from sklearn.cluster import DBSCAN

from nerfstudio.data.scene_box import OrientedBox

def ritters_enclosing_sphere(points):
    """
    Compute a near-minimal bounding sphere using Ritter's algorithm (two passes).
    
    Args:
        points (np.ndarray): (N,3) array of cluster points.
    Returns:
        center (np.ndarray): shape (3,), center of the sphere.
        radius (float): sphere radius.
    """
    # 1) Pick an arbitrary point p0
    p0 = points.mean(axis=0)  # use the centroid as a good initial point
    # 2) Find pA = farthest from p0
    dists0 = np.linalg.norm(points - p0[None, :], axis=1)
    pA = points[dists0.argmax()]
    # 3) Find pB = farthest from pA
    distsA = np.linalg.norm(points - pA[None, :], axis=1)
    pB = points[distsA.argmax()]
    # Initial sphere: center = midpoint, radius = half distance
    center = (pA + pB) / 2.0
    R = np.linalg.norm(pB - pA) / 2.0
    
    # 4) Grow sphere to include any outside points
    for p in points:
        d = np.linalg.norm(p - center)
        if d > R:
            # point is outside, grow sphere
            new_R = (R + d) / 2.0
            # shift center toward p
            center = center + ((d - R) / (2 * d)) * (p - center)
            R = new_R
    return center, R

def sphere_to_bbox(center, radius):
    # 2) Axis-aligned min/max corners
    x_min = center[0] - radius
    x_max = center[0] + radius
    y_min = center[1] - radius
    y_max = center[1] + radius
    z_min = center[2] - radius
    z_max = center[2] + radius

    # 3) List the 8 box corners
    corners = np.array([
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max],
    ])

    mask = np.array([x_min, x_max, y_min, y_max, z_min, z_max])
    return corners, mask

def bbox_edges_from_corners(corners):
    # 12 edges of a box, each as a pair of indices into the corners array
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]
    return np.array([[corners[i], corners[j]] for i, j in edges])

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
radiance_field = torch.load("/workspace/FruitProposal/attachment/RadianceCloud/radiance_field_20250602_125748.pt")
print(radiance_field.keys())

origins    = data["origins"]
directions = data["directions"]
labels     = data["semantic_labels"]
depth      = data["depth"]

radiance_field_origins = radiance_field["origins"]
radiance_field_directions = radiance_field["directions"]
radiance_field_rgb = radiance_field["rgb"]
radiance_field_depth = radiance_field["depth"]


points = origins + directions * depth
radiance_field_points = radiance_field_origins + radiance_field_directions * radiance_field_depth

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
filtered_labels = labels[mask_keep]

print(f"Filtered out {np.sum(~mask_keep)} noise points; kept {np.sum(mask_keep)} points in clusters.")

# Example usage:
filtered_pts = filtered_points # from density_filter above
min_samples = 10      # DBSCAN’s “core point” requirement (often same as k)
min_cluster_size = 10 # any cluster <20 points is too small to be a fruit

clusters, keep_clusters_mask = cluster_and_filter(filtered_pts, r_candidate, min_samples, min_cluster_size)
print(f"Found {len(clusters)} clusters of size ≥ {min_cluster_size}")

all_corners = []

for cidx, idx in clusters:
    # idx is an array of indices into filtered_pts
    cluster_points = filtered_pts[idx]          # <-- use filtered_pts here
    cluster_labels = filtered_labels[idx]       # <-- likewise for labels

    # compute bounding sphere (center, radius)
    center, radius = ritters_enclosing_sphere(cluster_points)

    # convert to 8 AABB corners
    corners, _ = sphere_to_bbox(center, radius + 1e-3)

    # Saving as points

    all_corners.append(corners)

# Concatenate all corners into a single array
all_corners = np.vstack(all_corners)


all_colors_corners = np.zeros((all_corners.shape[0], 3))  # black (0,0,0)

viser = viser.ViserServer(port=9009)

pcd_bbox = o3d.geometry.PointCloud()
pcd_bbox.points = o3d.utility.Vector3dVector(all_corners)
pcd_bbox.colors = o3d.utility.Vector3dVector(all_colors_corners)
viser.scene.add_point_cloud(
    name="bounding_boxes",
    points=np.asarray(pcd_bbox.points),
    colors=np.asarray(pcd_bbox.colors),
    point_size=0.001,
)

# After you have center and radius for each cluster:
for cidx, idx in clusters:
    cluster_points = filtered_pts[idx]
    center, radius = ritters_enclosing_sphere(cluster_points)

    corners, mask = sphere_to_bbox(center, radius + 1e-3)

    rf_pts = radiance_field_points  # (N_rf, 3)
    rf_rgb = radiance_field_rgb     # same length: (N_rf, 3) or (N_rf, 4)

    mask_in_box = (
        (rf_pts[:, 0] >= mask[0]) & (rf_pts[:, 0] <= mask[1]) &
        (rf_pts[:, 1] >= mask[2]) & (rf_pts[:, 1] <= mask[3]) &
        (rf_pts[:, 2] >= mask[4]) & (rf_pts[:, 2] <= mask[5])
    )
    # Keep only those radiance‐field points/rgb that lie inside this box
    inside_pts = rf_pts[mask_in_box]
    inside_rgb = rf_rgb[mask_in_box]

    pcd_radiance = o3d.geometry.PointCloud()
    pcd_radiance.points = o3d.utility.Vector3dVector(inside_pts.cpu().numpy())
    pcd_radiance.colors = o3d.utility.Vector3dVector(inside_rgb.cpu().numpy())

    viser.scene.add_point_cloud(
        name=f"cluster_{cidx}_points",
        points=np.asarray(pcd_radiance.points),
        colors=np.asarray(pcd_radiance.colors),
        point_size=0.001,
    )


fruit_pts = filtered_pts[keep_clusters_mask]
fruit_labels = labels[mask_keep][keep_clusters_mask]


pcd_filtered = o3d.geometry.PointCloud()
pcd_filtered.points = o3d.utility.Vector3dVector(fruit_pts.cpu().numpy())
colors = create_color_vector(fruit_labels.cpu().numpy())
pcd_filtered.colors = o3d.utility.Vector3dVector(colors)

viser.scene.add_point_cloud(
    name="filtered_points",
    points=np.asarray(pcd_filtered.points),
    colors=np.asarray(pcd_filtered.colors),
    point_size=0.001,
    visible=False
)



while True:
    x = input("Press Enter to continue...")
    if x == "":
        break


# VisualizePoints(fruit_pts, fruit_labels, port=9009, wait=False)
# VisualizePoints(filtered_points, labels[mask_keep], port=9099, wait=False)
# VisualizePoints(points, labels, port=7007)
