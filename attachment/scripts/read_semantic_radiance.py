import open3d as o3d
import viser
import torch
import numpy as np

data = torch.load("/workspace/FruitProposal/attachment/RadianceCloud/eval/semantic_radiance_field_20250602_191008.pt")

print(data.keys())

origins    = data["origins"]
directions = data["directions"]
labels     = data["semantic_labels"]
depth      = data["depth"]

points = origins + directions * depth


# Detele the 0 label points
mask = labels != 0
points = points[mask]
labels = labels[mask]

# creating color vector black and white for each label
def create_color_vector(labels):
    color_map = {
        0: [1, 1, 1],  # black
        1: [255, 0, 0],  # white
    }
    return [color_map[label] for label in labels]

# Plot points with open3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
colors = create_color_vector(labels.cpu().numpy())
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize with viser
ViserServer = viser.ViserServer()

# Add the point cloud to the viser server
ViserServer.scene.add_point_cloud(
    name="radiance_field",
    points=np.asarray(pcd.points),
    colors=np.asarray(pcd.colors),
    point_size=0.001,
)

while(True):
    continue
    