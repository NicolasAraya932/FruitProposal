import torch
import sys
sys.path.append('/workspace/RadianceFields')  # (Docker Version) Replace with the actual path to the scripts directory

from scripts import get_rgba_image

import open3d as o3d
import viser
import numpy as np
import pandas as pd
import argparse

# Define command-line arguments
parser = argparse.ArgumentParser(description="Visualize radiance field point cloud.")
parser.add_argument(
    "--file", 
    type=str, 
    required=True, 
    help="Path to the .pt file containing the radiance field data."
)

parser.add_argument(
    "--save-csv",
    action="store_true",
    help="To save point cloud as csv"
)

parser.add_argument(
    "--save-ply",
    action="store_true",
    help="To save point cloud as ply"
)

args = parser.parse_args()

data = torch.load(args.file)

print(data.keys())

points = data["points"]
rgbs = data["rgb"]
accumulation = data["accumulation"]
depth = data["depth"]
origins = data["origins"]
view_directions = data["view_directions"]
pixel_area = data["pixel_area"]

depth = data["depth"]

rgba = get_rgba_image(data, "rgb")

points = origins + view_directions * depth

# Filter points with opacity lower than 0.5
mask = rgba[..., -1] > 0.01
view_directions = view_directions[mask]
rgbs = rgba[mask][..., :3]

points = points[mask]
rgbs = rgbs[mask]
view_direction = view_directions[mask]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.double().numpy())
pcd.colors = o3d.utility.Vector3dVector(rgbs.double().numpy())

if args.save_csv:
    csv_data = points.numpy()
    csv_header = ["x", "y", "z"]
    csv_file = args.file.replace(".pt", "_point_cloud.csv")
    pd.DataFrame(csv_data, columns=csv_header).to_csv(csv_file, index=False)
    print(f"Point cloud saved to CSV: {csv_file}")

if args.save_ply:
    ply_file = args.file.replace(".pt", "_point_cloud.ply")
    o3d.io.write_point_cloud(ply_file, pcd)
    print(f"Point cloud saved to PLY: {ply_file}")

# Visualize with viser
ViserServer = viser.ViserServer()

# Add the point cloud to the viser server
ViserServer.scene.add_point_cloud(
    name="radiance_field",
    points=np.asarray(pcd.points),
    colors=np.asarray(pcd.colors),
    point_size=0.0005,
)


while True:
    continue