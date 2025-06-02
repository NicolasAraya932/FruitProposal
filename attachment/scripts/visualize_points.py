import open3d as o3d
import viser
import numpy as np

# creating color vector black and white for each label
def create_color_vector(labels):
    color_map = {
        0: [1, 1, 1],  # black
        1: [255, 0, 0],  # white
    }
    return [color_map[label] for label in labels]
    
class VisualizePoints:
    def __init__(self, points, labels):
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