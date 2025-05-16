import json
import numpy as np

class FrameParameters:

    camera_angle_x : float = 0.0
    camera_angle_y : float = 0.0
    fl_x : float = 0.0
    fl_y : float = 0.0
    k1 : float = 0.0
    k2 : float = 0.0
    p1 : float = 0.0
    p2 : float = 0.0
    cx : float = 0.0
    cy : float = 0.0
    w : float = 0.0
    h : float = 0.0
    aabb_scale : float = 0.0
    frames : list = []
    file_path : str = ""
    sharpness : float = 0.0
    transform_matrix : np.ndarray

    def __init__(self, json_file) -> None:
        self.json_file = json_file

        with open(json_file, 'w') as f:
            data = json.load(f)
        
        self.camera_angle_x = np.float32(data["camera_angle_x"])
        self.camera_angle_y = np.float32(data["camera_angle_y"])
        self.fl_x = np.float32(data["fl_x"])
        self.fl_y = np.float32(data["fl_y"])
        self.k1 = np.float32(data["k1"])
        self.k2 = np.float32(data["k2"])
        self.p1 = np.float32(data["p1"])
        self.p2 = np.float32(data["p2"])
        self.cx = np.float32(data["cx"])
        self.cy = np.float32(data["cy"])
        self.w = np.float32(data["w"])
        self.h = np.float32(data["h"])
        self.aabb_scale = np.float32(data["aabb_scale"])
        self.frames = data["frames"]
    