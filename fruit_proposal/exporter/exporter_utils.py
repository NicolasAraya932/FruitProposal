# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Export utils such as structs, point cloud generation, and rendering code.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pymeshlab
import torch
from jaxtyping import Float
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from torch import Tensor

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.exporter.exporter_utils import *

if TYPE_CHECKING:
    # Importing open3d can take ~1 second, so only do it below if we actually
    # need it.
    import open3d as o3d


def generate_radiance_fields_cloud(
    pipeline: Pipeline,
    num_points: int = 3500000,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
    normal_output_name: Optional[str] = None,
    crop_obb: Optional[OrientedBox] = None,
) -> Dict[str, torch.Tensor]:
    """Generate a radiance field dataset from a NeRF model.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        crop_obb: Optional oriented bounding box to crop points.

    Returns:
        A dictionary containing all radiance field data.
    """

    # Initialize progress bar
    progress = Progress(
        TextColumn(":cloud: Computing Radiance Field :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )

    # Initialize lists to store outputs
    points = []
    rgbs = []
    accumulations = []
    depths = []
    origins = []
    directions = []
    pixel_areas = []
    normals = []
    view_directions = []

    with progress as progress_bar:
        task = progress_bar.add_task("Generating Radiance Field", total=num_points)
        while not progress_bar.finished:
            normal = None

            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_train(0)
                assert isinstance(ray_bundle, RayBundle)
                outputs = pipeline.model(ray_bundle)

            # Validate outputs
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)

            rgba = pipeline.model.get_rgba_image(outputs, rgb_output_name)
            depth = outputs[depth_output_name]

            if normal_output_name is not None:
                if normal_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {normal_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --normal_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                normal = outputs[normal_output_name]
                assert (
                    torch.min(normal) >= 0.0 and torch.max(normal) <= 1.0
                ), "Normal values from method output must be in [0, 1]"
                normal = (normal * 2.0) - 1.0

            point = ray_bundle.origins + ray_bundle.directions * depth
            view_direction = ray_bundle.directions

            # Filter points with opacity lower than 0.01
            mask = rgba[..., -1] > 0.01
            point = point[mask]
            view_direction = view_direction[mask]
            rgb = rgba[mask][..., :3]
            if normal is not None:
                normal = normal[mask]

            if crop_obb is not None:
                mask = crop_obb.within(point)
                point = point[mask]
                rgb = rgb[mask]
                view_direction = view_direction[mask]
                if normal is not None:
                    normal = normal[mask]

            # Append data to lists
            points.append(point.cpu())
            rgbs.append(rgb.cpu())
            accumulations.append(outputs["accumulation"][mask].cpu())
            depths.append(outputs["depth"][mask].cpu())
            origins.append(ray_bundle.origins[mask].cpu())
            directions.append(ray_bundle.directions[mask].cpu())
            pixel_areas.append(ray_bundle.pixel_area[mask].cpu())
            view_directions.append(view_direction.cpu())

            if normal is not None:
                normals.append(normal.cpu())

            progress.advance(task, point.shape[0])

    # Combine lists into tensors on the CPU
    radiance_field_data = {
        "points": torch.cat(points, dim=0),
        "rgb": torch.cat(rgbs, dim=0),
        "accumulation": torch.cat(accumulations, dim=0),
        "depth": torch.cat(depths, dim=0),
        "origins": torch.cat(origins, dim=0),
        "directions": torch.cat(directions, dim=0),
        "pixel_area": torch.cat(pixel_areas, dim=0),
        "view_directions": torch.cat(view_directions, dim=0),
    }

    if normals:
        radiance_field_data["normals"] = torch.cat(normals, dim=0)

    return radiance_field_data

def generate_fruit_proposal_radiance_cloud(
    pipeline: Pipeline,
    num_points: int = 3500000,
    semantic_output_name: str = "semantic_labels",
    depth_output_name: str = "depth",
    normal_output_name: Optional[str] = None,
    crop_obb: Optional[OrientedBox] = None,
) -> Dict[str, torch.Tensor]:
    """Generate a radiance field dataset from a NeRF model.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        semantic_output_name: Name of the semantic output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        crop_obb: Optional oriented bounding box to crop points.

    Returns:
        A dictionary containing all radiance field data.
    """

    # Initialize progress bar
    progress = Progress(
        TextColumn(":cloud: Computing Radiance Field :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )

    points          = []
    accumulations   = []
    depths          = []
    origins         = []
    directions      = []
    view_directions = []
    normals         = []

    with progress as progress_bar:
        task = progress_bar.add_task("Generating Radiance Field", total=num_points)
        while not progress_bar.finished:
            normal = None

            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_train(0)
                assert isinstance(ray_bundle, RayBundle)
                outputs = pipeline.model(ray_bundle)

            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)

            if semantic_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {semantic_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --semantic_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)

            depth = outputs[depth_output_name]
            semantic_labels = outputs[semantic_output_name]

            print(semantic_labels)

            if normal_output_name is not None:
                if normal_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {normal_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --normal_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                normal = outputs[normal_output_name]
                assert (
                    torch.min(normal) >= 0.0 and torch.max(normal) <= 1.0
                ), "Normal values from method output must be in [0, 1]"
                normal = (normal * 2.0) - 1.0

            point = ray_bundle.origins + ray_bundle.directions * depth
            view_direction = ray_bundle.directions

            if normal is not None:
                normal = normal[mask]

            if crop_obb is not None:
                mask = crop_obb.within(point)
                point = point[mask]
                view_direction = view_direction[mask]
                if normal is not None:
                    normal = normal[mask]

            # Append data to lists
            points.append(point.cpu())
            accumulations.append(outputs["accumulation"][mask].cpu())
            depths.append(outputs["depth"][mask].cpu())
            origins.append(ray_bundle.origins[mask].cpu())
            directions.append(ray_bundle.directions[mask].cpu())
            view_directions.append(view_direction.cpu())

            if normal is not None:
                normals.append(normal.cpu())

            progress.advance(task, point.shape[0])

    # Combine lists into tensors on the CPU
    radiance_field_data = {
        "points": torch.cat(points, dim=0),
        "accumulation": torch.cat(accumulations, dim=0),
        "depth": torch.cat(depths, dim=0),
        "origins": torch.cat(origins, dim=0),
        "directions": torch.cat(directions, dim=0),
        "view_directions": torch.cat(view_directions, dim=0),
    }

    if normals:
        radiance_field_data["normals"] = torch.cat(normals, dim=0)

    return radiance_field_data
