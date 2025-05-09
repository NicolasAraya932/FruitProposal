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
Semantic NeRF-W implementation which should be fast enough to view in the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Literal

import sys
import numpy as np
import torch
from torch.nn import Parameter
from torch.nn import CrossEntropyLoss

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig

from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from fruit_proposal.fruit_proposal_field import FruitProposalField
from nerfstudio.model_components.losses import distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    SemanticRenderer,
    DepthRenderer
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

from nerfstudio.utils.rich_utils import CONSOLE

@dataclass
class FruitProposalModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: FruitProposalModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""

    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 1024
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""

    num_semantic_classes: int = 2
    """Number of semantic classes."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""

    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""

    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""

    average_init_density: float = 1.0
    """Average initial density output from MLP. """

    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""

class FruitProposalModel(Model):
    """Semantic NeRF model for binary semantics and density."""

    config: FruitProposalModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # Initialize the field
        self.field = FruitProposalField(
            aabb = self.scene_box.aabb,
            num_levels = self.config.num_levels,
            base_res = self.config.base_res,
            max_res = self.config.max_res,
            log2_hashmap_size = self.config.log2_hashmap_size,
            features_per_level = self.config.features_per_level,
            average_init_density = self.config.average_init_density,
            implementation = self.config.implementation
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    **prop_net_args,
                    average_init_density=self.config.average_init_density,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )
        # Collider
        self.collider = NearFarCollider(
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
        ) 
        
        # Renderers
        self.renderer_semantics   = SemanticRenderer()
        self.renderer_depth       = DepthRenderer()

        # Losses
        self.semantic_loss = CrossEntropyLoss() # Default reduction="mean")
        self.interlevel_loss = interlevel_loss

        import matplotlib.pyplot as plt
        
        # Initialize colormap using matplotlib
        cmap = plt.get_cmap("viridis", self.config.num_semantic_classes)
        self.colormap = torch.tensor(cmap.colors, dtype=torch.float32)

        # # Print dtype of every tensor in the class
        # for name, param in self.named_parameters():
        #     CONSOLE.print(f"{name} is {param.dtype}")

    def get_outputs(self, ray_bundle: RayBundle):
        """Compute outputs for semantics only."""
        
        outputs = {}
        # Sample points along rays using the proposal sampler
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )
    
        # Compute field outputs
        field_outputs = self.field(ray_samples)
    
        # Compute density weights
        weights_static = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights_static)
        # Renderers
        
        # Render depth
        depth = self.renderer_depth(weights=weights_static, ray_samples=ray_samples)

        # Render semantics
        semantic_weights = weights_static
        semantics = self.renderer_semantics(
            field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights
        )

        # Apply colormap for visualization
        semantic_labels = torch.argmax(torch.nn.functional.softmax(semantics, dim=-1), dim=-1)

        #print(semantic_labels.shape, semantic_labels[:10], semantics[:10])
        semantics_colormap = self.colormap.to(self.device)[semantic_labels]
    
        # Return only semantics-related outputs
        outputs = {
            "semantics": semantics,
            "semantic_labels": semantic_labels,
            "semantics_colormap": semantics_colormap,
            "depth": depth,
            "weights_list": weights_list
        }

        # print(semantics)

        return outputs


    def get_loss_dict(self, outputs, batch, metrics_dict=None):

        loss_dict = {}

        # Predictions
        pred_logits = outputs["semantics"]  # [N_rays, num_classes]
        pred_logits = torch.clamp(pred_logits, min=-3.8, max=7)

        # Convert summed_rgb to binary: 1 if non-zero, 0 otherwise
        binary_values = batch["binary_mask"][:,:3].to(self.device)
        summed_rgb = binary_values.sum(dim=-1)
        binary_mask = (summed_rgb != 0).long()

        assert torch.all((binary_mask == 0) | (binary_mask == 1)), "Ground truth cannot be interpreted as binary mask"
        gt_sem = binary_mask

        # Get ray_indices (if available)
        N_rays = pred_logits.shape[0]
        ray_indices = batch.get("ray_indices", torch.arange(N_rays, device=self.device))
        gt_labels = gt_sem[ray_indices]  # [N_rays]

        # Cross entropy loss
        loss_dict["semantic_loss"] = self.semantic_loss(pred_logits, gt_labels)

        return loss_dict


    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}


        pred_logits = outputs["semantics"]  # [N_rays, num_classes]
        pred_logits = torch.clamp(pred_logits, min=-3.8, max=7)
        N_rays = pred_logits.shape[0]

        rgb_values = batch["image"][:,:3].to(self.device)
        summed_rgb = rgb_values.sum(dim=-1)

        # Convert summed_rgb to binary: 1 if non-zero, 0 otherwise
        binary_mask = (summed_rgb != 0).long()
        
        assert torch.all((binary_mask == 0) | (binary_mask == 1)), "Ground truth cannot be interpreted as binary mask"

        gt_sem = binary_mask

        # Get ray_indices (if available)
        ray_indices = batch.get("ray_indices", torch.arange(N_rays, device=self.device))

        gt_labels = gt_sem[ray_indices]  # [N_rays]

        # Predictions
        pred_labels = torch.argmax(torch.nn.functional.softmax(pred_logits, dim=-1), dim=-1) # [N_rays]

        metrics_dict["semantic_accuracy"] = (pred_labels == gt_labels).float().mean()
        # To print the accuracy
        # print(f"Semantic accuracy: {metrics_dict['semantic_accuracy'].item()}")

        return metrics_dict


    def get_image_metrics_and_images(self, outputs, batch):
        """
        Return:
          - {} for scalar image‐metrics
          - {"semantics": ..., ...} for visuals
        """
        images_dict: Dict[str, torch.Tensor] = {}

        # Colorize the predicted semantics
        pred_logits = outputs["semantics"]                       # [B*H*W? or N_rays]
        # If we can reshape back to (B,H,W,2), do so; else assume H=W=?? for display
        # Here, we expect get_outputs produced [B,H,W,2]:
        if pred_logits.ndim == 2:
            # can't visualize per‐pixel grid if it's per‐ray only
            return {}, {}
        sem_logits = pred_logits                                 # [B,H,W,2]
        pred_labels = sem_logits.argmax(dim=-1).cpu().numpy()    # [B,H,W]

        # Apply a colormap for TensorBoard/viewer
        sem_vis = colormaps.apply_colormap(pred_labels)          # [B,H,W,3] uint8
        images_dict["semantics"] = sem_vis

        return {}, images_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """
        Return a dict mapping group names to lists of Parameters.
        Nerfstudio will merge this with the DataManager param groups.
        """
        param_groups: Dict[str, List[Parameter]] = {}

        # 1) Proposal networks
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())

        # 2) The semantic‐only field
        param_groups["fields"] = list(self.field.parameters())

        # 3) (Optional) any other modules you added, e.g. camera optimizer
        # param_groups["camera_optimizer"] = list(self.camera_optimizer.parameters())

        return param_groups