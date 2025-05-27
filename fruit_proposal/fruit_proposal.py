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
from typing      import (Dict, List, Tuple, Type, Literal, Optional)

import sys
import numpy as np
import torch
from torch.nn import Parameter
from torch.nn import CrossEntropyLoss

from nerfstudio.cameras.rays              import RayBundle, RaySamples
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig

from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)

from fruit_proposal.callbacks.FruitProposalCallback import FruitEarlyStopCallback

from nerfstudio.field_components.field_heads         import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields                import HashMLPDensityField
from nerfstudio.fields.nerfacto_field                import NerfactoField
from fruit_proposal.fruit_proposal_field             import FruitProposalField

from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers    import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
    SemanticRenderer,
)
from nerfstudio.model_components.shaders         import NormalsShader
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model                import Model, ModelConfig

from nerfstudio.utils                            import colormaps

from nerfstudio.utils.rich_utils import CONSOLE

@dataclass
class FruitProposalModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: FruitProposalModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "black"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
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
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_appearance_embedding: bool = True
    """Whether to use an appearance embedding."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""
    average_init_density: float = 1.0
    """Average initial density output from MLP. """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""

class FruitProposalModel(Model):
    """Semantic NeRF model for binary semantics and density."""

    config: FruitProposalModelConfig
    to_save: bool = False
    last_semantic_labels: Dict = None

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))
        
        """
        NERFACTO FIELD TO PROCESS FREEZEN RGB + DENSITY
        """
        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0
        self.nerfacto_field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
        )

        """
        FRUIT PROPOSAL FIELD
        """
        self.fruit_proposal_field = FruitProposalField(
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
                spatial_distortion=scene_contraction,
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
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    average_init_density=self.config.average_init_density,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            if self.step >= 800 and self.step < 810:
                self.to_save = True
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
        self.collider  = NearFarCollider(
            near_plane = self.config.near_plane,
            far_plane  = self.config.far_plane,
        ) 
        
        # Renderers
        self.renderer_rgb = RGBRenderer()
        self.renderer_normals = NormalsRenderer()
        self.normals_shader  = NormalsShader()

        self.renderer_semantics      = SemanticRenderer()
        self.renderer_accumulation   = AccumulationRenderer()
        self.renderer_depth          = DepthRenderer()

        # Losses
        self.semantic_loss   = CrossEntropyLoss() # Default reduction="mean")
        self.interlevel_loss = interlevel_loss
        self.step = 0

        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image      import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_param_groups(self):
        param_groups = {}
        param_groups["proposal_networks"]       = list(self.proposal_networks.parameters())
        param_groups["fields"]   = list(self.fruit_proposal_field.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks


    def get_outputs(self, ray_bundle: RayBundle):
        outputs = {}

        # 1) camera refinement
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)

        # 2) sample proposals (same geometry for both)
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )

        # 3) NerfactoField: first get density + feature embedding
        density, geo_feats = self.nerfacto_field.get_density(ray_samples)
        # compute the scalar weights from that density
        nerf_weights = ray_samples.get_weights(density)

        # now get the RGB (and any other heads) by passing the embedding back in
        nerf_out = self.nerfacto_field.get_outputs(
            ray_samples, density_embedding=geo_feats
        )
        rgb = self.renderer_rgb(
            rgb=nerf_out[FieldHeadNames.RGB], weights=nerf_weights
        )
        depth = self.renderer_depth(weights=nerf_weights, ray_samples=ray_samples)
        acc   = self.renderer_accumulation(weights=nerf_weights)

        outputs.update({
            "rgb": rgb,
            "depth": depth,
            "accumulation": acc,
        })

        # 4) FruitProposalField: reuse identical ray_samples + geo_feats
        sem_out = self.fruit_proposal_field.get_outputs(
            ray_samples, density_embedding=geo_feats
        )
        logits = sem_out[FieldHeadNames.SEMANTICS]
        semantics = self.renderer_semantics(logits, weights=nerf_weights)
        labels = torch.argmax(torch.nn.functional.softmax(semantics, dim=-1), dim=-1)

        outputs.update({
            "semantics": semantics,
            "semantic_labels": labels,
        })

        # 5) training‐only bookkeeping
        if self.training:
            outputs.update({
                "proposal_weights_list": weights_list + [nerf_weights],
                "ray_samples_list": ray_samples_list + [ray_samples],
            })

        return outputs



    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """
        Compute the loss for the model.
        """

        loss_dict = {}

        # Predictions
        pred_logits = outputs["semantics"]  # [N_rays, num_classes]
        pred_logits = torch.clamp(pred_logits, min=-3.8, max=7)

        # Convert summed_rgb to binary: 1 if non-zero, 0 otherwise
        binary_values = batch["binary_img"][:,:3].to(self.device)
        summed_rgb = binary_values.sum(dim=-1)
        binary_img = (summed_rgb != 0).long()

        assert torch.all((binary_img == 0) | (binary_img == 1)), "Ground truth cannot be interpreted as binary img"
        gt_sem = binary_img

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

        rgb_values = batch["binary_img"][:,:3].to(self.device)
        summed_rgb = rgb_values.sum(dim=-1)

        # Convert summed_rgb to binary: 1 if non-zero, 0 otherwise
        binary_img= (summed_rgb != 0).long()
        assert torch.all((binary_img == 0) | (binary_img == 1)), "Ground truth cannot be interpreted as binary img"
        gt_sem = binary_img

        # Get ray_indices (if available)
        ray_indices = batch.get("ray_indices", torch.arange(N_rays, device=self.device))

        gt_labels = gt_sem[ray_indices]  # [N_rays]

        # Predictions
        pred_labels = torch.argmax(torch.nn.functional.softmax(pred_logits, dim=-1), dim=-1) # [N_rays]
        
        metrics_dict.update({"semantic_accuracy": (pred_labels == gt_labels).float().mean()})
        metrics_dict.update({"semantic_psnr": self.psnr(pred_labels, gt_labels)})

        return metrics_dict


    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # all of these metrics will be logged as scalars
        metrics_dict          = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore

        images_dict = {"accumulation": combined_acc,
                       "depth": combined_depth}

        return metrics_dict, images_dict
