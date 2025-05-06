from typing import Dict, Optional, Tuple, Literal
import torch
from torch import Tensor, nn
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.field_components.field_heads import SemanticFieldHead, FieldHeadNames
from nerfstudio.fields.base_field import Field, get_normalized_directions


class FruitProposalField(Field):
    """Merged field for density and semantics using hash-based encoding.

    Args:
        aabb: Scene bounding box.
        num_levels: Number of levels in the hash encoding.
        base_res: Minimum resolution of the hash grid.
        max_res: Maximum resolution of the hash grid.
        log2_hashmap_size: Log2 size of the hash map.
        features_per_level: Number of features per level in the hash grid.
        num_layers: Number of layers in the MLP.
        layer_width: Width of each MLP layer.
        geo_feat_dim: Dimension of geometric features.
        num_semantic_classes: Number of semantic classes.
        spatial_distortion: Spatial distortion module.
        average_init_density: Initial density scaling factor.
        implementation: Backend implementation ("tcnn" or "torch").
    """

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_levels: int = 4,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        num_layers: int = 9,
        layer_width: int = 128,
        geo_feat_dim: int = 15,
        num_semantic_classes: int = 2,
        average_init_density: float = 1.0,
        num_layers_color: int = 3,  # COLOR
        hidden_dim_color: int = 64, # COLOR
        skip_connections: Optional[Tuple[int]] = (5,),
        appearance_embedding_dim: int = 32,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))
        self.num_images = num_images
        self.average_init_density = average_init_density
        self.geo_feat_dim = geo_feat_dim
        self.num_semantic_classes = num_semantic_classes
        self.appearance_embedding_dim = appearance_embedding_dim

        if self.appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        else:
            self.embedding_appearance = None

        # print("Inside the SemanticIEField constructor")
        # print(f"Using {num_levels} levels, base res {base_res}, max res {max_res}, log2_hashmap_size {log2_hashmap_size}")

        # Merge of encoding and mlp_base
        """
        args:
            def __init__(
            self,
            num_levels: int = 16,
            min_res: int = 16,
            max_res: int = 1024,
            log2_hashmap_size: int = 19,
            features_per_level: int = 2,
            hash_init_scale: float = 0.001,
            interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
            num_layers: int = 2,
            layer_width: int = 64,
            out_dim: Optional[int] = None,
            skip_connections: Optional[Tuple[int]] = None,
            activation: Optional[nn.Module] = nn.ReLU(),
            out_activation: Optional[nn.Module] = None,
            implementation: Literal["tcnn", "torch"] = "torch",

        Adding disponible args from the encoding and mlp_base
        """

        # Encoding for directions
        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        # Density base mlp with hash encoding
        self.mlp_base = MLPWithHashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            num_layers=num_layers,
            layer_width=layer_width,
            out_dim=1 + geo_feat_dim,  # 1 for density, rest for geometric features
            skip_connections=skip_connections, # skip to 5 layer as semantic is enough
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        # Color MLP
        self.mlp_color = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        # Semantic MLP for semantic logits
        self.mlp_semantic = MLP(
            in_dim=geo_feat_dim,
            num_layers=1,
            layer_width=128,
            out_dim=num_semantic_classes,
            activation=nn.ReLU(),
            out_activation=None,
        )

        # Semantic field head
        self.field_head_semantic = SemanticFieldHead(
            in_dim=self.mlp_semantic.get_out_dim(), num_classes=num_semantic_classes
        )


    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes density and geometric features."""
        
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]

        assert positions.numel() > 0, "positions is empty."

        self._sample_locations = positions
        positions_flat = positions.view(-1, 3)

        # Compute density and geometric features
        assert positions_flat.numel() > 0, "positions_flat is empty."
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        base_mlp_out = base_mlp_out.to(torch.float32)

        # Rectify density
        density = self.average_init_density * trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]

        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        """Get outputs for density and semantics."""
        assert density_embedding is not None, "density_embedding is None :C"

        outputs = {}

        # Directions for Color MLP training
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        embedded_appearance = None
        if self.embedding_appearance is not None:
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )

        # Compute semantic logits
        semantics_input = density_embedding.view(-1, self.geo_feat_dim).to(torch.float32)
        semantic_logits = self.mlp_semantic(semantics_input).view(*outputs_shape, -1).to(semantics_input)
        semantics = self.field_head_semantic(semantic_logits).to(semantics_input)
        outputs.update({FieldHeadNames.SEMANTICS: semantics})

        # Compute 
        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
            ]
            + (
                [embedded_appearance.view(-1, self.appearance_embedding_dim)] if embedded_appearance is not None else []
            ),
            dim=-1,
        )
        rgb = self.mlp_color(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB : rgb})

        return outputs