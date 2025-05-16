"""
Datamanager.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
)

import torch
import tyro
from torch import nn
from torch.nn import Parameter
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import TypeVar

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager

from fruit_proposal.data.fruit_proposal_dataparser import FruitProposalDataParserOutputs
from fruit_proposal.data.fruit_proposal_dataparser import FruitProposalDataParserConfig
from fruit_proposal.data.fruit_proposal_dataset import FruitProposalDataset


from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PatchPixelSamplerConfig, PixelSampler, PixelSamplerConfig
from nerfstudio.data.utils.dataloaders import CacheDataloader, FixedIndicesEvalDataloader, RandIndicesEvalDataloader
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper, get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.data.datamanagers.base_datamanager import (
    variable_res_collate,
    DataManager,
    DataManagerConfig)



@dataclass
class FruitDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: FruitDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = field(default_factory=FruitProposalDataParserConfig)
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    eval_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per eval iteration."""

TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)

class FruitDataManager(VanillaDataManager, Generic[TDataset]):
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: FruitDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset
    train_dataparser_outputs: FruitProposalDataParserOutputs
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: FruitDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        if self.config.images_on_gpu is True and "image" in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("binary_img")


    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return FruitProposalDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return FruitProposalDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )