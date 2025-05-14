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

"""A set of standard datasets."""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type

import torch
from jaxtyping import Float
from torch import Tensor

import nerfstudio.configs.base_config as cfg
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox

from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs

@dataclass
class FruitProposalDataparserOutputs(DataparserOutputs):
    binary_filenames: List[Path] = field(
        default=None,
        metadata={"doc": "Filenames for the binary images."},
    )


@dataclass
class FruitProposalDataParserConfig(DataParserConfig):
    """Basic dataset config"""

    _target: Type = field(default_factory=lambda: FruitProposalDataParser)
    """_target: target class to instantiate"""
    data: Path = Path()
    """Directory specifying location of data."""


@dataclass
class FruitProposalDataParser(DataParser):
    """A dataset.

    Args:
        config: datasetparser config containing all information needed to instantiate dataset

    Attributes:
        config: datasetparser config containing all information needed to instantiate dataset
        includes_time: Does the dataset include time information in the camera poses.
    """

    config: FruitProposalDataParserConfig
    includes_time: bool = False

    def __init__(self, config: FruitProposalDataParserConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def _generate_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> FruitProposalDataparserOutputs:
        """Abstract method that returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).
            kwargs: kwargs for generating dataparser outputs.

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """

    def get_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> FruitProposalDataparserOutputs:
        """Returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).
            kwargs: kwargs for generating dataparser outputs.

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """
        dataparser_outputs = self._generate_dataparser_outputs(split, **kwargs)
        return dataparser_outputs
