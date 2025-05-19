"""
Semantic dataset for handling images, masks, and binary images.
"""

from typing import Dict, Literal, Optional, Tuple, Type, Union
from pathlib import Path
from PIL import Image
from torch import Tensor
import torch
import numpy as np
from nerfstudio.data.datasets.base_dataset import InputDataset

from fruit_proposal.data.fruit_proposal_dataparser import FruitProposalDataParserOutputs

class FruitProposalDataset(InputDataset):
    """Dataset that returns images, masks, and binary images.

    Args:
        dataparser_outputs: Description of where and how to read input images.
        scale_factor: Factor to scale the images and masks.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["mask", "binary_img"]

    def __init__(self, dataparser_outputs: FruitProposalDataParserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx, "float32")
        binary = self.get_image_float32(image_idx)
        data["binary_img"] = binary
        return data
