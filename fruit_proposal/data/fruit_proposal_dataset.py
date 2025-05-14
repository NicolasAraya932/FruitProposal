"""
Semantic dataset for handling images, masks, and binary images.
"""

from typing import Dict, Literal, Float
from pathlib import Path
from PIL import Image
from torch import Tensor
import torch
import numpy as np
from nerfstudio.data.datasets.base_dataset import InputDataset

from fruit_proposal.data.dataparser.fruit_proposal_base_dataparser import FruitProposalDataparserOutputs

class FruitDataset(InputDataset):
    """Dataset that returns images, masks, and binary images.

    Args:
        dataparser_outputs: Description of where and how to read input images.
        scale_factor: Factor to scale the images and masks.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["mask", "binary_img"]

    def __init__(self, dataparser_outputs: FruitProposalDataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)


    def get_with_binary_data(self, image_idx: int,
                        binary_image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        # image_idx + image + mask (if exists)
        data = self.get_data(image_idx, binary_image_type)

        # To process the binary image and num_classes
        if binary_image_type == "float32":
            binary_img = self.get_image_float32(image_idx)
        elif binary_image_type == "uint8":
            binary_img = self.get_image_uint8(image_idx)
        else:
            raise NotImplementedError(f"image_type (={binary_image_type}) getter was not implemented, use uint8 or float32")
        
        data.update({"binary_img"  : binary_img})
        return data
