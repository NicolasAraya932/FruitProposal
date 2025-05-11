"""
Semantic dataset.
"""

from typing import Dict

import torch
from pathlib import Path
import numpy as np
from PIL import Image
from typing import List, Tuple, Union

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset


class FruitDataset(InputDataset):
    """Dataset that returns images and binary_img and masks.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["mask", "binary_img"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        self.binary_img = dataparser_outputs.metadata.get("binary_img", None)

        assert "binary_img" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["binary_img"], Semantics), "No semantic instance could be found! Is a semantic folder included in the input folder and transform.json file?"

        self.binary_img = self.metadata["binary_img"]

    def get_metadata(self, data: Dict) -> Dict:
        filepath = self.binary_img.filenames[data["image_idx"]]

        if image_type == "float32":
            image = self.get_image_float32(image_idx)
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")

        return {"binary_img": self.binary_img.filenames[data["image_idx"]]}
