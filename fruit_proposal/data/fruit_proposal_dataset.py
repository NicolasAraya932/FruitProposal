"""
Semantic dataset for handling images, masks, and binary images.
"""

from typing import Dict, Literal, Optional, Tuple, Type, Union
from jaxtyping import Float, UInt8
from pathlib import Path
from PIL import Image
from torch import Tensor
import torch
import numpy as np
import numpy.typing as npt
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

    def get_binary_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        binary_filename = self._dataparser_outputs.binary_filenames[image_idx]
        pil_image = Image.open(binary_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.Resampling.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_binary_float32(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in float32 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_binary_numpy_image(image_idx).astype("float32") / 255.0)
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        return image

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx, "float32")
        binary = self.get_binary_float32(image_idx)
        data["binary_img"] = binary
        return data
