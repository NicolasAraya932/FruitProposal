import torch 
from typing import Any, Dict, List, Optional, Tuple, Type, Union

def get_rgba_image(outputs: Dict[str, torch.Tensor], output_name: str = "rgb") -> torch.Tensor:
    """Returns the RGBA image from the outputs of the model.

    Args:
        outputs: Outputs of the model.

    Returns:
        RGBA image.
    """
    
    rgb = outputs[output_name]
    return torch.cat((rgb, torch.ones_like(rgb[..., :1])), dim=-1)
