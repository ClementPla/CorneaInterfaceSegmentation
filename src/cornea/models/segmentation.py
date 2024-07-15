from typing import Literal
from functools import lru_cache
import torch
import numpy as np
from cornea.models.hf_hubs import download_model
import torchvision.transforms.functional as Ftv
from kornia.contrib import connected_components

DEFAULT_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
DEFAULT_NORMALIZATION_STD = (0.229, 0.224, 0.225)


Architecture = Literal["unet"]
EncoderModel = Literal["resnet34"]


def clean_segment_mask(mask):
    ccl = connected_components(mask.unsqueeze(0).float(), mask.shape[-1])
    unique = torch.unique(ccl, sorted=True)
    mask = (ccl == unique[-1]).long().squeeze(0)
    return mask


def segment(
    image,
    image_resolution=(416, 1280),
    arch="unet",
    encoder="resnet34",
    autofit_resolution=True,
    reverse_autofit=True,
    device: torch.device = "cuda",
):
    image = (image / 255.0).astype(np.float32)

    tensor = torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(0).to(device)
    B, C, H, W = tensor.shape

    if autofit_resolution:
        tensor = torch.nn.functional.interpolate(
            tensor, size=image_resolution, mode="bilinear", align_corners=False
        )
    tensor = Ftv.normalize(
        tensor, mean=DEFAULT_NORMALIZATION_MEAN, std=DEFAULT_NORMALIZATION_STD
    )

    model = get_model(arch=arch, encoder=encoder, device=device)
    with torch.no_grad():
        prediction = model(tensor)
    prediction = torch.sigmoid(prediction)
    if reverse_autofit:
        prediction = torch.nn.functional.interpolate(
            prediction, size=(H, W), mode="bilinear", align_corners=False
        )

    return prediction.squeeze(0).squeeze(0) > 0.5


@lru_cache(maxsize=2)
def get_model(
    arch: Architecture = "unet",
    encoder: EncoderModel = "resnest50d",
    device: torch.device = "cuda",
):
    """Get segmentation model

    Args:
        arch (Architecture, optional): Defaults to 'unet'.
        encoder (EncoderModel, optional):  Defaults to 'resnest50d'.
        device (torch.device, optional): Defaults to "cuda".

    Returns:
        nn.Module: Torch segmentation model
    """
    model = download_model(arch, encoder).to(device=device)
    model.eval()
    return model
