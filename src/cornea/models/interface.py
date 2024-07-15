import torch
import numpy as np

from cornea.models.segmentation import segment, clean_segment_mask
from scipy.ndimage import label


def get_interface_line(image, arch="unet", encoder="resnet34", device="cuda"):
    mask = segment(image, arch=arch, encoder=encoder, device=device)
    mask = clean_segment_mask(mask)
    interface_line = extract_interface_line(mask.long()).cpu().numpy()
    interface_line = clean_interface_line(interface_line)
    return interface_line


def extract_interface_line(mask):
    """Get the interface line from the mask (batch)."""
    is_batch = True
    if mask.dim() == 2:
        is_batch = False
        mask = mask.unsqueeze(0)

    B, H, W = mask.shape

    # Get the interface line
    mask = torch.flip(mask, [-2])
    h_coord = torch.argmax(mask, dim=-2, keepdim=True)
    h_coord = H - h_coord - 1
    xx, yy = torch.meshgrid(
        torch.arange(H, device=mask.device), torch.arange(W, device=mask.device)
    )

    xx = xx.to(mask.device)

    xx = xx.unsqueeze(0).expand(B, H, W)

    interface_line = xx == h_coord

    if not is_batch:
        interface_line = interface_line.squeeze(0)

    return interface_line


def clean_interface_line(interface_line, threshold=0.95):
    result = np.zeros_like(interface_line)
    interface_line[-20:] = False
    line_x = np.sum(interface_line, axis=0)
    argwhere = np.argwhere(line_x)
    x1 = np.min(argwhere)
    x2 = np.max(argwhere)

    y = np.argmax(interface_line, axis=0)[x1:x2]
    x = np.arange(x1, x2)

    diff = np.abs(np.diff(y)) / np.diff(x)
    y = y[1:].astype(np.float32)
    x = x[1:]

    y[diff > np.quantile(diff, threshold)] = np.nan

    connected_components = np.ones_like(y)
    connected_components[np.isnan(y)] = 0

    a = np.array(connected_components)
    m = np.r_[False, a > a.mean(), False]
    idx = np.flatnonzero(m[:-1] != m[1:])

    I = (idx[1::2] - idx[::2]).argmax()

    xelems = x[idx[2 * I] : idx[2 * I + 1]]
    yelems = y[idx[2 * I] : idx[2 * I + 1]]

    result[yelems.astype(int), xelems.astype(int)] = 1
    return result


def polyfit(interface_line, deg, crop=0):
    interface_line = interface_line
    line_x = np.sum(interface_line, axis=0)

    argwhere = np.argwhere(line_x)
    x1 = np.min(argwhere)
    x2 = np.max(argwhere)

    x = np.arange(x1, x2)

    y = np.argmax(interface_line, axis=0)[x]
    coeffs = np.polyfit(x[crop : -(1 + crop)], y[crop : -(1 + crop)], deg=deg)
    return coeffs, x1, x2


def interpolate(coeffs, x):
    return np.polyval(coeffs, x)
