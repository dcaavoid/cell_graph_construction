"""
detection.py
Step 2.1: Use HoVer-Net and load the weights of MoNuSeg
Step 2.2: Inference: use normalized tiles from 1.3 as input, output instance maps

HoVer-Net model initialization and inference for nuclei segmentation.

Why MoNuSeg weights?
- MoNuSeg dataset contains diverse tissue types (breast, kidney, liver, prostate, etc.)
- Provides good generalization to brain tumor tissue where specific ground truth is unavailable
- Validated performance on histopathology nuclei segmentation benchmarks
"""

import torch
import numpy as np
from typing import Tuple
from scipy import ndimage
from scipy.ndimage import label as scipy_label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from hover_net.models.hovernet import create_model

from config import DEVICE, MIN_NUCLEUS_SIZE, NP_THRESHOLD


def load_hovernet_model(weights_path: str) -> torch.nn.Module:
    """
    Initialize HoVer-Net and load MoNuSeg pre-trained weights.

    Args:
        weights_path: Path to hovernet_fast_monuseg.tar

    Returns:
        HoVer-Net model in evaluation mode
    """
    # Create model architecture (fast mode for encoder-decoder)
    model = create_model(
        mode="fast",
        nr_types=None  # Instance segmentation only, no classification
    )

    # Load pre-trained MoNuSeg weights
    checkpoint = torch.load(weights_path, map_location=DEVICE)

    # Handle different checkpoint formats
    if "desc" in checkpoint:
        state_dict = checkpoint["desc"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"HoVer-Net loaded with MoNuSeg weights")
    print(f"  Device: {DEVICE}")
    print(f"  Parameters: {param_count:,}")

    return model


def preprocess_tile_for_model(tile: np.ndarray) -> torch.Tensor:
    """
    Preprocess normalized tile for HoVer-Net input.

    Args:
        tile: Normalized RGB numpy array (H, W, 3), uint8

    Returns:
        Tensor of shape (1, 3, H, W), float32, normalized to [0, 1]
    """
    # Normalize to [0, 1]
    tile_float = tile.astype(np.float32) / 255.0

    # Transpose (H, W, C) -> (C, H, W)
    tile_chw = np.transpose(tile_float, (2, 0, 1))

    # Add batch dimension
    tensor = torch.from_numpy(tile_chw).unsqueeze(0)

    return tensor


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    tile: np.ndarray
) -> np.ndarray:
    """
    Run HoVer-Net inference on a normalized tile.

    Args:
        model: Loaded HoVer-Net model
        tile: Normalized RGB tile from Step 1.3

    Returns:
        Instance map (H, W) where each nucleus has a unique integer ID
    """
    # Preprocess tile
    tile_tensor = preprocess_tile_for_model(tile)
    tile_tensor = tile_tensor.to(DEVICE)

    # Forward pass
    outputs = model(tile_tensor)

    # Extract outputs
    np_map = outputs["np"].squeeze().cpu().numpy()  # Nuclear probability
    hv_map = outputs["hv"].squeeze().cpu().numpy()  # Horizontal-Vertical maps

    # Post-process to instance map
    instance_map = _postprocess_hovernet(np_map, hv_map)

    return instance_map


def _postprocess_hovernet(
    np_map: np.ndarray,
    hv_map: np.ndarray
) -> np.ndarray:
    """
    Post-process HoVer-Net outputs to generate instance segmentation.
    Uses watershed on HV gradients to separate touching nuclei.
    """
    # Binarize nuclear probability
    binary_mask = np_map > NP_THRESHOLD

    if not np.any(binary_mask):
        return np.zeros_like(np_map, dtype=np.int32)

    # Calculate gradient magnitude of HV maps
    h_grad = ndimage.sobel(hv_map[0], axis=1)
    v_grad = ndimage.sobel(hv_map[1], axis=0)
    grad_magnitude = np.sqrt(h_grad**2 + v_grad**2)
    grad_magnitude = (grad_magnitude - grad_magnitude.min()) / (grad_magnitude.max() - grad_magnitude.min() + 1e-8)

    # Find watershed markers
    distance = ndimage.distance_transform_edt(binary_mask)
    local_max_coords = peak_local_max(
        distance,
        min_distance=5,
        labels=binary_mask,
        exclude_border=False
    )

    markers = np.zeros_like(binary_mask, dtype=np.int32)
    if len(local_max_coords) > 0:
        markers[tuple(local_max_coords.T)] = np.arange(1, len(local_max_coords) + 1)
    markers = scipy_label(markers)[0]

    # Apply watershed
    instance_map = watershed(
        grad_magnitude,
        markers=markers,
        mask=binary_mask,
        compactness=0.001
    )

    # Remove small objects
    instance_map = _remove_small_nuclei(instance_map)

    return instance_map


def _remove_small_nuclei(instance_map: np.ndarray) -> np.ndarray:
    """Remove nuclei smaller than MIN_NUCLEUS_SIZE pixels."""
    from skimage.measure import regionprops

    output = instance_map.copy()
    for prop in regionprops(instance_map):
        if prop.area < MIN_NUCLEUS_SIZE:
            output[output == prop.label] = 0

    # Relabel for consecutive IDs
    output, _ = scipy_label(output > 0)
    return output