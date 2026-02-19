"""
detection.py
Step 2.1: Use HoVer-Net and load the weights of Kumar (MoNuSeg)
Step 2.2: Inference: use normalized tiles from 1.3 as input, output instance maps

HoVer-Net model initialization and inference for nuclei segmentation.

Why Kumar weights?
- Kumar dataset contains diverse tissue types (breast, kidney, liver, prostate, etc.)
- Equivalent to MoNuSeg for multi-organ nuclei segmentation
- Provides good generalization to brain tumor tissue where specific ground truth is unavailable

IMPORTANT: Kumar checkpoint requires model_mode='original' (NOT 'fast')
- Input size: 270x270 pixels
- Output size: 80x80 pixels
"""

import torch
import numpy as np
from typing import Tuple
from scipy import ndimage
from scipy.ndimage import label as scipy_label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.transform import resize
import cv2

# Import HoVer-Net from the cloned repository (requires PYTHONPATH to be set)
# export PYTHONPATH=$WORK/brain_tumor/hover_net:$PYTHONPATH
from models.hovernet.net_desc import HoVerNet

from pipeline_config import DEVICE, MIN_NUCLEUS_SIZE, NP_THRESHOLD


def load_hovernet_model(weights_path: str) -> torch.nn.Module:
    """
    Initialize HoVer-Net and load Kumar pre-trained weights.

    Args:
        weights_path: Path to hovernet_original_kumar_notype_tf2pytorch.tar

    Returns:
        HoVer-Net model in evaluation mode
    """
    # Create model architecture
    # CRITICAL: Kumar checkpoint requires mode='original' (270x270 input, 80x80 output)
    # nr_types=None means segmentation only (no cell type classification)
    model = HoVerNet(
        input_ch=3,
        nr_types=None,  # None = segmentation only (NOT 0, which creates invalid layer)
        freeze=False,
        mode="original"  # Must be 'original' for Kumar weights
    )

    # Load pre-trained Kumar weights
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
    print(f"HoVer-Net loaded with Kumar weights")
    print(f"  Mode: original (270x270 input)")
    print(f"  Device: {DEVICE}")
    print(f"  Parameters: {param_count:,}")

    return model


def preprocess_tile_for_model(tile: np.ndarray) -> torch.Tensor:
    """
    Preprocess normalized tile for HoVer-Net input.

    HoVer-Net expects input in [0, 255] range as float32, NOT [0, 1].
    This is critical - wrong normalization causes model to output garbage.

    Args:
        tile: Normalized RGB numpy array (H, W, 3), uint8

    Returns:
        Tensor of shape (1, 3, H, W), float32, in [0, 255] range
    """
    # Keep in [0, 255] range - HoVer-Net expects this!
    # DO NOT divide by 255
    tile_float = tile.astype(np.float32)

    # Transpose (H, W, C) -> (C, H, W)
    tile_chw = np.transpose(tile_float, (2, 0, 1))

    # Add batch dimension
    tensor = torch.from_numpy(tile_chw).unsqueeze(0)

    return tensor


def upscale_for_40x(tile: np.ndarray, source_magnification: int = 10) -> np.ndarray:
    """
    Upscale a tile from lower magnification to simulate 40x.

    For 10x WSIs, extract center region and upscale 4x to match 40x scale.
    The Kumar model expects nuclei to be ~15-25 pixels. At 10x they are ~4-6 pixels.

    Args:
        tile: RGB tile of shape (270, 270, 3)
        source_magnification: Original WSI magnification (10, 20, etc.)

    Returns:
        Upscaled tile of shape (270, 270, 3)
    """
    if source_magnification >= 40:
        return tile  # No upscaling needed

    scale_factor = 40 // source_magnification  # 4x for 10x, 2x for 20x

    h, w = tile.shape[:2]

    # Calculate the center crop size
    # To get equivalent physical area, we crop a smaller region and upscale
    crop_size = h // scale_factor  # 270 // 4 = 67 for 10x

    # Extract center crop
    start = (h - crop_size) // 2
    center_crop = tile[start:start+crop_size, start:start+crop_size]

    # Upscale to original tile size using bilinear interpolation
    upscaled = cv2.resize(center_crop, (w, h), interpolation=cv2.INTER_LINEAR)

    return upscaled


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    tile: np.ndarray,
    source_magnification: int = 40
) -> np.ndarray:
    """
    Run HoVer-Net inference on a normalized tile.

    Args:
        model: Loaded HoVer-Net model
        tile: Normalized RGB tile from Step 1.3
        source_magnification: WSI magnification (10, 20, 40). If <40, tile will be upscaled.

    Returns:
        Instance map (H, W) where each nucleus has a unique integer ID
    """
    # Upscale if source magnification is less than 40x
    if source_magnification < 40:
        tile = upscale_for_40x(tile, source_magnification)

    # Preprocess tile
    tile_tensor = preprocess_tile_for_model(tile)
    tile_tensor = tile_tensor.to(DEVICE)

    # Forward pass
    outputs = model(tile_tensor)

    # Extract outputs
    # np output has shape (batch, 2, H, W) - contains raw logits, needs softmax
    np_logits = outputs["np"]  # Shape: (1, 2, H, W)

    # Debug: print raw output ranges
    print(f"      DEBUG - np logits shape: {np_logits.shape}")
    print(f"      DEBUG - np logits[0,0] (bg) range: {np_logits[0,0].min().item():.3f} to {np_logits[0,0].max().item():.3f}")
    print(f"      DEBUG - np logits[0,1] (fg) range: {np_logits[0,1].min().item():.3f} to {np_logits[0,1].max().item():.3f}")

    # Apply softmax to convert logits to probabilities
    np_probs = torch.softmax(np_logits, dim=1)  # Softmax over channel dimension
    np_probs = np_probs.squeeze().cpu().numpy()  # Shape: (2, H, W)
    np_map = np_probs[1]  # Take nuclear probability channel (index 1)

    print(f"      DEBUG - np_map (after softmax) range: {np_map.min():.3f} to {np_map.max():.3f}")
    print(f"      DEBUG - np_map mean: {np_map.mean():.3f}")

    # hv output has shape (batch, 2, H, W) - channel 0 is horizontal, channel 1 is vertical
    hv_map = outputs["hv"].squeeze().cpu().numpy()  # Shape: (2, H, W)
    print(f"      DEBUG - hv_map shape: {hv_map.shape}")

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
    # Use distance transform to find nucleus centers
    distance = ndimage.distance_transform_edt(binary_mask)

    # Combine distance transform with HV gradient for better separation
    # HV gradients point toward nucleus centers - use this information
    h_map, v_map = hv_map[0], hv_map[1]

    # Calculate "energy" map - high at nucleus centers, low at boundaries
    # Sobel on HV maps gives boundary strength
    sobelh = ndimage.sobel(h_map, axis=1)
    sobelv = ndimage.sobel(v_map, axis=0)
    boundary_energy = np.sqrt(sobelh**2 + sobelv**2)

    # Normalize and invert - we want peaks at nucleus centers
    boundary_energy = 1 - (boundary_energy / (boundary_energy.max() + 1e-8))

    # Combine distance and boundary energy
    marker_energy = distance * boundary_energy

    local_max_coords = peak_local_max(
        marker_energy,
        min_distance=3,  # Reduced from 5 to detect more separate nuclei
        labels=binary_mask,
        exclude_border=False,
        threshold_rel=0.3  # Only keep peaks that are 30% of max
    )

    markers = np.zeros_like(binary_mask, dtype=np.int32)
    if len(local_max_coords) > 0:
        markers[tuple(local_max_coords.T)] = np.arange(1, len(local_max_coords) + 1)
    markers = scipy_label(markers)[0]

    print(f"      DEBUG - Watershed markers found: {len(local_max_coords)}")
    print(f"      DEBUG - Binary mask coverage: {binary_mask.sum()}/{binary_mask.size} pixels ({100*binary_mask.mean():.1f}%)")

    # Apply watershed using NEGATIVE marker_energy as landscape
    # This makes each marker a local minimum (basin), ensuring proper separation
    # The gradient magnitude alone doesn't have strong enough ridges between touching nuclei
    watershed_landscape = -marker_energy  # Invert so peaks become valleys

    instance_map = watershed(
        watershed_landscape,
        markers=markers,
        mask=binary_mask,
        compactness=0
    )

    print(f"      DEBUG - Instances after watershed: {instance_map.max()}")

    # Remove small objects
    instance_map = _remove_small_nuclei(instance_map)
    print(f"      DEBUG - Instances after filtering small nuclei: {instance_map.max()}")

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