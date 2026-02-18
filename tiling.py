"""
tiling.py
Step 1.2: Sliding window tiling with overlap of 128 pixels and tile size is 1024*1024 pixels

Extracts tiles from WSI using a sliding window approach.
- Tile Size: 1024 x 1024 pixels
- Overlap: 128 pixels
- Stride: 1024 - 128 = 896 pixels
"""

import numpy as np
import openslide
from typing import Generator, Tuple

from config import TILE_SIZE, OVERLAP, STRIDE, MAGNIFICATION_LEVEL, TISSUE_THRESHOLD


def calculate_tile_grid(
    wsi_width: int,
    wsi_height: int
) -> Tuple[int, int]:
    """
    Calculate the number of tiles in each dimension.

    Args:
        wsi_width: Width of WSI at Level 0
        wsi_height: Height of WSI at Level 0

    Returns:
        Tuple of (num_tiles_x, num_tiles_y)
    """
    # Stride = Tile Size - Overlap = 1024 - 128 = 896
    num_tiles_x = int(np.ceil((wsi_width - OVERLAP) / STRIDE))
    num_tiles_y = int(np.ceil((wsi_height - OVERLAP) / STRIDE))

    total_tiles = num_tiles_x * num_tiles_y
    print(f"Tiling Configuration:")
    print(f"  Tile Size: {TILE_SIZE} x {TILE_SIZE} pixels")
    print(f"  Overlap: {OVERLAP} pixels")
    print(f"  Stride: {STRIDE} pixels")
    print(f"  Grid: {num_tiles_x} x {num_tiles_y} = {total_tiles} total tiles")

    return num_tiles_x, num_tiles_y


def generate_tile_coordinates(
    wsi_width: int,
    wsi_height: int
) -> Generator[Tuple[int, int, int, int], None, None]:
    """
    Generate coordinates for each tile using sliding window.

    Yields:
        Tuple of (tile_idx_x, tile_idx_y, x_start, y_start)
        - tile_idx_x, tile_idx_y: Tile indices in the grid
        - x_start, y_start: Top-left corner coordinates in WSI
    """
    num_tiles_x, num_tiles_y = calculate_tile_grid(wsi_width, wsi_height)

    for tile_idx_y in range(num_tiles_y):
        for tile_idx_x in range(num_tiles_x):
            # Calculate top-left corner using stride
            x_start = tile_idx_x * STRIDE
            y_start = tile_idx_y * STRIDE

            # Handle edge tiles
            x_start = min(x_start, max(0, wsi_width - TILE_SIZE))
            y_start = min(y_start, max(0, wsi_height - TILE_SIZE))

            yield tile_idx_x, tile_idx_y, x_start, y_start


def extract_tile(
    slide: openslide.OpenSlide,
    x_start: int,
    y_start: int
) -> np.ndarray:
    """
    Extract a single 1024x1024 tile from the WSI at Level 0 (40x).

    Args:
        slide: OpenSlide object
        x_start: Top-left x coordinate
        y_start: Top-left y coordinate

    Returns:
        RGB numpy array of shape (1024, 1024, 3)
    """
    # Read region at Level 0 (40x magnification)
    tile_pil = slide.read_region(
        location=(x_start, y_start),
        level=MAGNIFICATION_LEVEL,
        size=(TILE_SIZE, TILE_SIZE)
    )

    # Convert RGBA to RGB numpy array
    tile_rgb = np.array(tile_pil.convert("RGB"))

    return tile_rgb


def is_tissue_tile(tile: np.ndarray) -> bool:
    """
    Determine if a tile contains sufficient tissue (not mostly background).

    Args:
        tile: RGB numpy array of shape (H, W, 3)

    Returns:
        True if tile contains tissue, False if mostly background
    """
    # Convert to grayscale
    gray = np.mean(tile, axis=2)

    # Count non-white pixels (intensity < 220)
    tissue_pixels = np.sum(gray < 220)
    total_pixels = tile.shape[0] * tile.shape[1]

    tissue_fraction = tissue_pixels / total_pixels

    return tissue_fraction > TISSUE_THRESHOLD