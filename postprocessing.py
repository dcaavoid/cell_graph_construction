"""
postprocessing.py
Step 2.3: Extract centroid: use regionprops from scikit-image and output the centroid coordinates of each instance
Step 2.4: Transform the local coordinates on each tile into a global absolute oordinate on WSI
"""

import numpy as np
from typing import List, Dict, Tuple
from skimage.measure import regionprops
from config import TILE_SIZE, OVERLAP, VALID_MARGIN


def extract_centroids(instance_map: np.ndarray) -> List[Dict]:
    """
    Extract centroid coordinates for each nucleus using regionprops.

    Args:
        instance_map: (H, W) array where each nucleus has unique integer ID

    Returns:
        List of dicts with keys: 'id', 'x_local', 'y_local', 'area'
        Note: x = column, y = row (standard image coordinate system)
    """
    nuclei_list = []

    props = regionprops(instance_map)

    for prop in props:
        # regionprops centroid is (row, col) = (y, x)
        # Convert to (x, y) format
        centroid_row, centroid_col = prop.centroid

        nuclei_list.append({
            'id': prop.label,
            'x_local': centroid_col,  # Column = x coordinate
            'y_local': centroid_row,  # Row = y coordinate
            'area': prop.area,
        })

    return nuclei_list


# Valid region bounds (to avoid overlap duplicates)
VALID_MIN = VALID_MARGIN       # 64
VALID_MAX = TILE_SIZE - VALID_MARGIN  # 960


def is_in_valid_region(
    x_local: float,
    y_local: float,
    tile_idx_x: int,
    tile_idx_y: int,
    num_tiles_x: int,
    num_tiles_y: int
) -> bool:
    """
    Check if centroid is in the valid (non-overlapping) region.

    Edge tiles have extended valid regions since they have no neighbors
    on boundary sides.

    Args:
        x_local, y_local: Local centroid coordinates
        tile_idx_x, tile_idx_y: Tile indices in grid
        num_tiles_x, num_tiles_y: Total tiles in each dimension

    Returns:
        True if centroid should be kept
    """
    # Determine valid bounds based on tile position
    x_min = 0 if tile_idx_x == 0 else VALID_MIN
    x_max = TILE_SIZE if tile_idx_x == num_tiles_x - 1 else VALID_MAX

    y_min = 0 if tile_idx_y == 0 else VALID_MIN
    y_max = TILE_SIZE if tile_idx_y == num_tiles_y - 1 else VALID_MAX

    return (x_min <= x_local < x_max) and (y_min <= y_local < y_max)


def transform_to_global(
    nuclei_local: List[Dict],
    tile_idx_x: int,
    tile_idx_y: int,
    x_start: int,
    y_start: int,
    num_tiles_x: int,
    num_tiles_y: int,
    filter_overlap: bool = True
) -> List[Dict]:
    """
    Transform local tile coordinates to global WSI absolute coordinates.

    Formula:
        X_global = x_local + x_start
        Y_global = y_local + y_start

    Where x_start, y_start are the top-left corner of the tile in WSI space.

    Args:
        nuclei_local: List of nuclei with local coordinates from extract_centroids()
        tile_idx_x, tile_idx_y: Tile indices
        x_start, y_start: Tile top-left corner in global coordinates
        num_tiles_x, num_tiles_y: Total tiles for edge detection
        filter_overlap: If True, discard nuclei in overlap regions

    Returns:
        List of nuclei with global coordinates
    """
    nuclei_global = []

    for nucleus in nuclei_local:
        x_local = nucleus['x_local']
        y_local = nucleus['y_local']

        # Filter overlap regions to avoid duplicates
        if filter_overlap:
            if not is_in_valid_region(
                x_local, y_local,
                tile_idx_x, tile_idx_y,
                num_tiles_x, num_tiles_y
            ):
                continue

        # Transform to global coordinates
        x_global = x_local + x_start
        y_global = y_local + y_start

        nuclei_global.append({
            'id': nucleus['id'],
            'x_global': float(x_global),
            'y_global': float(y_global),
            'area': nucleus['area'],
            'tile_origin': (tile_idx_x, tile_idx_y)
        })

    return nuclei_global


def merge_all_nuclei(
    all_tile_nuclei: List[List[Dict]]
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Merge nuclei from all tiles into single coordinate array.

    Args:
        all_tile_nuclei: List of nuclei lists from each tile

    Returns:
        Tuple of:
        - coordinates: (N, 2) numpy array of [X_global, Y_global]
        - metadata: List of dicts with all properties
    """
    all_nuclei = []
    for tile_nuclei in all_tile_nuclei:
        all_nuclei.extend(tile_nuclei)

    if len(all_nuclei) == 0:
        return np.array([]).reshape(0, 2), []

    # Create coordinate array for HACT-Net
    coordinates = np.array([
        [n['x_global'], n['y_global']] for n in all_nuclei
    ], dtype=np.float32)

    # Reassign sequential IDs
    for i, nucleus in enumerate(all_nuclei):
        nucleus['id'] = i

    print(f"Total nuclei after merging: {len(all_nuclei)}")

    return coordinates, all_nuclei