"""
wsi_loader.py
Step 1.1: Load WSI with 40x magnification

Uses OpenSlide to load whole slide images at Level 0 (40x magnification).
DO NOT use 20x - nuclei detection requires full resolution.
"""

import openslide
from pathlib import Path
from typing import Dict, Tuple

from config import MAGNIFICATION_LEVEL


def load_wsi(wsi_path: str) -> openslide.OpenSlide:
    """
    Load a Whole Slide Image at 40x magnification (Level 0).

    Args:
        wsi_path: Path to .svs or .ndpi file

    Returns:
        OpenSlide object for the WSI

    Raises:
        FileNotFoundError: If WSI file does not exist
        ValueError: If WSI cannot be opened
    """
    wsi_path = Path(wsi_path)
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    slide = openslide.OpenSlide(str(wsi_path))

    # Verify and report magnification
    objective_power = slide.properties.get(
        openslide.PROPERTY_NAME_OBJECTIVE_POWER, None
    )
    if objective_power:
        print(f"WSI Objective Power: {objective_power}x")
        if float(objective_power) != 40.0:
            print(f"WARNING: Expected 40x magnification, found {objective_power}x")

    # Report Level 0 dimensions
    level_0_dims = slide.level_dimensions[MAGNIFICATION_LEVEL]
    print(f"WSI Dimensions at Level 0 (40x): {level_0_dims[0]} x {level_0_dims[1]} pixels")

    return slide


def get_wsi_dimensions(slide: openslide.OpenSlide) -> Tuple[int, int]:
    """
    Get WSI dimensions at Level 0 (40x).

    Returns:
        Tuple of (width, height) in pixels
    """
    return slide.level_dimensions[MAGNIFICATION_LEVEL]


def get_wsi_metadata(slide: openslide.OpenSlide) -> Dict:
    """
    Extract relevant WSI metadata.

    Returns:
        Dictionary with WSI properties
    """
    return {
        "dimensions": slide.level_dimensions[MAGNIFICATION_LEVEL],
        "level_count": slide.level_count,
        "level_downsamples": slide.level_downsamples,
        "mpp_x": slide.properties.get(openslide.PROPERTY_NAME_MPP_X, "unknown"),
        "mpp_y": slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, "unknown"),
        "vendor": slide.properties.get(openslide.PROPERTY_NAME_VENDOR, "unknown"),
        "objective_power": slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, "unknown"),
    }