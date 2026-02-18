"""
normalization.py
Step 1.3: Stain normalization with Macenko Normalization

Applies Macenko stain normalization to standardize H&E appearance
across all tiles before inference. This improves model generalization.
"""

import numpy as np
from tiatoolbox.tools.stainnorm import MacenkoNormalizer

# Global normalizer instance (initialized once)
_NORMALIZER = None


def initialize_normalizer(reference_image: np.ndarray = None) -> MacenkoNormalizer:
    """
    Initialize the Macenko normalizer with reference stain matrix.

    Args:
        reference_image: Optional reference image for target stain extraction.
                        If None, uses standard reference values.

    Returns:
        Initialized MacenkoNormalizer
    """
    global _NORMALIZER

    normalizer = MacenkoNormalizer()

    if reference_image is not None:
        normalizer.fit(reference_image)
    else:
        # Standard reference stain matrix from Macenko paper
        normalizer.stain_matrix_target = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])
        normalizer.target_concentrations = np.array([1.9705, 1.0308])

    _NORMALIZER = normalizer
    print("Macenko normalizer initialized")

    return normalizer


def get_normalizer() -> MacenkoNormalizer:
    """
    Get the global normalizer instance, initializing if needed.
    """
    global _NORMALIZER
    if _NORMALIZER is None:
        initialize_normalizer()
    return _NORMALIZER


def normalize_tile(tile: np.ndarray) -> np.ndarray:
    """
    Apply Macenko stain normalization to a single tile.

    Args:
        tile: RGB numpy array of shape (H, W, 3), dtype uint8

    Returns:
        Normalized RGB numpy array of same shape and dtype
    """
    normalizer = get_normalizer()

    # Skip normalization for mostly white/background tiles
    if np.mean(tile) > 240:
        return tile

    try:
        normalized = normalizer.transform(tile)
        return normalized.astype(np.uint8)

    except Exception as e:
        # Some tiles may fail (e.g., insufficient color variation)
        print(f"Normalization failed, returning original: {e}")
        return tile