import torch
from pathlib import Path

# =============================================================================
# CONFIGURATION VARIABLES - MODIFY THESE PATHS FOR YOUR TACC ENVIRONMENT
# =============================================================================

# Input: Directory containing raw .svs whole slide images
# TODO: change
RAW_WSI_DIR = Path("/scratch/projects/brain_tumor/raw_wsi/")

# Output: Root directory for processed nuclei detection results
# TODO: change
OUTPUT_DIR = Path("/scratch/projects/brain_tumor/wsi_processed/nuclei_detection/40x_1024px_128px_overlap/")

# HoVer-Net pre-trained weights (MoNuSeg)
# TODO: change
MONUSEG_WEIGHTS = Path("/scratch/projects/brain_tumor/models/hovernet_weights/hovernet_fast_monuseg.tar")

# =============================================================================
# FIXED PARAMETERS - DO NOT MODIFY UNLESS YOU UNDERSTAND THE IMPLICATIONS
# =============================================================================

# Magnification level (Level 0 = highest resolution, typically 40x)
MAGNIFICATION_LEVEL = 0  # Use Level 0 for 40x

# Tile dimensions
TILE_SIZE = 1024  # pixels
OVERLAP = 128     # pixels
STRIDE = TILE_SIZE - OVERLAP  # 1024 - 128 = 896 pixels

# Central region for valid predictions (to avoid edge artifacts)
VALID_MARGIN = OVERLAP // 2  # 64 pixels from each edge

# Tissue detection threshold
TISSUE_THRESHOLD = 0.1  # Minimum fraction of non-white pixels

# Nuclei detection parameters
MIN_NUCLEUS_SIZE = 10  # Minimum nucleus area in pixels
NP_THRESHOLD = 0.5     # Nuclear probability threshold

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # Adjust based on GPU memory (A100: 8-16, V100: 4-8)