import torch
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION VARIABLES - MODIFY THESE PATHS FOR YOUR TACC ENVIRONMENT
# =============================================================================

TACC_USER = "zeyuancao"
SCRATCH = os.environ.get('SCRATCH')
WORK = os.environ.get('WORK')

# Input: Directory containing raw .svs whole slide images
RAW_WSI_DIR = Path(f"{SCRATCH}/raw_wsi")

# Output: Root directory for processed nuclei detection results
# TODO: change
OUTPUT_DIR = Path(f"{WORK}/brain_tumor/output/nuclei_detection/")

# HoVer-Net pre-trained weights (MoNuSeg)
# TODO: change
HOVERNET_WEIGHTS = Path(f"{WORK}/brain_tumor/weights/hovernet_original_kumar_notype_tf2pytorch.tar")

# ============================================================================
# HOVERNET MODEL CONFIGURATION
# ============================================================================
# CRITICAL: Kumar checkpoint requires model_mode='original'
MODEL_MODE = "original"  # DO NOT change to 'fast' - incompatible with Kumar weights
NR_TYPES = None  # None = segmentation only (NOT 0, which creates invalid layer)

# Model mode specifications:
# 'original' mode: 270x270 input -> 80x80 output (for Kumar, CPM17, CoNSeP)
# 'fast' mode: 256x256 input -> 164x164 output (for PanNuke, MoNuSAC)

# =============================================================================
# FIXED PARAMETERS - DO NOT MODIFY UNLESS YOU UNDERSTAND THE IMPLICATIONS
# =============================================================================

# Magnification level (Level 0 = highest resolution, typically 40x)
MAGNIFICATION_LEVEL = 0  # Use Level 0 for 40x

# Tile dimensions
TILE_SIZE = 270  # Must be 270 for 'original' mode
OVERLAP = 80     # Output size for 'original' mode
STRIDE = TILE_SIZE - OVERLAP  # 270 - 80 = 190 pixels

# Central region for valid predictions (to avoid edge artifacts)
VALID_MARGIN = OVERLAP // 2  # 40 pixels from each edge

# Tissue detection threshold
TISSUE_THRESHOLD = 0.1  # Minimum fraction of non-white pixels

# Nuclei detection parameters
MIN_NUCLEUS_SIZE = 10  # Minimum nucleus area in pixels 10 -> 5 -> 3
NP_THRESHOLD = 0.5     # Nuclear probability threshold

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16