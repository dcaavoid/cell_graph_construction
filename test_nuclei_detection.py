"""
test_nuclei_detection.py
Quick test to verify nuclei detection pipeline and visualize results.
Run this on TACC interactive GPU session before submitting batch jobs.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for TACC
import matplotlib.pyplot as plt
from pathlib import Path

# Import pipeline modules
from wsi_loader import load_wsi, get_wsi_dimensions
from tiling import extract_tile, is_tissue_tile
from normalization import initialize_normalizer, normalize_tile
from detection import load_hovernet_model, run_inference, upscale_for_40x
from postprocessing import extract_centroids

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
SCRATCH = os.environ.get('SCRATCH')
WORK = os.environ.get('WORK')

WEIGHTS_PATH = f"{WORK}/brain_tumor/weights/hovernet_original_kumar_notype_tf2pytorch.tar"
WSI_PATH = f"{SCRATCH}/raw_wsi/WSI_000001.svs"  # <-- CHANGE THIS to your WSI file
OUTPUT_DIR = Path("./test_output")

# ============================================================================

def test_single_tile():
    """Test nuclei detection on a single tile and visualize results."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("NUCLEI DETECTION QUICK TEST")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Load WSI
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading WSI...")
    print(f"      Path: {WSI_PATH}")

    if not Path(WSI_PATH).exists():
        print(f"ERROR: WSI file not found: {WSI_PATH}")
        print("Please update WSI_PATH in this script.")
        return

    slide = load_wsi(WSI_PATH)
    width, height = get_wsi_dimensions(slide)
    print(f"      WSI dimensions: {width} x {height} pixels")

    # Get magnification for proper scaling
    import openslide
    mag_str = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, "40")
    source_magnification = int(float(mag_str)) if mag_str else 40
    print(f"      Source magnification: {source_magnification}x")
    if source_magnification < 40:
        print(f"      NOTE: Will upscale {source_magnification}x to 40x for inference")

    # -------------------------------------------------------------------------
    # Step 2: Find a tissue tile
    # -------------------------------------------------------------------------
    print("\n[2/6] Searching for tissue tile...")

    tile = None
    tile_x, tile_y = 0, 0

    # Try multiple positions to find tissue
    test_positions = [
        (width // 4, height // 4),
        (width // 2, height // 2),
        (width // 3, height // 3),
        (width * 2 // 3, height // 3),
        (width // 3, height * 2 // 3),
    ]

    for x, y in test_positions:
        # Ensure coordinates are within bounds
        x = min(x, width - 270)
        y = min(y, height - 270)

        test_tile = extract_tile(slide, x, y)
        if is_tissue_tile(test_tile):
            tile = test_tile
            tile_x, tile_y = x, y
            print(f"      Found tissue tile at position ({x}, {y})")
            break

    if tile is None:
        print("ERROR: No tissue tile found in sampled positions.")
        print("Try manually specifying coordinates in the script.")
        slide.close()
        return

    # -------------------------------------------------------------------------
    # Step 3: Apply stain normalization
    # -------------------------------------------------------------------------
    print("\n[3/6] Applying Macenko stain normalization...")
    initialize_normalizer()
    tile_normalized = normalize_tile(tile)
    print("      Normalization complete")

    # -------------------------------------------------------------------------
    # Step 4: Load HoVer-Net model
    # -------------------------------------------------------------------------
    print("\n[4/6] Loading HoVer-Net model (Kumar checkpoint)...")
    print(f"      Weights: {WEIGHTS_PATH}")

    if not Path(WEIGHTS_PATH).exists():
        print(f"ERROR: Weights file not found: {WEIGHTS_PATH}")
        slide.close()
        return

    model = load_hovernet_model(WEIGHTS_PATH)
    print("      Model loaded successfully")

    # -------------------------------------------------------------------------
    # Step 5: Run nuclei detection
    # -------------------------------------------------------------------------
    print("\n[5/6] Running nuclei detection inference...")
    if source_magnification < 40:
        print(f"      Upscaling {source_magnification}x tile to simulate 40x...")
    instance_map = run_inference(model, tile_normalized, source_magnification=source_magnification)
    num_nuclei = int(instance_map.max())
    print(f"      Detected {num_nuclei} nuclei in this tile")

    # -------------------------------------------------------------------------
    # Step 6: Extract centroids
    # -------------------------------------------------------------------------
    print("\n[6/6] Extracting centroid coordinates...")
    centroids = extract_centroids(instance_map)
    print(f"      Extracted {len(centroids)} centroid coordinates")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("GENERATING VISUALIZATIONS")
    print("-" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Panel 1: Original tile
    axes[0, 0].imshow(tile)
    axes[0, 0].set_title(f"Original Tile ({source_magnification}x)\n(Position: {tile_x}, {tile_y})", fontsize=12)
    axes[0, 0].axis('off')

    # Panel 2: Show what model sees (upscaled if needed)
    if source_magnification < 40:
        tile_for_model = upscale_for_40x(tile_normalized, source_magnification)
        axes[0, 1].imshow(tile_for_model)
        axes[0, 1].set_title(f"Upscaled to Simulate 40x\n(Center crop {270//( 40//source_magnification)}x{270//(40//source_magnification)} → 270x270)", fontsize=12)
    else:
        axes[0, 1].imshow(tile_normalized)
        axes[0, 1].set_title("After Macenko Normalization", fontsize=12)
    axes[0, 1].axis('off')

    # Panel 3: Instance segmentation map
    im = axes[1, 0].imshow(instance_map, cmap='nipy_spectral')
    axes[1, 0].set_title(f"Instance Segmentation Map\n({num_nuclei} nuclei detected)", fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Panel 4: Centroid overlay
    axes[1, 1].imshow(tile)
    for c in centroids:
        axes[1, 1].scatter(c['x_local'], c['y_local'], c='red', s=15, marker='x', linewidths=1)
    axes[1, 1].set_title(f"Detected Nuclei Centroids\n({len(centroids)} points)", fontsize=12)
    axes[1, 1].axis('off')

    plt.tight_layout()

    # Save figure
    fig_path = OUTPUT_DIR / "nuclei_detection_test.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    # -------------------------------------------------------------------------
    # Save centroid data
    # -------------------------------------------------------------------------

    # Save as numpy array
    coords = np.array([[c['x_local'], c['y_local']] for c in centroids], dtype=np.float32)
    npy_path = OUTPUT_DIR / "test_centroids.npy"
    np.save(npy_path, coords)
    print(f"Saved: {npy_path}")

    # Save detailed info as text
    txt_path = OUTPUT_DIR / "test_centroids.txt"
    with open(txt_path, 'w') as f:
        f.write("ID\tX_local\tY_local\tArea\n")
        for c in centroids:
            f.write(f"{c['id']}\t{c['x_local']:.2f}\t{c['y_local']:.2f}\t{c['area']}\n")
    print(f"Saved: {txt_path}")

    # -------------------------------------------------------------------------
    # Print sample results
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("SAMPLE CENTROID COORDINATES (first 10)")
    print("-" * 60)
    print(f"{'ID':<6} {'X':<10} {'Y':<10} {'Area':<8}")
    print("-" * 34)
    for c in centroids[:10]:
        print(f"{c['id']:<6} {c['x_local']:<10.2f} {c['y_local']:<10.2f} {c['area']:<8}")

    if len(centroids) > 10:
        print(f"... and {len(centroids) - 10} more")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    slide.close()

    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)
    print(f"""
Results Summary:
  - WSI: {Path(WSI_PATH).name}
  - Tile position: ({tile_x}, {tile_y})
  - Tile size: {tile.shape[0]} x {tile.shape[1]}
  - Nuclei detected: {num_nuclei}
  - Centroids extracted: {len(centroids)}

Output files:
  - {OUTPUT_DIR}/nuclei_detection_test.png  (visualization)
  - {OUTPUT_DIR}/test_centroids.npy         (coordinates for HACT-Net)
  - {OUTPUT_DIR}/test_centroids.txt         (human-readable)

To view the visualization on your local machine:
  scp username@ls6.tacc.utexas.edu:{OUTPUT_DIR.absolute()}/nuclei_detection_test.png .
""")


if __name__ == "__main__":
    test_single_tile()