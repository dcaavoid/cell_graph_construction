#!/usr/bin/env python3

# ========================================================================
# Goal: generate nuclei centroid coordinates from WSIs
# 1. WSI preprocess
# 2. Nuclei detection to get the coordinates of centroids
# ========================================================================


# ========================================================================
# Setup
# ========================================================================

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import (
    RAW_WSI_DIR, OUTPUT_DIR, MONUSEG_WEIGHTS,
    TILE_SIZE, OVERLAP, STRIDE, DEVICE
)
from wsi_loader import load_wsi, get_wsi_dimensions, get_wsi_metadata
from tiling import generate_tile_coordinates, extract_tile, is_tissue_tile, calculate_tile_grid
from normalization import initialize_normalizer, normalize_tile
from detection import load_hovernet_model, run_inference
from postprocessing import extract_centroids, transform_to_global, merge_all_nuclei

# ========================================================================
def process_wsi(wsi_path: Path, model, output_dir: Path) -> dict:
    """Process a single WSI through the complete pipeline."""
    print(f"\n{'='*60}")
    print(f"Processing: {wsi_path.name}")
    print(f"{'='*60}")

    # Step 1.1: Load WSI with 40x magnification
    slide = load_wsi(str(wsi_path))
    wsi_width, wsi_height = get_wsi_dimensions(slide)
    metadata = get_wsi_metadata(slide)

    # Calculate tile grid
    num_tiles_x, num_tiles_y = calculate_tile_grid(wsi_width, wsi_height)

    # Process all tiles
    all_tile_nuclei = []
    tile_coords = list(generate_tile_coordinates(wsi_width, wsi_height))

    for tile_idx_x, tile_idx_y, x_start, y_start in tqdm(tile_coords, desc="Tiles"):
        # Step 1.2: Extract tile (1024x1024, 128px overlap)
        tile = extract_tile(slide, x_start, y_start)

        if not is_tissue_tile(tile):
            continue

        # Step 1.3: Stain normalization with Macenko
        tile_normalized = normalize_tile(tile)

        # Step 2.1 & 2.2: HoVer-Net inference -> instance maps
        instance_map = run_inference(model, tile_normalized)

        # Step 2.3: Extract centroids using regionprops
        nuclei_local = extract_centroids(instance_map)

        # Step 2.4: Transform to global coordinates
        nuclei_global = transform_to_global(
            nuclei_local,
            tile_idx_x, tile_idx_y,
            x_start, y_start,
            num_tiles_x, num_tiles_y
        )
        all_tile_nuclei.append(nuclei_global)

    slide.close()

    # Merge all nuclei
    coordinates, nuclei_metadata = merge_all_nuclei(all_tile_nuclei)

    # Prepare output
    output = {
        "wsi_name": wsi_path.name,
        "magnification": "40x",
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "stride": STRIDE,
        "wsi_dimensions": [wsi_width, wsi_height],
        "total_nuclei": len(nuclei_metadata),
        "nuclei": [
            {"id": n['id'], "x_global": round(n['x_global'], 2),
             "y_global": round(n['y_global'], 2), "area": n['area']}
            for n in nuclei_metadata
        ]
    }

    # Save outputs
    stem = wsi_path.stem
    json_path = output_dir / f"{stem}_nuclei.json"
    npy_path = output_dir / f"{stem}_centroids.npy"

    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    np.save(npy_path, coordinates)

    print(f"Saved: {json_path.name}, {npy_path.name}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Nuclei Detection for HACT-Net")
    parser.add_argument("--wsi_dir", type=str, default=str(RAW_WSI_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--weights", type=str, default=str(MONUSEG_WEIGHTS))
    args = parser.parse_args()

    wsi_dir = Path(args.wsi_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    print("Initializing pipeline...")
    initialize_normalizer()
    model = load_hovernet_model(args.weights)

    # Find WSI files
    wsi_files = list(wsi_dir.glob("*.svs")) + list(wsi_dir.glob("*.ndpi"))
    print(f"Found {len(wsi_files)} WSI files")

    # Process each WSI
    results = []
    for wsi_path in wsi_files:
        result = process_wsi(wsi_path, model, output_dir)
        results.append(result)

    # Save summary
    summary = {
        "pipeline": "Nuclei Detection for HACT-Net",
        "total_wsi": len(results),
        "total_nuclei": sum(r['total_nuclei'] for r in results),
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nComplete. Processed {len(results)} WSIs.")


if __name__ == "__main__":
    main()