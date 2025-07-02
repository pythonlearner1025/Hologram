#!/usr/bin/env python3
"""
Export optimized acoustic lenses to STL format for 3D printing.

This script reads the optimized parameters from a JAX optimization run
and converts them into printable STL files.
"""

import numpy as np
import json
import os
from pathlib import Path
import trimesh
import argparse

# Global variables for physical parameters (will be loaded from JSON)
NUM_LENSES = None
LENS_WIDTH_MM = None
LENS_THICKNESS_MM = None
BACKING_THICKNESS_MM = None
GAP_BETWEEN_LENSES_MM = None
VOXEL_SIZE_MM = None
NUM_XY_SEGMENTS = None
TOTAL_VOXELS_PER_LENS = None

# Binarization threshold
THRESHOLD = 0.5  # Material fraction above this becomes solid


def load_parameters(params_path, json_path):
    """Load optimized parameters and metadata."""
    global NUM_LENSES, LENS_WIDTH_MM, LENS_THICKNESS_MM, BACKING_THICKNESS_MM
    global GAP_BETWEEN_LENSES_MM, VOXEL_SIZE_MM, NUM_XY_SEGMENTS, TOTAL_VOXELS_PER_LENS
    
    params = np.load(params_path)
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # Try to load physical parameters from new format first
    if 'physical_params' in metadata:
        phys = metadata['physical_params']
        NUM_LENSES = phys['num_lenses']
        LENS_WIDTH_MM = phys['lens_width_mm']
        LENS_THICKNESS_MM = phys['lens_thickness_mm']
        BACKING_THICKNESS_MM = phys['backing_thickness_mm']
        GAP_BETWEEN_LENSES_MM = phys['gap_between_lenses_mm']
        VOXEL_SIZE_MM = phys['voxel_size_mm']
        NUM_XY_SEGMENTS = phys['num_xy_segments']
        TOTAL_VOXELS_PER_LENS = phys['total_voxels_per_lens']
        print(f"Loaded physical parameters from JSON:")
        print(f"  - {NUM_LENSES} lenses")
        print(f"  - Lens size: {LENS_WIDTH_MM}×{LENS_WIDTH_MM}mm")
        print(f"  - Voxel grid: {NUM_XY_SEGMENTS}×{NUM_XY_SEGMENTS}")
        print(f"  - Voxel size: {VOXEL_SIZE_MM}mm")
    else:
        # Fall back to inferring from parameter array size for old format
        total_params = params.size
        print(f"Warning: Old JSON format detected. Inferring parameters from array size ({total_params} elements)")
        
        # Common configurations:
        # If total_params = 10000, then 4 lenses × 50×50 = 10000 (voxel_size = 0.4mm)
        # If total_params = 62500, then 4 lenses × 125×125 = 62500 (voxel_size = 0.16mm)
        # If total_params = 14400, then 4 lenses × 60×60 = 14400 (voxel_size = 0.333mm)
        
        # Set defaults (can be overridden below)
        NUM_LENSES = 4
        LENS_WIDTH_MM = 20.0
        LENS_THICKNESS_MM = 0.5
        BACKING_THICKNESS_MM = 1.0
        GAP_BETWEEN_LENSES_MM = 0.01
        
        # Infer NUM_XY_SEGMENTS from total parameter count
        TOTAL_VOXELS_PER_LENS = total_params // NUM_LENSES
        NUM_XY_SEGMENTS = int(np.sqrt(TOTAL_VOXELS_PER_LENS))
        
        if NUM_XY_SEGMENTS ** 2 != TOTAL_VOXELS_PER_LENS:
            raise ValueError(f"Cannot determine grid size from {total_params} parameters")
        
        VOXEL_SIZE_MM = LENS_WIDTH_MM / NUM_XY_SEGMENTS
        print(f"Inferred: {NUM_LENSES} lenses, {NUM_XY_SEGMENTS}×{NUM_XY_SEGMENTS} voxels, voxel size = {VOXEL_SIZE_MM:.3f}mm")
    
    return params, metadata


def params_to_lens_designs(params):
    """Convert parameter vector to lens material designs."""
    # Apply sigmoid to get material fractions [0, 1]
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    material_fractions = sigmoid(params)
    
    # Reshape to (NUM_LENSES, NUM_XY_SEGMENTS, NUM_XY_SEGMENTS)
    lens_designs = material_fractions.reshape(NUM_LENSES, NUM_XY_SEGMENTS, NUM_XY_SEGMENTS)
    
    # Binarize: 1 = solid silicone, 0 = void (water channel)
    binary_designs = (lens_designs > THRESHOLD).astype(np.float32)
    
    return binary_designs


def create_lens_mesh(binary_design, lens_index, include_backing=True):
    """Create a 3D mesh for a single lens."""
    voxels = []
    
    # Create voxels for the optimized layer
    for i in range(NUM_XY_SEGMENTS):
        for j in range(NUM_XY_SEGMENTS):
            if binary_design[i, j] > 0.5:  # Solid voxel
                x = i * VOXEL_SIZE_MM
                y = j * VOXEL_SIZE_MM
                z = 0
                
                # Create a box for this voxel
                box = trimesh.creation.box(
                    extents=[VOXEL_SIZE_MM, VOXEL_SIZE_MM, LENS_THICKNESS_MM],
                    transform=trimesh.transformations.translation_matrix([
                        x + VOXEL_SIZE_MM/2,
                        y + VOXEL_SIZE_MM/2,
                        z + LENS_THICKNESS_MM/2
                    ])
                )
                voxels.append(box)
    
    # Add backing layer if requested
    if include_backing:
        backing = trimesh.creation.box(
            extents=[LENS_WIDTH_MM, LENS_WIDTH_MM, BACKING_THICKNESS_MM],
            transform=trimesh.transformations.translation_matrix([
                LENS_WIDTH_MM/2,
                LENS_WIDTH_MM/2,
                LENS_THICKNESS_MM + BACKING_THICKNESS_MM/2
            ])
        )
        voxels.append(backing)
    
    # Combine all voxels into a single mesh
    if voxels:
        mesh = trimesh.util.concatenate(voxels)
        # Remove duplicate vertices and faces
        mesh.merge_vertices()
        return mesh
    else:
        # Empty design - create a thin frame
        print(f"Warning: Lens {lens_index + 1} has no solid voxels in optimized layer!")
        return create_frame_only(include_backing)


def create_frame_only(include_backing=True):
    """Create a frame-only lens (for completely empty designs)."""
    # Create a hollow box frame
    outer_box = trimesh.creation.box(
        extents=[LENS_WIDTH_MM, LENS_WIDTH_MM, LENS_THICKNESS_MM]
    )
    inner_box = trimesh.creation.box(
        extents=[LENS_WIDTH_MM - 2*VOXEL_SIZE_MM, 
                 LENS_WIDTH_MM - 2*VOXEL_SIZE_MM, 
                 LENS_THICKNESS_MM + 0.1]
    )
    frame = outer_box.difference(inner_box)
    frame.apply_translation([LENS_WIDTH_MM/2, LENS_WIDTH_MM/2, LENS_THICKNESS_MM/2])
    
    meshes = [frame]
    
    if include_backing:
        backing = trimesh.creation.box(
            extents=[LENS_WIDTH_MM, LENS_WIDTH_MM, BACKING_THICKNESS_MM],
            transform=trimesh.transformations.translation_matrix([
                LENS_WIDTH_MM/2,
                LENS_WIDTH_MM/2,
                LENS_THICKNESS_MM + BACKING_THICKNESS_MM/2
            ])
        )
        meshes.append(backing)
    
    return trimesh.util.concatenate(meshes)


def create_stacked_assembly(binary_designs):
    """Create a single STL with all lenses stacked."""
    meshes = []
    current_z = 0
    
    for i in range(NUM_LENSES):
        # Create lens mesh
        mesh = create_lens_mesh(binary_designs[i], i, include_backing=True)
        
        # Translate to correct Z position
        mesh.apply_translation([0, 0, current_z])
        meshes.append(mesh)
        
        # Update Z position for next lens
        current_z += LENS_THICKNESS_MM + BACKING_THICKNESS_MM + GAP_BETWEEN_LENSES_MM
    
    # Combine all lenses
    assembly = trimesh.util.concatenate(meshes)
    return assembly


def export_lenses(params_path, json_path, output_dir):
    """Main export function."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load parameters
    print(f"Loading parameters from {params_path}")
    params, metadata = load_parameters(params_path, json_path)
    
    # Convert to binary lens designs
    print("\nConverting parameters to lens designs...")
    binary_designs = params_to_lens_designs(params)
    
    # Print statistics
    print("\nLens statistics:")
    for i in range(NUM_LENSES):
        solid_fraction = np.mean(binary_designs[i])
        print(f"  Lens {i+1}: {solid_fraction*100:.1f}% solid")
    
    # Export individual lenses
    print("\nExporting individual lenses...")
    for i in range(NUM_LENSES):
        mesh = create_lens_mesh(binary_designs[i], i, include_backing=True)
        filename = output_dir / f"lens_{i+1}.stl"
        mesh.export(filename)
        print(f"  Saved: {filename}")
    
    # Export stacked assembly
    print("\nExporting stacked assembly...")
    assembly = create_stacked_assembly(binary_designs)
    assembly_file = output_dir / "lens_assembly.stl"
    assembly.export(assembly_file)
    print(f"  Saved: {assembly_file}")
    
    # Export optimized layers only (without backing)
    print("\nExporting optimized layers only...")
    for i in range(NUM_LENSES):
        mesh = create_lens_mesh(binary_designs[i], i, include_backing=False)
        filename = output_dir / f"lens_{i+1}_optimized_only.stl"
        mesh.export(filename)
        print(f"  Saved: {filename}")
    
    # Save binary designs as images for reference
    print("\nSaving binary designs as images...")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, NUM_LENSES, figsize=(4*NUM_LENSES, 4))
    if NUM_LENSES == 1:
        axes = [axes]
    
    for i in range(NUM_LENSES):
        axes[i].imshow(binary_designs[i], cmap='RdBu_r', vmin=0, vmax=1)
        axes[i].set_title(f'Lens {i+1}')
        axes[i].set_xlabel('X segments')
        axes[i].set_ylabel('Y segments')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'binary_designs.png', dpi=200)
    plt.close()
    
    # Save export info
    export_info = {
        "source_params": str(params_path),
        "source_json": str(json_path),
        "physical_params": {
            "num_lenses": NUM_LENSES,
            "lens_width_mm": LENS_WIDTH_MM,
            "lens_thickness_mm": LENS_THICKNESS_MM,
            "backing_thickness_mm": BACKING_THICKNESS_MM,
            "gap_between_lenses_mm": GAP_BETWEEN_LENSES_MM,
            "voxel_size_mm": VOXEL_SIZE_MM,
            "num_xy_segments": NUM_XY_SEGMENTS,
            "total_voxels_per_lens": TOTAL_VOXELS_PER_LENS
        },
        "threshold": THRESHOLD,
        "solid_fractions": [float(np.mean(binary_designs[i])) for i in range(NUM_LENSES)]
    }
    
    with open(output_dir / "export_info.json", "w") as f:
        json.dump(export_info, f, indent=2)
    
    print(f"\nExport complete! Files saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export optimized acoustic lenses to STL")
    parser.add_argument("params_path", help="Path to optimised_params.npy file")
    parser.add_argument("json_path", help="Path to run.json file")
    parser.add_argument("-o", "--output", default="exported_lenses", 
                        help="Output directory (default: exported_lenses)")
    
    args = parser.parse_args()
    
    export_lenses(args.params_path, args.json_path, args.output) 