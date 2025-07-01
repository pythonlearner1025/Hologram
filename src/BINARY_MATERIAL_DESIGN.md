# Binary Material Optimization for 3D-Printable Acoustic Lenses

This implementation uses a continuous relaxation approach to optimize binary material distributions for 3D-printable acoustic lenses.

## Overview

The optimization method allows gradient-based optimization of designs that must ultimately be binary (either silicone or water/void) for 3D printing with a single material.

## Key Concepts

### 1. Continuous Relaxation

Instead of directly optimizing binary variables (which is computationally difficult), we:
- Use continuous optimization variables `γ` for each voxel
- Apply sigmoid function: `σ(γ) = 1 / (1 + exp(-γ))`
- This maps `γ ∈ (-∞, +∞)` to material fraction `σ(γ) ∈ (0, 1)`

### 2. Material Property Interpolation

Effective acoustic properties are calculated as:
```
c_eff = (c_silicone - c_water) * σ(γ) + c_water
ρ_eff = (ρ_silicone - ρ_water) * σ(γ) + ρ_water  
α_eff = (α_silicone - α_water) * σ(γ) + α_water
```

### 3. Natural Binarization

During optimization:
- The algorithm naturally pushes `γ` values to extremes
- `γ >> 0` → `σ(γ) ≈ 1` (silicone)
- `γ << 0` → `σ(γ) ≈ 0` (water/void)
- Results in nearly binary solutions without explicit constraints

### 4. Design Structure

The lens consists of:
- **Optimizable layer**: 0.2mm thick, binary material distribution
- **Backing layer**: 0.5mm thick, uniform silicone (non-optimizable)

## Configuration

Key parameters in `focus.py`:
```python
# Material properties
SOUND_SPEED_WATER = 1500  # m/s
SOUND_SPEED_SILICONE = 1000  # m/s
DENSITY_WATER = 1000  # kg/m³
DENSITY_SILICONE = 1100  # kg/m³

# Lens geometry
LENS_THICKNESS_MM = 0.2  # Optimizable layer
BACKING_THICKNESS_MM = 0.5  # Non-optimizable backing
```

## Usage

1. **Run optimization**:
   ```bash
   cd /workspace/hologram
   source jaxenv/bin/activate
   python3 focus.py
   ```

2. **Test with fewer iterations**:
   ```bash
   python3 test_binary_material.py
   ```

## Output Files

The optimization generates:
- `optimized_binary_design.png`: Visualization of material distribution
- `optimized_binary_design.csv`: Voxel coordinates for 3D printing
- `optimized_binary_design_with_backing.csv`: Complete design including backing

## Interpreting Results

### Binary Statistics
The optimizer reports binarization progress:
```
Binary material distribution: 87.5% (Water: 105, Silicone: 35, Total: 160)
```

Higher percentages indicate better convergence to binary solutions.

### CSV Format
The exported CSV contains:
- `x_mm`, `y_mm`, `z_mm`: Voxel center coordinates
- `voxel_size_mm`: Size of each voxel

### Visualization
- **Red/Blue gradient**: Continuous material fractions (0=water, 1=silicone)
- **Black/White**: Binary design after thresholding

## 3D Printing Workflow

1. **Export binary design** from optimized parameters
2. **Convert CSV** to your printer's format (e.g., STL)
3. **Print with silicone** where voxels = 1
4. **Leave voids** where voxels = 0 (filled with water in use)

## Customization

### Material Properties
Adjust based on your printing material:
```python
SOUND_SPEED_SILICONE = 1000  # Measure for your material
DENSITY_SILICONE = 1100  # From material datasheet
```

### Lens Geometry
Modify dimensions as needed:
```python
LENS_WIDTH_MM = 15.0  # Lateral size
LENS_THICKNESS_MM = 0.2  # Optimizable thickness
```

### Optimization Behavior
- **Learning rate**: Controls convergence speed
- **Iterations**: More iterations → better binarization
- **Objective function**: Choose based on application needs

## Troubleshooting

1. **Poor binarization** (<80% binary):
   - Increase iterations
   - Adjust learning rate
   - Check if objective conflicts with binary structure

2. **Convergence issues**:
   - Reduce learning rate
   - Initialize with small random values instead of zeros
   - Check material property contrast

3. **Fabrication constraints**:
   - Ensure voxel size matches printer resolution
   - Consider minimum feature sizes
   - Account for material shrinkage 