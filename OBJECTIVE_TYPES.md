# Acoustic Lens Optimization: Objective Function Types

This implementation provides several objective function types for optimizing acoustic lenses with different goals.

## Configuration

To select an objective type, modify the `OBJECTIVE_TYPE` parameter in `focus.py`:

```python
# Options: "point", "area_contrast", "uniform", "sidelobe"
OBJECTIVE_TYPE = "area_contrast"
```

## Available Objective Types

### 1. Point-Based Objective (`"point"`)
- **Goal**: Maximize pressure at a single target point
- **Use case**: Precise focusing applications, single-point therapy
- **Loss function**: `-pressure_at_target_point`

### 2. Area Contrast Objective (`"area_contrast"`)
- **Goal**: Maximize contrast between target area and surrounding regions
- **Use case**: Therapy applications where you want to minimize off-target effects
- **Loss function**: `-(mean_pressure_in_target / mean_pressure_outside_target)`
- **Parameters**:
  - `TARGET_RADIUS_MM`: Radius of the circular target area

### 3. Uniform Area Objective (`"uniform"`)
- **Goal**: Create uniform pressure distribution within target area
- **Use case**: Extended treatment zones, uniform heating applications
- **Loss function**: `-(mean_pressure - 0.1 * pressure_variance)`
- **Parameters**:
  - `TARGET_RADIUS_MM`: Radius of the circular target area

### 4. Sidelobe Suppression Objective (`"sidelobe"`)
- **Goal**: Maximize target pressure while suppressing sidelobes
- **Use case**: Imaging applications, reducing artifacts
- **Loss function**: `-(target_pressure - 0.3 * sidelobe_pressure)`
- **Parameters**:
  - `TARGET_RADIUS_MM`: Radius of the target area
  - `SIDELOBE_RADIUS_MM`: Outer radius for sidelobe suppression region

## Usage Example

```bash
# Run with default area contrast objective
python3 focus.py

# Test all objective types
python3 test_objectives.py
```

## Visualization

Each objective type adds appropriate visualization elements:
- **Point**: Shows target point with pressure value
- **Area-based**: Shows circular target area
- **Sidelobe**: Shows both target area and sidelobe suppression region

## Customization

You can easily add new objective functions by:
1. Creating a new function following the pattern of existing objectives
2. Adding it to the objective selection in `main()`
3. Updating visualization as needed

## Parameter Tuning

- **Learning rate**: Adjust in `main()` (default: 0.1)
- **Weights**: Modify the weighting factors in objective functions
- **Target sizes**: Adjust `TARGET_RADIUS_MM` and `SIDELOBE_RADIUS_MM` 