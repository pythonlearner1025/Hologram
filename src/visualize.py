import os
import csv
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from src.losses import compute_field
from src.config import (
    NUM_LENSES, NUM_X_SEGMENTS, LENS_WIDTH_MM, LENS_THICKNESS_MM,
    MAX_X_RESOLUTION_MM, BACKING_THICKNESS_MM, GAP_BETWEEN_LENSES_MM,
    LENS_START_Z_MM, GRID_SIZE_MM, FOCAL_DISTANCE_MM, TARGET_RADIUS_MM
)
