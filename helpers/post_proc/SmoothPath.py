from scipy.interpolate import make_interp_spline
from shapely.geometry import LineString
import numpy as np

def simplify_line(coords, tolerance=1.0):
    """
    Simplifies a line defined by a list of (y, x) tuples using Shapely.

    Args:
        coords: A list of (y, x) tuples representing the line.
        tolerance: The simplification tolerance (higher = more simplification).

    Returns:
        A list of (y, x) tuples representing the simplified line.
    """
    # Shapely uses (x, y) coordinates, so we need to swap
    swapped_coords = [(x, y) for y, x in coords]

    line = LineString(swapped_coords)
    simplified_line = line.simplify(tolerance, preserve_topology=False)  # preserve_topology = false is generally preferred.

    # Swap back to (y, x) coordinates
    simplified_coords = [(y, x) for x, y in simplified_line.coords]

    return simplified_coords