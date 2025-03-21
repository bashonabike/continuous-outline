from scipy.interpolate import make_interp_spline
from shapely.geometry import LineString
import numpy as np

def simplify_line(coords, tolerance=1.0, preserve_topology=False):
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
    # preserve_topology = false is generally preferred.
    simplified_line = line.simplify(tolerance, preserve_topology=preserve_topology)

    # Swap back to (y, x) coordinates
    simplified_coords = [(int(y), int(x)) for x, y in simplified_line.coords]

    return simplified_coords