# from scipy.interpolate import make_interp_spline
from shapely.geometry import LineString, Point
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


def intelligent_simplify_line(parent_inkex, coords, straitening_cutoff_length, test_tolerance):
    """
    Intelligently simplifies a line defined by a list of (y, x) tuples using Shapely.

    Simplification is done by testing the line for straight sections longer than
    `straitening_cutoff_length` and then reinserting the original points in those sections.
    The remaining points are then tested for proximity and collapsed if necessary.

    Args:
        coords: A list of (y, x) tuples representing the line.
        straitening_cutoff_length: The length of straight sections to look for in the line.
        test_tolerance: The simplification tolerance (higher = more simplification).

    Returns:
        A list of (y, x) tuples representing the simplified line.
    """
    # Shapely uses (x, y) coordinates, so we need to swap
    swapped_coords = [(x, y) for y, x in coords]

    orig_line = LineString(swapped_coords)
    # preserve_topology = false is generally preferred.
    test_simplified_line = orig_line.simplify(test_tolerance, preserve_topology=False)

    # Swap back to (y, x) coordinates
    test_simplified_coords = [(int(y), int(x)) for x, y in test_simplified_line.coords]
    if len(test_simplified_coords) < 2:
        return coords

    # Convert to numpy array, test for straight sections
    simplified_np = np.array(test_simplified_coords)
    diffs = np.diff(simplified_np, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    long_stretch_indices = np.where(distances > straitening_cutoff_length)[0]

    # Determine stretches
    diffs = np.diff(long_stretch_indices)
    split_indices = np.where(diffs != 1)[0] + 1  # Find where ranges break

    if not split_indices.size:  # All contiguous
        return [(long_stretch_indices[0], long_stretch_indices[-1])]

    long_stretch_ranges, long_stretch_lowers, long_stretch_uppers = [], [], []
    lower_bounds = np.concatenate(([0], split_indices))
    upper_bounds = np.concatenate((split_indices - 1, [len(long_stretch_indices) - 1]))

    for lower, upper in zip(lower_bounds, upper_bounds):
        long_stretch_ranges.append((long_stretch_indices[lower], long_stretch_indices[upper] + 1))
        long_stretch_lowers.append(long_stretch_indices[lower])
        long_stretch_uppers.append(long_stretch_indices[upper] + 1)

    def get_lowest_index_within_tolerance_norm(parent_inkex, nd_point, list_of_points, start_idx, tolerance):
        """
        Gets the indices of all points in list_of_points that are within a tolerance of nd_point using np.linalg.norm.

        Args:
            nd_point: NumPy array representing the nd point.
            list_of_points: NumPy array of shape (n, m) where each row is an m-dimensional point.
            tolerance: The tolerance for Euclidean distance.

        Returns:
            NumPy array of indices.
        """

        for incr_start_idx in range(start_idx, len(list_of_points), 50):
            incr_end_idx = min(incr_start_idx + 50, len(list_of_points))
            diffs = list_of_points[incr_start_idx:incr_end_idx] - nd_point  # Calculate differences using broadcasting
            distances = np.linalg.norm(diffs, axis=1)  # Calculate Euclidean distances

            indices = np.where(distances <= tolerance)[0]

            if indices.size > 0:
                min_index = np.min(indices)
                return incr_start_idx + min_index
        return None  # Or any other value to indicate no indices found

    # Determine link-up likely points
    final_path, orig_path_ctr = [], 0
    orig_nd = np.array(coords)
    for i in range(len(long_stretch_ranges)):
        lower_nd, upper_nd = simplified_np[long_stretch_lowers[i]], simplified_np[long_stretch_uppers[i]]
        orig_exit_pt = get_lowest_index_within_tolerance_norm(parent_inkex, lower_nd, orig_nd, orig_path_ctr,
                                                              test_tolerance)
        if orig_exit_pt is not None:
            orig_entry_pt = get_lowest_index_within_tolerance_norm(parent_inkex, upper_nd, orig_nd, orig_path_ctr + 1,
                                                                   test_tolerance)
            if orig_entry_pt is not None:
                final_path.extend(coords[orig_path_ctr:orig_exit_pt + 1])
                final_path.extend(test_simplified_coords[long_stretch_lowers[i] + 1:long_stretch_uppers[i]])
                orig_path_ctr = orig_entry_pt
        # parent_inkex.msg(f"intellibound {orig_exit_pt} in {orig_entry_pt} out")

    if orig_path_ctr is not None and orig_path_ctr < len(coords): final_path.extend(coords[orig_path_ctr:])

    return final_path
