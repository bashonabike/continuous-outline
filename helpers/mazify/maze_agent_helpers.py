import numpy as np

def process_points_in_quadrant_boxes_to_weighted_centroids(point, image, radius):
    """
    Finds True points within quadrant bounding boxes, computes centroids,
    and creates vectors pointing to centroids.

    Args:
        point: A tuple (y, x) representing the point's coordinates.
        image: A 2D NumPy array (binary image) with True/False values.
        radius: The search radius, defining the bounding box size.

    Returns:
        A list of tuples (centroid_y, centroid_x, vector_y, vector_x).
    """

    y, x = point
    rows, cols = image.shape
    results = []

    # Define quadrant bounding boxes
    quadrant_boxes = {
        1: (max(0, int(y - radius)), int(y), int(x), min(cols, int(x + radius + 1))),  # Top-right
        2: (max(0, int(y - radius)), int(y), max(0, int(x - radius)), int(x)),  # Top-left
        3: (int(y), min(rows, int(y + radius + 1)), max(0, int(x - radius)), int(x)),  # Bottom-left
        4: (int(y), min(rows, int(y + radius + 1)), int(x), min(cols, int(x + radius + 1)))  # Bottom-right
    }

    for quadrant, (min_y, max_y, min_x, max_x) in quadrant_boxes.items():
        # Find points within quadrant bounding box
        found_points = find_true_points_in_bbox(image, (min_y, max_y, min_x, max_x))

        if found_points.size > 0:
            centroid_y = np.mean(found_points[:, 0])
            centroid_x = np.mean(found_points[:, 1])
            distance = np.sqrt((centroid_y - y)**2 + (centroid_x - x)**2)

            # Avoid division by zero
            vector_y = (centroid_y - y) / (distance + 1.0)
            vector_x = (centroid_x - x) / (distance + 1.0)
            results.append({'quadrant': quadrant, 'cent_y': centroid_y, 'cent_x': centroid_x, 'vec_y': vector_y,
                            'vec_x': vector_x})
        else:
            results.append({'quadrant': quadrant, 'cent_y': 0.0, 'cent_x': 0.0, 'vec_y': 0.0, 'vec_x': 0.0})
    return results

def find_true_points_in_bbox(array, bbox):
    """
    Finds all True points within a bounding box in a NumPy ndarray.

    Args:
        array: The NumPy ndarray (boolean).
        bbox: A tuple (min_y, max_y, min_x, max_x) defining the bounding box.

    Returns:
        A NumPy array of (y, x) coordinates of True points.
    """

    min_y, max_y, min_x, max_x = bbox
    sub_array = array[min_y:max_y, min_x:max_x]
    y_indices, x_indices = np.where(sub_array)

    # Adjust indices to the original array's coordinates
    y_indices += min_y
    x_indices += min_x

    return np.column_stack((y_indices, x_indices))

def compute_compass_from_quadrant_vectors(quadrant_vectors):
    """
   Compute compass directions for input to network

   Args:
       quadrant_vectors: list(dict).

   Returns:
       Dict of compass directions
   """
    compass = {'N':0.0, 'S':0.0, 'E':0.0, 'W':0.0}
    for quadrant in quadrant_vectors:
        if quadrant['quadrant'] in (1, 2):
            compass['N'] += quadrant['vec_y']
        else:
            compass['S'] += quadrant['vec_y']

        if quadrant['quadrant'] in (1, 4):
            compass['E'] += quadrant['vec_x']
        else:
            compass['W'] += quadrant['vec_x']

    return compass




