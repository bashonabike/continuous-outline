import cv2
import numpy as np

def connect_paths(image, paths, obliteration_radius=5):
    """Connects paths in a continuous line, minimizing connection length and
       obliterating regions around connections.

    Args:
        paths: A list of NumPy arrays, where each array represents a path.
        obliteration_radius: The radius around the connection line to obliterate.

    Returns:
        A NumPy array representing the connected path, or None if no paths
        could be connected.  Also returns a mask of obliterated regions.
    """

    #TODO: try to minimize sudden changes in pen direction
    #TODO: maybe try breaking up the constituent paths a bit more so have more to select from

    if not paths:
        return None, None

    connected_path = []
    used_paths = [False] * len(paths)
    obliterated_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8) #Image assumed to exist globally

    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def find_nearest_path(current_point):
        min_dist = float('inf')
        nearest_path_index = -1
        nearest_point = None

        for i, path in enumerate(paths):
            if not used_paths[i]:
                for point in path:
                    if obliterated_mask[int(point[0][1]), int(point[0][0])] == 0: #Check if obliterated
                        dist = distance(current_point, point)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_path_index = i
                            nearest_point = point
        return nearest_path_index, nearest_point

    # Start with the first available path
    start_path_index = 0
    for i in range(len(paths)):
        if not used_paths[i]:
            start_path_index = i
            break

    current_path = paths[start_path_index]
    connected_path.extend(current_path)
    used_paths[start_path_index] = True
    last_point = current_path[-1]

    for _ in range(len(paths) - 1):  # Connect the remaining paths
        nearest_path_index, nearest_point = find_nearest_path(last_point)

        if nearest_path_index != -1:
            connection_line = np.array([last_point, nearest_point])

            # Obliterate the region around the connection line
            for i in range(2):
                for x in range(min(int(connection_line[i][0][0]), int(connection_line[1][0][0])) - obliteration_radius,
                               max(int(connection_line[i][0][0]), int(connection_line[1][0][0])) + obliteration_radius + 1):
                    for y in range(min(int(connection_line[i][0][1]), int(connection_line[1][0][1])) - obliteration_radius,
                                   max(int(connection_line[i][0][1]), int(connection_line[1][0][1])) + obliteration_radius + 1):
                        if (0 <= x < image.shape[1] and 0 <= y < image.shape[0] and distance((x, y), connection_line[i])
                                <= obliteration_radius):
                            obliterated_mask[y, x] = 1

            connected_path.extend(paths[nearest_path_index])
            used_paths[nearest_path_index] = True
            last_point = paths[nearest_path_index][-1]
        else:
            break #No more paths could be found

    return np.array(connected_path), obliterated_mask


def draw_path(image, path, color=(0, 0, 255), thickness=2):
    """Draws a path on an image."""
    image_with_path = image.copy()
    if path is not None and len(path) > 1:
        for i in range(len(path) - 1):
            start_point = tuple(path[i][0].astype(int))
            end_point = tuple(path[i+1][0].astype(int))
            cv2.line(image_with_path, start_point, end_point, color, thickness)
    return image_with_path

def draw_obliterated(image, mask, color=(0, 255, 0)):
    """Draws the obliterated region."""
    image_with_obliterated = image.copy()
    y_indices, x_indices = np.where(mask == 1)
    for x, y in zip(x_indices, y_indices):
        cv2.circle(image_with_obliterated, (x, y), 1, color, -1)
    return image_with_obliterated
