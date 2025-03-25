

def remove_repeated_coords(coords):
    import numpy as np
    """
    Removes repeated coordinates from a list of (y, x) tuples.

    Args:
        coords: A list of (y, x) tuples representing the line.

    Returns:
        A list of (y, x) tuples representing the line without repeated coordinates.
    """
    if len(coords) <= 1:
        return coords
    coords_nd = np.array(coords)


    diffs = np.diff(coords_nd, axis=0)  # Calculate differences between consecutive points
    mask = np.any(diffs != 0, axis=1)  # find rows where any element is not zero.
    mask = np.concatenate(([True], mask))  # add true to beginning of mask

    unique_nd = coords_nd[mask]
    return unique_nd.tolist()

# def remove_intersecting_gross_squiggle(path, )

def remove_inout(parent_inkex, path, manhatten_max_thickness=0, acuteness_threshold=0.15):
    import numpy as np

    def manhatten_dist(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def vectorized_triangle_areas(coordinates_nd):
        """Calculates the area of triangles formed by consecutive triplets of coordinates."""
        n = len(coordinates_nd)
        if n < 3:
            return np.array([])  # Not enough points for triangles

        x = coordinates_nd[:, 1]
        y = coordinates_nd[:, 0]

        # Roll the arrays to create the coordinate triplets
        x1 = x
        y1 = y
        x2 = np.roll(x, 1)
        y2 = np.roll(y, 1)
        x3 = np.roll(x, 2)
        y3 = np.roll(y, 2)

        # Apply the Shoelace Formula (vectorized)
        areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        #Determine perimeters
        p1, p2, p3 = np.stack((y1, x1), axis=1),np.stack((y2, x2), axis=1),np.stack((y3, x3), axis=1)
        side1 = np.linalg.norm(p2 - p1, axis=1)
        side2 = np.linalg.norm(p3 - p2, axis=1)
        side3 = np.linalg.norm(p1 - p3, axis=1)
        perimeters = side1 + side2 + side3

        #Find closeness of in and out vectors using projection
        v1 = p1 - p2  # v1 = p2 -> p1
        v2 = p3 - p2  # v2 = p2 -> p3

        # Calculate vector lengths
        v1_lengths = np.linalg.norm(v1, axis=1)
        v2_lengths = np.linalg.norm(v2, axis=1)

        # Find min and max length vectors pointwise
        min_length_vectors = np.where(v1_lengths[:, np.newaxis] < v2_lengths[:, np.newaxis], v1, v2)
        max_length_vectors = np.where(v1_lengths[:, np.newaxis] > v2_lengths[:, np.newaxis], v1, v2)

        # Find dot product pointwise
        dot_products = np.sum(min_length_vectors * max_length_vectors, axis=1)

        # Scale the max length vector to the dot product's projection
        projection_scalars = dot_products / np.linalg.norm(max_length_vectors, axis=1) ** 2
        projection_vectors = max_length_vectors * projection_scalars[:, np.newaxis]

        # Calculate distance between dot product endpoint and min vector endpoint
        projected_vectors_distances = np.linalg.norm(min_length_vectors - projection_vectors, axis=1)

        return areas, perimeters, projected_vectors_distances

    #Find turnarounds.  Build triangle with each set of 3 points, if area/perimeter < 0.1
    # and distance smallest in-path to dot product with larges is small, then acute
    coordinates_nd = np.array(path)
    triangle_areas, triangle_perimeters, projected_vectors_distances = vectorized_triangle_areas(coordinates_nd)
    accuteness_ratio = triangle_areas / triangle_perimeters
    accuteness_mask = accuteness_ratio < acuteness_threshold
    distance_mask = projected_vectors_distances < 0.5 * manhatten_max_thickness
    combined_mask = np.logical_and(accuteness_mask, distance_mask)
    blip_apex_indices = (np.where(combined_mask)[0] - 1).tolist()
    if len(blip_apex_indices) == 0:
        return path

    def calculate_directions(path):
        """
        Calculates slopes between consecutive points and between points and their previous points.

        Args:
            path: NumPy array of shape (n, 2) where each row is [x, y].

        Returns:
            Tuple of two NumPy arrays:
                - forward_slopes: Slopes between each point and the next.
                - backward_slopes: Slopes between each point and the previous.
        """

        if path.shape[0] < 2:
            return np.array([]), np.array([])  # Return empty arrays for paths with less than 2 points

        x_diffs = np.diff(path[:, 1])
        y_diffs = np.diff(path[:, 0])

        # Calculate forward directions (point to next)
        forward_directions = (2*np.pi + np.arctan2(y_diffs, x_diffs)) % (2*np.pi)

        # Calculate backward directions (point to previous)
        backward_directions = (2*np.pi + np.arctan2(-y_diffs, -x_diffs)) % (2*np.pi)

        # Pad the arrays to match the original path length
        forward_directions = np.hstack((forward_directions, np.array(0)))
        backward_directions = np.hstack((np.array(0), backward_directions))

        return forward_directions, backward_directions

    #Also add in sharp points to consider
    sharpness_cutoff = 6*acuteness_threshold
    forward_directions, backward_directions = calculate_directions(coordinates_nd)
    sharp_points = np.abs(((forward_directions - backward_directions) + np.pi) % (2 * np.pi) - np.pi) <= sharpness_cutoff
    sharp_points_idxs = np.where(sharp_points)[0]
    blip_apex_indices = list(set(blip_apex_indices + sharp_points_idxs.tolist()))

    #Process inouts, check direction once deviating consider endpoints of blip
    processed_inouts = []
    dir_thresh = sharpness_cutoff/2
    def angle_abs_disp(angle1, angle2):
        from math import pi
        displacement = angle2 - angle1
        displacement = (displacement + pi) % (2 * pi) - pi
        return abs(displacement)

    for blip_apex in blip_apex_indices:
        if blip_apex > len(path) - 2 or blip_apex < 1: continue
        lower, upper = blip_apex - 1, blip_apex + 1

        #Go until not same direction
        iters_since_manhatten = 0
        while lower > 0 and upper < len(path) - 1 and iters_since_manhatten < 5:
            #NOTE: Assuming first rev always true
            if angle_abs_disp(forward_directions[upper], backward_directions[lower]) <= dir_thresh:
                lower -= 1
                upper += 1
            elif angle_abs_disp(forward_directions[upper], backward_directions[lower + 1]) <= dir_thresh:
                #Allow upper to catch up
                upper += 1
            elif angle_abs_disp(forward_directions[upper - 1], backward_directions[lower]) <= dir_thresh:
                #Allow lower to catch up
                lower -= 1
            else: break

            #Make sure not running away!
            manhatten_dist_cur = manhatten_dist(path[lower], path[upper])
            if manhatten_dist_cur > manhatten_max_thickness:
                if manhatten_dist_cur > manhatten_max_thickness * 2: break
                iters_since_manhatten += 1
            else:
                iters_since_manhatten = 0

        #Spread until manhatten distance exceeded
        while lower > 0 and upper < len(path) - 1:
            if manhatten_dist(path[lower], path[upper]) <= manhatten_max_thickness:
                lower -= 1
                upper += 1
            elif manhatten_dist(path[lower + 1], path[upper]) <= manhatten_max_thickness:
                #Allow upper to catch up
                upper += 1
            elif manhatten_dist(path[lower], path[upper - 1]) <= manhatten_max_thickness:
                #Allow lower to catch up
                lower -= 1
            else: break
        processed_inouts.append([lower, upper])

    #TODO: SMooth dir look for blips in that??

    #Look for blippey deviations that missed the accuteness and sharpness checks
    abrupt_turns = np.abs(((forward_directions - (np.pi + backward_directions) % (2 * np.pi)) + np.pi)
                           % (2 * np.pi) - np.pi) > np.pi/3
    abrupt_turns_idxs = np.where(abrupt_turns)[0]

    for abrupt_turn in abrupt_turns_idxs:
        if abrupt_turn > len(path) - 2 or abrupt_turn < 1: continue

        in_dir = (np.pi + backward_directions[abrupt_turn]) % (2 * np.pi)
        out_dir_checks = forward_directions[abrupt_turn + 1:abrupt_turn + 100]
        out_dir_matches = np.abs(((out_dir_checks - in_dir) + np.pi) % (2 * np.pi) - np.pi) <= 0.9
        out_dir_match_indices = np.where(out_dir_matches)[0]
        if out_dir_match_indices.size > 0:
            sorted_indices = np.sort(out_dir_match_indices)
            for idx in sorted_indices:
                if manhatten_dist(path[abrupt_turn], path[abrupt_turn + 1 + idx]) <= manhatten_max_thickness:
                    processed_inouts.append([abrupt_turn, abrupt_turn + 1 + idx])
                    break

    #Conjoin as needed
    # Sort the index pairs by start index
    processed_inouts.sort(key=lambda x: x[0])

    conjoined_inouts = []
    current_pair = processed_inouts[0]

    for pair in processed_inouts[1:]:
        if pair[0] <= current_pair[1]:  # Overlapping
            current_pair[1] = max(current_pair[1], pair[1])  # Extend end index
        else:  # No overlap
            conjoined_inouts.append(current_pair)
            current_pair = pair

    conjoined_inouts.append(current_pair)  # Add the last pair

    #Re-build path without inouts
    processed_path = path[0:conjoined_inouts[0][0] + 1]
    for i in range(len(conjoined_inouts)):
        #NOTE: including the removed boundary so doesn't chop up path too much
        # processed_path.append([0,0])
        startpoint, endpoint = (conjoined_inouts[i][1],
                                conjoined_inouts[i + 1][0] + 1 if i < len(conjoined_inouts) - 1 else len(path))
        processed_path.extend(path[startpoint:endpoint])

    return processed_path
