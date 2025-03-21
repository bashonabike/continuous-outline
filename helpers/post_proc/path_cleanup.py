def remove_inout(path, manhatten_max_thickness=0, acuteness_threshold=0.15):
    import numpy as np
    def manhatten_dist(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def vectorized_triangle_areas(coordinates_nd):
        """Calculates the area of triangles formed by consecutive triplets of coordinates."""
        n = len(coordinates_nd)
        if n < 3:
            return np.array([])  # Not enough points for triangles

        x = coordinates_nd[:, 0]
        y = coordinates_nd[:, 1]

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
        p1, p2, p3 = np.stack((x1, y1), axis=1),np.stack((x2, y2), axis=1),np.stack((x3, y3), axis=1)
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
    blip_apex_indices = (np.where(combined_mask)[0] + 1).tolist()
    if len(blip_apex_indices) == 0:
        return path

    #Process inouts
    processed_inouts = []
    for blip_apex in blip_apex_indices:
        if blip_apex > len(path) - 2: continue
        apex_offset, apex_hardstop = 1, min(blip_apex, len(path) - 1 - blip_apex)
        while apex_offset < apex_hardstop:
            if manhatten_dist(path[blip_apex - apex_offset], path[blip_apex + apex_offset]) > manhatten_max_thickness:
                break
            apex_offset += 1
        # processed_inouts.append({"start_idx": blip_apex - apex_offset, "end_idx": blip_apex + apex_offset})
        processed_inouts.append([blip_apex - apex_offset, blip_apex + apex_offset])

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
    processed_path = path[0:conjoined_inouts[0][0]]
    for i in range(len(conjoined_inouts)):
        #NOTE: including 1 of the removed boundary so doesn't chop up path too much
        startpoint, endpoint = (conjoined_inouts[i][1],
                                conjoined_inouts[i + 1][0] if i < len(conjoined_inouts) - 1 else len(path))
        processed_path.extend(path[startpoint:endpoint])

    return processed_path
